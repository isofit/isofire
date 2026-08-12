#!/usr/bin/env python
#
#  Copyright 2024 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Authors: william.r.keely@jpl.nasa.gov
"""AVIRIS-3 multi-scene fire-temperature watcher (orthorectified products).

Reads one ``--input_dir`` holding ENVI files for one or more AV3 scenes, grouped
by scene id (``AV3<date><time>[_<chunk>]``), and processes a scene once its RDN,
LOC, GLT, BANDMASK/sat and OBS ENVI pairs are all present. Writes the
fire-temperature (E[T]) GeoTIFF to ``--fire_temp_dir`` as
``<scene_id>_fireTemp.tif`` and the uncertainty (sigma) GeoTIFF to
``--fire_temp_unc_dir`` as ``<scene_id>_fireTempUnc.tif``, optionally copying
both into ``--downlink_dir``.

This is the entry point used for the HPC runs in
https://arxiv.org/abs/2608.07580, and it is also the parent module for
onboard.py, which imports its shared I/O helpers. Algorithm and physics live in
retrieval.py and rt.py; this file is I/O and orchestration only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from spectral.io import envi

# Silence noisy JAX float64 warnings before importing jax.
warnings.filterwarnings("ignore", message=".*float64.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*64-bit.*", category=UserWarning)

import jax
jax.config.update("jax_enable_x64", False)

from .rt import radiative_transfer_engine
from .retrieval import (
    BAD_RANGES_NM_DEFAULT,
    DEFAULT_ALPHA_TUNING,
    DEFAULT_SIGMA_MIN_K,
    ETA_SWIR1_OVERRIDE_RANGES_NM_DEFAULT,
    ETA_SWIR2_DEFAULT_NM,
    HOT_RANGES_NM_DEFAULT,
    MixtureRetrievalConfig,
    SMOLDER_RANGES_NM_DEFAULT,
    SURFACE_RANGES_NM_DEFAULT,
    estimate_scene_water_vapor,
    load_fire_library_resampled,
    make_bitwise_bad_range_mask_1d,
    parse_water_vapors_csv,
    ranges_to_mask,
    run_moment_matched_retrieval,
)


# Auxiliary data file lookup

FIRE_LIB_FILENAME = "fire_library.csv"
SOLAR_IRR_FILENAME = "kurudz_0.1nm.dat"
SENSOR_WL_FWHM_FILENAME = "aviris3_wl_fwhm.npz"


def _resolve_repo_data_file(filename: str) -> Optional[str]:
    """Return the first on-disk path for an auxiliary data file, or None.

    Searches $ISOFIRE_DATA_DIR, the bundled package data dir, then the legacy
    in-repo data/ directory.
    """
    here = Path(__file__).resolve()
    dirs = []
    env = os.environ.get("ISOFIRE_DATA_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(here.parent / "data")        # bundled package data
    dirs.append(here.parents[2] / "data")    # legacy repo-root data/
    for d in dirs:
        p = d / filename
        if p.exists():
            return str(p)
    return None


DEFAULT_FIRE_LIB_CSV = _resolve_repo_data_file(FIRE_LIB_FILENAME)
DEFAULT_SOLAR_IRR_FILE = _resolve_repo_data_file(SOLAR_IRR_FILENAME)
DEFAULT_SENSOR_WL_FWHM_FILE = _resolve_repo_data_file(SENSOR_WL_FWHM_FILENAME)


# User knobs: defaults for commonly-tweaked CLI flags, all CLI-overridable.

# Single input dir holding all ENVI files for one or more scenes.
DEFAULT_INPUT_DIR: Optional[str] = None

# Output dirs for the fireTemp (E[T]) and fireTempUnc (sigma) GeoTIFFs.
DEFAULT_FIRE_TEMP_DIR: Optional[str] = None
DEFAULT_FIRE_TEMP_UNC_DIR: Optional[str] = None

# Optional extra staging dir; when set, both products are also copied here.
DEFAULT_DOWNLINK_DIR: str = ""

# Watcher poll interval; the watcher runs until killed.
USER_POLL_SECONDS: float = 10.0

# Single-Gaussian post-processing (sigma scaling).
USER_ALPHA_TUNING: float = DEFAULT_ALPHA_TUNING
USER_SIGMA_MIN_K: float = DEFAULT_SIGMA_MIN_K

# Last-band go/no-go: pixel retrieved only if rdn[..., -1] > USER_DARK_THRESH.
USER_DARK_THRESH: float = 0.15
USER_SKIP_DARK_PIXELS: bool = True

# Fill for in-scene pixels that were not retrieved; real E[T] and sigma are
# strictly positive, so -1 leaves 0 free to mark out-of-swath ortho pixels.
USER_FTP_FILL_K: float = -1.0    # fireTemp (E[T])
USER_FUQ_FILL_K: float = -1.0    # fireTempUnc (sigma)


# Sensor metadata helpers

def _ensure_nm(wl: np.ndarray) -> np.ndarray:
    """Return wavelengths in nanometers, converting from micrometers if needed."""
    wl = np.asarray(wl, dtype=np.float64)
    return wl * 1000.0 if np.nanmax(wl) < 20.0 else wl


def _ensure_fwhm_nm(fwhm: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    """Return FWHM in nanometers, converting from micrometers if needed."""
    fwhm = np.asarray(fwhm, dtype=np.float64)
    if np.nanmax(fwhm) < 0.1 and np.nanmax(wl_nm) > 100.0:
        return fwhm * 1000.0
    return fwhm


def _load_sensor_wl_fwhm(npz_path: str):
    """Load (wl_nm, fwhm_nm) from a .npz cache, forcing both to nanometers."""
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in ("wl_nm", "fwhm_nm") if k not in z.files]
        if missing:
            raise KeyError(
                f"{npz_path} missing required keys: {missing}. "
                f"Regenerate the cache from a header that carries fwhm."
            )
        wl   = _ensure_nm(np.asarray(z["wl_nm"],   dtype=np.float64))
        fwhm = _ensure_fwhm_nm(np.asarray(z["fwhm_nm"], dtype=np.float64), wl)
    if wl.shape != fwhm.shape:
        raise ValueError(
            f"{npz_path}: wl_nm and fwhm_nm length mismatch "
            f"({wl.shape} vs {fwhm.shape})"
        )
    return wl, fwhm


def _resolve_geometry(
    obs_full: np.ndarray,
    loc_full: np.ndarray,
    sza_arg: Optional[float],
    elev_arg: Optional[float],
    alt_arg: Optional[float],
) -> Tuple[float, float, float, float]:
    """Return (sza_deg, csza, elev_km, alt_km) from the OBS/LOC cubes.

    OBS is required and supplies the canonical SZA (band index 4); ``--sza_deg``
    may override it but the OBS cube must still be present.
    """
    if obs_full is None or obs_full.ndim != 3 or obs_full.shape[2] < 5:
        raise RuntimeError(
            "OBS cube is required to resolve scene geometry and is missing "
            "or malformed (need a 3-D BIP cube with at least 5 bands; SZA "
            "is read from band index 4)."
        )

    if sza_arg is not None and np.isfinite(sza_arg):
        sza = float(sza_arg)
    else:
        sza = float(np.nanmean(obs_full[..., 4]))
    if not np.isfinite(sza):
        raise RuntimeError("Resolved SZA is not finite.")
    csza = max(float(np.cos(np.deg2rad(sza))), 1e-3)

    if elev_arg is not None:
        elev_km = float(elev_arg)
    else:
        try:
            elev_km = float(np.nanmean(loc_full[..., 2])) / 1000.0
        except Exception:
            elev_km = 0.0
    if not np.isfinite(elev_km):
        elev_km = 0.0

    if alt_arg is not None:
        alt_km = float(alt_arg)
    else:
        try:
            mid = int(loc_full.shape[1] // 2)
            alt_km = float(np.nanmean(loc_full[:, mid, 2] + obs_full[:, mid, 0])) / 1000.0
            if not np.isfinite(alt_km):
                raise ValueError
        except Exception:
            alt_km = 5.0

    return sza, csza, elev_km, alt_km


# Scratch / LUT directory management

def _resolve_lut_dir(user_lut_dir: Optional[str]) -> str:
    """Return a fresh per-process scratch dir for 6S LUT inputs and outputs.

    The leaf name is unique across concurrent SLURM jobs/tasks and PIDs.
    """
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    task = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"
    token = f"job{job}_task{task}_pid{os.getpid()}"
    if user_lut_dir:
        d = os.path.join(str(user_lut_dir), token)
    else:
        d = os.path.join("/tmp", "isofire_lut", token)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


# GLT-based ortho output

# Out-of-swath / invalid GLT entries are flagged with 0 (hpc / older pyortho) or
# -9999 (onboard rtsp); neither is ever a valid 1-based source index.
GLT_INVALID_SENTINELS = (0, -9999)


def ortho_data(
    data: np.ndarray,
    glt: np.ndarray,
    glt_mask: Optional[np.ndarray] = None,
    glt_nodata=0,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """Orthorectify ``data`` through a stacked-band GLT (pyortho convention).

    The source index is ``abs(entry) - 1``, so negative (interpolated) entries
    stay valid, while entries equal to 0, -9999 or ``glt_nodata`` are filled
    with ``nodata_value``. ``data`` is (rows, cols, bands) or (rows, cols) and
    ``glt`` is (out_rows, out_cols, 2) with channel 0 = column, 1 = row.
    """
    do_squeeze = False
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
        do_squeeze = True

    out = np.zeros(
        (glt.shape[0], glt.shape[1], data.shape[2]), dtype=data.dtype
    ) + nodata_value

    passed = (glt_nodata,) if np.isscalar(glt_nodata) else tuple(glt_nodata)
    sentinels = {int(s) for s in (*GLT_INVALID_SENTINELS, *passed)}
    invalid_glt = np.zeros(glt.shape[:2], dtype=bool)
    for s in sentinels:
        invalid_glt |= np.any(glt == s, axis=-1)
    valid_glt = ~invalid_glt
    if glt_mask is not None:
        valid_glt = np.logical_and(valid_glt, glt_mask)

    glt_tmp = np.abs(glt).astype(int)
    glt_tmp[valid_glt] -= 1

    out[valid_glt, :] = data[
        np.abs(glt_tmp[valid_glt, 1]),  # source row (line)
        np.abs(glt_tmp[valid_glt, 0]),  # source col (sample)
        :,
    ]
    if do_squeeze:
        out = np.squeeze(out)
    return out


def export_ortho_geotiff(
    tif_file: str,
    data: np.ndarray,
    glt_samples: np.ndarray,
    glt_lines: np.ndarray,
    geotransform: Tuple[float, float, float, float, float, float],
    projection_wkt: str,
    nodata_value: float = 0.0,
    glt_nodata: float = 0.0,
) -> None:
    """Write a single-band float32 COG, ortho-warped via the GLT.

    NaNs are mapped to nodata and no flip is applied, since the on-disk L1B RDN
    is already in the orientation the GLT references. ``nodata=0`` marks
    out-of-swath: retrieved mu/sigma are strictly positive and non-retrieved
    pixels carry -1, so 0 is unambiguous.
    """
    data_clean = np.asarray(data, dtype=np.float32).copy()
    data_clean[~np.isfinite(data_clean)] = float(nodata_value)

    glt = np.stack([
        np.asarray(glt_samples, dtype=np.int32),  # column index
        np.asarray(glt_lines,   dtype=np.int32),  # row index
    ], axis=-1)

    out = ortho_data(
        data_clean, glt,
        glt_nodata=int(glt_nodata) if float(glt_nodata).is_integer() else float(glt_nodata),
        nodata_value=float(nodata_value),
    )
    out_rows, out_cols = out.shape

    # CRS policy: use the GLT WKT when it resolves to a known EPSG code,
    # otherwise normalize to EPSG:3857 so QGIS overlays against other products.
    if projection_wkt:
        try:
            wkt_crs = rasterio.crs.CRS.from_wkt(projection_wkt)
            wkt_epsg = wkt_crs.to_epsg()
            if wkt_epsg is not None:
                crs = wkt_crs
                crs_label = f"EPSG:{wkt_epsg}"
                crs_source = "GLT file"
            else:
                print(f"[ortho] WARNING: GLT WKT does not resolve to a known "
                      f"EPSG code (QGIS will see it as 'custom CRS 4096'); "
                      f"normalizing to EPSG:3857.")
                crs = rasterio.crs.CRS.from_epsg(3857)
                crs_label = "EPSG:3857"
                crs_source = "normalized from unresolvable GLT WKT"
        except Exception as _e_crs:
            print(f"[ortho] WARNING: GLT WKT failed to parse ({_e_crs}); "
                  f"falling back to EPSG:3857.")
            crs = rasterio.crs.CRS.from_epsg(3857)
            crs_label = "EPSG:3857"
            crs_source = "hardcoded fallback (GLT WKT unparseable)"
    else:
        crs = rasterio.crs.CRS.from_epsg(3857)
        crs_label = "EPSG:3857"
        crs_source = "hardcoded fallback (GLT had no CRS)"

    # Per-output summary: georeferencing, coverage and value distribution, so a
    # silent dtype/scaling/lookup bug shows up as out-of-range values.
    n_total = out_rows * out_cols
    nd = float(nodata_value)
    valid_mask = (out != nd) & np.isfinite(out)
    n_valid = int(valid_mask.sum())
    pct_valid = 100.0 * n_valid / max(n_total, 1)
    px_x = abs(float(geotransform[1]))
    px_y = abs(float(geotransform[5]))
    ulx = float(geotransform[0])
    uly = float(geotransform[3])
    basename = os.path.basename(tif_file)

    if n_valid > 0:
        valid_vals = out[valid_mask]
        v_min = float(valid_vals.min())
        v_med = float(np.median(valid_vals))
        v_max = float(valid_vals.max())
        v_summary = f"vals[min/med/max]=({v_min:.1f}, {v_med:.1f}, {v_max:.1f})"

        # Physical-range sanity, tagged by filename. _fireTempUnc contains
        # _fireTemp as a substring, so the Unc case MUST be tested first.
        bl = basename.lower()
        if "_firetempunc" in bl:
            lo, hi, kind = 0.0, 1000.0, "fireTempUnc (sigma, K)"
        elif "_firetemp" in bl:
            lo, hi, kind = 300.0, 2000.0, "fireTemp (effective fire T, K)"
        else:
            lo, hi, kind = None, None, None

        range_warn = ""
        if kind is not None:
            # Check only strictly-positive values; -1 (not retrieved) and 0
            # (out-of-swath) are not physical T/sigma.
            phys_vals = valid_vals[valid_vals > 0.0]
            if phys_vals.size > 0 and (float(phys_vals.min()) < lo
                                       or float(phys_vals.max()) > hi):
                range_warn = (f"  WARNING: {kind} values outside expected "
                              f"[{lo:.0f}, {hi:.0f}] K range -- possible "
                              f"dtype/scaling/GLT bug")
    else:
        v_summary = "vals[min/med/max]=(no valid pixels)"
        range_warn = ""

    print(f"[ortho] {basename}: shape={out_rows}x{out_cols} CRS={crs_label} "
          f"({crs_source}) px=({px_x:.2f}, {px_y:.2f})m ul=({ulx:.2f}, {uly:.2f}) "
          f"valid={n_valid}/{n_total} ({pct_valid:.2f}%) nodata={nd:g}  "
          f"{v_summary}{range_warn}")

    transform = rasterio.transform.Affine.from_gdal(*geotransform)
    os.makedirs(os.path.dirname(os.path.abspath(tif_file)), exist_ok=True)

    cog_profile = dict(
        driver="COG",
        height=out_rows,
        width=out_cols,
        count=1,
        dtype="float32",
        nodata=float(nodata_value),
        compress="deflate",
        predictor=3,
        blocksize=256,
        transform=transform,
        crs=crs,
    )
    with rasterio.open(tif_file, "w", **cog_profile) as dst:
        dst.write(out, 1)

    size_mb = os.path.getsize(tif_file) / 1024.0 / 1024.0
    print(f"[ortho] {basename}: wrote COG (DEFLATE+PREDICTOR=3), {size_mb:.2f} MB")


def _read_envi_header_field(hdr_path: str, field_name: str) -> Optional[str]:
    """Return the raw string value of an ENVI header field, or None.

    Handles single-line and braced multi-line values with a case-insensitive key
    match; braces are stripped and internal whitespace collapsed.
    """
    try:
        with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    key = field_name.strip().lower()
    for i, line in enumerate(lines):
        lhs, sep, rhs = line.partition("=")
        if not sep or lhs.strip().lower() != key:
            continue
        rhs = rhs.strip()
        if not rhs.startswith("{"):
            return rhs or None
        # Braced value; accumulate following lines until the closing brace.
        buf = rhs
        j = i
        while "}" not in buf and j + 1 < len(lines):
            j += 1
            buf += " " + lines[j].strip()
        start = buf.find("{")
        end = buf.find("}", start)
        inner = buf[start + 1: end] if end != -1 else buf[start + 1:]
        return " ".join(inner.split()) or None
    return None


def _check_glt_band_order(
    glt_samples: np.ndarray,
    glt_lines: np.ndarray,
    raw_rows: int,
    raw_cols: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Auto-correct a GLT whose sample/line bands are swapped vs. the RDN.

    Detected purely from the index ranges: if the arrays don't fit the
    (cols, rows) assignment but do fit the swapped one, they are swapped.
    Returns ``(samples, lines, swapped)``.
    """
    s_max = int(np.abs(glt_samples).max()) if glt_samples.size else 0
    l_max = int(np.abs(glt_lines).max()) if glt_lines.size else 0

    fits_as_is   = (s_max <= raw_cols) and (l_max <= raw_rows)
    fits_swapped = (l_max <= raw_cols) and (s_max <= raw_rows)

    if raw_rows != raw_cols and not fits_as_is and fits_swapped:
        return glt_lines, glt_samples, True
    return glt_samples, glt_lines, False


def _open_glt(glt_hdr: str) -> Tuple[np.ndarray, np.ndarray, Tuple, str, float]:
    """Read a 2-band ENVI GLT into (samples, lines, geotransform, wkt, nodata).

    Band 1 is samples and band 2 is lines; the caller runs
    :func:`_check_glt_band_order` once it knows the raw image dims. The CRS comes
    from rasterio, falling back to the ENVI 'coordinate system string'.
    """
    img_path = os.path.splitext(glt_hdr)[0]
    if not os.path.exists(img_path):
        for cand in (img_path, img_path + ".img", img_path + ".dat"):
            if os.path.exists(cand):
                img_path = cand
                break

    with rasterio.open(img_path) as src:
        if src.count < 2:
            raise RuntimeError(f"GLT {glt_hdr} has only {src.count} bands; expected >=2.")
        glt_samples = src.read(1).astype(np.int64)
        glt_lines   = src.read(2).astype(np.int64)
        gt = src.transform.to_gdal()
        wkt = src.crs.to_wkt() if src.crs is not None else ""

    # Fall back to the ENVI 'coordinate system string' when rasterio gave no CRS.
    if not wkt:
        css = _read_envi_header_field(glt_hdr, "coordinate system string")
        if css:
            wkt = css

    # Parse the 'data ignore value' (the GLT nodata sentinel); default 0.
    raw_nodata = _read_envi_header_field(glt_hdr, "data ignore value")
    try:
        glt_nodata = float(raw_nodata) if raw_nodata is not None else 0.0
    except ValueError:
        glt_nodata = 0.0
    return glt_samples, glt_lines, gt, wkt, glt_nodata


# HFDI pre-fit fire screen

# Hyperspectral Fire Detection Index:
#     HFDI = (L(2430 nm) - L(2061 nm)) / (L(2430 nm) + L(2061 nm))
# Non-fire surfaces are brighter at 2061 nm, so HFDI sits well below zero, while
# thermal emission from fire lifts the 2430 nm end and drives the index up.
HFDI_LO_NM_DEFAULT = 2061.0
HFDI_HI_NM_DEFAULT = 2430.0
HFDI_MIN_DEFAULT = -0.1

# Nearest band must land within this distance of each target, else the screen is
# disabled rather than indexing the wrong part of the spectrum.
HFDI_BAND_DISTANCE_MAX_NM = 5.0


def _nearest_band(wl_nm: np.ndarray, target_nm: float) -> Tuple[int, float]:
    """(index, actual wavelength) of the sensor band nearest ``target_nm``."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    idx = int(np.argmin(np.abs(wl - float(target_nm))))
    return idx, float(wl[idx])


def _band_saturated(bm_flat: np.ndarray, bands: int, band_idx: int) -> np.ndarray:
    """Per-pixel saturation flag for ONE band, without unpacking the full mask.

    Packed bits are indexed directly (band i is byte i // 8, bit 7 - (i % 8),
    matching numpy's MSB-first unpackbits), avoiding a full per-band array.
    """
    bm = np.asarray(bm_flat, dtype=np.uint8)
    bm = bm.reshape(bm.shape[0], -1) if bm.ndim > 1 else bm[:, None]
    if bm.shape[1] == int(bands):
        return bm[:, int(band_idx)] != 0
    byte = int(band_idx) // 8
    bit = 7 - (int(band_idx) % 8)
    if byte >= bm.shape[1]:
        raise RuntimeError(
            f"bandmask has {bm.shape[1]} bytes; band {band_idx} of {bands} "
            f"needs byte {byte}."
        )
    return ((bm[:, byte] >> bit) & 1).astype(bool)


def compute_hfdi(
    rdn: np.ndarray,
    wl_nm: np.ndarray,
    *,
    lo_nm: float = HFDI_LO_NM_DEFAULT,
    hi_nm: float = HFDI_HI_NM_DEFAULT,
) -> Tuple[np.ndarray, Tuple[int, float], Tuple[int, float]]:
    """Per-pixel HFDI at the sensor bands nearest each target.

    Returns (hfdi_flat, (lo_idx, lo_actual_nm), (hi_idx, hi_actual_nm)); HFDI is
    NaN where the denominator collapses, which the screen drops as non-fire.
    """
    n_bands = int(np.asarray(wl_nm).size)
    rdn2 = np.asarray(rdn, dtype=np.float32).reshape(-1, n_bands)
    lo_idx, lo_act = _nearest_band(wl_nm, lo_nm)
    hi_idx, hi_act = _nearest_band(wl_nm, hi_nm)

    L_lo = rdn2[:, lo_idx].astype(np.float64)
    L_hi = rdn2[:, hi_idx].astype(np.float64)
    denom = L_hi + L_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        hfdi = np.where(np.abs(denom) > 1e-12, (L_hi - L_lo) / denom, np.nan)
    return hfdi.astype(np.float32), (lo_idx, lo_act), (hi_idx, hi_act)


def hfdi_screen(
    rdn: np.ndarray,
    wl_nm: np.ndarray,
    bm: np.ndarray,
    *,
    hfdi_min: float = HFDI_MIN_DEFAULT,
    lo_nm: float = HFDI_LO_NM_DEFAULT,
    hi_nm: float = HFDI_HI_NM_DEFAULT,
    band_distance_max_nm: float = HFDI_BAND_DISTANCE_MAX_NM,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-fit fire screen. Returns (keep_flat, hfdi_flat), both (n_pix,).

    A pixel is kept when ``HFDI > hfdi_min``; saturation in either index band
    auto-passes, since clipped radiance makes HFDI meaningless and a pixel
    saturating at 2430 nm is a strong fire candidate. If either target is further
    than ``band_distance_max_nm`` from the nearest band the screen is disabled.
    """
    n_bands = int(np.asarray(wl_nm).size)
    hfdi, (lo_idx, lo_act), (hi_idx, hi_act) = compute_hfdi(
        rdn, wl_nm, lo_nm=lo_nm, hi_nm=hi_nm
    )

    d_lo = abs(lo_act - float(lo_nm))
    d_hi = abs(hi_act - float(hi_nm))
    if d_lo > float(band_distance_max_nm) or d_hi > float(band_distance_max_nm):
        if verbose:
            print(f"[hfdi] WARNING: nearest sensor bands are far from the index "
                  f"targets ({lo_nm:.1f} nm -> {lo_act:.2f}, dist={d_lo:.2f}; "
                  f"{hi_nm:.1f} nm -> {hi_act:.2f}, dist={d_hi:.2f}); "
                  f"HFDI screen DISABLED for this scene.")
        return np.ones(hfdi.shape, dtype=bool), hfdi

    bm_flat = np.asarray(bm, dtype=np.uint8).reshape(hfdi.size, -1)
    sat_either = (_band_saturated(bm_flat, n_bands, lo_idx)
                  | _band_saturated(bm_flat, n_bands, hi_idx))

    finite = np.isfinite(hfdi)
    keep = (finite & (hfdi > float(hfdi_min))) | sat_either

    if verbose:
        n_tot = int(hfdi.size)
        n_keep = int(keep.sum())
        n_sat = int((sat_either & ~(finite & (hfdi > float(hfdi_min)))).sum())
        med_keep = float(np.median(hfdi[keep & finite])) if int((keep & finite).sum()) else float("nan")
        med_drop = float(np.median(hfdi[~keep & finite])) if int((~keep & finite).sum()) else float("nan")
        print(f"[hfdi] L({hi_act:.2f} nm) vs L({lo_act:.2f} nm), threshold "
              f"HFDI > {float(hfdi_min):g}: keeping {n_keep}/{n_tot} pixels "
              f"({100.0 * n_keep / max(n_tot, 1):.2f}%); {n_sat} kept on the "
              f"saturation auto-pass; median HFDI kept={med_keep:.3f} "
              f"dropped={med_drop:.3f}")
    return keep, hfdi


# Scene discovery + watcher

# Scene id from filename: AV3 base (date+time) plus an optional _NNN chunk, so
# chunked and unchunked siblings stay distinct.
_SCENE_RE = re.compile(r"^(AV3\d{8}[tT]\d{6})(?:_(\d{3,}))?")

# Free-text AV3 scene id search inside a GLT header 'description'.
_GLT_DESC_SCENE_RE = re.compile(r"AV3\d{8}[tT]\d{6}(?:_\d{3,})?")

REQUIRED_KIND_KEYS = ("rdn", "loc", "glt", "mask", "obs")

# Suffix aliases per kind (case-insensitive). BANDMASK and sat both map to mask,
# and RDN excludes _BANDMASK/_SAT/_FLATFIELD tails so those route to mask.
KIND_SUFFIX_ALIASES = {
    "rdn":  ("_RDN",),
    "loc":  ("_LOC",),
    "glt":  ("_GLT",),
    "mask": ("_BANDMASK", "_SAT"),
    "obs":  ("_OBS",),
}
_RDN_EXCLUDE_TAILS = ("_bandmask", "_sat", "_flatfield")


def _parse_scene_id(name: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return (scene_id, base, chunk) for a filename, or None if not AV3.

    ``scene_id`` is ``base`` alone, or ``base_chunk`` when a chunk number is
    present.
    """
    m = _SCENE_RE.match(os.path.basename(name))
    if not m:
        return None
    base = m.group(1)
    chunk = m.group(2)
    scene_id = f"{base}_{chunk}" if chunk else base
    return scene_id, base, chunk


def _scene_id_from_glt_description(glt_hdr: str) -> Optional[str]:
    """Pull an AV3 scene id out of the GLT header 'description', or None.

    The description records the canonical scene id, used for output naming only;
    file pairing stays on the filename id.
    """
    desc = _read_envi_header_field(glt_hdr, "description")
    if not desc:
        return None
    m = _GLT_DESC_SCENE_RE.search(desc)
    return m.group(0) if m else None


def _classify_hdr(hdr_path: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """Classify an ENVI .hdr filename into (kind, scene_id, base, chunk).

    Returns None if the name isn't an AV3 scene or lacks a known kind suffix.
    """
    bn = os.path.basename(hdr_path)
    bn_lower = bn.lower()
    if not bn_lower.endswith(".hdr"):
        return None
    stem = bn[: bn_lower.rindex(".hdr")]
    parsed = _parse_scene_id(stem)
    if parsed is None:
        return None
    scene_id, base, chunk = parsed
    stem_lower = stem.lower()
    for kind, suffixes in KIND_SUFFIX_ALIASES.items():
        for suf in suffixes:
            if stem_lower.endswith(suf.lower()):
                if kind == "rdn" and any(
                    stem_lower.endswith(bad) for bad in _RDN_EXCLUDE_TAILS
                    if bad != suf.lower()
                ):
                    continue
                return kind, scene_id, base, chunk
    return None


def _envi_binary_for_hdr(hdr_path: str) -> Optional[str]:
    """Return the ENVI binary cube paired with an .hdr, or None if absent.

    Tries the no-extension companion first, then common extension variants.
    """
    if not hdr_path.lower().endswith(".hdr"):
        return None
    stem = hdr_path[: hdr_path.lower().rindex(".hdr")]
    candidates = [stem] + [stem + ext for ext in (".img", ".dat", ".bin",
                                                  ".bil", ".bsq", ".bip")]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def discover_ready_chunks(*, input_dir: str) -> Tuple[dict, dict]:
    """Return (ready, partial) scene maps from a single input directory.

    Groups ENVI .hdr files by (scene_id, kind), counting one only when its binary
    cube exists, and drops any pair matching more than one file. A scene missing
    a required kind lands in ``partial[scene_id]`` as the list of missing kinds.
    """
    import glob as _glob
    if not input_dir or not os.path.isdir(input_dir):
        return {}, {}

    hits = sorted(set(_glob.glob(os.path.join(input_dir, "*.hdr"))
                      + _glob.glob(os.path.join(input_dir, "*.HDR"))))

    by_scene: Dict[str, Dict[str, str]] = {}
    dupe_keys: set = set()
    for h in hits:
        cls = _classify_hdr(h)
        if cls is None:
            continue
        kind, scene_id, _base, _chunk = cls
        # Register only when the binary cube is present.
        if _envi_binary_for_hdr(h) is None:
            continue
        slot = by_scene.setdefault(scene_id, {})
        if kind in slot:
            dupe_keys.add((scene_id, kind))
            continue
        slot[kind] = h

    # Drop ambiguous (scene_id, kind) pairs.
    for scene_id, kind in dupe_keys:
        by_scene.get(scene_id, {}).pop(kind, None)

    ready: dict = {}
    partial: dict = {}
    for scene_id, kinds in sorted(by_scene.items()):
        missing = [k for k in REQUIRED_KIND_KEYS if k not in kinds]
        if missing:
            partial[scene_id] = missing
            continue
        ready[scene_id] = {k: kinds[k] for k in REQUIRED_KIND_KEYS}
    return ready, partial


def _resolve_output_scene_id(rdn_scene_id: str, glt_hdr: Optional[str]) -> str:
    """Scene id used to NAME the output fireTemp/fireTempUnc products.

    Products are named from the GLT 'description' when it parses, falling back to
    the filename id; pairing and the per-date 6S LUT cache key always use the
    filename id.
    """
    if glt_hdr:
        desc_id = _scene_id_from_glt_description(glt_hdr)
        if desc_id is not None:
            return desc_id
    return rdn_scene_id


def _output_paths_for_chunk(scene_id: str, fire_temp_dir: str,
                            fire_temp_unc_dir: str) -> Tuple[str, str]:
    """Return the (fireTemp, fireTempUnc) tif paths for a scene id."""
    return (
        os.path.join(fire_temp_dir,     f"{scene_id}_fireTemp.tif"),
        os.path.join(fire_temp_unc_dir, f"{scene_id}_fireTempUnc.tif"),
    )


def _delete_scene_inputs(paths: dict) -> int:
    """Delete the ENVI inputs (.hdr + binary cube) for a processed scene.

    Best-effort: a failed delete logs a WARNING and does not abort. Returns the
    number of files removed.
    """
    removed = []
    for hdr in paths.values():
        if not hdr:
            continue
        targets = [hdr]
        binary = _envi_binary_for_hdr(hdr)
        if binary and binary not in targets:
            targets.append(binary)
        for t in targets:
            try:
                os.remove(t)
                removed.append(t)
            except OSError as e:
                print(f"[cleanup] WARNING: could not delete {t}: {e!r}")
    if removed:
        print(f"[cleanup] deleted {len(removed)} L1B input file(s): "
              + ", ".join(sorted(os.path.basename(p) for p in removed)))
    return len(removed)


# Per-scene pipeline

class _RteBuilder:
    """Date-cached 6S RTE builder.

    Builds one 6S LUT per calendar date and reuses it for every scene flown that
    date, logging how far each scene's geometry has drifted from the geometry the
    LUT was built at. Cached scratch dirs live until :meth:`cleanup`.
    """

    # Drift thresholds (per cache hit) above which a warning is logged.
    SZA_DRIFT_WARN_DEG: float = 5.0
    ELEV_DRIFT_WARN_KM: float = 0.5
    ALT_DRIFT_WARN_KM: float = 1.0

    def __init__(self, lut_root: str, sixs_dir: str, water_vapors: list,
                 water_aot550: float, water_o3: float):
        self.lut_root = lut_root
        self.sixs_dir = sixs_dir
        self.water_vapors = water_vapors
        self.water_aot550 = water_aot550
        self.water_o3 = water_o3
        self._cache: Dict[str, dict] = {}   # date_key -> rte/lut/geometry/signature

    @staticmethod
    def _date_key(scene_id: str) -> str:
        """Extract YYYYMMDD from a scene_id ``AV3<YYYYMMDD>[tT]...``.

        The separator is matched case-insensitively so hpc (GPS-time) and onboard
        (system-time) ids share a date key; falls back to the full scene_id.
        """
        m = re.match(r"AV3(\d{8})[tT]", str(scene_id))
        return m.group(1) if m else str(scene_id)

    @staticmethod
    def _spectral_signature(wl_nm: np.ndarray) -> tuple:
        """Fingerprint a sensor spectral grid to catch a build mixup on a cache hit."""
        wl = np.asarray(wl_nm, dtype=np.float64)
        return (int(wl.size), round(float(wl[0]), 3), round(float(wl[-1]), 3))

    def build(self, scene_id: str, sza: float, elev_km: float, alt_km: float,
              wl_nm: np.ndarray, fwhm_nm: np.ndarray) -> Tuple[dict, str]:
        """Return a date-cached RTE for this scene, building on a cache miss.

        Returns (rte, lut_subdir); the caller must not rmtree lut_subdir, which
        the builder owns until :meth:`cleanup`.
        """
        date_key = self._date_key(scene_id)
        sig_now = self._spectral_signature(wl_nm)

        entry = self._cache.get(date_key)
        if entry is not None:
            # Spectral grid changing mid-day means a sensor reconfig; rebuild.
            if entry["spectral_signature"] != sig_now:
                print(f"[rt] WARNING: date {date_key} cached LUT has spectral "
                      f"signature {entry['spectral_signature']} but scene "
                      f"{scene_id} reports {sig_now}. Rebuilding from scratch.")
                self._evict(date_key)
                entry = None
            else:
                # Geometry-drift warnings; the cache is still served.
                dsza = abs(float(sza) - entry["sza"])
                delev = abs(float(elev_km) - entry["elev_km"])
                dalt = abs(float(alt_km) - entry["alt_km"])
                drift_parts = []
                if dsza > self.SZA_DRIFT_WARN_DEG:
                    drift_parts.append(f"SZA {entry['sza']:.2f}->{sza:.2f} deg (|d|={dsza:.2f})")
                if delev > self.ELEV_DRIFT_WARN_KM:
                    drift_parts.append(f"elev {entry['elev_km']:.2f}->{elev_km:.2f} km (|d|={delev:.2f})")
                if dalt > self.ALT_DRIFT_WARN_KM:
                    drift_parts.append(f"alt {entry['alt_km']:.2f}->{alt_km:.2f} km (|d|={dalt:.2f})")
                if drift_parts:
                    print(f"[rt] cache HIT date={date_key} for scene={scene_id}, but "
                          f"geometry drifted from cached build: " + "; ".join(drift_parts)
                          + ". Cached LUT reused; if accuracy matters here, tighten "
                          f"the cache key to (date, sza_bin).")
                else:
                    print(f"[rt] cache HIT date={date_key} for scene={scene_id}")
                return entry["rte"], entry["lut_subdir"]

        # Cache miss (or invalidated): build a fresh LUT for this date.
        sub = os.path.join(
            self.lut_root,
            f"date{date_key}_sza{sza:.4f}_elev{elev_km:.4f}_alt{alt_km:.4f}",
        )
        os.makedirs(sub, exist_ok=True)
        print(f"[rt] cache MISS date={date_key} for scene={scene_id}; "
              f"building LUT at {sub}")
        sixs_dict = dict(
            aermodel=1, AOT550=float(self.water_aot550), O3=float(self.water_o3),
            H2OSTR=float(self.water_vapors[0]),
            day=1, month=1,
            elev=float(elev_km), alt=float(alt_km),
            wlinf=float(wl_nm[0] / 1000.0), wlsup=float(wl_nm[-1] / 1000.0),
            solzen=float(sza), viewzen=0.0, solaz=0.0, viewaz=0.0,
        )
        rte = radiative_transfer_engine(
            sub, [self.water_vapors], ["H2OSTR"], sixs_dict, wl_nm, fwhm_nm,
            self.sixs_dir, multipart_transmittance=True,
        )
        self._cache[date_key] = dict(
            rte=rte,
            lut_subdir=sub,
            sza=float(sza),
            elev_km=float(elev_km),
            alt_km=float(alt_km),
            spectral_signature=sig_now,
            built_for_scene=str(scene_id),
        )
        return rte, sub

    def _evict(self, date_key: str) -> None:
        """Drop a cached entry and rmtree its scratch dir; safe on unknown keys."""
        entry = self._cache.pop(date_key, None)
        if entry is None:
            return
        sub = entry.get("lut_subdir")
        if sub and os.path.isdir(sub):
            shutil.rmtree(sub, ignore_errors=True)

    def cleanup(self) -> None:
        """Release every cached RTE and rmtree all cached LUT scratch dirs."""
        if not self._cache:
            return
        n = len(self._cache)
        for date_key in list(self._cache.keys()):
            self._evict(date_key)
        print(f"[rt] cleanup: removed {n} cached LUT scratch dir(s)")


def process_chunk(
    chunk: str,
    paths: dict,
    args,
    rte_builder: "_RteBuilder",
) -> Tuple[Optional[str], Optional[str]]:
    """Run the retrieval for one ready scene and write its fireTemp/fireTempUnc TIFFs.

    Returns (fire_temp_path, fire_temp_unc_path) on success; raises if a required
    ENVI cube is missing or malformed.
    """
    t_chunk = time.perf_counter()
    print("")
    print(f"[scene] === {chunk} ===")
    print(f"  rdn  = {paths['rdn']}")
    print(f"  loc  = {paths['loc']}")
    print(f"  glt  = {paths['glt']}")
    print(f"  mask = {paths['mask']}")
    print(f"  obs  = {paths['obs']}")

    if not paths.get("obs"):
        # Defensive: discover_ready_chunks keeps OBS-less scenes in `partial`.
        raise RuntimeError(
            f"scene {chunk}: OBS file is required but not present in `paths`. "
            "discover_ready_chunks should have left this scene in `partial`."
        )

    def _scene_id_from_path(p: str) -> Optional[str]:
        parsed = _parse_scene_id(os.path.basename(p))
        return parsed[0] if parsed else None

    scene_id_from_rdn = _scene_id_from_path(paths["rdn"])
    if scene_id_from_rdn is None:
        raise RuntimeError(
            f"scene {chunk}: could not parse a scene id from RDN filename "
            f"{os.path.basename(paths['rdn'])!r}; expected pattern "
            f"AV3<YYYYMMDD>[tT]<HHMMSS>[_NNN]."
        )

    # The filename id drives file pairing and the per-date 6S LUT cache key; the
    # GLT 'description' id, when present, names the output TIFFs.
    scene_id = scene_id_from_rdn

    glt_desc_id = _scene_id_from_glt_description(paths["glt"]) if paths.get("glt") else None
    if glt_desc_id is not None:
        output_scene_id = glt_desc_id
        if glt_desc_id != scene_id_from_rdn:
            print(f"[scene] naming outputs from GLT 'description' scene_id="
                  f"{glt_desc_id!r} (RDN/system-time id is {scene_id_from_rdn!r}).")
        else:
            print(f"[scene] GLT 'description' scene_id={glt_desc_id!r} matches the "
                  f"RDN/system-time id; using it for output naming.")
    else:
        output_scene_id = scene_id_from_rdn
        raw_desc = _read_envi_header_field(paths["glt"], "description") if paths.get("glt") else None
        print(f"[scene] no parseable scene id in the GLT 'description' "
              f"(description={raw_desc!r}); naming outputs from the RDN/system-time "
              f"id {scene_id_from_rdn!r}.")

    # Every input file should derive the same filename scene_id; disagreement is
    # almost certainly an upstream-naming bug.
    per_kind_ids = {kind: _scene_id_from_path(paths[kind]) for kind in paths}
    mismatched = {k: sid for k, sid in per_kind_ids.items() if sid != scene_id_from_rdn}
    if mismatched:
        print(f"[scene] WARNING: scene id mismatch across input files. "
              f"RDN derives scene_id={scene_id_from_rdn!r}, but: "
              + ", ".join(f"{k}={sid!r}" for k, sid in mismatched.items())
              + ". Using the RDN's scene_id for pairing; investigate the upstream "
              "writer if this fires.")

    if scene_id_from_rdn != chunk:
        print(f"[scene] WARNING: watcher grouped this scene under chunk={chunk!r}, "
              f"but the RDN filename derives scene_id={scene_id_from_rdn!r}.")
    print(f"[scene] processing scene_id = {scene_id!r} (system time); "
          f"output scene_id = {output_scene_id!r}  "
          f"(outputs: {output_scene_id}_fireTemp.tif, {output_scene_id}_fireTempUnc.tif)")

    rdn_ds = envi.open(paths["rdn"])
    mask_ds = envi.open(paths["mask"])
    loc_ds = envi.open(paths["loc"])
    obs_ds = envi.open(paths["obs"])

    rdn_full = rdn_ds.open_memmap(interleave="bip")
    bm_full = mask_ds.open_memmap(interleave="bip")
    loc_full = loc_ds.open_memmap(interleave="bip")
    obs_full = obs_ds.open_memmap(interleave="bip")

    wl_nm = _ensure_nm(np.array([float(x) for x in rdn_ds.metadata["wavelength"]], dtype=np.float64))

    # FWHM source priority: the RDN header 'fwhm' field, else the
    # --sensor_wl_fwhm .npz cache (interpolated if band counts differ), else error.
    if "fwhm" in rdn_ds.metadata:
        fwhm_nm = _ensure_fwhm_nm(
            np.array([float(x) for x in rdn_ds.metadata["fwhm"]], dtype=np.float64),
            wl_nm,
        )
        print(f"[spec] using header fwhm ({wl_nm.size} bands)")
    else:
        wl_npz = getattr(args, "sensor_wl_fwhm", None)
        if not wl_npz or not os.path.exists(wl_npz):
            raise ValueError(
                f"RDN header {paths['rdn']} has no 'fwhm' field and no "
                f"--sensor_wl_fwhm cache was supplied. Onboard AVIRIS-3 L1B "
                f"headers do not carry FWHM; generate a cache (.npz with keys "
                f"wl_nm, fwhm_nm) and pass its path via --sensor_wl_fwhm."
            )
        wl_cache_npz, fwhm_cache_npz = _load_sensor_wl_fwhm(wl_npz)
        if fwhm_cache_npz.size == wl_nm.size:
            fwhm_nm = fwhm_cache_npz.copy()
            print(f"[spec] header lacks fwhm; cache shapes match, using cache directly: "
                  f"{wl_npz} ({fwhm_nm.size} bands)")
        else:
            # np.interp needs ascending xp (the hpc-derived cache is); out-of-
            # range queries clip to endpoints, safe at the ~1 nm grid-edge offsets.
            fwhm_nm = np.interp(wl_nm, wl_cache_npz, fwhm_cache_npz)
            donor_max = float(np.abs(
                wl_nm - wl_cache_npz[np.abs(wl_nm[:, None] - wl_cache_npz[None, :]).argmin(axis=1)]
            ).max())
            print(f"[spec] header lacks fwhm; interpolated cache fwhm "
                  f"({wl_cache_npz.size} bands) onto header wl ({wl_nm.size} bands); "
                  f"max donor distance {donor_max:.3f} nm")

    # FWHM/WL pairing sanity print. AVIRIS-3 fwhm is nearly uniform so pairing
    # can't be verified rigorously, but this catches wrong units and unpairing.
    _step_nm_med = float(np.median(np.abs(np.diff(wl_nm)))) if wl_nm.size >= 2 else 0.0
    _fwhm_min = float(np.min(fwhm_nm)); _fwhm_med = float(np.median(fwhm_nm))
    _fwhm_max = float(np.max(fwhm_nm))
    print(f"[spec] wl/fwhm pairing: wl[0]={wl_nm[0]:.2f} nm fwhm[0]={fwhm_nm[0]:.3f} nm | "
          f"wl[-1]={wl_nm[-1]:.2f} nm fwhm[-1]={fwhm_nm[-1]:.3f} nm")
    print(f"[spec] fwhm stats (nm): min={_fwhm_min:.3f} median={_fwhm_med:.3f} "
          f"max={_fwhm_max:.3f}; median band step={_step_nm_med:.3f}; "
          f"fwhm/step ratio (median)={_fwhm_med / _step_nm_med if _step_nm_med > 0 else float('nan'):.3f}")
    _fwhm_warns = []
    if _fwhm_min < 0.5:
        _fwhm_warns.append(f"min fwhm={_fwhm_min:.3f} nm is implausibly small (<0.5 nm); "
                           f"unit conversion or pairing may be wrong")
    if _fwhm_max > 50.0:
        _fwhm_warns.append(f"max fwhm={_fwhm_max:.3f} nm is implausibly large (>50 nm)")
    if _step_nm_med > 0 and not (0.5 <= (_fwhm_med / _step_nm_med) <= 3.0):
        _fwhm_warns.append(f"fwhm/step ratio {_fwhm_med / _step_nm_med:.3f} is outside typical "
                           f"0.5-3.0 range; fwhm may be in wrong units or unpaired from wl")
    if _fwhm_warns:
        for _w in _fwhm_warns:
            print(f"[spec] WARNING: {_w}")

    # Onboard RDN headers list wavelengths SWIR-first, but downstream code assumes
    # ascending, so reverse wl/fwhm, the radiance cube and the bandmask. A
    # bit-packed bandmask is unpacked, reversed and repacked, since reversing
    # whole bytes does not reverse the bits within each byte.
    if wl_nm.size >= 2 and wl_nm[0] > wl_nm[-1]:
        print(f"[spec] RDN bands stored in descending wavelength order "
              f"({wl_nm[0]:.2f} -> {wl_nm[-1]:.2f} nm); reversing band axis "
              f"so downstream code sees ascending wavelengths.")
        wl_nm = wl_nm[::-1].copy()
        fwhm_nm = fwhm_nm[::-1].copy()
        rdn_full = rdn_full[..., ::-1]
        nb_full = int(wl_nm.size)
        bm_arr = np.asarray(bm_full)
        if bm_arr.ndim >= 1 and bm_arr.shape[-1] == nb_full:
            bm_full = bm_arr[..., ::-1]
        else:
            orig_lead_shape = bm_arr.shape[:-1]
            bits = np.unpackbits(bm_arr.astype(np.uint8), axis=-1)[..., :nb_full]
            bits = bits[..., ::-1]
            pad = (-nb_full) % 8
            if pad:
                pad_shape = orig_lead_shape + (pad,)
                bits = np.concatenate(
                    [bits, np.zeros(pad_shape, dtype=np.uint8)], axis=-1,
                )
            bm_full = np.packbits(bits, axis=-1)

    sza, csza, elev_km, alt_km = _resolve_geometry(
        obs_full, loc_full, args.sza_deg, args.elev_km, args.alt_km
    )
    print(f"[geom] SZA={sza:.3f} deg  cos(SZA)={csza:.4f}  elev={elev_km:.3f} km  alt={alt_km:.3f} km")

    irr = np.loadtxt(args.solar_irr_file, comments="#")
    iwl, irr = irr.T
    irr = irr / 10.0
    from .rt import resample_spectrum
    solar_irr = resample_spectrum(irr, iwl, wl_nm, fwhm_nm).astype(np.float64)

    # Build or fetch the date-cached RTE. The returned dict is the in-memory
    # per-pixel lookup cache; rte_builder owns the on-disk LUT subdir.
    rte, lut_subdir = rte_builder.build(scene_id, sza, elev_km, alt_km, wl_nm, fwhm_nm)

    try:
        print("[water] estimating per-pixel water vapor and applying super-pixel median")
        water = estimate_scene_water_vapor(
            rdn_full=np.asarray(rdn_full, dtype=np.float32),
            wl_nm=wl_nm,
            rte=rte,
            solar_irr=solar_irr,
            sza_deg=sza,
            water_vapors=rte_builder.water_vapors,
            superpixel_size=int(args.water_superpixel_size),
        )

        lib_spectra = load_fire_library_resampled(args.fire_lib_csv, wl_nm, fwhm_nm).astype(np.float64)

        bit_mask_1d = make_bitwise_bad_range_mask_1d(wl_nm, BAD_RANGES_NM_DEFAULT)
        surface_1d = bit_mask_1d & ranges_to_mask(wl_nm, _parse_ranges(args.surface_window_ranges_nm, SURFACE_RANGES_NM_DEFAULT))
        hot_1d = bit_mask_1d & ranges_to_mask(wl_nm, _parse_ranges(args.hot_window_ranges_nm, HOT_RANGES_NM_DEFAULT))
        smolder_1d = bit_mask_1d & ranges_to_mask(wl_nm, _parse_ranges(args.smolder_window_ranges_nm, SMOLDER_RANGES_NM_DEFAULT))
        eta_swir2_1d = ranges_to_mask(
            wl_nm,
            np.array([[float(args.eta_swir2_lo_nm), float(args.eta_swir2_hi_nm)]], dtype=np.float64),
        )
        eta_swir1_1d = ranges_to_mask(wl_nm, ETA_SWIR1_OVERRIDE_RANGES_NM_DEFAULT)

        cfg = MixtureRetrievalConfig(
            Tmin=args.Tmin, Tmax=args.Tmax, N_T=args.N_T, T_step1_init=args.T_step1_init,
            mu_hot_min_T=args.mu_hot_min_T, mu_hot_max_T=args.mu_hot_max_T,
            mu_smol_min_T=args.mu_smol_min_T, mu_smol_max_T=args.mu_smol_max_T,
            sigma_hot_min=args.sigma_hot_min, sigma_hot_max=args.sigma_hot_max,
            sigma_smol_min=args.sigma_smol_min, sigma_smol_max=args.sigma_smol_max,
            alpha1_min=args.alpha1_min, alpha1_max=args.alpha1_max,
            alpha2_min=args.alpha2_min, alpha2_max=args.alpha2_max,
            eta_min=args.eta_min, eta_max=args.eta_max,
        )

        rdn = np.asarray(rdn_full, dtype=np.float32)
        bm_raw = np.asarray(bm_full, dtype=np.uint8)

        # Pre-fit HFDI screen. Failing pixels are dropped by setting their water
        # vapor to NaN, the retrieval's existing 'do not retrieve' channel, so
        # they land in the products as the non-retrieved fill value.
        if bool(args.hfdi_filter):
            keep_flat, _hfdi = hfdi_screen(
                rdn, wl_nm, bm_raw,
                hfdi_min=float(args.hfdi_min),
                lo_nm=float(args.hfdi_lo_nm),
                hi_nm=float(args.hfdi_hi_nm),
                verbose=True,
            )
            n_before = int(np.isfinite(water).sum())
            water = np.where(keep_flat.reshape(water.shape), water,
                             np.float32(np.nan)).astype(np.float32)
            n_after = int(np.isfinite(water).sum())
            print(f"[hfdi] retrievable pixels after screen: {n_after} "
                  f"(was {n_before}; {n_before - n_after} dropped)")

        out = run_moment_matched_retrieval(
            rdn=rdn,
            wv=water,
            bm=bm_raw,
            cfg=cfg,
            rte=rte,
            solar_irr=solar_irr,
            csza_val=csza,
            wl_nm=wl_nm,
            lib_spectra=lib_spectra,
            base_surface_1d=surface_1d,
            base_hot_1d=hot_1d,
            base_smolder_1d=smolder_1d,
            eta_swir2_window_1d=eta_swir2_1d,
            eta_swir1_override_window_1d=eta_swir1_1d,
            eta_sat_frac_low=float(args.eta_sat_frac_low),
            eta_sat_frac_high=float(args.eta_sat_frac_high),
            eta_floor_swir2_only_sat=float(args.eta_floor_swir2_only_sat),
            eta_floor_swir1_and_swir2_sat=float(args.eta_floor_swir1_and_swir2_sat),
            eta_ceiling_low_sat=float(args.eta_ceiling_low_sat),
            eta_floor_max=float(args.eta_floor_max),
            skip_dark_pixels=bool(args.skip_dark_pixels),
            dark_thresh=float(args.dark_thresh),
            use_cold_prior=bool(args.use_cold_prior),
            mu0_T=args.mu0_T, lambda_mu_cold=args.lambda_mu_cold, alpha_cold_thresh=args.alpha_cold_thresh,
            hot_window_weight=args.hot_window_weight, smolder_window_weight=args.smolder_window_weight,
            batch_size=int(args.batch_size),
            step1_adam_steps=int(args.step1_adam_steps), step1_adam_lr=float(args.step1_adam_lr),
            step2_adam_steps=int(args.step2_adam_steps), step2_adam_lr=float(args.step2_adam_lr),
            adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps, adam_grad_clip=args.adam_grad_clip,
            wv_quant_step=args.wv_quant_step,
            alpha_tuning=float(args.alpha_tuning),
            sigma_min_k=float(args.sigma_min_k),
            qc_band_diff_filter=bool(args.qc_band_diff_filter),
            qc_band_lo_nm=float(args.qc_band_lo_nm),
            qc_band_hi_nm=float(args.qc_band_hi_nm),
            qc_band_diff_max=float(args.qc_band_diff_max),
            verbose=True,
        )
        n_qc = int(out.get("n_qc_masked", 0))
        qc_str = f"; qc_masked={n_qc}" if args.qc_band_diff_filter else ""
        print(f"[retrieval] valid pixels: {out['n_valid']} in {out['elapsed']:.2f}s{qc_str}")

        # Per-scene temperature stats over finite pixels only. mu_map is E[T] (K)
        # and sigma_map the moment-matched single-Gaussian sigma_post (K).
        mu_arr = np.asarray(out["mu_map"], dtype=np.float32)
        sigma_arr = np.asarray(out["sigma_map"], dtype=np.float32)
        mu_finite = np.isfinite(mu_arr)
        n_mu_finite = int(mu_finite.sum())
        if n_mu_finite > 0:
            mu_vals = mu_arr[mu_finite]
            sigma_vals = sigma_arr[np.isfinite(sigma_arr)]
            mu_pcts = np.percentile(mu_vals, [5, 50, 95])
            print(
                f"[stats] E[T] (K) over {n_mu_finite} valid pixels: "
                f"min={float(mu_vals.min()):.1f}  median={float(mu_pcts[1]):.1f}  "
                f"max={float(mu_vals.max()):.1f}  p05={float(mu_pcts[0]):.1f}  "
                f"p95={float(mu_pcts[2]):.1f}"
            )
            if sigma_vals.size > 0:
                print(
                    f"[stats] sigma_post (K): "
                    f"min={float(sigma_vals.min()):.1f}  "
                    f"median={float(np.median(sigma_vals)):.1f}  "
                    f"max={float(sigma_vals.max()):.1f}"
                )
        else:
            print(f"[stats] no finite E[T] pixels in this scene")

        glt_samples, glt_lines, gt, wkt, glt_nodata = _open_glt(paths["glt"])
        raw_rows, raw_cols = rdn.shape[:2]

        # Band-swap fallback for flight-computer GLTs that ship sample/line bands
        # swapped; see _check_glt_band_order.
        glt_samples, glt_lines, _swapped = _check_glt_band_order(
            glt_samples, glt_lines, raw_rows, raw_cols
        )
        if _swapped:
            print(f"[glt] NOTE: GLT sample/line bands looked swapped vs the raw "
                  f"image (rows={raw_rows}, cols={raw_cols}); swapped so "
                  f"band1->samples(col), band2->lines(row). Expected fallback for "
                  f"pre-fix flight-computer GLTs.")

        print(f"[glt] ortho grid {glt_samples.shape}, raw image ({raw_rows}, {raw_cols}); "
              f"glt nodata = {glt_nodata!r}")

        def _band_stats(arr, label, axis_size, axis_name):
            """One-line index-range summary for a GLT band."""
            abs_arr = np.abs(arr)
            return (
                f"[glt] {label}: min={int(arr.min())}, max={int(arr.max())}, "
                f"|max|={int(abs_arr.max())} (raw_{axis_name}={axis_size}); "
                f"negatives={int((arr < 0).sum())}, "
                f"== glt_nodata ({int(glt_nodata)})={int((arr == int(glt_nodata)).sum())}, "
                f"|val|>{axis_name}={int((abs_arr > axis_size).sum())}"
            )

        print(_band_stats(
            glt_samples, "band 1 = samples (col index)", raw_cols, "cols",
        ))
        print(_band_stats(
            glt_lines,   "band 2 = lines (row index)",   raw_rows, "rows",
        ))

        # Bounds-clip: GLT entries whose |value| exceeds the raw image dim are
        # overwritten with glt_nodata in both channels. Sentinel entries mark
        # out-of-swath rather than a chunk-size mismatch, so they never warn.
        all_sentinels = tuple({0, -9999, int(glt_nodata)})
        sentinel_px = (np.isin(glt_samples, all_sentinels)
                       | np.isin(glt_lines, all_sentinels))
        s_oob = (np.abs(glt_samples) > raw_cols) & ~sentinel_px
        l_oob = (np.abs(glt_lines)   > raw_rows) & ~sentinel_px
        n_oob = int((s_oob | l_oob).sum())
        if n_oob > 0:
            print(f"[glt] WARNING: {n_oob} ortho pixels have GLT entries pointing "
                  f"outside the raw image ("
                  f"|samples|>{raw_cols}: {int(s_oob.sum())}; "
                  f"|lines|>{raw_rows}: {int(l_oob.sum())}). "
                  f"Marking as nodata. Most likely cause: GLT and RDN have "
                  f"different chunk sizes (onboard vs hpc pipelines).")
            mask_oob = s_oob | l_oob
            glt_samples = glt_samples.copy()
            glt_lines = glt_lines.copy()
            glt_samples[mask_oob] = int(glt_nodata)
            glt_lines[mask_oob]   = int(glt_nodata)

        # Fill in-scene non-retrieved pixels with the -1 sentinel before
        # orthorectification so they survive the GLT lookup.
        mu_filled = np.asarray(out["mu_map"], dtype=np.float32)
        mu_filled = np.where(np.isfinite(mu_filled), mu_filled,
                             np.float32(USER_FTP_FILL_K))
        sigma_filled = np.asarray(out["sigma_map"], dtype=np.float32)
        sigma_filled = np.where(np.isfinite(sigma_filled), sigma_filled,
                                np.float32(USER_FUQ_FILL_K))

        fire_temp_path, fire_temp_unc_path = _output_paths_for_chunk(
            output_scene_id, args.fire_temp_dir, args.fire_temp_unc_dir
        )
        os.makedirs(args.fire_temp_dir, exist_ok=True)
        os.makedirs(args.fire_temp_unc_dir, exist_ok=True)
        export_ortho_geotiff(fire_temp_path, mu_filled,
                             glt_samples, glt_lines, gt, wkt,
                             glt_nodata=glt_nodata)
        export_ortho_geotiff(fire_temp_unc_path, sigma_filled,
                             glt_samples, glt_lines, gt, wkt,
                             glt_nodata=glt_nodata)

        print(f"[output] fireTemp    = {fire_temp_path}")
        print(f"[output] fireTempUnc = {fire_temp_unc_path}")

        # Optional copy into a downlink staging dir; best-effort, and a blank
        # --downlink_dir is a no-op.
        dl_dir = (getattr(args, "downlink_dir", "") or "").strip()
        if dl_dir:
            os.makedirs(dl_dir, exist_ok=True)
            for src in (fire_temp_path, fire_temp_unc_path):
                dst = os.path.join(dl_dir, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                    print(f"[downlink] copied {os.path.basename(src)} -> {dst}")
                except OSError as e:
                    print(f"[downlink] WARNING: could not copy {src} -> {dst}: {e!r}")

        print(f"[scene] {chunk} done in {time.perf_counter() - t_chunk:.2f}s")
        return fire_temp_path, fire_temp_unc_path
    finally:
        # rte and lut_subdir belong to rte_builder's date-keyed cache, which
        # releases everything in cleanup() at watcher exit.
        pass


def run_watcher(args) -> dict:
    """Discover and process AV3 scenes from a single ``--input_dir``.

    Polls every ``args.poll_seconds`` and runs until SIGINT/SIGTERM, logging
    'waiting for scenes' once per quiet stretch. Returns a summary dict of
    counters.
    """
    # Auxiliary data must resolve to real files (auto-discovered or via flag).
    for fld, lbl in [("fire_lib_csv", FIRE_LIB_FILENAME), ("solar_irr_file", SOLAR_IRR_FILENAME)]:
        v = getattr(args, fld)
        if not v or not os.path.exists(v):
            raise FileNotFoundError(
                f"--{fld} not provided and no '{lbl}' found under "
                f"$ISOFIRE_DATA_DIR or <repo>/data. Pass --{fld} explicitly "
                f"or commit the file to the repo's data/ directory."
            )

    if not args.input_dir:
        raise FileNotFoundError(
            "--input_dir is required (single directory containing the ENVI files for each scene)."
        )
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"--input_dir does not exist: {args.input_dir}")

    if not args.fire_temp_dir:
        raise ValueError("--fire_temp_dir is required.")
    if not args.fire_temp_unc_dir:
        raise ValueError("--fire_temp_unc_dir is required.")
    os.makedirs(args.fire_temp_dir, exist_ok=True)
    os.makedirs(args.fire_temp_unc_dir, exist_ok=True)

    dl_dir = (getattr(args, "downlink_dir", "") or "").strip()

    print("[watch]")
    print(f"  input_dir         = {args.input_dir}")
    print(f"  fire_temp_dir     = {args.fire_temp_dir}")
    print(f"  fire_temp_unc_dir = {args.fire_temp_unc_dir}")
    if dl_dir:
        print(f"  downlink_dir      = {dl_dir}")
    print(f"[knobs] dark_thresh={args.dark_thresh}  alpha_tuning={args.alpha_tuning}  "
          f"sigma_min_k={args.sigma_min_k}  skip_dark_pixels={bool(args.skip_dark_pixels)}  "
          f"poll_seconds={args.poll_seconds}  overwrite={bool(args.overwrite)}  "
          f"debug={bool(args.debug)}")
    if bool(args.debug):
        print("[cleanup] --debug set: L1B input files will be KEPT after processing.")
    else:
        print("[cleanup] L1B input files (RDN/LOC/GLT/MASK/OBS) will be DELETED after "
              "each scene's products are written (pass --debug to keep them).")
    if float(args.eta_floor_max) > 0.0:
        ceiling_note = (
            f"; ceiling {args.eta_ceiling_low_sat:.2f} when swir2_sat < {args.eta_sat_frac_low:.3f}"
            if float(args.eta_ceiling_low_sat) < 1.0
            else " (no ceiling)"
        )
        print(
            f"[knobs] pre-fit eta hard floor: "
            f"sat threshold high={args.eta_sat_frac_high:.3f}; "
            f"floor(swir2_only={args.eta_floor_swir2_only_sat:.2f}, "
            f"swir1+swir2={args.eta_floor_swir1_and_swir2_sat:.2f})"
            f"{ceiling_note}"
        )

    # 6S LUTs are built once per calendar date and reused across every scene
    # flown that date; see _RteBuilder.
    water_vapors = parse_water_vapors_csv(args.water_vapors)
    lut_root = _resolve_lut_dir(args.lut_dir)
    print(f"[rt] scratch LUT root: {lut_root}  (one LUT cached per calendar date)")
    rte_builder = _RteBuilder(
        lut_root=lut_root,
        sixs_dir=args.sixs_dir,
        water_vapors=water_vapors,
        water_aot550=float(args.water_aot550),
        water_o3=float(args.water_o3),
    )

    seen_chunks: set = set()
    processed: list = []
    skipped: list = []
    failed: list = []

    # Poll indefinitely until SIGINT/SIGTERM; the finally block always cleans up
    # cached LUT scratch dirs.
    try:
        waiting_logged = False
        while True:
            ready, partial = discover_ready_chunks(input_dir=args.input_dir)
            new_ready = [c for c in sorted(ready) if c not in seen_chunks]

            if new_ready:
                print(f"[watch] {len(new_ready)} new ready scene(s): {new_ready}")
                waiting_logged = False

            for chunk in new_ready:
                seen_chunks.add(chunk)

                out_scene_id = _resolve_output_scene_id(chunk, ready[chunk].get("glt"))
                ft_done, ftu_done = _output_paths_for_chunk(
                    out_scene_id, args.fire_temp_dir, args.fire_temp_unc_dir
                )
                if not bool(args.overwrite) and os.path.exists(ft_done) and os.path.exists(ftu_done):
                    print(f"[watch] {chunk}: outputs exist (as "
                          f"{out_scene_id}_fireTemp.tif / {out_scene_id}_fireTempUnc.tif), "
                          f"skipping (use --overwrite to redo)")
                    skipped.append((chunk, "outputs already exist"))
                    continue

                try:
                    ft_path, ftu_path = process_chunk(chunk, ready[chunk], args, rte_builder)
                    if ft_path is None:
                        skipped.append((chunk, "no fireTemp/fireTempUnc returned"))
                    else:
                        processed.append((chunk, ft_path, ftu_path))
                        # Drop the L1B inputs unless --debug asked us to keep them.
                        if not bool(args.debug):
                            _delete_scene_inputs(ready[chunk])
                except Exception as e:
                    # One bad scene must not kill the watcher; inputs are kept
                    # on failure so the scene can be retried.
                    warnings.warn(f"scene {chunk} failed: {e!r}", stacklevel=2)
                    failed.append((chunk, repr(e)))

            if partial:
                print(f"[watch] partial scenes (some required kind missing): "
                      f"{ {c: partial[c] for c in sorted(partial)[:20]} }"
                      + (f" (+{len(partial)-20} more)" if len(partial) > 20 else ""))

            if not new_ready and not waiting_logged:
                # Print the headline once after activity, then stay quiet until
                # new ready scenes show up.
                print(f"[watch] scene complete, waiting for scenes "
                      f"(seen={len(seen_chunks)} processed={len(processed)} "
                      f"skipped={len(skipped)} failed={len(failed)}; "
                      f"polling every {float(args.poll_seconds):.1f}s; "
                      f"Ctrl-C / SIGTERM to exit)",
                      flush=True)
                waiting_logged = True

            time.sleep(float(args.poll_seconds))
    except KeyboardInterrupt:
        print(f"[watch] interrupted by user (Ctrl-C); exiting cleanly.")
    finally:
        # Single point where scratch disk is released.
        rte_builder.cleanup()

    summary = dict(
        seen=len(seen_chunks),
        processed=len(processed),
        skipped=len(skipped),
        failed=len(failed),
        processed_chunks=[c for c, _, _ in processed],
        skipped_chunks=skipped,
        failed_chunks=failed,
    )
    print("")
    print("[watch] === final summary ===")
    print(f"  scenes seen:      {summary['seen']}")
    print(f"  scenes processed: {summary['processed']}")
    print(f"  scenes skipped:   {summary['skipped']}")
    print(f"  scenes failed:    {summary['failed']}")
    return summary


# CLI helpers

def _parse_ranges(s, default):
    """Parse a list-of-pairs spec like '[[900.0, 1800.0]]', or pass arrays through."""
    if s is None:
        return np.asarray(default, dtype=np.float64).reshape(-1, 2)
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode("utf-8")
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return np.asarray(default, dtype=np.float64).reshape(-1, 2)
        try:
            import ast
            s = ast.literal_eval(s)
        except Exception:
            return np.asarray(default, dtype=np.float64).reshape(-1, 2)
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    return arr


# Argparse + main entry

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the watcher."""
    p = argparse.ArgumentParser(
        description="AVIRIS-3 two-component Gaussian temperature retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input directory.
    g_paths = p.add_argument_group(
        "input directory (single dir with all ENVI files for one or more scenes)"
    )
    g_paths.add_argument("--input_dir", default=DEFAULT_INPUT_DIR,
                         help="Directory containing AV3 ENVI files. Files are grouped by "
                              "scene id (AV3<date><time>[_<chunk>]). Each scene must contain "
                              "RDN/rdn, LOC/loc, GLT/glt, BANDMASK/sat (case-insensitive), "
                              "and OBS/obs ENVI pairs. OBS is REQUIRED (not optional).")

    # Output directories.
    g_out = p.add_argument_group(
        "output directories (one fireTemp + one fireTempUnc GeoTIFF per ready scene)"
    )
    g_out.add_argument("--fire_temp_dir", default=DEFAULT_FIRE_TEMP_DIR,
                       help="Output directory for the fire-temperature (E[T]) GeoTIFF, "
                            "named '<scene_id>_fireTemp.tif' "
                            "(e.g. '/data/rtsp/Layers/fireTemp/science').")
    g_out.add_argument("--fire_temp_unc_dir", default=DEFAULT_FIRE_TEMP_UNC_DIR,
                       help="Output directory for the uncertainty (sigma) GeoTIFF, "
                            "named '<scene_id>_fireTempUnc.tif' "
                            "(e.g. '/data/rtsp/Layers/fireTempUnc/science').")
    g_out.add_argument("--downlink_dir", default=DEFAULT_DOWNLINK_DIR,
                       help="Optional extra directory to ALSO write the fireTemp/fireTempUnc "
                            "GeoTIFFs into, in addition to --fire_temp_dir / "
                            "--fire_temp_unc_dir (e.g. '/data/rtsp/Layers/to_downlink'). "
                            "Defaults to an empty string, which disables the extra copy.")
    g_out.add_argument("--overwrite", action="store_true",
                       help="Re-process scenes even when output TIFFs already exist.")
    g_out.add_argument("--debug", action="store_true",
                       help="Keep the L1B input ENVI files (RDN/LOC/GLT/MASK/OBS, "
                            ".hdr + binary cube) after processing. By DEFAULT (no "
                            "--debug) those inputs are DELETED once a scene's "
                            "fireTemp/fireTempUnc products are written, so the input "
                            "directory doesn't grow without bound. Failures and "
                            "already-done skips never delete.")

    # Watcher behavior.
    g_watch = p.add_argument_group("watcher behavior (runs until killed)")
    g_watch.add_argument("--poll_seconds", type=float, default=USER_POLL_SECONDS,
                         help="Seconds between input-directory polls. The watcher runs "
                              "indefinitely (Ctrl-C / SIGTERM to stop); after each pass "
                              "with no new ready scenes it logs 'waiting for scenes' and "
                              "sleeps one poll interval.")

    # Last-band darkness threshold (go/no-go).
    g_skip = p.add_argument_group("go/no-go: last-band darkness threshold")
    g_skip.add_argument("--dark_thresh", type=float, default=USER_DARK_THRESH,
                        help="Pixel is RETRIEVED only if rdn[..., -1] > dark_thresh.")
    g_skip.add_argument("--skip_dark_pixels",    dest="skip_dark_pixels", action="store_true",  default=USER_SKIP_DARK_PIXELS)
    g_skip.add_argument("--no_skip_dark_pixels", dest="skip_dark_pixels", action="store_false")

    # Sigma scaling (analytic moment-matched single Gaussian).
    g_sigma = p.add_argument_group("sigma scaling (analytic moment-matched single Gaussian)")
    g_sigma.add_argument("--alpha_tuning", type=float, default=USER_ALPHA_TUNING,
                         help="sigma_post = sigma_single * (1 + alpha_tuning * max(alpha2, 0))")
    g_sigma.add_argument("--sigma_min_k",  type=float, default=USER_SIGMA_MIN_K,
                         help="Lower clamp on sigma_post (Kelvin).")

    g_hfdi = p.add_argument_group("QC: HFDI fire screen (pre-fit)")
    g_hfdi.add_argument("--hfdi_filter", dest="hfdi_filter",
                        action="store_true", default=True,
                        help="Drop pixels whose Hyperspectral Fire Detection Index, "
                             "HFDI = (L(hi) - L(lo)) / (L(hi) + L(lo)), is at or below "
                             "--hfdi_min. Saturation in either index band auto-passes.")
    g_hfdi.add_argument("--no_hfdi_filter", dest="hfdi_filter", action="store_false")
    g_hfdi.add_argument("--hfdi_lo_nm", type=float, default=HFDI_LO_NM_DEFAULT,
                        help="Lower index wavelength (nm); nearest sensor band is used.")
    g_hfdi.add_argument("--hfdi_hi_nm", type=float, default=HFDI_HI_NM_DEFAULT,
                        help="Upper index wavelength (nm); nearest sensor band is used.")
    g_hfdi.add_argument("--hfdi_min", type=float, default=HFDI_MIN_DEFAULT,
                        help="Keep a pixel only if HFDI > this value.")

    g_qc = p.add_argument_group(
        "QC: legacy band-diff filter (pre-fit; superseded by the HFDI screen)")
    g_qc.add_argument("--qc_band_diff_filter",    dest="qc_band_diff_filter", action="store_true",  default=False,
                      help="Legacy absolute SWIR1-minus-SWIR2 filter, off by default. "
                           "Enabling it ANDs with the HFDI screen; kept for A/B "
                           "comparison against earlier band-diff runs.")
    g_qc.add_argument("--no_qc_band_diff_filter", dest="qc_band_diff_filter", action="store_false")
    g_qc.add_argument("--qc_band_lo_nm", type=float, default=1096.9,
                      help="Target wavelength for the SWIR1 diagnostic band; nearest sensor band is used.")
    g_qc.add_argument("--qc_band_hi_nm", type=float, default=2249.4,
                      help="Target wavelength for the SWIR2 diagnostic band; nearest sensor band is used.")
    g_qc.add_argument("--qc_band_diff_max", type=float, default=5.0,
                      help="Drop pixel if obs[lo_band] - obs[hi_band] exceeds this (W/m^2/sr/nm).")

    # Pre-fit eta floor: more saturation means a hotter pixel, so eta_lo is
    # raised toward the hot Gaussian; --eta_floor_max=0 disables it.
    g_etafloor = p.add_argument_group("Eta floor (pre-fit, saturation-driven, hard categorical)")
    g_etafloor.add_argument("--eta_sat_frac_low", type=float, default=0.02,
                            help="SWIR2 sat fraction below which a pixel is treated as "
                                 "having essentially no thermal signature (triggers "
                                 "eta_hi clamp to --eta_ceiling_low_sat).")
    g_etafloor.add_argument("--eta_sat_frac_high", type=float, default=0.08,
                            help="SWIR2 (or SWIR1) sat fraction at or above which a "
                                 "pixel is treated as having strong thermal saturation "
                                 "in that window. Drives the categorical floor levels.")
    g_etafloor.add_argument("--eta_floor_swir2_only_sat", type=float, default=0.75,
                            help="Floor on eta_lo when SWIR2 is highly saturated but "
                                 "SWIR1 is not.")
    g_etafloor.add_argument("--eta_floor_swir1_and_swir2_sat", type=float, default=0.85,
                            help="Floor on eta_lo when BOTH SWIR2 and SWIR1 are highly "
                                 "saturated.")
    g_etafloor.add_argument("--eta_ceiling_low_sat", type=float, default=1.0,
                            help="Ceiling on eta_hi when SWIR2 sat fraction is below "
                                 "--eta_sat_frac_low. Default 1.0 (no ceiling). Set to "
                                 "e.g. 0.5 to clamp cold pixels away from hot-dominant.")
    g_etafloor.add_argument("--eta_floor_max", type=float, default=0.0,
                            help="Master gate on the pre-fit floor. 0.0 (default) "
                                 "disables both the floor and the ceiling entirely. "
                                 "Any value > 0 enables the categorical thresholds above.")

    # Geometry overrides (defaults come from OBS/LOC; OBS itself is required).
    g_geom = p.add_argument_group("geometry overrides (defaults from OBS/LOC)")
    g_geom.add_argument("--sza_deg", type=float, default=None)
    g_geom.add_argument("--elev_km", type=float, default=None)
    g_geom.add_argument("--alt_km",  type=float, default=None)

    # Auxiliary inputs.
    g_aux = p.add_argument_group("auxiliary inputs (default to <repo>/data/)")
    g_aux.add_argument("--fire_lib_csv",   default=DEFAULT_FIRE_LIB_CSV,
                       help=f"Fire endmember library CSV. Defaults to <repo>/data/{FIRE_LIB_FILENAME} "
                            "or $ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--solar_irr_file", default=DEFAULT_SOLAR_IRR_FILE,
                       help=f"Top-of-atmosphere solar irradiance file. Defaults to "
                            f"<repo>/data/{SOLAR_IRR_FILENAME} or $ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--sensor_wl_fwhm",  default=DEFAULT_SENSOR_WL_FWHM_FILE,
                       help=f"Optional .npz cache of sensor wavelengths and FWHM in nm "
                            f"(keys: wl_nm, fwhm_nm). Used as the FWHM source ONLY when "
                            f"the RDN header is missing the 'fwhm' field (onboard "
                            f"AVIRIS-3 headers). When the cache band count differs from "
                            f"the cube, FWHM is linearly interpolated onto the cube "
                            f"wavelengths. Wavelengths always come from the RDN header. "
                            f"Defaults to <repo>/data/{SENSOR_WL_FWHM_FILENAME} or "
                            f"$ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--sixs_dir",       default=os.environ.get("ISOFIRE_SIXS_DIR", "./sixs/Sixs/"),
                       help="Directory containing the compiled 6S binary (sixsV2.1). "
                            "Defaults to $ISOFIRE_SIXS_DIR if set (apptainer/podman "
                            "images export this to the baked-in build path), otherwise "
                            "to the in-repo vendored 6S at ./sixs/Sixs/; run "
                            "`cd sixs/Sixs && make` once before invoking natively.")
    g_aux.add_argument("--lut_dir",        default=None,
                       help="Per-process scratch parent for 6S LUT (default: /tmp/isofire_lut)")

    # Water vapor.
    g_wv = p.add_argument_group("water vapor")
    g_wv.add_argument("--water_vapors", default="0.2,0.5,1.0,2.0,3.0,4.0")
    g_wv.add_argument("--water_aot550", type=float, default=0.01)
    g_wv.add_argument("--water_o3",     type=float, default=0.30)
    g_wv.add_argument("--water_superpixel_size", type=int, default=12)

    # Spectral fitting windows.
    g_win = p.add_argument_group("spectral fitting windows")
    g_win.add_argument("--surface_window_ranges_nm", default=str(SURFACE_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--hot_window_ranges_nm",     default=str(HOT_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--smolder_window_ranges_nm", default=str(SMOLDER_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--eta_swir2_lo_nm",          type=float, default=ETA_SWIR2_DEFAULT_NM[0])
    g_win.add_argument("--eta_swir2_hi_nm",          type=float, default=ETA_SWIR2_DEFAULT_NM[1])

    # Mixture retrieval config.
    g_cfg = p.add_argument_group("mixture retrieval config")
    g_cfg.add_argument("--Tmin",          type=float, default=200.0)
    g_cfg.add_argument("--Tmax",          type=float, default=2400.0)
    g_cfg.add_argument("--N_T",           type=int,   default=240)
    g_cfg.add_argument("--T_step1_init",  type=float, default=900.0)
    g_cfg.add_argument("--mu_hot_min_T",  type=float, default=500.0)
    g_cfg.add_argument("--mu_hot_max_T",  type=float, default=2100.0)
    g_cfg.add_argument("--mu_smol_min_T", type=float, default=450.0)
    g_cfg.add_argument("--mu_smol_max_T", type=float, default=850.0)
    g_cfg.add_argument("--sigma_hot_min", type=float, default=50.0)
    g_cfg.add_argument("--sigma_hot_max", type=float, default=400.0)
    g_cfg.add_argument("--sigma_smol_min",type=float, default=100.0)
    g_cfg.add_argument("--sigma_smol_max",type=float, default=400.0)
    g_cfg.add_argument("--alpha1_min",    type=float, default=1e-4)
    g_cfg.add_argument("--alpha1_max",    type=float, default=10.0)
    g_cfg.add_argument("--alpha2_min",    type=float, default=0.0)
    g_cfg.add_argument("--alpha2_max",    type=float, default=10.0)
    g_cfg.add_argument("--eta_min",       type=float, default=0.0)
    g_cfg.add_argument("--eta_max",       type=float, default=1.0)
    g_cfg.add_argument("--hot_window_weight",     type=float, default=1.0)
    g_cfg.add_argument("--smolder_window_weight", type=float, default=1.0)

    # Optional cold prior.
    g_cold = p.add_argument_group("optional cold prior")
    g_cold.add_argument("--use_cold_prior",    action="store_true")
    g_cold.add_argument("--mu0_T",             type=float, default=500.0)
    g_cold.add_argument("--lambda_mu_cold",    type=float, default=1e-3)
    g_cold.add_argument("--alpha_cold_thresh", type=float, default=1e-3)

    # Optimizer.
    g_opt = p.add_argument_group("optimizer")
    g_opt.add_argument("--batch_size",        type=int,   default=8000)
    g_opt.add_argument("--step1_adam_steps",  type=int,   default=30)
    g_opt.add_argument("--step1_adam_lr",     type=float, default=0.03)
    g_opt.add_argument("--step2_adam_steps",  type=int,   default=100)
    g_opt.add_argument("--step2_adam_lr",     type=float, default=0.015)
    g_opt.add_argument("--adam_beta1",        type=float, default=0.9)
    g_opt.add_argument("--adam_beta2",        type=float, default=0.999)
    g_opt.add_argument("--adam_eps",          type=float, default=1e-8)
    g_opt.add_argument("--adam_grad_clip",    type=float, default=5.0)
    g_opt.add_argument("--wv_quant_step",     type=float, default=0.02)

    return p


def main(argv=None) -> int:
    """CLI entry point. Returns 0 unless a scene failed outright."""
    args = _build_parser().parse_args(argv)
    summary = run_watcher(args)
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())