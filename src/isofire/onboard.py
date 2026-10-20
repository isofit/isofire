#!/usr/bin/env python
#
#  Copyright 2024 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Authors: william.r.keely@jpl.nasa.gov
"""AVIRIS-3 onboard fire-temperature watcher (pre-ortho).

Flight-computer entry point. Reads one ``--input_dir`` holding ENVI files for
one or more AVIRIS-3 or AVIRIS-5 scenes, grouped by scene id
(``AV[35]<date><time>[_<chunk>]``). Behaviour differs by instrument in two
places and nowhere else: AVIRIS-5 screens on a wide-span normalised
difference (1650/2430) plus windowed saturation rather than HFDI, and FRP is skipped for AVIRIS-5 because the model is
AVIRIS-3 only. A
scene is processed once its RDN, LOC, BANDMASK/sat and OBS ENVI pairs are all
present. Products are written unwarped into ``--to_ortho_dir`` as ENVI pairs
(binary with no extension + ``.hdr``) for a downstream ortho step to warp
through the GLT: ``<scene_id>_fireTemp_toOrtho``,
``<scene_id>_fireTempUnc_toOrtho``, and with ``--frp_model`` also
``<scene_id>_fireFRP_toOrtho`` and ``<scene_id>_fireFRPUnc_toOrtho``.

The scene id, including its ``T<HHMMSS>`` system-time stamp, is taken verbatim
from the RDN filename; no time conversion is performed. L1B inputs are never
deleted. Fire pixels are screened pre-fit with the HFDI index. A scene whose
inputs stay incomplete for ``--scene_timeout_seconds`` is abandoned, and with
``--exit_on_timeout`` the watcher exits nonzero for a supervisor to restart.

Algorithm/physics live in retrieval.py and rt.py; shared I/O helpers come from
airborne.py. This file is orchestration only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from spectral.io import envi

"""Silence noisy JAX float64 warnings before importing jax."""
warnings.filterwarnings("ignore", message=".*float64.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*64-bit.*", category=UserWarning)

import jax
jax.config.update("jax_enable_x64", False)

"""Shared helpers from the airborne entry point. Move these to a common.py to
drop the rasterio dependency; none are ortho-specific."""
from .airborne import (
    DEFAULT_FIRE_LIB_CSV,
    DEFAULT_SENSOR_WL_FWHM_FILE,
    DEFAULT_SOLAR_IRR_FILE,
    FIRE_LIB_FILENAME,
    SENSOR_WL_FWHM_FILENAME,
    SOLAR_IRR_FILENAME,
    _RteBuilder,
    _classify_hdr,
    _ensure_fwhm_nm,
    _ensure_nm,
    _envi_binary_for_hdr,
    _load_sensor_wl_fwhm,
    _parse_ranges,
    _parse_scene_id,
    _resolve_geometry,
    _resolve_lut_dir,
)
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
    WV_TEMP_CLIP_MIN,
    decode_sat_ok,
    estimate_scene_water_vapor,
    load_fire_library_resampled,
    make_bitwise_bad_range_mask_1d,
    parse_water_vapors_csv,
    ranges_to_mask,
    run_moment_matched_retrieval,
)


# User knobs -----------------------------------------------------------------
"""Defaults for commonly-tweaked CLI flags; the CLI overrides any of these."""

"""Input dir with every scene's ENVI files; output dir for every product."""
DEFAULT_INPUT_DIR: Optional[str] = None
DEFAULT_TO_ORTHO_DIR: Optional[str] = None

"""Poll interval, scene-assembly limit, and the age an input must reach before
it counts as fully written."""
USER_POLL_SECONDS: float = 10.0
USER_SCENE_TIMEOUT_SECONDS: float = 300.0
USER_SETTLE_SECONDS: float = 5.0

"""Single-Gaussian post-processing (sigma scaling)."""
USER_ALPHA_TUNING: float = DEFAULT_ALPHA_TUNING
USER_SIGMA_MIN_K: float = DEFAULT_SIGMA_MIN_K

"""Last-band go/no-go: pixel retrieved only if rdn[..., -1] > USER_DARK_THRESH."""
USER_DARK_THRESH: float = 0.15
USER_SKIP_DARK_PIXELS: bool = True

"""Night / twilight handling. Above NIGHT_SZA_DEG the solar path carries no
signal: solar irradiance is zeroed (which makes the forward model's solar
baseline exactly 0, leaving a thermal-only fit) and the 940 nm water retrieval
is replaced by a fixed H2OSTR, since it measures reflected sunlight. 6S is not
meaningful at large solar zenith either, so the LUT is always built at no more
than MAX_LUT_SZA_DEG; upwelling transmittance, the only RT term a night scene
uses, is insensitive to solar zenith."""
USER_NIGHT_SZA_DEG: float = 90.0
USER_MAX_LUT_SZA_DEG: float = 85.0
USER_NIGHT_WATER_VAPOR: float = 1.0

"""Fill for non-retrieved pixels, echoed in the header as 'data ignore value'.
-1 leaves 0 free for the downstream ortho step to mark out-of-swath; real E[T]
and sigma are strictly positive, so both are unambiguous."""
USER_FILL_VALUE: float = -1.0


class WaterVaporCarry:
    """Remembers the last daylight scene's median H2OSTR for night scenes.

    The 940 nm retrieval needs reflected sunlight, so a night scene cannot
    measure water vapor at all. Atmospheric water changes slowly, so a value
    carried from earlier in the same flight beats any fixed guess. The value is
    also written to disk under the LUT scratch root, so a watcher restarted
    after dark still has it.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.value: Optional[float] = None
        self.source: Optional[str] = None
        self.sza: Optional[float] = None
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            v = float(d["water_vapor"])
            if np.isfinite(v):
                self.value, self.source = v, d.get("scene_id")
                self.sza = d.get("sza")
                print(f"[water] carried H2OSTR {v:.3f} g/cm^2 loaded from "
                      f"{self.path} (from scene {self.source!r})")
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[water] WARNING: could not read {self.path} ({e!r}); "
                  f"starting with no carried value.")

    def update(self, water: np.ndarray, scene_id: str, sza: float) -> None:
        """Record a daylight scene's median H2OSTR. No-op if none is finite."""
        arr = np.asarray(water, dtype=np.float64)
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            return
        self.value = float(np.median(fin))
        self.source, self.sza = str(scene_id), float(sza)
        print(f"[water] median H2OSTR {self.value:.3f} g/cm^2 retained for any "
              f"night scenes later this flight")
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"water_vapor": self.value, "scene_id": self.source,
                           "sza": self.sza}, f)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[water] WARNING: could not persist to {self.path}: {e!r}")


# Scene discovery ------------------------------------------------------------

"""GLT is absent: orthorectification happens downstream."""
REQUIRED_KIND_KEYS = ("rdn", "loc", "mask", "obs")

"""Product suffixes. Binary carries no extension, header is <stem>.hdr, e.g.
AV320260602T213058_fireTempUnc_toOrtho[.hdr]"""
PRODUCT_SUFFIXES: Dict[str, str] = {
    "fire_temp": "_fireTemp_toOrtho",
    "fire_temp_unc": "_fireTempUnc_toOrtho",
    "frp": "_fireFRP_toOrtho",
    "frp_unc": "_fireFRPUnc_toOrtho",
}


def _file_settled(path: str, settle_seconds: float, now: float) -> bool:
    """True if ``path`` exists and is at least ``settle_seconds`` old.

    Keeps a cube the flight computer is still writing from being opened.
    ``settle_seconds <= 0`` disables the check.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    if settle_seconds <= 0.0:
        return True
    return (now - float(st.st_mtime)) >= float(settle_seconds)


AV5_PRODUCT_SUBDIRS = ("l1b_rdn", "l1b_ort")


def _search_dirs(input_dir: str) -> list:
    """Directories to scan.

    Flight writes every product into one directory. The AVIRIS-5 archive splits
    a line into l1b_rdn (RDN, BANDMASK) and l1b_ort (OBS, LOC, GLT); those are
    scanned when present. A flight inbox has neither and falls through.
    """
    sub = [os.path.join(input_dir, d) for d in AV5_PRODUCT_SUBDIRS
           if os.path.isdir(os.path.join(input_dir, d))]
    return sub if sub else [input_dir]


def discover_ready_scenes(
    *,
    input_dir: str,
    required: Sequence[str] = REQUIRED_KIND_KEYS,
    settle_seconds: float = 0.0,
    require_chunk: bool = False,
) -> Tuple[dict, dict]:
    """Return (ready, partial) scene maps from a single input directory.

    Groups ENVI .hdr files by (scene_id, kind), counting one only when its
    binary cube exists and both files have settled, and drops any pair matching
    more than one file so a scene is never picked from ambiguous candidates.
    ``ready[scene_id]`` has every kind in ``required``; anything missing lands
    in ``partial[scene_id]``. GLT files are ignored.
    """
    import glob as _glob
    if not input_dir or not os.path.isdir(input_dir):
        return {}, {}

    now = time.time()
    hits = []
    for d in _search_dirs(input_dir):
        hits += _glob.glob(os.path.join(d, "*.hdr"))
        hits += _glob.glob(os.path.join(d, "*.HDR"))
    hits = sorted(set(hits))

    by_scene: Dict[str, Dict[str, str]] = {}
    dupe_keys: set = set()
    for h in hits:
        cls = _classify_hdr(h)
        if cls is None:
            continue
        kind, scene_id, _base, _chunk = cls
        # The AVIRIS-5 archive holds unchunked full-line OBS/LOC/GLT with
        # no matching RDN, which would wait in 'partial' until timeout.
        # Off by default: a flight scene may have no _NNN suffix.
        if require_chunk and _chunk is None:
            continue
        if kind not in required:
            continue
        binary = _envi_binary_for_hdr(h)
        if binary is None:
            continue
        if not (_file_settled(h, settle_seconds, now)
                and _file_settled(binary, settle_seconds, now)):
            """Still being written; the scene stays partial this pass."""
            by_scene.setdefault(scene_id, {})
            continue
        slot = by_scene.setdefault(scene_id, {})
        if kind in slot:
            dupe_keys.add((scene_id, kind))
            continue
        slot[kind] = h

    for scene_id, kind in dupe_keys:
        by_scene.get(scene_id, {}).pop(kind, None)

    ready: dict = {}
    partial: dict = {}
    for scene_id, kinds in sorted(by_scene.items()):
        missing = [k for k in required if k not in kinds]
        if missing:
            partial[scene_id] = missing
            continue
        ready[scene_id] = {k: kinds[k] for k in required}
    return ready, partial


def output_stem(scene_id: str, product: str, to_ortho_dir: str) -> str:
    """Path stem (no extension) for a product; header is ``<stem>.hdr``.

    ``scene_id`` comes off the RDN filename, so its ``T<HHMMSS>`` system-time
    stamp carries through untouched.
    """
    try:
        suffix = PRODUCT_SUFFIXES[product]
    except KeyError:
        raise KeyError(
            f"unknown product {product!r}; expected one of "
            f"{sorted(PRODUCT_SUFFIXES)}"
        ) from None
    return os.path.join(to_ortho_dir, f"{scene_id}{suffix}")


def _product_exists(stem: str) -> bool:
    """True when both the ENVI binary and its header are on disk."""
    return os.path.exists(stem) and os.path.exists(stem + ".hdr")


# ENVI product writing -------------------------------------------------------

ENVI_DATA_TYPE_FLOAT32 = 4


def write_envi_single_band(
    stem: str,
    data: np.ndarray,
    *,
    description: str,
    band_name: str,
    data_ignore_value: float,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Write a single-band float32 BSQ ENVI pair, unwarped, to ``stem``.

    Binary goes to ``stem``, header to ``stem + '.hdr'``. Both are staged as
    ``.part`` and renamed into place, binary first, so a downstream watcher
    keying on the header never sees one pointing at a half-written cube.
    Non-finite values become ``data_ignore_value``.

    Returns (binary_path, header_path).
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, cols) array, got shape {arr.shape}")
    arr = np.where(np.isfinite(arr), arr, np.float32(data_ignore_value))
    arr = np.ascontiguousarray(arr, dtype="<f4")
    rows, cols = arr.shape

    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    hdr_path = stem + ".hdr"
    bin_tmp = stem + ".part"
    hdr_tmp = hdr_path + ".part"

    lines = [
        "ENVI",
        f"description = {{{description}}}",
        f"samples = {cols}",
        f"lines = {rows}",
        "bands = 1",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {ENVI_DATA_TYPE_FLOAT32}",
        "interleave = bsq",
        "byte order = 0",
        f"data ignore value = {float(data_ignore_value):g}",
        f"band names = {{{band_name}}}",
    ]
    for key, val in (extra_metadata or {}).items():
        lines.append(f"{key} = {{{val}}}")

    arr.tofile(bin_tmp)
    os.replace(bin_tmp, stem)
    with open(hdr_tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(hdr_tmp, hdr_path)

    """Per-output summary: shape plus the value distribution over retrieved
    pixels, so a dtype/scaling bug shows as an out-of-range value."""
    basename = os.path.basename(stem)
    size_mb = os.path.getsize(stem) / 1024.0 / 1024.0
    fill = np.float32(data_ignore_value)
    real = arr[arr != fill]
    if real.size > 0:
        v_min, v_med, v_max = (float(real.min()), float(np.median(real)),
                               float(real.max()))
        summary = (f"vals[min/med/max]=({v_min:.1f}, {v_med:.1f}, {v_max:.1f})")
        bl = basename.lower()
        """_fireTempUnc contains _fireTemp as a substring: test Unc FIRST."""
        if "_firetempunc" in bl:
            lo, hi, kind = 0.0, 1000.0, "fireTempUnc (sigma, K)"
        elif "_firetemp" in bl:
            lo, hi, kind = 300.0, 2000.0, "fireTemp (effective fire T, K)"
        else:
            lo = hi = kind = None
        if kind is not None:
            phys = real[real > 0.0]
            if phys.size > 0 and (float(phys.min()) < lo or float(phys.max()) > hi):
                summary += (f"  WARNING: {kind} values outside expected "
                            f"[{lo:.0f}, {hi:.0f}] K range -- possible "
                            f"dtype/scaling bug")
    else:
        summary = "vals[min/med/max]=(no retrieved pixels)"

    print(f"[envi] {basename}: {rows}x{cols} float32 BSQ, {size_mb:.2f} MB, "
          f"fill={float(data_ignore_value):g}  {summary}")
    return stem, hdr_path


# HFDI pre-fit fire screen ---------------------------------------------------

"""Hyperspectral Fire Detection Index:

    HFDI = (L(2430 nm) - L(2061 nm)) / (L(2430 nm) + L(2061 nm))

Non-fire surfaces are brighter at 2061 nm than at 2430 nm, so HFDI sits well
below zero; thermal emission from fire lifts the 2430 nm end and drives the
index up. Pixels at or below --hfdi_min are dropped before the fit."""
HFDI_LO_NM_DEFAULT = 2061.0
HFDI_HI_NM_DEFAULT = 2430.0
HFDI_MIN_DEFAULT = -0.1

"""Nearest band must land within this distance of each target, else the screen
is disabled rather than indexing the wrong part of the spectrum."""
HFDI_BAND_DISTANCE_MAX_NM = 5.0


def _nearest_band(wl_nm: np.ndarray, target_nm: float) -> Tuple[int, float]:
    """(index, actual wavelength) of the sensor band nearest ``target_nm``."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    idx = int(np.argmin(np.abs(wl - float(target_nm))))
    return idx, float(wl[idx])


def _band_saturated(bm_flat: np.ndarray, bands: int, band_idx: int) -> np.ndarray:
    """Per-pixel saturation flag for ONE band, without unpacking the full mask.

    ``decode_sat_ok`` builds an (n_pix, bands) bool array (~285 MB on 1 Mpix)
    and the retrieval already pays that once. Only two bands are needed here, so
    packed bits are indexed directly: band i is byte i // 8, bit 7 - (i % 8),
    matching numpy's MSB-first unpackbits.
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

    Returns (hfdi_flat, (lo_idx, lo_actual_nm), (hi_idx, hi_actual_nm)). HFDI is
    NaN where the denominator collapses (both bands dark); such pixels are not
    fire and the screen drops them.
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


WIDE_LO_NM_DEFAULT = 1650.0      # AVIRIS-5 screen, SWIR1
WIDE_HI_NM_DEFAULT = 2430.0      # AVIRIS-5 screen, SWIR2
WIDE_INDEX_MIN_DEFAULT = -0.50
SAT_LO_NM_DEFAULT = 2300.0
SAT_HI_NM_DEFAULT = 2500.0
SAT_MIN_FRAC_DEFAULT = 0.25
SAT_BAD_COL_RATE_DEFAULT = 0.5


def scene_instrument(scene_id: str) -> str:
    """'AV3', 'AV5', or 'unknown' from a scene id."""
    m = re.match(r"(AV[35])", str(scene_id))
    return m.group(1) if m else "unknown"


def wide_index_screen(
    rdn: np.ndarray,
    wl_nm: np.ndarray,
    bm: np.ndarray,
    *,
    index_min: float = WIDE_INDEX_MIN_DEFAULT,
    lo_nm: float = WIDE_LO_NM_DEFAULT,
    hi_nm: float = WIDE_HI_NM_DEFAULT,
    sat_lo_nm: float = SAT_LO_NM_DEFAULT,
    sat_hi_nm: float = SAT_HI_NM_DEFAULT,
    sat_min_frac: float = SAT_MIN_FRAC_DEFAULT,
    sat_bad_col_rate: float = SAT_BAD_COL_RATE_DEFAULT,
    band_distance_max_nm: float = HFDI_BAND_DISTANCE_MAX_NM,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-fit fire screen for AVIRIS-5. Returns (keep_flat, index_flat).

    Keeps a pixel when

        (L_hi - L_lo) / (L_hi + L_lo) > index_min

    with lo = 1650 nm and hi = 2430 nm, OR when it is saturated over a SWIR2
    window.

    The same normalised-difference form as HFDI, on a wider band pair. HFDI
    uses 2061/2430, both inside one atmospheric window, so a cloud is
    spectrally flat across them and scores near zero with nothing burning: on
    an AVIRIS-5 line over pyrocumulus that flagged 264,731 cloud pixels. Across
    1650 to 2430, water and ice absorb strongly near 2.4 um, so cloud
    reflectance falls while fire emission rises. Measured on that line, cloud
    sits below -0.70 and fire above -0.50.

    The saturation term exists because the index goes blind on the hottest
    pixels: where the bands clip, the numerator collapses and a fire core reads
    as background. It is tested over a window rather than one band, because a
    single bad detector pixel flags an entire column under a single-band test,
    and columns saturating on more than sat_bad_col_rate of their lines are
    rejected as artifacts.
    """
    wl = np.asarray(wl_nm, dtype=np.float64)
    n_bands = int(wl.size)
    shape2d = np.asarray(rdn).shape[:2]
    rdn2 = np.asarray(rdn, dtype=np.float32).reshape(-1, n_bands)

    lo_idx, lo_act = _nearest_band(wl, lo_nm)
    hi_idx, hi_act = _nearest_band(wl, hi_nm)
    d_lo, d_hi = abs(lo_act - float(lo_nm)), abs(hi_act - float(hi_nm))
    if d_lo > float(band_distance_max_nm) or d_hi > float(band_distance_max_nm):
        if verbose:
            print(f"[screen] WARNING: nearest sensor bands are far from the "
                  f"index targets ({lo_nm:.1f} -> {lo_act:.2f}, d={d_lo:.2f}; "
                  f"{hi_nm:.1f} -> {hi_act:.2f}, d={d_hi:.2f}); screen "
                  f"DISABLED for this scene.")
        return np.ones(rdn2.shape[0], dtype=bool), np.full(rdn2.shape[0], np.nan)

    L_lo = rdn2[:, lo_idx].astype(np.float64)
    L_hi = rdn2[:, hi_idx].astype(np.float64)
    denom = L_hi + L_lo
    index = np.full(L_lo.shape, np.nan)
    ok = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    index[ok] = (L_hi[ok] - L_lo[ok]) / denom[ok]
    c_idx = np.isfinite(index) & (index > float(index_min))

    c_sat = np.zeros_like(c_idx)
    n_bad_col = 0
    win = np.flatnonzero((wl >= float(sat_lo_nm)) & (wl <= float(sat_hi_nm)))
    if win.size and bm is not None:
        bm_flat = np.asarray(bm, dtype=np.uint8).reshape(c_idx.size, -1)
        sat = np.stack([_band_saturated(bm_flat, n_bands, int(k)) for k in win],
                       axis=-1)
        c_sat = sat.mean(axis=-1) >= float(sat_min_frac)
        # A dead column fires on nearly every line; a fire core does not.
        if len(shape2d) == 2 and shape2d[0] * shape2d[1] == c_sat.size:
            s2 = c_sat.reshape(shape2d)
            bad = s2.mean(axis=0) > float(sat_bad_col_rate)
            n_bad_col = int(bad.sum())
            if n_bad_col:
                c_sat = (s2 & ~bad[None, :]).reshape(-1)

    keep = c_idx | c_sat

    if verbose:
        n = int(c_idx.size)
        print(f"[screen] (L{hi_act:.0f} - L{lo_act:.0f})/(L{hi_act:.0f} + "
              f"L{lo_act:.0f}) > {index_min:g} OR saturated over "
              f"{sat_lo_nm:.0f}-{sat_hi_nm:.0f} nm ({win.size} bands, "
              f">= {sat_min_frac:.0%})")
        print(f"[screen]   index      {int(c_idx.sum()):9,} / {n:,}")
        print(f"[screen]   saturation {int(c_sat.sum()):9,}"
              + (f"   ({n_bad_col} bad column(s) rejected)" if n_bad_col else ""))
        print(f"[screen]   kept       {int(keep.sum()):9,} "
              f"({100.0 * keep.sum() / max(n, 1):.3f}%)")
        print(f"[screen]   recovered by saturation: "
              f"{int((c_sat & ~c_idx).sum()):,}")
        # What the legacy HFDI screen would have kept, for comparison.
        h_lo, _ = _nearest_band(wl, HFDI_LO_NM_DEFAULT)
        h_hi, _ = _nearest_band(wl, HFDI_HI_NM_DEFAULT)
        a, b = rdn2[:, h_lo].astype(np.float64), rdn2[:, h_hi].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            hf = (b - a) / (b + a)
        print(f"[screen]   for reference, HFDI > {HFDI_MIN_DEFAULT} would keep "
              f"{int(np.nansum(hf > HFDI_MIN_DEFAULT)):,}")
    return keep, index


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

    Keeps a pixel when ``HFDI > hfdi_min``. Saturation in either index band
    auto-passes: clipped radiance makes HFDI meaningless, and a pixel saturating
    at 2430 nm is a strong fire candidate that must not be dropped on a bad
    number. If either target is further than ``band_distance_max_nm`` from the
    nearest band, the screen is disabled and everything kept.
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


# FRP retrieval (NGBoost on the forward-modeled emissive radiance) -----------

def load_frp_model(model_path: str):
    """Load a pickled NGBoost regressor from a joblib file.

    Unpickling reconstructs the estimator, so ngboost and the scikit-learn/scipy
    versions it was pickled against must be importable here. Version skew shows
    up as an AttributeError or ModuleNotFoundError; the caller catches it and
    disables FRP for the run.
    """
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"FRP model not found: {model_path!r}")
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "joblib is required for --frp_model but is not installed in this "
            "environment."
        ) from e
    model = joblib.load(model_path)
    probabilistic = hasattr(model, "pred_dist")
    print(f"[frp] loaded {type(model).__name__} from {model_path} "
          f"(probabilistic={'yes' if probabilistic else 'no'})")
    return model


def _planck_grid(wl_nm: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """(N_T, bands) blackbody radiance (emissivity 1) in uW/cm^2/sr/nm."""
    import jax.numpy as jnp
    from .rt import emissive_radiance
    wl_j = jnp.asarray(wl_nm, dtype=jnp.float32)
    ones = jnp.ones_like(wl_j)
    grid = jax.vmap(lambda T: emissive_radiance(ones, T, wl_j))(
        jnp.asarray(T_grid, dtype=jnp.float32)
    )
    return np.asarray(grid, dtype=np.float32)


def transm_up_table(
    rte: dict,
    wv: np.ndarray,
    bands: int,
    *,
    wv_quant_step: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize H2OSTR and return (table, index) for upwelling transmittance.

    ``table`` is (n_unique, bands), ``index`` is (n_pix,), so per-pixel
    transmittance is ``table[index]`` with no (n_pix, bands) array. Uses the
    retrieval's quantization step, so this is the transmittance the fit saw.
    """
    step = float(max(wv_quant_step, 1e-6))
    wv = np.asarray(wv, dtype=np.float32)
    wv = np.where(np.isfinite(wv), wv, np.float32(1.0))
    q = np.maximum(np.round(wv / step) * step,
                   np.float32(WV_TEMP_CLIP_MIN)).astype(np.float32)
    keys = np.unique(q)
    table = np.zeros((keys.size, int(bands)), dtype=np.float32)
    for i, key in enumerate(keys):
        x = np.asarray([float(key)], dtype=np.float32)
        table[i] = (
            np.asarray(rte["transm_up_dif"](x), dtype=np.float32).reshape(-1)
            + np.asarray(rte["transm_up_dir"](x), dtype=np.float32).reshape(-1)
        )
    index = np.searchsorted(keys, q).astype(np.int32)
    return table, index


def forward_emissive_radiance(
    *,
    mu_hot: np.ndarray,
    sigma_hot: np.ndarray,
    mu_smol: np.ndarray,
    sigma_smol: np.ndarray,
    eta: np.ndarray,
    alpha2: np.ndarray,
    cfg: MixtureRetrievalConfig,
    wl_nm: np.ndarray,
    transm_table: Optional[np.ndarray] = None,
    transm_index: Optional[np.ndarray] = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Rebuild the forward model's emissive-radiance term, per pixel.

    Same math as ``retrieval._step2_forward``: moment-match the mixture to a
    single Gaussian over ``cfg.T_grid``, integrate Planck against it with the
    trapezoid weights ``cfg.w_T``, scale by alpha2. Pass ``transm_table`` and
    ``transm_index`` for the at-sensor thermal term
    (``alpha2 * E[B(T)] * transm_up``), the quantity the fit adds to the solar
    baseline; omit them for surface-level radiance.

    sigma_single is recomputed from the components, not read off ``sigma_map``:
    the saved sigma is the inflated posterior
    ``sigma_single * (1 + alpha_tuning * alpha2)``, which the forward model
    never used, and would widen the pdf and bias the radiance.

    Returns (n_pix, bands) float32 in uW/cm^2/sr/nm, matching the L1B cube.
    """
    mu_hot = np.asarray(mu_hot, dtype=np.float32).reshape(-1)
    sigma_hot = np.asarray(sigma_hot, dtype=np.float32).reshape(-1)
    mu_smol = np.asarray(mu_smol, dtype=np.float32).reshape(-1)
    sigma_smol = np.asarray(sigma_smol, dtype=np.float32).reshape(-1)
    eta = np.clip(np.asarray(eta, dtype=np.float32).reshape(-1), 0.0, 1.0)
    alpha2 = np.maximum(np.asarray(alpha2, dtype=np.float32).reshape(-1), 0.0)

    n_pix = mu_hot.size
    n_bands = int(np.asarray(wl_nm).size)
    T = np.asarray(cfg.T_grid, dtype=np.float32)
    wT = np.asarray(cfg.w_T, dtype=np.float32)
    """Fold the quadrature weights into Planck once: (p_T * w_T) @ planck."""
    wplanck = _planck_grid(wl_nm, T) * wT[:, None]

    out = np.empty((n_pix, n_bands), dtype=np.float32)
    bs = int(max(batch_size, 1))
    for i0 in range(0, n_pix, bs):
        i1 = min(n_pix, i0 + bs)
        e_b = eta[i0:i1]
        mu_s = e_b * mu_hot[i0:i1] + (1.0 - e_b) * mu_smol[i0:i1]
        var_s = (
            e_b * sigma_hot[i0:i1] ** 2
            + (1.0 - e_b) * sigma_smol[i0:i1] ** 2
            + e_b * (1.0 - e_b) * (mu_hot[i0:i1] - mu_smol[i0:i1]) ** 2
        )
        sig_s = np.sqrt(np.maximum(var_s, 1e-6))
        z = (T[None, :] - mu_s[:, None]) / np.maximum(sig_s[:, None], 1e-12)
        p = np.exp(-0.5 * z * z)
        p /= (np.sum(wT[None, :] * p, axis=1, keepdims=True) + 1e-12)
        e_T = (p @ wplanck) * alpha2[i0:i1][:, None]
        if transm_table is not None and transm_index is not None:
            e_T = e_T * transm_table[transm_index[i0:i1]]
        out[i0:i1] = e_T.astype(np.float32, copy=False)
    return out


"""FRPD model feature spec, from the model handoff notebook. Order is load
bearing: the seven base bands, then for each pair in ``combinations`` order the
sum, difference, ratio, and normalized difference."""
FRP_TARGET_BANDS_NM = (1025.0, 1200.0, 1550.0, 1700.0, 2100.0, 2260.0, 2360.0)

"""Emitted radiance leaves forward_emissive_radiance in uW/cm^2/sr/nm; the model
was trained in W/m^2/sr/um. The two differ by exactly this factor."""
FRP_RDN_UNIT_SCALE = 10.0

"""AVIRIS-3 instantaneous field of view, radians. Pixel area is
(IFOV * path_length)^2, which converts FRPD (W/m^2) to FRP (W)."""
FRP_IFOV_RAD = 0.56e-3

"""Nearest band must land within this distance of each target, else the model is
being fed the wrong part of the spectrum and FRP is skipped."""
FRP_BAND_DISTANCE_MAX_NM = 15.0


def frp_feature_names(target_bands_nm=FRP_TARGET_BANDS_NM) -> List[str]:
    """Feature names in model order; index i names column i of the matrix."""
    from itertools import combinations
    base = [f"b{int(round(b))}" for b in target_bands_nm]
    names = list(base)
    for v1, v2 in combinations(base, 2):
        names += [f"{v1}_plus_{v2}", f"{v1}_minus_{v2}",
                  f"{v1}_div_{v2}", f"{v1}_nd_{v2}"]
    return names


def build_frp_features(
    emissive_rdn: np.ndarray,
    wl_nm: np.ndarray,
    *,
    target_bands_nm=FRP_TARGET_BANDS_NM,
    unit_scale: float = FRP_RDN_UNIT_SCALE,
    band_distance_max_nm: float = FRP_BAND_DISTANCE_MAX_NM,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build the (n_pix, 91) FRPD feature matrix from emitted radiance.

    Selects the sensor band nearest each target, scales to the model's units,
    then emits the base bands followed by the four pairwise derived features per
    pair. Division by zero becomes NaN, which the HistGradientBoosting base
    learner handles natively.

    ``emissive_rdn`` is (n_pix, bands) in uW/cm^2/sr/nm. Raises if any target
    has no sensor band within ``band_distance_max_nm``.
    """
    from itertools import combinations

    X_full = np.asarray(emissive_rdn, dtype=np.float64)
    wl = np.asarray(wl_nm, dtype=np.float64)
    if X_full.ndim != 2 or X_full.shape[1] != wl.size:
        raise ValueError(
            f"emissive_rdn shape {X_full.shape} does not match {wl.size} bands."
        )

    idx, far = [], []
    for t in target_bands_nm:
        i = int(np.argmin(np.abs(wl - float(t))))
        d = abs(float(wl[i]) - float(t))
        if d > float(band_distance_max_nm):
            far.append(f"{t:.0f} nm -> {wl[i]:.2f} nm (dist {d:.2f})")
        idx.append(i)
    if far:
        raise ValueError(
            "FRP target bands are not represented on this sensor grid: "
            + "; ".join(far)
        )

    base_names = [f"b{int(round(b))}" for b in target_bands_nm]
    cols = {n: X_full[:, i] * float(unit_scale) for n, i in zip(base_names, idx)}

    feats = dict(cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        for v1, v2 in combinations(base_names, 2):
            x1, x2 = cols[v1], cols[v2]
            feats[f"{v1}_plus_{v2}"]  = x1 + x2
            feats[f"{v1}_minus_{v2}"] = x1 - x2
            feats[f"{v1}_div_{v2}"]   = x1 / x2
            feats[f"{v1}_nd_{v2}"]    = (x1 - x2) / (x1 + x2)

    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])
    X = np.where(np.isfinite(X), X, np.nan)

    if verbose:
        print(f"[frp] features: {X.shape[1]} columns from bands "
              + ", ".join(f"{t:.0f}->{wl[i]:.2f}nm"
                          for t, i in zip(target_bands_nm, idx))
              + f"; radiance scaled x{unit_scale:g}")
    return X, names


def pixel_area_m2(
    obs_full: np.ndarray,
    *,
    ifov_rad: float = FRP_IFOV_RAD,
    path_length_band: int = 0,
) -> np.ndarray:
    """Per-pixel ground area in m^2 from the OBS path length.

    OBS band 0 is path length in meters (the same band airborne.py adds to the
    LOC ground elevation to recover sensor altitude). Pixel size is
    ``ifov_rad * path_length``; area is its square. Non-finite or non-positive
    path lengths yield NaN, which drops those pixels from FRP.
    """
    obs = np.asarray(obs_full)
    if obs.ndim != 3 or obs.shape[2] <= int(path_length_band):
        raise RuntimeError(
            f"OBS cube shape {obs.shape} has no band {path_length_band} "
            f"(expected path length in meters)."
        )
    pl = np.asarray(obs[..., int(path_length_band)], dtype=np.float64)
    pl = np.where(np.isfinite(pl) & (pl > 0.0), pl, np.nan)
    return (float(ifov_rad) * pl) ** 2


def _ngboost_loc_scale(dist) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pull (loc, scale) out of an NGBoost predicted distribution.

    Tries the params dict (Normal -> {'loc', 'scale'}), then attributes, then
    ``mean()``. ``scale`` is None when the distribution has no spread parameter.
    """
    params = getattr(dist, "params", None)
    if isinstance(params, dict):
        loc = params.get("loc", params.get("mean"))
        scale = params.get("scale", params.get("s"))
        if loc is not None:
            return (np.asarray(loc, dtype=np.float64).reshape(-1),
                    None if scale is None
                    else np.asarray(scale, dtype=np.float64).reshape(-1))

    loc = getattr(dist, "loc", None)
    scale = getattr(dist, "scale", None)
    if loc is None and hasattr(dist, "mean"):
        loc = dist.mean()
    if loc is None:
        raise RuntimeError(
            f"could not extract a location parameter from a "
            f"{type(dist).__name__}; extend _ngboost_loc_scale for this "
            f"distribution."
        )
    return (np.asarray(loc, dtype=np.float64).reshape(-1),
            None if scale is None
            else np.asarray(scale, dtype=np.float64).reshape(-1))


def retrieve_frp(
    *,
    model,
    features: np.ndarray,
    valid_flat_idx: np.ndarray,
    map_shape: Tuple[int, int],
    area_m2: Optional[np.ndarray] = None,
    fill_value: float = USER_FILL_VALUE,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict per-pixel FRP with the NGBoost FRPD model.

    The model returns fire radiative power DENSITY in W/m^2. Passing
    ``area_m2`` (per-pixel ground area, from :func:`pixel_area_m2`) converts to
    FRP in megawatts; omit it to return FRPD unchanged. The predicted sigma
    scales by the same factor, since area enters as a linear multiplier.

    ``features`` is (n_retrieved, n_feat); ``valid_flat_idx`` maps those rows
    back into the row-major flattened (rows, cols) grid. Returns (frp_map,
    unc_map), both (rows, cols) float32 filled with ``fill_value`` outside the
    retrieved set. ``unc_map`` is None for a non-probabilistic model.
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
    idx = np.asarray(valid_flat_idx, dtype=np.int64).reshape(-1)
    if X.shape[0] != idx.size:
        raise ValueError(
            f"feature rows ({X.shape[0]}) do not match the retrieved-pixel "
            f"count ({idx.size})."
        )

    """One pass over the ensemble, not two: pred_dist already carries the mean
    as its location parameter, so calling predict() as well would double the
    tree traversal -- the dominant cost of FRP on CPU. predict() is the
    fallback only when the model exposes no distribution."""
    t_pred = time.perf_counter()
    loc, scale = None, None
    if hasattr(model, "pred_dist"):
        try:
            loc, scale = _ngboost_loc_scale(model.pred_dist(X))
        except Exception as e:
            print(f"[frp] WARNING: pred_dist failed ({e!r}); falling back to "
                  f"the point estimate.")
    if loc is None:
        loc = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    if verbose:
        print(f"[frp] model inference on {X.shape[0]} pixels took "
              f"{time.perf_counter() - t_pred:.2f}s")
    if loc.size != idx.size:
        raise ValueError(
            f"model returned {loc.size} predictions for {idx.size} pixels."
        )

    """FRPD (W/m^2) -> FRP (MW). Area is a linear factor, so sigma scales with
    it; pixels whose path length was unusable carry NaN area and drop out."""
    unit = "FRPD (W/m^2)"
    if area_m2 is not None:
        a = np.asarray(area_m2, dtype=np.float64).reshape(-1)[idx]
        loc = loc * a / 1.0e6
        if scale is not None:
            scale = scale * a / 1.0e6
        unit = "FRP (MW)"

    n_total = int(map_shape[0]) * int(map_shape[1])

    def _scatter(vals):
        out = np.full((n_total,), np.float32(fill_value), dtype=np.float32)
        v = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(v)
        out[idx[ok]] = v[ok].astype(np.float32)
        return out.reshape(map_shape)

    frp_map = _scatter(loc)
    unc_map = _scatter(scale) if scale is not None else None

    if verbose:
        fin = loc[np.isfinite(loc)]
        if fin.size:
            print(f"[frp] {unit} over {fin.size} pixels: "
                  f"min={fin.min():.4g} median={np.median(fin):.4g} "
                  f"max={fin.max():.4g} total={fin.sum():.4g}")
        else:
            print(f"[frp] no finite {unit} predictions")
    return frp_map, unc_map


def _write_frp_products(
    frp_map: np.ndarray,
    frp_unc_map: Optional[np.ndarray],
    scene_id: str,
    args,
    fill: float,
    common_meta: Dict[str, str],
    written: Dict[str, str],
    *,
    unit_txt: str,
) -> None:
    """Write the fireFRP / fireFRPUnc ENVI pair.

    Shared by the normal path and the no-fire path so a quiet scene yields the
    same four products as a burning one. ``frp_unc_map`` of None means the model
    is not probabilistic, in which case an all-fill sigma is written rather than
    omitting the file, keeping the product set fixed.
    """
    bin_path, _ = write_envi_single_band(
        output_stem(scene_id, "frp", args.to_ortho_dir), frp_map,
        description=f"{scene_id} fire radiative power ({unit_txt}), pre-ortho",
        band_name="fireFRP",
        data_ignore_value=fill,
        extra_metadata=common_meta,
    )
    written["frp"] = bin_path

    if frp_unc_map is None:
        print("[frp] model exposes no predictive distribution; writing an "
              "all-fill fireFRPUnc so the product set stays fixed.")
        frp_unc_map = np.full(np.asarray(frp_map).shape, np.float32(fill),
                              dtype=np.float32)
    bin_path, _ = write_envi_single_band(
        output_stem(scene_id, "frp_unc", args.to_ortho_dir), frp_unc_map,
        description=(f"{scene_id} fire radiative power 1-sigma "
                     f"({unit_txt}), pre-ortho"),
        band_name="fireFRPUnc",
        data_ignore_value=fill,
        extra_metadata=common_meta,
    )
    written["frp_unc"] = bin_path


# Per-scene pipeline ---------------------------------------------------------

def process_scene(
    scene_id: str,
    paths: dict,
    args,
    rte_builder: "_RteBuilder",
    frp_model=None,
    wv_carry: Optional["WaterVaporCarry"] = None,
) -> Dict[str, str]:
    """Run the retrieval for one ready scene and write its pre-ortho ENVI products.

    Returns a dict of product key -> binary path. Raises if a required ENVI cube
    is missing or malformed (callers in run_watcher catch and log).
    """
    t_scene = time.perf_counter()
    print("")
    print(f"[scene] === {scene_id} ===")
    for kind in REQUIRED_KIND_KEYS:
        print(f"  {kind:<4} = {paths[kind]}")

    if not paths.get("obs"):
        raise RuntimeError(
            f"scene {scene_id}: OBS file is required but not present in `paths`. "
            "discover_ready_scenes should have left this scene in `partial`."
        )

    def _scene_id_from_path(p: str) -> Optional[str]:
        parsed = _parse_scene_id(os.path.basename(p))
        return parsed[0] if parsed else None

    """Scene id, and with it the T<HHMMSS> stamp, comes verbatim from the RDN
    filename. No GLT 'description' lookup and no GPS/UTC conversion: whatever
    the flight computer stamped is what the products are named with."""
    scene_id_from_rdn = _scene_id_from_path(paths["rdn"])
    if scene_id_from_rdn is None:
        raise RuntimeError(
            f"scene {scene_id}: could not parse a scene id from RDN filename "
            f"{os.path.basename(paths['rdn'])!r}; expected pattern "
            f"AV3<YYYYMMDD>[tT]<HHMMSS>[_NNN]."
        )

    per_kind_ids = {kind: _scene_id_from_path(paths[kind]) for kind in paths}
    mismatched = {k: sid for k, sid in per_kind_ids.items() if sid != scene_id_from_rdn}
    if mismatched:
        print(f"[scene] WARNING: scene id mismatch across input files. "
              f"RDN derives scene_id={scene_id_from_rdn!r}, but: "
              + ", ".join(f"{k}={sid!r}" for k, sid in mismatched.items())
              + ". Using the RDN's scene_id; investigate the upstream writer "
              "if this fires.")
    if scene_id_from_rdn != scene_id:
        print(f"[scene] WARNING: watcher grouped this scene under "
              f"{scene_id!r}, but the RDN filename derives "
              f"{scene_id_from_rdn!r}.")
    scene_id = scene_id_from_rdn

    ft_stem = output_stem(scene_id, "fire_temp", args.to_ortho_dir)
    ftu_stem = output_stem(scene_id, "fire_temp_unc", args.to_ortho_dir)
    print(f"[scene] processing scene_id = {scene_id!r} (system time from the RDN "
          f"filename; outputs: {os.path.basename(ft_stem)}[.hdr], "
          f"{os.path.basename(ftu_stem)}[.hdr])")

    rdn_ds = envi.open(paths["rdn"])
    mask_ds = envi.open(paths["mask"])
    loc_ds = envi.open(paths["loc"])
    obs_ds = envi.open(paths["obs"])

    rdn_full = rdn_ds.open_memmap(interleave="bip")
    bm_full = mask_ds.open_memmap(interleave="bip")
    loc_full = loc_ds.open_memmap(interleave="bip")
    obs_full = obs_ds.open_memmap(interleave="bip")

    wl_nm = _ensure_nm(np.array([float(x) for x in rdn_ds.metadata["wavelength"]],
                                dtype=np.float64))

    """FWHM source priority: the RDN header 'fwhm' field; else the
    --sensor_wl_fwhm .npz cache, used directly when band counts match and
    linearly interpolated otherwise; else error. Onboard AVIRIS-3 L1B headers
    do not carry FWHM."""
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
            print(f"[spec] header lacks fwhm; cache shapes match, using cache "
                  f"directly: {wl_npz} ({fwhm_nm.size} bands)")
        else:
            """np.interp needs ascending xp, which the hpc-derived cache is.
            Out-of-range queries clip to endpoints, safe at the ~1 nm offsets
            between the hpc and onboard grids."""
            fwhm_nm = np.interp(wl_nm, wl_cache_npz, fwhm_cache_npz)
            donor_max = float(np.abs(
                wl_nm - wl_cache_npz[np.abs(wl_nm[:, None] - wl_cache_npz[None, :]).argmin(axis=1)]
            ).max())
            print(f"[spec] header lacks fwhm; interpolated cache fwhm "
                  f"({wl_cache_npz.size} bands) onto header wl ({wl_nm.size} bands); "
                  f"max donor distance {donor_max:.3f} nm")

    """FWHM/WL pairing sanity print. AVIRIS-3 fwhm is nearly uniform so pairing
    can't be verified rigorously, but endpoints and aggregate stats catch wrong
    units, negatives, off-scale values, and unpairing."""
    _step_nm_med = float(np.median(np.abs(np.diff(wl_nm)))) if wl_nm.size >= 2 else 0.0
    _fwhm_min = float(np.min(fwhm_nm)); _fwhm_med = float(np.median(fwhm_nm))
    _fwhm_max = float(np.max(fwhm_nm))
    print(f"[spec] wl/fwhm pairing: wl[0]={wl_nm[0]:.2f} nm fwhm[0]={fwhm_nm[0]:.3f} nm | "
          f"wl[-1]={wl_nm[-1]:.2f} nm fwhm[-1]={fwhm_nm[-1]:.3f} nm")
    print(f"[spec] fwhm stats (nm): min={_fwhm_min:.3f} median={_fwhm_med:.3f} "
          f"max={_fwhm_max:.3f}; median band step={_step_nm_med:.3f}; "
          f"fwhm/step ratio (median)="
          f"{_fwhm_med / _step_nm_med if _step_nm_med > 0 else float('nan'):.3f}")
    _fwhm_warns = []
    if _fwhm_min < 0.5:
        _fwhm_warns.append(f"min fwhm={_fwhm_min:.3f} nm is implausibly small (<0.5 nm); "
                           f"unit conversion or pairing may be wrong")
    if _fwhm_max > 50.0:
        _fwhm_warns.append(f"max fwhm={_fwhm_max:.3f} nm is implausibly large (>50 nm)")
    if _step_nm_med > 0 and not (0.5 <= (_fwhm_med / _step_nm_med) <= 3.0):
        _fwhm_warns.append(f"fwhm/step ratio {_fwhm_med / _step_nm_med:.3f} is outside "
                           f"typical 0.5-3.0 range; fwhm may be in wrong units or "
                           f"unpaired from wl")
    for _w in _fwhm_warns:
        print(f"[spec] WARNING: {_w}")

    """Onboard RDN headers list wavelengths SWIR-first; downstream code assumes
    ascending. Reverse wl/fwhm, the radiance cube, and the bandmask. A
    bit-packed bandmask is unpacked, reversed, and repacked, since reversing
    whole bytes does not reverse the bits inside each byte."""
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
    print(f"[geom] SZA={sza:.3f} deg  cos(SZA)={csza:.4f}  elev={elev_km:.3f} km  "
          f"alt={alt_km:.3f} km")

    """Night decision. 6S is given a capped solar zenith either way, so a
    twilight or night scene cannot blow up the LUT build; past
    --night_sza_deg the scene is additionally treated as having no solar
    path at all."""
    is_night = bool(args.night_mode) and sza > float(args.night_sza_deg)
    lut_sza = min(float(sza), float(args.max_lut_sza_deg))
    if lut_sza < float(sza):
        print(f"[geom] SZA {sza:.3f} deg exceeds --max_lut_sza_deg "
              f"{float(args.max_lut_sza_deg):.1f}; building the 6S LUT at the cap. "
              f"Upwelling transmittance is what a thermal fit uses and is "
              f"insensitive to solar zenith.")
    if is_night:
        print(f"[geom] NIGHT scene (SZA {sza:.3f} > "
              f"{float(args.night_sza_deg):.1f} deg): solar irradiance zeroed, "
              f"solar baseline drops out of both fit stages, water vapor fixed "
              f"at {float(args.night_water_vapor):.2f} g/cm^2. Retrieval is "
              f"thermal-only.")

    irr = np.loadtxt(args.solar_irr_file, comments="#")
    iwl, irr = irr.T
    irr = irr / 10.0
    from .rt import resample_spectrum
    solar_irr = resample_spectrum(irr, iwl, wl_nm, fwhm_nm).astype(np.float64)
    if is_night:
        """mult = solar_irr / cos(SZA) / pi feeds _baseline_solar, so zeroing
        the irradiance makes the solar baseline identically 0 in both stages.
        The surface endmember weights stop affecting the loss and stay at their
        simplex initialization; only the thermal terms are fit."""
        solar_irr = np.zeros_like(solar_irr)

    """Build or fetch the date-cached RTE. The returned dict is the per-pixel
    lookup cache; rte_builder owns the on-disk LUT subdir and releases it at
    watcher exit, not per scene."""
    """A night scene's LUT is built at the capped solar zenith, so it must not
    share the per-date cache slot with daytime scenes flown the same date --
    otherwise whichever came first would hand its atmosphere to the other. The
    builder keys on the date parsed from the scene id and falls back to the
    whole string when that fails, so a 'NIGHT-<date>' id gives night scenes
    their own per-date slot."""
    lut_key_id = scene_id
    if is_night:
        m_date = re.match(r"AV[35](\d{8})[tT]", str(scene_id))
        lut_key_id = f"NIGHT-{m_date.group(1) if m_date else scene_id}"
    rte, _lut_subdir = rte_builder.build(
        lut_key_id, lut_sza, elev_km, alt_km, wl_nm, fwhm_nm
    )

    if is_night:
        """The 940 nm band-depth retrieval reads reflected sunlight, so it
        returns noise or NaN at night -- and NaN water vapor is the retrieval's
        do-not-retrieve flag, which would empty the scene. Prefer the median
        carried from the last daylight scene of this flight; fall back to
        --night_water_vapor when the watcher started after dark."""
        carried = wv_carry.value if wv_carry is not None else None
        if carried is not None and np.isfinite(carried):
            wv_req, wv_src = float(carried), (
                f"carried from daylight scene {wv_carry.source!r}"
                + (f" at SZA {wv_carry.sza:.1f} deg" if wv_carry.sza is not None else "")
            )
        else:
            wv_req, wv_src = float(args.night_water_vapor), (
                "--night_water_vapor (no daylight scene seen this flight)"
            )

        wv_lut = np.asarray(rte_builder.water_vapors, dtype=np.float64)
        wv_night = float(np.clip(wv_req,
                                 max(float(wv_lut.min()), WV_TEMP_CLIP_MIN),
                                 float(wv_lut.max())))
        if wv_night != wv_req:
            print(f"[water] H2OSTR {wv_req:.2f} is outside the LUT range "
                  f"[{wv_lut.min():.2f}, {wv_lut.max():.2f}]; clipped to "
                  f"{wv_night:.2f}.")
        water = np.full(rdn_full.shape[:2], np.float32(wv_night), dtype=np.float32)
        print(f"[water] night scene: constant H2OSTR {wv_night:.2f} g/cm^2, "
              f"{wv_src}. Water vapor still scales upwelling transmittance, so "
              f"this affects the thermal amplitude and therefore FRP.")
    else:
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

    if not is_night and wv_carry is not None:
        wv_carry.update(water, scene_id, sza)

    lib_spectra = load_fire_library_resampled(args.fire_lib_csv, wl_nm, fwhm_nm).astype(np.float64)

    bit_mask_1d = make_bitwise_bad_range_mask_1d(wl_nm, BAD_RANGES_NM_DEFAULT)
    surface_1d = bit_mask_1d & ranges_to_mask(
        wl_nm, _parse_ranges(args.surface_window_ranges_nm, SURFACE_RANGES_NM_DEFAULT))
    hot_1d = bit_mask_1d & ranges_to_mask(
        wl_nm, _parse_ranges(args.hot_window_ranges_nm, HOT_RANGES_NM_DEFAULT))
    smolder_1d = bit_mask_1d & ranges_to_mask(
        wl_nm, _parse_ranges(args.smolder_window_ranges_nm, SMOLDER_RANGES_NM_DEFAULT))
    eta_swir2_1d = ranges_to_mask(
        wl_nm,
        np.array([[float(args.eta_swir2_lo_nm), float(args.eta_swir2_hi_nm)]],
                 dtype=np.float64),
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

    """Pre-fit HFDI screen. Failing pixels are dropped by setting their water
    vapor to NaN, the retrieval's existing 'do not retrieve' channel, so
    retrieval.py is untouched. Screened pixels never enter a batch's valid set
    and land in the products as --fill_value."""
    if bool(args.hfdi_filter):
        # AVIRIS-3 keeps hfdi_screen unchanged. AVIRIS-5 uses the
        # 1650/2430 wide index plus windowed saturation.
        _inst = scene_instrument(scene_id)
        if _inst == "AV5":
            keep_flat, _hfdi = wide_index_screen(
                rdn, wl_nm, bm_raw,
                index_min=float(getattr(args, "wide_index_min",
                                        WIDE_INDEX_MIN_DEFAULT)),
                verbose=True,
            )
        else:
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
        mu0_T=args.mu0_T, lambda_mu_cold=args.lambda_mu_cold,
        alpha_cold_thresh=args.alpha_cold_thresh,
        hot_window_weight=args.hot_window_weight,
        smolder_window_weight=args.smolder_window_weight,
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
    qc_str = f"; band_diff_masked={n_qc}" if args.qc_band_diff_filter else ""
    print(f"[retrieval] valid pixels: {out['n_valid']} in {out['elapsed']:.2f}s{qc_str}")

    """Per-scene stats over finite pixels only. mu_map is E[T] (K), sigma_map
    the moment-matched single-Gaussian sigma_post (K)."""
    mu_arr = np.asarray(out["mu_map"], dtype=np.float32)
    sigma_arr = np.asarray(out["sigma_map"], dtype=np.float32)
    mu_finite = np.isfinite(mu_arr)
    n_mu_finite = int(mu_finite.sum())
    if n_mu_finite > 0:
        mu_vals = mu_arr[mu_finite]
        sigma_vals = sigma_arr[np.isfinite(sigma_arr)]
        mu_pcts = np.percentile(mu_vals, [5, 50, 95])
        print(f"[stats] E[T] (K) over {n_mu_finite} valid pixels: "
              f"min={float(mu_vals.min()):.1f}  median={float(mu_pcts[1]):.1f}  "
              f"max={float(mu_vals.max()):.1f}  p05={float(mu_pcts[0]):.1f}  "
              f"p95={float(mu_pcts[2]):.1f}")
        if sigma_vals.size > 0:
            print(f"[stats] sigma_post (K): min={float(sigma_vals.min()):.1f}  "
                  f"median={float(np.median(sigma_vals)):.1f}  "
                  f"max={float(sigma_vals.max()):.1f}")
    else:
        print("[stats] no finite E[T] pixels in this scene")

    """Products are written unwarped, in raw (line, sample) geometry, so the
    ortho step can apply this scene's GLT directly. Non-retrieved pixels carry
    --fill_value, echoed in the header as 'data ignore value'."""
    fill = float(args.fill_value)
    common_meta = {
        "scene id": scene_id,
        "sensor type": "AVIRIS-3",
        "isofire product level": "pre-ortho (apply GLT downstream)",
    }
    written: Dict[str, str] = {}

    os.makedirs(args.to_ortho_dir, exist_ok=True)
    bin_path, _ = write_envi_single_band(
        ft_stem, out["mu_map"],
        description=f"{scene_id} fire temperature E[T] (K), pre-ortho",
        band_name="fireTemp",
        data_ignore_value=fill,
        extra_metadata=common_meta,
    )
    written["fire_temp"] = bin_path

    bin_path, _ = write_envi_single_band(
        ftu_stem, out["sigma_map"],
        description=f"{scene_id} fire temperature uncertainty sigma (K), pre-ortho",
        band_name="fireTempUnc",
        data_ignore_value=fill,
        extra_metadata=common_meta,
    )
    written["fire_temp_unc"] = bin_path

    """Optional FRP product. The FRPD model consumes emitted radiance, rebuilt
    here from the retrieved mixture state for the same pixels the temperature
    products cover. An FRP failure is logged; the temperature products still
    succeed."""
    # FRP is AVIRIS-3 only. The NGBoost model was trained on AVIRIS-3
    # radiance and AVIRIS-5's response differs (FWHM ~5.8 vs ~7.4 nm).
    if frp_model is not None and scene_instrument(scene_id) == "AV5":
        print(f"[frp] skipping FRP for {scene_id}: the model is AVIRIS-3 only.")
        frp_model = None

    if frp_model is not None:
        try:
            succ = np.asarray(out["success_map"], dtype=bool)
            idx = np.where(succ.reshape(-1))[0]
            if idx.size == 0:
                """No fire in the scene is a valid result, not a failure. Write
                all-fill products anyway so every scene yields the same four
                files and the ortho step downstream never has to special-case a
                quiet scene."""
                print(f"[frp] no retrieved pixels in {scene_id}; writing "
                      f"all-fill FRP products ({fill:g}).")
                frp_map = np.full(out["mu_map"].shape, np.float32(fill),
                                  dtype=np.float32)
                frp_unc_map = frp_map.copy()
                _write_frp_products(
                    frp_map, frp_unc_map, scene_id, args, fill,
                    common_meta, written,
                    unit_txt="W/m^2" if bool(args.frp_density) else "MW/pixel",
                )
            else:
                at_sensor = str(args.frp_radiance_level).lower() == "toa"
                t_tab = t_idx = None
                if at_sensor:
                    t_tab, t_idx = transm_up_table(
                        rte,
                        np.asarray(water, dtype=np.float32).reshape(-1)[idx],
                        int(wl_nm.size),
                        wv_quant_step=float(args.wv_quant_step),
                    )
                emis = forward_emissive_radiance(
                    mu_hot=out["mu_hot_map"].reshape(-1)[idx],
                    sigma_hot=out["sigma_hot_map"].reshape(-1)[idx],
                    mu_smol=out["mu_smol_map"].reshape(-1)[idx],
                    sigma_smol=out["sigma_smol_map"].reshape(-1)[idx],
                    eta=out["eta_map"].reshape(-1)[idx],
                    alpha2=out["alpha2_map"].reshape(-1)[idx],
                    cfg=cfg,
                    wl_nm=wl_nm,
                    transm_table=t_tab,
                    transm_index=t_idx,
                    batch_size=int(args.batch_size),
                )
                print(f"[frp] {'at-sensor' if at_sensor else 'surface'} emitted "
                      f"radiance for {idx.size} retrieved pixels")

                X, _names = build_frp_features(
                    emis, wl_nm,
                    unit_scale=float(args.frp_rdn_unit_scale),
                    verbose=True,
                )

                """Area from the OBS path length turns FRPD into FRP. Skipped
                with --frp_density, which writes W/m^2 instead."""
                area = None
                if not bool(args.frp_density):
                    area = pixel_area_m2(
                        obs_full, ifov_rad=float(args.frp_ifov_rad)
                    )
                    a_ok = np.isfinite(area.reshape(-1)[idx])
                    print(f"[frp] pixel area from OBS path length "
                          f"(IFOV={float(args.frp_ifov_rad):.3g} rad): "
                          f"usable on {int(a_ok.sum())}/{idx.size} pixels; "
                          f"median={np.nanmedian(area):.2f} m^2")

                frp_map, frp_unc_map = retrieve_frp(
                    model=frp_model,
                    features=X,
                    valid_flat_idx=idx,
                    map_shape=out["mu_map"].shape,
                    area_m2=area,
                    fill_value=fill,
                    verbose=True,
                )

                _write_frp_products(
                    frp_map, frp_unc_map, scene_id, args, fill,
                    common_meta, written,
                    unit_txt="W/m^2" if bool(args.frp_density) else "MW/pixel",
                )
        except Exception as e:
            print(f"[frp] SKIPPED for {scene_id}: {e!r}")

    for key, path in written.items():
        print(f"[output] {key:<13} = {path} (+ .hdr)")
    print(f"[scene] {scene_id} done in {time.perf_counter() - t_scene:.2f}s")
    return written


# Watcher --------------------------------------------------------------------

class SceneTimeout(RuntimeError):
    """Raised when a scene's required inputs never arrived in time."""


def run_watcher(args) -> dict:
    """Discover and process AV3 scenes from a single ``--input_dir``.

    Polls every ``args.poll_seconds``. A scene still missing required kinds
    ``args.scene_timeout_seconds`` after its first file appeared is abandoned,
    and with ``args.exit_on_timeout`` the watcher stops and exits nonzero. L1B
    inputs are never deleted. Returns a summary dict.
    """
    for fld, lbl in [("fire_lib_csv", FIRE_LIB_FILENAME),
                     ("solar_irr_file", SOLAR_IRR_FILENAME)]:
        v = getattr(args, fld)
        if not v or not os.path.exists(v):
            raise FileNotFoundError(
                f"--{fld} not provided and no '{lbl}' found under "
                f"$ISOFIRE_DATA_DIR or <repo>/data. Pass --{fld} explicitly "
                f"or commit the file to the repo's data/ directory."
            )

    if not args.input_dir:
        raise FileNotFoundError(
            "--input_dir is required (single directory containing the ENVI "
            "files for each scene)."
        )
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"--input_dir does not exist: {args.input_dir}")
    if not args.to_ortho_dir:
        raise ValueError("--to_ortho_dir is required.")
    os.makedirs(args.to_ortho_dir, exist_ok=True)

    print("[watch]")
    print(f"  input_dir     = {args.input_dir}")
    print(f"  to_ortho_dir  = {args.to_ortho_dir}")
    print(f"[knobs] dark_thresh={args.dark_thresh}  alpha_tuning={args.alpha_tuning}  "
          f"sigma_min_k={args.sigma_min_k}  skip_dark_pixels={bool(args.skip_dark_pixels)}  "
          f"poll_seconds={args.poll_seconds}  overwrite={bool(args.overwrite)}")
    print(f"[knobs] scene_timeout={float(args.scene_timeout_seconds):.0f}s  "
          f"settle={float(args.settle_seconds):.1f}s  "
          f"exit_on_timeout={bool(args.exit_on_timeout)}  "
          f"fill_value={float(args.fill_value):g}")
    print("[cleanup] L1B inputs (RDN/LOC/BANDMASK/OBS) are NEVER deleted by this "
          "entry point.")
    print("[ortho] disabled: products are written unwarped as ENVI pairs for a "
          "downstream ortho step.")
    if float(args.eta_floor_max) > 0.0:
        ceiling_note = (
            f"; ceiling {args.eta_ceiling_low_sat:.2f} when swir2_sat < "
            f"{args.eta_sat_frac_low:.3f}"
            if float(args.eta_ceiling_low_sat) < 1.0 else " (no ceiling)"
        )
        print(f"[knobs] pre-fit eta hard floor: sat threshold "
              f"high={args.eta_sat_frac_high:.3f}; "
              f"floor(swir2_only={args.eta_floor_swir2_only_sat:.2f}, "
              f"swir1+swir2={args.eta_floor_swir1_and_swir2_sat:.2f})"
              f"{ceiling_note}")

    """FRP model loaded once at startup; a load failure disables FRP for the
    run rather than killing the temperature products."""
    frp_model = None
    if getattr(args, "frp_model", None):
        try:
            frp_model = load_frp_model(args.frp_model)
        except Exception as e:
            print(f"[frp] WARNING: could not load {args.frp_model!r} "
                  f"({e!r}); FRP retrieval disabled for this run.")
            frp_model = None

    """6S LUTs are built once per calendar date and reused across every scene
    flown that date (see _RteBuilder)."""
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

    """Persisted beside the LUT scratch root, which is bound to real disk on the
    flight computer, so a watcher restarted after dark still finds the last
    daylight scene's water vapor."""
    wv_carry_path = getattr(args, "wv_carry_file", None)
    if not wv_carry_path:
        wv_carry_path = os.path.join(
            str(args.lut_dir) if args.lut_dir else "/tmp/isofire_lut",
            "wv_carry.json",
        )
    wv_carry = WaterVaporCarry(wv_carry_path)
    print(f"[water] daylight H2OSTR carry file: {wv_carry_path}")

    seen_scenes: set = set()
    first_seen: Dict[str, float] = {}
    processed: List[Tuple[str, Dict[str, str]]] = []
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []
    timed_out: List[Tuple[str, List[str], float]] = []

    timeout_s = float(args.scene_timeout_seconds)
    stop_reason = "interrupted"

    try:
        waiting_logged = False
        while True:
            now = time.time()
            ready, partial = discover_ready_scenes(
                input_dir=args.input_dir,
                settle_seconds=float(args.settle_seconds),
            )

            """Start the timeout clock when a scene's first file shows up."""
            for sid in list(ready) + list(partial):
                first_seen.setdefault(sid, now)

            new_ready = [c for c in sorted(ready)
                         if c not in seen_scenes
                         and c not in {s for s, _m, _a in timed_out}]
            if new_ready:
                print(f"[watch] {len(new_ready)} new ready scene(s): {new_ready}")
                waiting_logged = False

            for scene_id in new_ready:
                seen_scenes.add(scene_id)

                stems = [output_stem(scene_id, p, args.to_ortho_dir)
                         for p in ("fire_temp", "fire_temp_unc")]
                if not bool(args.overwrite) and all(_product_exists(s) for s in stems):
                    print(f"[watch] {scene_id}: products exist "
                          f"({os.path.basename(stems[0])} / "
                          f"{os.path.basename(stems[1])}), skipping "
                          f"(use --overwrite to redo)")
                    skipped.append((scene_id, "products already exist"))
                    continue

                try:
                    written = process_scene(scene_id, ready[scene_id], args,
                                            rte_builder, frp_model=frp_model,
                                            wv_carry=wv_carry)
                    processed.append((scene_id, written))
                except Exception as e:
                    """One bad scene must not kill the watcher."""
                    warnings.warn(f"scene {scene_id} failed: {e!r}", stacklevel=2)
                    failed.append((scene_id, repr(e)))

            """Input-wait timeout: a scene partial for longer than
            --scene_timeout_seconds is abandoned and never retried, so files
            landing later can't wake the watcher hours after the fact."""
            timed_out_ids = {s for s, _m, _a in timed_out}
            for sid in sorted(partial):
                if sid in seen_scenes or sid in timed_out_ids:
                    continue
                age = now - first_seen.get(sid, now)
                if age >= timeout_s:
                    print(f"[watch] TIMEOUT: scene {sid} still missing "
                          f"{partial[sid]} after {age:.0f}s "
                          f"(limit {timeout_s:.0f}s); abandoning it.")
                    timed_out.append((sid, list(partial[sid]), age))
                    if bool(args.exit_on_timeout):
                        raise SceneTimeout(
                            f"scene {sid} incomplete after {age:.0f}s "
                            f"(missing {partial[sid]})"
                        )

            if partial:
                pending = {c: partial[c] for c in sorted(partial)[:20]}
                print(f"[watch] partial scenes (waiting on required kinds): {pending}"
                      + (f" (+{len(partial) - 20} more)" if len(partial) > 20 else ""))

            if not new_ready and not waiting_logged:
                print(f"[watch] waiting for scenes "
                      f"(seen={len(seen_scenes)} processed={len(processed)} "
                      f"skipped={len(skipped)} failed={len(failed)} "
                      f"timed_out={len(timed_out)}; "
                      f"polling every {float(args.poll_seconds):.1f}s; "
                      f"scene assembly limit {timeout_s:.0f}s; "
                      f"Ctrl-C / SIGTERM to exit)", flush=True)
                waiting_logged = True

            time.sleep(float(args.poll_seconds))
    except KeyboardInterrupt:
        print("[watch] interrupted by user (Ctrl-C); exiting cleanly.")
    except SceneTimeout as e:
        stop_reason = "scene timeout"
        print(f"[watch] STOPPING: {e}. Pass --no_exit_on_timeout to keep "
              f"polling after an incomplete scene.")
    finally:
        """Single point where scratch disk is released."""
        rte_builder.cleanup()

    summary = dict(
        seen=len(seen_scenes),
        processed=len(processed),
        skipped=len(skipped),
        failed=len(failed),
        timed_out=len(timed_out),
        stop_reason=stop_reason,
        processed_scenes=[s for s, _w in processed],
        skipped_scenes=skipped,
        failed_scenes=failed,
        timed_out_scenes=timed_out,
    )
    print("")
    print("[watch] === final summary ===")
    print(f"  scenes seen:      {summary['seen']}")
    print(f"  scenes processed: {summary['processed']}")
    print(f"  scenes skipped:   {summary['skipped']}")
    print(f"  scenes failed:    {summary['failed']}")
    print(f"  scenes timed out: {summary['timed_out']}")
    return summary


# Argparse + main entry ------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AVIRIS-3 two-component Gaussian temperature retrieval "
                    "(onboard / pre-ortho entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_paths = p.add_argument_group(
        "input directory (single dir with all ENVI files for one or more scenes)"
    )
    g_paths.add_argument("--input_dir", default=DEFAULT_INPUT_DIR,
                         help="Directory containing AV3 ENVI files. Files are grouped by "
                              "scene id (AV3<date><time>[_<chunk>]). Each scene must "
                              "contain RDN, LOC, BANDMASK/sat and OBS ENVI pairs "
                              "(case-insensitive). GLT is NOT required -- "
                              "orthorectification happens downstream. Inputs are never "
                              "deleted.")

    g_out = p.add_argument_group("output directory (all products, pre-ortho ENVI)")
    g_out.add_argument("--to_ortho_dir", default=DEFAULT_TO_ORTHO_DIR,
                       help="Directory for ALL products, written as unwarped ENVI pairs "
                            "named '<scene_id>_fireTemp_toOrtho[.hdr]', "
                            "'<scene_id>_fireTempUnc_toOrtho[.hdr]' and, when FRP is "
                            "enabled, '<scene_id>_fireFRP_toOrtho[.hdr]' "
                            "(e.g. '/data/rtsp/Layers/to_ortho').")
    g_out.add_argument("--fill_value", type=float, default=USER_FILL_VALUE,
                       help="Value written for pixels that were not retrieved; also "
                            "recorded in the ENVI header as 'data ignore value'.")
    g_out.add_argument("--overwrite", action="store_true",
                       help="Re-process scenes even when output products already exist.")

    g_watch = p.add_argument_group("watcher behavior")
    g_watch.add_argument("--poll_seconds", type=float, default=USER_POLL_SECONDS,
                         help="Seconds between input-directory polls.")
    g_watch.add_argument("--scene_timeout_seconds", type=float,
                         default=USER_SCENE_TIMEOUT_SECONDS,
                         help="Give up on a scene whose required ENVI files are still "
                              "incomplete this many seconds after its FIRST file "
                              "appeared. Applies to waiting for inputs only; a running "
                              "retrieval is not preempted.")
    g_watch.add_argument("--exit_on_timeout", dest="exit_on_timeout",
                         action="store_true", default=True,
                         help="Stop the watcher (exit code 1) on the first scene "
                              "timeout, so a supervisor can restart it.")
    g_watch.add_argument("--no_exit_on_timeout", dest="exit_on_timeout",
                         action="store_false",
                         help="Abandon a timed-out scene but keep polling for others.")
    g_watch.add_argument("--settle_seconds", type=float, default=USER_SETTLE_SECONDS,
                         help="An input file counts as present only once its mtime is "
                              "this old, so a cube still being written is not opened. "
                              "Set 0 to disable.")

    g_frp = p.add_argument_group("FRP retrieval (NGBoost FRPD model)")
    g_frp.add_argument("--frp_model", default=None,
                       help="Path to the joblib-pickled NGBoost FRPD regressor. When "
                            "set, '<scene_id>_fireFRP_toOrtho' is written alongside "
                            "the temperature products, plus "
                            "'<scene_id>_fireFRPUnc_toOrtho' (the predicted 1-sigma) "
                            "when the model exposes pred_dist. FRP covers exactly the "
                            "pixels the temperature products retrieved. Requires "
                            "joblib, ngboost and scikit-learn in the image; a load "
                            "failure disables FRP without affecting temperature.")
    g_frp.add_argument("--frp_radiance_level", default="surface",
                       choices=("surface", "toa"),
                       help="Which forward-modeled emitted radiance feeds the model. "
                            "'surface' is alpha2 * E[B(T)], the radiance leaving the "
                            "fire; 'toa' multiplies by upwelling transmittance. MUST "
                            "match the ISOFire emitted-radiance product the model was "
                            "trained against.")
    g_frp.add_argument("--frp_rdn_unit_scale", type=float,
                       default=FRP_RDN_UNIT_SCALE,
                       help="Multiplier applied to emitted radiance before feature "
                            "construction. Default 10 converts uW/cm^2/sr/nm to the "
                            "W/m^2/sr/um the model was trained in.")
    g_frp.add_argument("--frp_ifov_rad", type=float, default=FRP_IFOV_RAD,
                       help="Instantaneous field of view in radians. Pixel area is "
                            "(IFOV * OBS path length)^2, converting FRPD to FRP.")
    g_frp.add_argument("--frp_density", action="store_true",
                       help="Write fire radiative power DENSITY (W/m^2) straight from "
                            "the model instead of converting to FRP (MW/pixel) via "
                            "pixel area. Use when OBS path length is unavailable or "
                            "suspect.")

    g_skip = p.add_argument_group("go/no-go: last-band darkness threshold")
    g_skip.add_argument("--dark_thresh", type=float, default=USER_DARK_THRESH,
                        help="Pixel is RETRIEVED only if rdn[..., -1] > dark_thresh.")
    g_skip.add_argument("--skip_dark_pixels", dest="skip_dark_pixels",
                        action="store_true", default=USER_SKIP_DARK_PIXELS)
    g_skip.add_argument("--no_skip_dark_pixels", dest="skip_dark_pixels",
                        action="store_false")

    g_sigma = p.add_argument_group("sigma scaling (analytic moment-matched single Gaussian)")
    g_sigma.add_argument("--alpha_tuning", type=float, default=USER_ALPHA_TUNING,
                         help="sigma_post = sigma_single * (1 + alpha_tuning * max(alpha2, 0))")
    g_sigma.add_argument("--sigma_min_k", type=float, default=USER_SIGMA_MIN_K,
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
    g_hfdi.add_argument("--wide_index_min", type=float,
                        default=WIDE_INDEX_MIN_DEFAULT,
                        help="AVIRIS-5 ONLY: keep a pixel when "
                             "(L2430 - L1650)/(L2430 + L1650) exceeds this, or "
                             "when it is saturated over 2300-2500 nm. The "
                             "wide band pair separates cloud from fire where "
                             "HFDI's 2061/2430 pair does not. Ignored for "
                             "AVIRIS-3.")
    g_hfdi.add_argument("--require_chunk", action="store_true",
                        help="Ignore scenes whose filename carries no _NNN "
                             "chunk number. For the AVIRIS-5 cluster archive, "
                             "whose unchunked full-line OBS/LOC/GLT have no "
                             "matching RDN. Leave off for flight.")
    g_hfdi.add_argument("--hfdi_min", type=float, default=HFDI_MIN_DEFAULT,
                        help="Keep a pixel only if HFDI > this value.")

    g_qc = p.add_argument_group(
        "QC: legacy band-diff filter (pre-fit; superseded by the HFDI screen)")
    g_qc.add_argument("--qc_band_diff_filter", dest="qc_band_diff_filter",
                      action="store_true", default=False,
                      help="Legacy absolute SWIR1-minus-SWIR2 filter, off by default "
                           "here. Enabling it ANDs with the HFDI screen; kept for A/B "
                           "comparison against earlier band-diff runs.")
    g_qc.add_argument("--no_qc_band_diff_filter", dest="qc_band_diff_filter",
                      action="store_false")
    g_qc.add_argument("--qc_band_lo_nm", type=float, default=1096.9,
                      help="Target wavelength for the SWIR1 diagnostic band; nearest "
                           "sensor band is used.")
    g_qc.add_argument("--qc_band_hi_nm", type=float, default=2249.4,
                      help="Target wavelength for the SWIR2 diagnostic band; nearest "
                           "sensor band is used.")
    g_qc.add_argument("--qc_band_diff_max", type=float, default=5.0,
                      help="Drop pixel if obs[lo_band] - obs[hi_band] exceeds this "
                           "(W/m^2/sr/nm).")

    g_etafloor = p.add_argument_group("Eta floor (pre-fit, saturation-driven, hard categorical)")
    g_etafloor.add_argument("--eta_sat_frac_low", type=float, default=0.02)
    g_etafloor.add_argument("--eta_sat_frac_high", type=float, default=0.08)
    g_etafloor.add_argument("--eta_floor_swir2_only_sat", type=float, default=0.75)
    g_etafloor.add_argument("--eta_floor_swir1_and_swir2_sat", type=float, default=0.85)
    g_etafloor.add_argument("--eta_ceiling_low_sat", type=float, default=1.0)
    g_etafloor.add_argument("--eta_floor_max", type=float, default=0.0,
                            help="Master gate on the pre-fit floor. 0.0 disables both "
                                 "the floor and the ceiling entirely.")

    g_night = p.add_argument_group("night / twilight handling")
    g_night.add_argument("--night_mode", dest="night_mode", action="store_true",
                         default=True,
                         help="Above --night_sza_deg, zero the solar irradiance (which "
                              "removes the solar baseline from both fit stages, leaving "
                              "a thermal-only retrieval) and replace the 940 nm water "
                              "retrieval with --night_water_vapor. Fire radiance is "
                              "unaffected, so night scenes still retrieve.")
    g_night.add_argument("--no_night_mode", dest="night_mode", action="store_false",
                         help="Treat every scene as daylight. A true night scene will "
                              "then retrieve nothing, since its water-vapor field comes "
                              "back NaN.")
    g_night.add_argument("--night_sza_deg", type=float, default=USER_NIGHT_SZA_DEG,
                         help="Solar zenith above which a scene is treated as night.")
    g_night.add_argument("--max_lut_sza_deg", type=float, default=USER_MAX_LUT_SZA_DEG,
                         help="Cap on the solar zenith handed to 6S. Applies to every "
                              "scene, so a twilight or night geometry can never blow up "
                              "the LUT build. Night scenes get their own per-date LUT "
                              "cache slot so a capped build is not reused for daylight.")
    g_night.add_argument("--night_water_vapor", type=float,
                         default=USER_NIGHT_WATER_VAPOR,
                         help="Fallback constant H2OSTR (g/cm^2) for night scenes, used "
                              "ONLY when no daylight scene has been processed this "
                              "flight. Otherwise the median from the last daylight "
                              "scene is carried forward, which is a far better estimate. "
                              "Clipped into the --water_vapors LUT range.")
    g_night.add_argument("--wv_carry_file", default=None,
                         help="Where to persist the last daylight scene's median "
                              "H2OSTR, so a watcher restarted after dark still has it. "
                              "Defaults to wv_carry.json under --lut_dir.")

    g_geom = p.add_argument_group("geometry overrides (defaults from OBS/LOC)")
    g_geom.add_argument("--sza_deg", type=float, default=None)
    g_geom.add_argument("--elev_km", type=float, default=None)
    g_geom.add_argument("--alt_km", type=float, default=None)

    g_aux = p.add_argument_group("auxiliary inputs (default to <repo>/data/)")
    g_aux.add_argument("--fire_lib_csv", default=DEFAULT_FIRE_LIB_CSV,
                       help=f"Fire endmember library CSV. Defaults to "
                            f"<repo>/data/{FIRE_LIB_FILENAME} or "
                            f"$ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--solar_irr_file", default=DEFAULT_SOLAR_IRR_FILE,
                       help=f"Top-of-atmosphere solar irradiance file. Defaults to "
                            f"<repo>/data/{SOLAR_IRR_FILENAME} or "
                            f"$ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--sensor_wl_fwhm", default=DEFAULT_SENSOR_WL_FWHM_FILE,
                       help=f"Optional .npz cache of sensor wavelengths and FWHM in nm "
                            f"(keys: wl_nm, fwhm_nm). Used as the FWHM source ONLY when "
                            f"the RDN header lacks 'fwhm' (the onboard case). Defaults "
                            f"to <repo>/data/{SENSOR_WL_FWHM_FILENAME} or "
                            f"$ISOFIRE_DATA_DIR/<filename>.")
    g_aux.add_argument("--sixs_dir", default=os.environ.get("ISOFIRE_SIXS_DIR", "./sixs/Sixs/"),
                       help="Directory containing the compiled 6S binary (sixsV2.1).")
    g_aux.add_argument("--lut_dir", default=None,
                       help="Per-process scratch parent for 6S LUT (default: /tmp/isofire_lut)")

    g_wv = p.add_argument_group("water vapor")
    g_wv.add_argument("--water_vapors", default="0.2,0.5,1.0,2.0,3.0,4.0")
    g_wv.add_argument("--water_aot550", type=float, default=0.01)
    g_wv.add_argument("--water_o3", type=float, default=0.30)
    g_wv.add_argument("--water_superpixel_size", type=int, default=12)

    g_win = p.add_argument_group("spectral fitting windows")
    g_win.add_argument("--surface_window_ranges_nm", default=str(SURFACE_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--hot_window_ranges_nm", default=str(HOT_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--smolder_window_ranges_nm", default=str(SMOLDER_RANGES_NM_DEFAULT.tolist()))
    g_win.add_argument("--eta_swir2_lo_nm", type=float, default=ETA_SWIR2_DEFAULT_NM[0])
    g_win.add_argument("--eta_swir2_hi_nm", type=float, default=ETA_SWIR2_DEFAULT_NM[1])

    g_cfg = p.add_argument_group("mixture retrieval config")
    g_cfg.add_argument("--Tmin", type=float, default=200.0)
    g_cfg.add_argument("--Tmax", type=float, default=2400.0)
    g_cfg.add_argument("--N_T", type=int, default=240)
    g_cfg.add_argument("--T_step1_init", type=float, default=900.0)
    g_cfg.add_argument("--mu_hot_min_T", type=float, default=500.0)
    g_cfg.add_argument("--mu_hot_max_T", type=float, default=2100.0)
    g_cfg.add_argument("--mu_smol_min_T", type=float, default=450.0)
    g_cfg.add_argument("--mu_smol_max_T", type=float, default=850.0)
    g_cfg.add_argument("--sigma_hot_min", type=float, default=50.0)
    g_cfg.add_argument("--sigma_hot_max", type=float, default=400.0)
    g_cfg.add_argument("--sigma_smol_min", type=float, default=100.0)
    g_cfg.add_argument("--sigma_smol_max", type=float, default=400.0)
    g_cfg.add_argument("--alpha1_min", type=float, default=1e-4)
    g_cfg.add_argument("--alpha1_max", type=float, default=10.0)
    g_cfg.add_argument("--alpha2_min", type=float, default=0.0)
    g_cfg.add_argument("--alpha2_max", type=float, default=10.0)
    g_cfg.add_argument("--eta_min", type=float, default=0.0)
    g_cfg.add_argument("--eta_max", type=float, default=1.0)
    g_cfg.add_argument("--hot_window_weight", type=float, default=1.0)
    g_cfg.add_argument("--smolder_window_weight", type=float, default=1.0)

    g_cold = p.add_argument_group("optional cold prior")
    g_cold.add_argument("--use_cold_prior", action="store_true")
    g_cold.add_argument("--mu0_T", type=float, default=500.0)
    g_cold.add_argument("--lambda_mu_cold", type=float, default=1e-3)
    g_cold.add_argument("--alpha_cold_thresh", type=float, default=1e-3)

    g_opt = p.add_argument_group("optimizer")
    g_opt.add_argument("--batch_size", type=int, default=8000)
    g_opt.add_argument("--step1_adam_steps", type=int, default=30)
    g_opt.add_argument("--step1_adam_lr", type=float, default=0.03)
    g_opt.add_argument("--step2_adam_steps", type=int, default=100)
    g_opt.add_argument("--step2_adam_lr", type=float, default=0.015)
    g_opt.add_argument("--adam_beta1", type=float, default=0.9)
    g_opt.add_argument("--adam_beta2", type=float, default=0.999)
    g_opt.add_argument("--adam_eps", type=float, default=1e-8)
    g_opt.add_argument("--adam_grad_clip", type=float, default=5.0)
    g_opt.add_argument("--wv_quant_step", type=float, default=0.02)

    return p


def main(argv=None) -> int:
    """CLI entry point. Returns 0 unless a scene failed or timed out."""
    args = _build_parser().parse_args(argv)
    summary = run_watcher(args)
    bad = int(summary.get("failed", 0)) + int(summary.get("timed_out", 0))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
