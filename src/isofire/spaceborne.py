#!/usr/bin/env python
#
#  Copyright 2026 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Authors: william.r.keely@jpl.nasa.gov
"""ISOFire for EMIT: sub-pixel fire temperature from spaceborne VSWIR.

Per scene:

1. Day or night from the median to-sun zenith (night > ``--night_sza``, day <
   ``--day_sza_max``; twilight in between is skipped unless ``--mode`` forces it).
2. Screen for hot pixels.
   * Night: L(2400) >= 0.01 and L(2249)/L(1097) >= 0.5, the recipe from the night
     scan notebook (the standard HFDI misfires on cloud at night).
   * Day: HFDI = (L2430 - L2061)/(L2430 + L2061) > ``--hfdi_min``, optionally OR a
     scene-robust HFDI z-score above ``--hfdi_z`` (catches small flares whose HFDI
     stays negative over bright ground).
   Saturation at the screen bands auto-passes when a bandmask is present.
3. Interpolate the EMIT VSWIR ISOFIT/MODTRAN LUT (the zarr from the
   EMIT+ECOSTRESS work, epoch picked by date) at the candidates' median
   geometry, elevation, AOT and CO2, keeping the H2OSTR axis. At night the
   solar terms are zeroed and only the upward transmittance is used.
4. Water vapor: day from the 940 nm band depth (super-pixel median), night a
   fixed ``--night_h2ostr`` (no reflected sun to measure it from).
5. Retrieve only the candidate pixels with ``pca_surface.run_pca_fire_retrieval``:
   day fits 8 PCA surface scores plus the thermal mixture and converts the
   scores back to reflectance; night fits the thermal mixture alone, seeded with
   the two-band (2249/1097) Planck temperature.
6. Write a per-pixel CSV, an npz with surface reflectance (day), full-scene maps,
   and ortho GeoTIFFs through the GLT when one is present.

Usage
-----
    isofire-spaceborne --scene emit20260925t075553 \\
        --acq_root /store/emit/ops/data/acquisitions \\
        --pca_basis /path/pca_surface_nonoise_balanced_PC8.npz \\
        --vswir_lut /path/to/emit_vswir_luts \\
        --out_dir results/emit

``--scene`` takes a scene dir, its ``l1b/`` dir, a file in it, or a bare fid; it
can be repeated, and ``--scene_list`` reads one per line. ``--pca_basis`` is
required for day scenes only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*float64.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*64-bit.*", category=UserWarning)

import jax
jax.config.update("jax_enable_x64", False)

try:
    from scipy import ndimage
except ImportError:          # scipy is a hard dependency of isofire; kept defensive
    ndimage = None

from .emit_io import (
    EmitScene,
    _clean,
    find_emit_scene,
    loc_band_index,
    nearest_band,
    obs_band_index,
    read_glt,
    scene_median_sza,
)
from .pca_surface import (
    PCASurfaceBasis,
    planck_stable,
    ratio_temperature,
    run_pca_fire_retrieval,
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
    create_water_estimator,
    decode_sat_ok,
    estimate_water_vapor,
    eta_bounds_from_saturation,
    get_water_wavelengths,
    make_bitwise_bad_range_mask_1d,
    ranges_to_mask,
    water_superpixel_nanmedian,
)
from .rt import resample_spectrum

FILL_NOT_RETRIEVED = -1.0    # in-scene, not retrieved (0 = out of swath in ortho)


# --------------------------------------------------------------- helpers

def _data_file(name: str) -> Optional[str]:
    from .airborne import _resolve_repo_data_file
    return _resolve_repo_data_file(name)


def _parse_ranges(s, default):
    from .airborne import _parse_ranges as _pr
    return _pr(s, default)


def _sat_map(bm_cube, n_bands: int, band_idx: int) -> np.ndarray:
    """(rows, cols) True where ``band_idx`` is flagged in a packed or per-band bandmask."""
    if bm_cube.nb == n_bands:
        return bm_cube.band(band_idx) != 0
    byte, bit = band_idx // 8, 7 - (band_idx % 8)
    return ((bm_cube.band(byte).astype(np.uint8) >> bit) & 1).astype(bool)


def _blob_filter(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1 or not mask.any() or ndimage is None:
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(min_px)
    keep[0] = False
    return keep[lab]


def _robust_z(v: np.ndarray) -> np.ndarray:
    f = np.isfinite(v)
    if f.sum() < 100:
        return np.full(v.shape, np.nan, np.float32)
    med = np.nanmedian(v[f])
    mad = 1.4826 * np.nanmedian(np.abs(v[f] - med))
    return ((v - med) / max(mad, 1e-6)).astype(np.float32)


# --------------------------------------------------------------- EMIT LUT

LUT_KEYS = ("rhoatm", "sphalb", "transm_down_dir", "transm_down_dif", "transm_up_dir", "transm_up_dif")


def pick_lut_store(path, dt) -> Path:
    """A zarr/netCDF LUT, or the store in a directory whose _<start>-<end> covers ``dt``."""
    p = Path(path).expanduser()
    if p.suffix in (".zarr", ".nc", ".h5"):
        return p
    stores = sorted(p.glob("*.zarr")) + sorted(p.glob("*.nc"))
    if not stores:
        raise FileNotFoundError(f"no .zarr or .nc LUT in {p}")
    ymd = dt.strftime("%Y%m%d") if dt else "99999999"
    for st in stores:
        m = re.search(r"_(\d{8})-(\d{8})\.(zarr|nc)$", st.name)
        if m and m.group(1) <= ymd <= m.group(2):
            return st
    print(f"[lut] WARNING: no LUT epoch covers {ymd}; using {stores[-1].name}")
    return stores[-1]


def _h2o_interp(h: np.ndarray, A: np.ndarray):
    """Callable H2OSTR -> (n, wl), linear along the LUT's H2OSTR axis, clamped."""
    def f(x):
        x = np.clip(np.atleast_1d(np.asarray(x, np.float64)).ravel(), h[0], h[-1])
        i = np.clip(np.searchsorted(h, x) - 1, 0, h.size - 2)
        w = ((x - h[i]) / (h[i + 1] - h[i]))[:, None]
        return A[i] * (1.0 - w) + A[i + 1] * w
    return f


def load_emit_lut(path, dt, wl, *, sza, vza, raz, elev_km, aot, co2, night):
    """Interpolate the EMIT ISOFIT/MODTRAN VSWIR LUT at one scene state.

    Every axis except H2OSTR is interpolated at the given state (clamped to the
    grid, with the clamping reported); H2OSTR is kept so water vapour can vary
    per pixel. Returns (rte, solar_irr_or_None, h2o_grid, info), where ``rte``
    maps each RT term to a callable H2OSTR -> (n, wl), the interface the
    retrieval and the water estimator expect.

    At night the solar terms (rhoatm, sphalb, both downward transmittances) are
    zeroed and only the upward transmittance is used, as in the EMIT+ECOSTRESS
    work; the LUT's solar_zenith axis stops at 75 deg, so it is sampled there.
    """
    import xarray as xr
    store = pick_lut_store(path, dt)
    ds = xr.open_zarr(store) if store.suffix == ".zarr" else xr.open_dataset(store)
    lut_wl = np.asarray(ds["wl"].values, np.float64)
    if lut_wl.size != wl.size or np.abs(lut_wl - wl).max() > 0.5:
        raise RuntimeError(f"{store.name}: LUT wavelengths ({lut_wl.size}) do not match the scene "
                           f"({wl.size}); max |diff| {np.abs(lut_wl - wl).max() if lut_wl.size == wl.size else 'n/a'}")
    want = dict(AOT550=aot, observer_zenith=vza, solar_zenith=sza, relative_azimuth=raz,
                surface_elevation_km=elev_km)
    sel, clamped = {}, {}
    for ax, v in want.items():
        if ax not in ds.sizes:
            continue
        g = np.asarray(ds[ax].values, np.float64)
        vc = float(np.clip(v, g.min(), g.max()))
        sel[ax] = vc
        if abs(vc - v) > 1e-6:
            clamped[ax] = (round(float(v), 3), vc)
    sub = ds[list(LUT_KEYS)]
    if "CO2" in ds.sizes:
        sub = sub.sel(CO2=float(co2), method="nearest")
    sub = sub.interp(**sel).load()
    h = np.asarray(sub["H2OSTR"].values, np.float64)
    arrs = {k: np.asarray(sub[k].transpose("H2OSTR", "wl").values, np.float64) for k in LUT_KEYS}
    if night:
        for k in ("rhoatm", "sphalb", "transm_down_dir", "transm_down_dif"):
            arrs[k] = np.zeros_like(arrs[k])
    rte = {k: _h2o_interp(h, A) for k, A in arrs.items()}
    solar = None
    if "solar_irr" in ds:
        si = ds["solar_irr"]
        si = si.isel({d: 0 for d in si.dims if d != "wl"})
        solar = np.asarray(si.values, np.float64)
    info = dict(store=str(store), state=sel, clamped=clamped,
                co2=float(sub["CO2"].values) if "CO2" in sub.coords else None,
                h2o_grid=[float(h[0]), float(h[-1]), int(h.size)], night_solar_zeroed=bool(night))
    return rte, solar, list(h), info


# --------------------------------------------------------------- screens

def night_screen(rad, wl, bm, args):
    gi, gwl = nearest_band(wl, args.night_gate_nm)
    hi, hwl = nearest_band(wl, args.night_ratio_hi_nm)
    lo, lwl = nearest_band(wl, args.night_ratio_lo_nm)
    B = _clean(rad.bands([gi, hi, lo]))
    gate, L_hi, L_lo = B[..., 0], B[..., 1], B[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(L_lo > 1e-6, L_hi / L_lo, np.nan).astype(np.float32)
    keep = np.isfinite(gate) & (gate >= args.night_gate_thr) & np.isfinite(ratio) & (ratio >= args.night_ratio_thr)
    n_sat = 0
    if bm is not None:
        # Saturation overrides only the ratio test, never the radiance gate: the
        # EMIT bandmask flags whole lines that are not hot (seen as ~10240 extra
        # px per scene), and a truly saturated fire is far above the gate anyway.
        sat = (_sat_map(bm, wl.size, gi) | _sat_map(bm, wl.size, hi)) \
            & np.isfinite(gate) & (gate >= args.night_gate_thr)
        n_sat = int((sat & ~keep).sum())
        keep |= sat
    info = dict(metric=f"L{hwl:.0f}/L{lwl:.0f}", gate_band_nm=gwl, ratio_hi_nm=hwl, ratio_lo_nm=lwl,
                n_sat_autopass=n_sat)
    return keep, ratio, info


def day_screen(rad, wl, bm, args):
    lo, lwl = nearest_band(wl, args.hfdi_lo_nm)
    hi, hwl = nearest_band(wl, args.hfdi_hi_nm)
    B = _clean(rad.bands([lo, hi]))
    L_lo, L_hi = B[..., 0], B[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = L_hi + L_lo
        hfdi = np.where(np.abs(d) > 1e-12, (L_hi - L_lo) / d, np.nan).astype(np.float32)
    keep = np.isfinite(hfdi) & (hfdi > args.hfdi_min)
    n_z = 0
    if args.hfdi_z and args.hfdi_z > 0:
        z = _robust_z(hfdi)
        extra = np.isfinite(z) & (z > args.hfdi_z) & ~keep
        n_z = int(extra.sum())
        keep |= extra
    n_sat = 0
    if bm is not None:
        # as at night: saturation passes only pixels that are also bright at 2430 nm
        bright = np.isfinite(L_hi) & (L_hi > np.nanpercentile(L_hi, 99.5))
        sat = (_sat_map(bm, wl.size, lo) | _sat_map(bm, wl.size, hi)) & bright
        n_sat = int((sat & ~keep).sum())
        keep |= sat
    info = dict(metric=f"HFDI {hwl:.0f}/{lwl:.0f}", hfdi_lo_nm=lwl, hfdi_hi_nm=hwl,
                n_z_added=n_z, n_sat_autopass=n_sat)
    return keep, hfdi, info


# --------------------------------------------------------------- one scene

def process_scene(scene: EmitScene, args, basis_cache: Dict[str, PCASurfaceBasis]) -> dict:
    t0 = time.perf_counter()
    fid = scene.fid
    out_dir = Path(args.out_dir) / fid
    csv_path = out_dir / f"{fid}_isofire_pixels.csv"
    if csv_path.exists() and not args.overwrite:
        print(f"[scene] {fid}: outputs exist, skipping (--overwrite to redo)")
        return dict(fid=fid, status="skipped (exists)")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[scene] === {fid} ({scene.fmt}) ===")
    for k, v in scene.files.items():
        print(f"  {k:9s}= {v}")

    rad = scene.radiance()
    obs = scene.obs()
    loc = scene.loc()
    bm = None if args.no_bandmask else scene.bandmask()
    wl = np.asarray(rad.wl_nm, np.float64)
    fwhm = np.asarray(rad.fwhm_nm, np.float64)
    if wl.size != rad.nb or fwhm.size != rad.nb:
        raise RuntimeError(f"wavelength/fwhm lists ({wl.size}/{fwhm.size}) do not match {rad.nb} bands")
    if wl[0] > wl[-1]:
        raise RuntimeError("descending wavelengths: not expected for EMIT L1B")
    rows, cols, nb = rad.shape
    if bm is None:
        print("[scene] no bandmask found: saturation auto-pass and eta saturation bounds are off")

    # ---- day / night
    sza_med = scene_median_sza(scene)
    if args.mode == "auto":
        if sza_med >= args.night_sza:
            mode = "night"
        elif sza_med <= args.day_sza_max:
            mode = "day"
        else:
            print(f"[scene] {fid}: median SZA {sza_med:.1f} deg is twilight "
                  f"({args.day_sza_max}-{args.night_sza}); skipping. Force with --mode.")
            return dict(fid=fid, status="twilight", sza=sza_med)
    else:
        mode = args.mode
    print(f"[scene] median SZA {sza_med:.2f} deg -> {mode}")
    # The PCA basis also defines the band set every fit uses, day and night: bands outside
    # the cover-class/specmix grid (water-band edges, spectral ends) are fully masked, as in
    # cover-class. Night needs no surface, but loads the basis for its band set when given.
    basis = None
    if args.pca_basis:
        key = f"{args.pca_basis}|{args.n_pc}|{wl.size}|{wl[0]:.3f}"
        if key not in basis_cache:
            basis_cache[key] = PCASurfaceBasis.from_npz(args.pca_basis, wl, n_pc=args.n_pc)
            b = basis_cache[key]
            print(f"[pca] {Path(b.source).name}: {b.n_pc} PCs, approach {b.approach!r}, "
                  f"{b.grid_match}, {int(b.valid_1d.sum())}/{wl.size} bands defined")
        basis = basis_cache[key]
    if mode == "day" and basis is None:
        raise RuntimeError("day scene needs --pca_basis")
    if basis is None:
        print("[bands] WARNING: no --pca_basis, so night fits use every band outside the fixed bad "
              "ranges, including the water-band edges; pass --pca_basis to use the cover-class band set")

    # ---- screen
    t_s = time.perf_counter()
    keep, metric, sinfo = (night_screen if mode == "night" else day_screen)(rad, wl, bm, args)
    n_raw = int(keep.sum())
    min_px = args.min_blob_px if args.min_blob_px is not None else (2 if mode == "night" else 1)
    keep = _blob_filter(keep, min_px)
    if args.dilate and args.dilate > 0 and ndimage is not None and keep.any():
        keep = ndimage.binary_dilation(keep, iterations=int(args.dilate))
    if args.bbox:
        la0, la1, lo0, lo1 = (float(x) for x in args.bbox.split(","))
        lat = _clean(loc.band(loc_band_index(loc, "lat")))
        lon = _clean(loc.band(loc_band_index(loc, "lon")))
        keep &= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    rr, cc = np.nonzero(keep)
    if rr.size > args.max_candidates:
        order = np.argsort(-np.nan_to_num(metric[rr, cc], nan=-np.inf))[: args.max_candidates]
        print(f"[screen] {rr.size} candidates > --max_candidates {args.max_candidates}; "
              f"keeping the highest {sinfo['metric']}")
        rr, cc = rr[order], cc[order]
    print(f"[screen] {sinfo['metric']}: {n_raw} px pass, {rr.size} after blob>={min_px}"
          f"{', dilation' if args.dilate else ''}{', bbox' if args.bbox else ''} "
          f"({time.perf_counter() - t_s:.1f}s) {sinfo}")

    run_meta = dict(fid=fid, mode=mode, sza_median=sza_med, screen=sinfo, n_screen_raw=n_raw,
                    n_candidates=int(rr.size), files={k: str(v) for k, v in scene.files.items()},
                    args={k: v for k, v in vars(args).items() if k not in ("scene",)})
    if rr.size == 0:
        (out_dir / f"{fid}_isofire_run.json").write_text(json.dumps(run_meta, indent=1, default=str))
        pd.DataFrame(columns=["row", "col"]).to_csv(csv_path, index=False)
        print(f"[scene] {fid}: no candidates")
        return dict(fid=fid, status="no candidates", mode=mode, sza=sza_med, n=0)

    # ---- per-candidate inputs
    obs_px = _clean(rad.pixels(rr, cc))
    ob_px = _clean(obs.pixels(rr, cc))
    lc_px = _clean(loc.pixels(rr, cc))
    sza_px = ob_px[:, obs_band_index(obs, "sun_zen")]
    vza_px = ob_px[:, obs_band_index(obs, "sensor_zen")]
    saz_px = ob_px[:, obs_band_index(obs, "sun_az")]
    vaz_px = ob_px[:, obs_band_index(obs, "sensor_az")]
    esd_i = obs_band_index(obs, "earth_sun_dist")
    esd = float(np.nanmedian(ob_px[:, esd_i])) if ob_px.shape[1] > esd_i else 1.0
    if not (0.95 < esd < 1.05):
        print(f"[geom] earth-sun distance {esd} looks wrong; using 1.0")
        esd = 1.0
    lat_px = lc_px[:, loc_band_index(loc, "lat")]
    lon_px = lc_px[:, loc_band_index(loc, "lon")]
    elev_px = lc_px[:, loc_band_index(loc, "elev")]
    if bm is not None:
        sat_ok = decode_sat_ok(bm.pixels(rr, cc), nb)
    else:
        sat_ok = np.ones((rr.size, nb), bool)

    sza = float(np.nanmedian(sza_px))
    vza = float(np.nanmedian(vza_px))
    elev_km = float(np.clip(np.nanmedian(elev_px) / 1000.0, 0.0, 8.0)) if np.isfinite(elev_px).any() else 0.0
    dt = scene.datetime
    print(f"[geom] candidates: SZA {sza:.2f}, VZA {vza:.2f}, elev {elev_km:.3f} km, "
          f"earth-sun {esd:.4f} AU, date {dt:%Y-%m-%d}")

    # ---- EMIT VSWIR LUT at the candidates' median state
    saz = float(np.nanmedian(saz_px))
    vaz = float(np.nanmedian(vaz_px))
    raz = abs(saz - vaz) % 360.0
    raz = 360.0 - raz if raz > 180.0 else raz
    t_rt = time.perf_counter()
    rte, lut_solar, water_vapors, lut_info = load_emit_lut(
        args.vswir_lut, dt, wl, sza=sza, vza=vza, raz=raz, elev_km=elev_km,
        aot=args.aot550, co2=args.co2, night=(mode == "night"))
    print(f"[lut] {Path(lut_info['store']).name} at {lut_info['state']} "
          f"({time.perf_counter() - t_rt:.1f}s)" + (f"; clamped {lut_info['clamped']}" if lut_info['clamped'] else ""))
    if args.solar_source == "lut":
        if lut_solar is None:
            raise RuntimeError("--solar_source lut but the LUT has no solar_irr")
        solar_irr = lut_solar / esd ** 2
    else:
        irr = np.loadtxt(args.solar_irr_file, comments="#")
        iwl, irr = irr.T
        solar_irr = resample_spectrum(irr / 10.0, iwl, wl, fwhm).astype(np.float64) / esd ** 2

    # ---- water vapor
    if mode == "day":
        wb, wwl = get_water_wavelengths(wl)
        interp, _ = create_water_estimator(water_vapors, np.asarray(wwl), rte, solar_irr[wb],
                                           np.deg2rad(sza), wb)
        W = _clean(rad.bands(wb))
        wv_map = water_superpixel_nanmedian(
            estimate_water_vapor(W, interp, np.asarray(wwl)), k=int(args.water_superpixel_size))
        lo_wv = max(WV_TEMP_CLIP_MIN, min(water_vapors))
        wv_map = np.clip(wv_map, lo_wv, max(water_vapors))
        wv_px = wv_map[rr, cc]
        fill = float(np.nanmedian(wv_map)) if np.isfinite(wv_map).any() else 1.5
        wv_px = np.where(np.isfinite(wv_px), wv_px, fill).astype(np.float32)
        print(f"[water] H2OSTR at candidates: median {np.median(wv_px):.2f}, "
              f"range {wv_px.min():.2f}-{wv_px.max():.2f} g/cm^2")
    else:
        wv_px = np.full(rr.size, float(args.night_h2ostr), np.float32)
        print(f"[water] night: fixed H2OSTR {args.night_h2ostr} g/cm^2")

    # ---- windows and eta bounds
    good = make_bitwise_bad_range_mask_1d(wl, BAD_RANGES_NM_DEFAULT)
    surface_1d = good & ranges_to_mask(wl, _parse_ranges(args.surface_window_ranges_nm, SURFACE_RANGES_NM_DEFAULT))
    hot_1d = good & ranges_to_mask(wl, _parse_ranges(args.hot_window_ranges_nm, HOT_RANGES_NM_DEFAULT))
    smol_1d = good & ranges_to_mask(wl, _parse_ranges(args.smolder_window_ranges_nm, SMOLDER_RANGES_NM_DEFAULT))
    if basis is not None:
        surface_1d &= basis.valid_1d
        hot_1d &= basis.valid_1d
        smol_1d &= basis.valid_1d
        print(f"[bands] fit bands (cover-class set & window & not bad): surface {int(surface_1d.sum())}, "
              f"hot {int(hot_1d.sum())}, smoulder {int(smol_1d.sum())}; bandmask-flagged bands are "
              f"removed per pixel on top")
    swir2_1d = ranges_to_mask(wl, np.array([[args.eta_swir2_lo_nm, args.eta_swir2_hi_nm]]))
    swir1_1d = ranges_to_mask(wl, ETA_SWIR1_OVERRIDE_RANGES_NM_DEFAULT)
    sat2, sat1, eta_lo, eta_hi = eta_bounds_from_saturation(
        (~sat_ok).astype(np.uint8), swir2_1d, swir1_1d, nb, args.eta_min, args.eta_max,
        eta_sat_frac_low=args.eta_sat_frac_low, eta_sat_frac_high=args.eta_sat_frac_high,
        eta_floor_swir2_only_sat=args.eta_floor_swir2_only_sat,
        eta_floor_swir1_and_swir2_sat=args.eta_floor_swir1_and_swir2_sat,
        eta_ceiling_low_sat=args.eta_ceiling_low_sat, eta_floor_max=args.eta_floor_max)

    fitwin = hot_1d | smol_1d | (surface_1d if mode == "day" else False)
    flag_frac = 1.0 - sat_ok[:, fitwin].mean(axis=1)
    too_flagged = flag_frac > args.max_flagged_frac
    print(f"[bandmask] candidates with any flagged fit band: {int((flag_frac > 0).sum())}/{rr.size}; "
          f"> {args.max_flagged_frac:.0%} flagged (skipped): {int(too_flagged.sum())}")
    if too_flagged.any():
        obs_px = obs_px.copy()
        obs_px[too_flagged] = np.nan          # the solver skips non-finite pixels

    night = mode == "night"
    cfg = MixtureRetrievalConfig(
        Tmin=args.Tmin, Tmax=args.Tmax if args.Tmax else (2700.0 if night else 2400.0),
        N_T=args.N_T, T_step1_init=args.T_step1_init,
        mu_hot_min_T=args.mu_hot_min_T,
        mu_hot_max_T=args.mu_hot_max_T if args.mu_hot_max_T else (2400.0 if night else 2100.0),
        mu_smol_min_T=args.mu_smol_min_T, mu_smol_max_T=args.mu_smol_max_T,
        sigma_hot_min=args.sigma_hot_min, sigma_hot_max=args.sigma_hot_max,
        sigma_smol_min=args.sigma_smol_min, sigma_smol_max=args.sigma_smol_max,
        eta_min=args.eta_min, eta_max=args.eta_max)

    # ---- initial guess: night from the two-band Planck ratio
    T1_init = la1_init = None
    if night:
        hi, hwl = nearest_band(wl, args.night_ratio_hi_nm)
        lo, lwl = nearest_band(wl, args.night_ratio_lo_nm)
        T1_init = ratio_temperature(obs_px[:, hi], obs_px[:, lo], hwl, lwl,
                                    t_min=cfg.mu_hot_min, t_max=cfg.mu_hot_max)
        tup = np.asarray(rte["transm_up_dir"](np.array([args.night_h2ostr], np.float32))
                         + rte["transm_up_dif"](np.array([args.night_h2ostr], np.float32))).reshape(-1)
        B_hi = np.array([float(planck_stable(t, hwl)) for t in T1_init])
        with np.errstate(divide="ignore", invalid="ignore"):
            la1_init = np.log10(np.clip(obs_px[:, hi] / (B_hi * max(tup[hi], 1e-3)),
                                        10.0 ** args.log_alpha_min, 10.0 ** args.log_alpha_max))
        la1_init = np.where(np.isfinite(la1_init), la1_init, -3.0).astype(np.float32)

    mu0 = np.cos(np.deg2rad(np.where(np.isfinite(sza_px), sza_px, sza))).astype(np.float32)
    res = run_pca_fire_retrieval(
        obs_px, wv_px, sat_ok, cfg=cfg, rte=rte, solar_irr=solar_irr, mu0=mu0, wl_nm=wl,
        basis=basis if mode == "day" else None, surface_1d=surface_1d, hot_1d=hot_1d, smolder_1d=smol_1d,
        eta_lo=eta_lo, eta_hi=eta_hi, include_solar=not night,
        solar_scaling=args.solar_scaling, T1_init=T1_init, log_alpha1_init=la1_init,
        log_alpha_min=args.log_alpha_min, log_alpha_max=args.log_alpha_max,
        noise_frac=args.noise_frac, noise_floor=args.noise_floor,
        prior_weight=args.pca_prior_weight, rfl_bounds_weight=args.rfl_bounds_weight,
        refine_surface_stage2=args.refine_surface_stage2,
        hot_window_weight=args.hot_window_weight, smolder_window_weight=args.smolder_window_weight,
        batch_size=args.batch_size, lm_iters_stage1=args.lm_iters_stage1,
        lm_iters_stage2=args.lm_iters_stage2, alpha_tuning=args.alpha_tuning,
        sigma_min_k=args.sigma_min_k, verbose=True)

    # ---- outputs
    df = pd.DataFrame(dict(
        row=rr, col=cc, lat=lat_px, lon=lon_px, elev_m=elev_px, sza=sza_px, vza=vza_px,
        screen_metric=metric[rr, cc], h2ostr=wv_px, success=res["success"],
        mu=res["mu"], sigma=res["sigma"], mu_hot=res["mu_hot"], sigma_hot=res["sigma_hot"],
        mu_smol=res["mu_smol"], sigma_smol=res["sigma_smol"], eta=res["eta"],
        T1=res["T1"], alpha1=res["alpha1"], alpha2=res["alpha2"], rmse=res["rmse"],
        chi2_stage1=res["chi2_stage1"], chi2_stage2=res["chi2_stage2"],
        swir2_sat_frac=sat2, swir1_sat_frac=sat1))
    if res["rfl"] is not None:
        for k in range(res["scores"].shape[1]):
            df[f"pc{k + 1}"] = res["scores"][:, k]
        np.savez_compressed(out_dir / f"{fid}_isofire_surface.npz", wavelengths=wl, row=rr, col=cc,
                            rfl=res["rfl"], scores=res["scores"], radiance=obs_px,
                            basis_valid=basis.valid_1d, basis_source=str(basis.source))
    df.to_csv(csv_path, index=False, float_format="%.6g")

    mu_map = np.full((rows, cols), np.nan, np.float32)
    sg_map = np.full((rows, cols), np.nan, np.float32)
    ok = res["success"].astype(bool)
    mu_map[rr[ok], cc[ok]] = res["mu"][ok]
    sg_map[rr[ok], cc[ok]] = res["sigma"][ok]
    if args.save_maps:
        np.savez_compressed(out_dir / f"{fid}_isofire_maps.npz", mu=mu_map, sigma=sg_map,
                            screen=keep, metric=metric)

    if not args.no_geotiff:
        glt = read_glt(scene)
        if glt is None:
            print("[ortho] no GLT: GeoTIFFs skipped")
        else:
            from .airborne import _check_glt_band_order, export_ortho_geotiff
            gx, gy, gt, wkt, nd = glt
            gx, gy, _ = _check_glt_band_order(gx, gy, rows, cols)
            for arr, name in ((mu_map, "fireTemp"), (sg_map, "fireTempUnc")):
                filled = np.where(np.isfinite(arr), arr, np.float32(FILL_NOT_RETRIEVED))
                export_ortho_geotiff(str(out_dir / f"{fid}_{name}.tif"), filled, gx, gy, gt, wkt,
                                     glt_nodata=nd)

    good_mu = res["mu"][ok]
    stats = dict(n_retrieved=int(ok.sum()),
                 mu_p05_p50_p95=[float(x) for x in np.percentile(good_mu, [5, 50, 95])] if ok.any() else None,
                 elapsed_s=time.perf_counter() - t0, retrieval_s=float(res["elapsed"]))
    run_meta.update(stats, geometry=dict(sza=sza, vza=vza, elev_km=elev_km, earth_sun_au=esd),
                    lut=lut_info,
                    basis=None if basis is None else dict(source=basis.source, n_pc=basis.n_pc,
                                                          approach=basis.approach, grid=basis.grid_match))
    (out_dir / f"{fid}_isofire_run.json").write_text(json.dumps(run_meta, indent=1, default=str))
    msg = (f"E[T] p05/p50/p95 = {stats['mu_p05_p50_p95'][0]:.0f}/{stats['mu_p05_p50_p95'][1]:.0f}/"
           f"{stats['mu_p05_p50_p95'][2]:.0f} K" if ok.any() else "nothing retrieved")
    print(f"[scene] {fid} ({mode}): {stats['n_retrieved']}/{rr.size} retrieved, {msg}, "
          f"{stats['elapsed_s']:.1f}s -> {out_dir}")
    return dict(fid=fid, status="ok", mode=mode, sza=sza_med, n=int(rr.size), n_ok=stats["n_retrieved"])


# --------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("inputs and outputs")
    g.add_argument("--scene", action="append", default=[],
                   help="scene dir, its l1b/ dir, a file in it, or a fid (repeatable)")
    g.add_argument("--scene_list", default=None, help="text file, one --scene value per line")
    g.add_argument("--acq_root", default=os.environ.get("EMIT_ACQ_ROOT", "/store/emit/ops/data/acquisitions"),
                   help="ops acquisitions tree, used to resolve bare fids")
    g.add_argument("--out_dir", required=True)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--no_geotiff", action="store_true", help="skip the GLT ortho GeoTIFFs")
    g.add_argument("--save_maps", action="store_true", help="also save full-scene mu/sigma/screen maps (npz)")
    g.add_argument("--no_bandmask", action="store_true",
                   help="ignore the L1B bandmask (no saturation auto-pass, all bands treated as unsaturated)")
    g.add_argument("--max_flagged_frac", type=float, default=0.5,
                   help="skip candidates with more than this fraction of fit-window bands flagged in the bandmask")

    g = p.add_argument_group("day / night")
    g.add_argument("--mode", choices=("auto", "day", "night"), default="auto")
    g.add_argument("--night_sza", type=float, default=95.0, help="median SZA at or above this = night")
    g.add_argument("--day_sza_max", type=float, default=75.0,
                   help="median SZA at or below this = day (the LUT solar_zenith axis ends at 75)")

    g = p.add_argument_group("PCA surface (day)")
    g.add_argument("--pca_basis", default=None,
                   help="pca_surface_*.npz from the specmix notebook; required by day, and at night it "
                        "sets the fit band set (bands outside the cover-class grid are fully masked)")
    g.add_argument("--n_pc", type=int, default=8)
    g.add_argument("--pca_prior_weight", type=float, default=1.0, help="weight on the N(0,I) whitened-score prior")
    g.add_argument("--rfl_bounds_weight", type=float, default=1.0e4, help="penalty on reflectance outside [0,1]")
    g.add_argument("--no_refine_surface_stage2", dest="refine_surface_stage2", action="store_false",
                   help="hold the stage-1 surface fixed in stage 2, as the airborne retrieval does")
    g.add_argument("--solar_scaling", choices=("cos", "inv_cos"), default="cos",
                   help="solar path factor: solar*cos(SZA)/pi (ISOFIT convention) or the airborne solar/cos(SZA)/pi")

    g = p.add_argument_group("night screen")
    g.add_argument("--night_gate_nm", type=float, default=2400.0)
    g.add_argument("--night_gate_thr", type=float, default=0.01)
    g.add_argument("--night_ratio_hi_nm", type=float, default=2249.4)
    g.add_argument("--night_ratio_lo_nm", type=float, default=1096.9)
    g.add_argument("--night_ratio_thr", type=float, default=0.5)
    g.add_argument("--night_h2ostr", type=float, default=2.0, help="fixed column water vapor at night, g/cm^2")

    g = p.add_argument_group("day screen")
    g.add_argument("--hfdi_lo_nm", type=float, default=2061.0)
    g.add_argument("--hfdi_hi_nm", type=float, default=2430.0)
    g.add_argument("--hfdi_min", type=float, default=-0.1)
    g.add_argument("--hfdi_z", type=float, default=0.0,
                   help="also keep pixels whose HFDI is this many robust SDs above the scene median (0 = off)")

    g = p.add_argument_group("candidates")
    g.add_argument("--min_blob_px", type=int, default=None, help="default 2 at night (as in the scan), 1 by day")
    g.add_argument("--dilate", type=int, default=0, help="grow the candidate mask by this many pixels")
    g.add_argument("--bbox", default=None, help="lat0,lat1,lon0,lon1: only retrieve inside this box")
    g.add_argument("--max_candidates", type=int, default=50000)

    g = p.add_argument_group("noise model")
    g.add_argument("--noise_frac", type=float, default=0.01)
    g.add_argument("--noise_floor", type=float, default=0.005, help="uW/cm^2/sr/nm")

    g = p.add_argument_group("windows")
    g.add_argument("--surface_window_ranges_nm", default=str(SURFACE_RANGES_NM_DEFAULT.tolist()))
    g.add_argument("--hot_window_ranges_nm", default=str(HOT_RANGES_NM_DEFAULT.tolist()))
    g.add_argument("--smolder_window_ranges_nm", default=str(SMOLDER_RANGES_NM_DEFAULT.tolist()))
    g.add_argument("--eta_swir2_lo_nm", type=float, default=ETA_SWIR2_DEFAULT_NM[0])
    g.add_argument("--eta_swir2_hi_nm", type=float, default=ETA_SWIR2_DEFAULT_NM[1])

    g = p.add_argument_group("mixture config (None = mode default)")
    g.add_argument("--Tmin", type=float, default=200.0)
    g.add_argument("--Tmax", type=float, default=None, help="default 2400 day, 2700 night")
    g.add_argument("--N_T", type=int, default=240)
    g.add_argument("--T_step1_init", type=float, default=900.0)
    g.add_argument("--mu_hot_min_T", type=float, default=500.0)
    g.add_argument("--mu_hot_max_T", type=float, default=None, help="default 2100 day, 2400 night (flares)")
    g.add_argument("--mu_smol_min_T", type=float, default=450.0)
    g.add_argument("--mu_smol_max_T", type=float, default=850.0)
    g.add_argument("--sigma_hot_min", type=float, default=50.0)
    g.add_argument("--sigma_hot_max", type=float, default=400.0)
    g.add_argument("--sigma_smol_min", type=float, default=100.0)
    g.add_argument("--sigma_smol_max", type=float, default=400.0)
    g.add_argument("--eta_min", type=float, default=0.0)
    g.add_argument("--eta_max", type=float, default=1.0)
    g.add_argument("--log_alpha_min", type=float, default=-7.0)
    g.add_argument("--log_alpha_max", type=float, default=0.3)
    g.add_argument("--hot_window_weight", type=float, default=1.0)
    g.add_argument("--smolder_window_weight", type=float, default=1.0)

    g = p.add_argument_group("saturation-driven eta bounds (off unless --eta_floor_max > 0)")
    g.add_argument("--eta_sat_frac_low", type=float, default=0.02)
    g.add_argument("--eta_sat_frac_high", type=float, default=0.08)
    g.add_argument("--eta_floor_swir2_only_sat", type=float, default=0.75)
    g.add_argument("--eta_floor_swir1_and_swir2_sat", type=float, default=0.85)
    g.add_argument("--eta_ceiling_low_sat", type=float, default=1.0)
    g.add_argument("--eta_floor_max", type=float, default=0.0)

    g = p.add_argument_group("solver")
    g.add_argument("--lm_iters_stage1", type=int, default=40)
    g.add_argument("--lm_iters_stage2", type=int, default=60)
    g.add_argument("--batch_size", type=int, default=2048)
    g.add_argument("--alpha_tuning", type=float, default=DEFAULT_ALPHA_TUNING)
    g.add_argument("--sigma_min_k", type=float, default=DEFAULT_SIGMA_MIN_K)

    g = p.add_argument_group("atmosphere and auxiliary data")
    g.add_argument("--vswir_lut", default=os.environ.get("ISOFIRE_EMIT_LUT"), required=os.environ.get("ISOFIRE_EMIT_LUT") is None,
                   help="EMIT VSWIR LUT (.zarr/.nc) or a directory of *_<start>-<end>.zarr epochs ($ISOFIRE_EMIT_LUT)")
    g.add_argument("--aot550", type=float, default=0.1)
    g.add_argument("--co2", type=float, default=410.0, help="nearest LUT CO2 value is used")
    g.add_argument("--water_superpixel_size", type=int, default=12)
    g.add_argument("--solar_source", choices=("kurucz", "lut"), default="kurucz",
                   help="Kurucz file (as the airborne code) or the LUT's solar_irr")
    g.add_argument("--solar_irr_file", default=_data_file("kurudz_0.1nm.dat"))
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    specs = list(args.scene)
    if args.scene_list:
        specs += [ln.split("#")[0].strip() for ln in Path(args.scene_list).read_text().splitlines()]
    specs = [s for s in specs if s]
    if not specs:
        print("no scenes given (--scene / --scene_list)")
        return 2
    print(f"[isofire-spaceborne] {len(specs)} scene(s); JAX backend {jax.default_backend()}")
    cache: Dict[str, PCASurfaceBasis] = {}
    summary = []
    for spec in specs:
        try:
            scene = find_emit_scene(spec, args.acq_root)
            summary.append(process_scene(scene, args, cache))
        except Exception as err:
            traceback.print_exc()
            summary.append(dict(fid=str(spec), status=f"error: {type(err).__name__}: {err}"))
    print("\n[summary]")
    for s in summary:
        print("  " + "  ".join(f"{k}={v}" for k, v in s.items()))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(Path(args.out_dir) / f"summary_{time.strftime('%Y%m%dT%H%M%S')}.csv",
                                 index=False)
    return 0 if all(not str(s.get("status", "")).startswith("error") for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
