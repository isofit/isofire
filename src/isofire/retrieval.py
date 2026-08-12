#!/usr/bin/env python
#
#  Copyright 2024 California Institute of Technology
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


"""Temperature retrieval algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress

import jax
import jax.numpy as jnp
import optax

from .rt import (
    algebraic_forward,
    combined_multipart_transmittance,
    emissive_radiance,
    resample_spectrum,
    rte_eval_1d,
)


# Constants

# Floor on H2OSTR fed into the temperature retrieval, avoiding the known-bad
# low-H2OSTR endpoint of the multipart RT LUT.
WV_TEMP_CLIP_MIN = 0.21

# Default fitting windows (nm).
SURFACE_RANGES_NM_DEFAULT = np.array([[350.0, 2500.0]], dtype=np.float64)
HOT_RANGES_NM_DEFAULT = np.array([[900.0, 1800.0]], dtype=np.float64)
SMOLDER_RANGES_NM_DEFAULT = np.array([[1200.0, 2500.0]], dtype=np.float64)
ETA_SWIR2_DEFAULT_NM = (2000.0, 2500.0)
ETA_SWIR1_OVERRIDE_RANGES_NM_DEFAULT = np.array([[1500.0, 1700.0]], dtype=np.float64)

# Always-bad spectral regions (water bands etc.) intersected with all windows.
BAD_RANGES_NM_DEFAULT = np.array(
    [[0.0, 405.0], [1275.0, 1321.0], [2455.0, 9999.0]],
    dtype=np.float64,
)

# Default sigma-scaling tuning; runtime values come from the CLI flags
# --alpha_tuning and --sigma_min_k.
DEFAULT_ALPHA_TUNING = 0.25
DEFAULT_SIGMA_MIN_K = 1.0


# Mask utilities

def make_bitwise_bad_range_mask_1d(wl_nm: np.ndarray, bad_ranges_nm: np.ndarray) -> np.ndarray:
    """Boolean spectral mask: True where a band is OUTSIDE every bad range."""
    good = np.ones_like(wl_nm, dtype=bool)
    for lo, hi in np.asarray(bad_ranges_nm, dtype=np.float64):
        good &= ~((wl_nm >= lo) & (wl_nm <= hi))
    return good


def ranges_to_mask(wl_nm: np.ndarray, ranges_nm) -> np.ndarray:
    """Boolean spectral mask: True where a band is INSIDE any range."""
    m = np.zeros_like(wl_nm, dtype=bool)
    for lo, hi in np.asarray(ranges_nm, dtype=np.float64):
        m |= (wl_nm >= float(lo)) & (wl_nm <= float(hi))
    return m


def decode_sat_ok(bm_flat: np.ndarray, bands: int) -> np.ndarray:
    """ENVI-style bandmask -> per-pixel boolean, True where a band is unsaturated.

    Accepts both packed-bit and per-band-byte layouts.
    """
    bm_all = np.asarray(bm_flat, dtype=np.uint8)
    if bm_all.ndim == 1:
        bm_all = bm_all[:, None]
    elif bm_all.ndim > 2:
        bm_all = bm_all.reshape(bm_all.shape[0], -1)
    if bm_all.shape[1] == bands:
        return ~(bm_all != 0)
    bits = np.unpackbits(bm_all.astype(np.uint8), axis=1)
    if bits.shape[1] < bands:
        raise RuntimeError(f"Bandmask bits ({bits.shape[1]}) < bands ({bands}).")
    return ~bits[:, :bands].astype(bool)


def _bm_flat_to_masked_all(bm_flat: np.ndarray, bands: int) -> np.ndarray:
    """Per-pixel, per-band boolean, True where a band is flagged as masked."""
    bm_all = np.asarray(bm_flat, dtype=np.uint8)
    if bm_all.ndim == 1:
        bm_all = bm_all[:, None]
    elif bm_all.ndim > 2:
        bm_all = bm_all.reshape(bm_all.shape[0], -1)
    if bm_all.shape[1] == bands:
        return bm_all != 0
    bits = np.unpackbits(bm_all.astype(np.uint8), axis=1)
    if bits.shape[1] < bands:
        raise RuntimeError(f"Bandmask bits ({bits.shape[1]}) < bands ({bands}).")
    return bits[:, :bands].astype(bool)


def masked_fraction_from_bm_flat(bm_flat: np.ndarray, window_1d: np.ndarray, bands: int) -> np.ndarray:
    """Fraction of bands within ``window_1d`` flagged as masked, per pixel."""
    window_1d = np.asarray(window_1d, dtype=bool)
    denom = int(np.count_nonzero(window_1d))
    n_pix = int(np.asarray(bm_flat).shape[0])
    if denom <= 0:
        return np.zeros((n_pix,), dtype=np.float32)
    masked_all = _bm_flat_to_masked_all(bm_flat, bands)
    masked_count = np.count_nonzero(masked_all[:, window_1d], axis=1).astype(np.float32)
    return masked_count / float(denom)


def eta_bounds_from_saturation(
    bm_flat: np.ndarray,
    swir2_window_1d: np.ndarray,
    swir1_override_window_1d: np.ndarray,
    bands: int,
    eta_min: float,
    eta_max: float,
    *,
    # Hard categorical floor knobs.
    eta_sat_frac_low: float = 0.02,
    eta_sat_frac_high: float = 0.08,
    eta_floor_swir2_only_sat: float = 0.75,
    eta_floor_swir1_and_swir2_sat: float = 0.85,
    eta_ceiling_low_sat: float = 1.0,
    eta_floor_max: float = 0.0,
    # Legacy soft-ramp knobs, accepted for backward compatibility but ignored.
    eta_floor_swir2_ramp_lo: float = 0.10,
    eta_floor_swir2_ramp_hi: float = 0.50,
    eta_floor_swir1_ramp_lo: float = 0.05,
    eta_floor_swir1_ramp_hi: float = 0.30,
    eta_floor_gate_lo: float = 0.40,
    eta_floor_gate_hi: float = 0.60,
    eta_floor_swir2_only_max: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel (swir2_frac, swir1_frac, eta_lo, eta_hi) from band-mask saturation.

    Saturation at or above ``eta_sat_frac_high`` in both SWIR windows raises
    eta_lo to ``eta_floor_swir1_and_swir2_sat`` and in SWIR2 alone to
    ``eta_floor_swir2_only_sat``, while SWIR2 saturation below
    ``eta_sat_frac_low`` clamps eta_hi to ``eta_ceiling_low_sat``. Setting
    ``eta_floor_max=0.0`` (the default) disables both the floor and the ceiling.
    """
    sat_frac_swir2 = masked_fraction_from_bm_flat(bm_flat, swir2_window_1d, bands)
    eta_lo = np.full(sat_frac_swir2.shape, float(eta_min), dtype=np.float32)
    eta_hi = np.full(sat_frac_swir2.shape, float(eta_max), dtype=np.float32)

    if int(np.count_nonzero(np.asarray(swir1_override_window_1d, dtype=bool))) > 0:
        sat_frac_swir1 = masked_fraction_from_bm_flat(
            bm_flat, np.asarray(swir1_override_window_1d, dtype=bool), bands
        )
    else:
        sat_frac_swir1 = np.zeros_like(sat_frac_swir2, dtype=np.float32)

    if float(eta_floor_max) > 0.0:
        sat_swir2_high = sat_frac_swir2 >= float(eta_sat_frac_high)
        sat_swir1_high = sat_frac_swir1 >= float(eta_sat_frac_high)
        both_high  = sat_swir2_high & sat_swir1_high
        only_swir2 = sat_swir2_high & ~sat_swir1_high

        # Clamp floor levels so eta_lo cannot collide with eta_hi=eta_max.
        feasible_cap = max(float(eta_max) - 1e-3, float(eta_min))
        floor_both  = float(min(float(eta_floor_swir1_and_swir2_sat), feasible_cap))
        floor_swir2 = float(min(float(eta_floor_swir2_only_sat),       feasible_cap))

        eta_lo = np.where(
            both_high,
            np.float32(floor_both),
            np.where(only_swir2, np.float32(floor_swir2), np.float32(float(eta_min))),
        ).astype(np.float32, copy=False)

        ceiling_low = float(eta_ceiling_low_sat)
        if ceiling_low < float(eta_max) and float(eta_sat_frac_low) > 0.0:
            sat_swir2_low = sat_frac_swir2 < float(eta_sat_frac_low)
            ceiling_low = max(ceiling_low, float(eta_min) + 1e-3)
            eta_hi = np.where(
                sat_swir2_low,
                np.float32(ceiling_low),
                np.float32(float(eta_max)),
            ).astype(np.float32, copy=False)

    eta_hi = np.maximum(eta_hi, eta_lo).astype(np.float32, copy=False)
    return (
        sat_frac_swir2.astype(np.float32, copy=False),
        sat_frac_swir1.astype(np.float32, copy=False),
        eta_lo,
        eta_hi,
    )


# Water vapor

def parse_water_vapors_csv(water_vapors_csv: str) -> List[float]:
    """Parse a comma-separated H2OSTR LUT-points string."""
    vals = [float(x.strip()) for x in str(water_vapors_csv).split(",") if str(x).strip() != ""]
    if len(vals) < 3:
        raise ValueError(
            f"Need >=3 H2OSTR LUT points for quadratic water interpolation, got {vals}"
        )
    return vals


def get_water_wavelengths(
    wl_nm: np.ndarray,
    wl_start: float = 879.0,
    wl_stop: float = 995.0,
) -> Tuple[List[int], List[float]]:
    """Indices and wavelengths inside the 940 nm water absorption band."""
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    bands = np.where((wl_nm >= wl_start) & (wl_nm <= wl_stop))[0].tolist()
    return bands, wl_nm[bands].tolist()


def create_water_estimator(
    water_vapors: List[float],
    water_wavelengths: np.ndarray,
    rte: dict,
    irradiance: np.ndarray,
    theta_s: float,
    bands: List[int],
):
    """Build an interp1d mapping integrated water-band depth to H2OSTR.

    Uses the same combined multipart transmittance the temperature retrieval
    sees, so the water field stays consistent with the forward model.
    """
    water_depths = []
    water_wavelengths = np.asarray(water_wavelengths, dtype=np.float64)
    bands = np.asarray(bands, dtype=int)

    for wv in water_vapors:
        rhoatm = rte_eval_1d(rte, "rhoatm", float(wv))[bands]
        sphalb = rte_eval_1d(rte, "sphalb", float(wv))[bands]
        transm = combined_multipart_transmittance(rte, float(wv))[bands]

        rdn_sens = algebraic_forward(
            rhoatm, sphalb, transm, 0.4,
            np.asarray(irradiance, dtype=np.float64),
            float(theta_s),
        )
        m = (rdn_sens[-1] - rdn_sens[0]) / (water_wavelengths[-1] - water_wavelengths[0])
        b = rdn_sens[-1] - water_wavelengths[-1] * m
        continuum = water_wavelengths * m + b
        depth = 1.0 - rdn_sens / continuum
        water_depths.append(np.abs(np.trapezoid(depth)))

    water_depths = np.asarray(water_depths, dtype=np.float64)
    water_vapors = np.asarray(water_vapors, dtype=np.float64)
    order = np.argsort(water_depths)
    return interp1d(
        water_depths[order],
        water_vapors[order],
        kind="quadratic",
        fill_value="extrapolate",
    ), water_depths


def estimate_water_vapor(
    radiance: np.ndarray,
    water_interpolator,
    water_wavelengths: np.ndarray,
) -> np.ndarray:
    """Apply the water-depth to H2OSTR interpolator over a (rows, cols, bands) cube."""
    water_wavelengths = np.asarray(water_wavelengths, dtype=np.float64)
    radiance = np.asarray(radiance, dtype=np.float32)

    m = (radiance[:, :, 0] - radiance[:, :, -1]) / (water_wavelengths[0] - water_wavelengths[-1])
    b = radiance[:, :, -1] - water_wavelengths[-1] * m
    continuum = np.einsum("i,mj->mji", water_wavelengths, m) + np.expand_dims(b, axis=2)
    depth = 1.0 - radiance / continuum
    return water_interpolator(np.trapezoid(depth, axis=2))[()].astype(np.float32)


def water_superpixel_nanmedian(water_image: np.ndarray, k: int = 12) -> np.ndarray:
    """Replace each pixel with the nanmedian over its k x k super-pixel.

    Edge tiles use the nearest full k x k window, so edge pixels reuse interior
    neighbors instead of dropping a partial block.
    """
    k = int(k)
    water_image = np.asarray(water_image, dtype=np.float32)
    if k <= 1:
        return water_image.astype(np.float32, copy=False)

    H, W = water_image.shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    for r0_out in range(0, H, k):
        r1_out = min(r0_out + k, H)
        r0_win = 0 if H <= k else min(r0_out, H - k)
        r1_win = min(r0_win + k, H)
        for c0_out in range(0, W, k):
            c1_out = min(c0_out + k, W)
            c0_win = 0 if W <= k else min(c0_out, W - k)
            c1_win = min(c0_win + k, W)
            blk = water_image[r0_win:r1_win, c0_win:c1_win]
            finite = np.isfinite(blk)
            if np.any(finite):
                out[r0_out:r1_out, c0_out:c1_out] = np.nanmedian(blk[finite]).astype(np.float32)
    return out


def estimate_scene_water_vapor(
    *,
    rdn_full: np.ndarray,
    wl_nm: np.ndarray,
    rte: dict,
    solar_irr: np.ndarray,
    sza_deg: float,
    water_vapors: List[float],
    superpixel_size: int = 12,
) -> np.ndarray:
    """End-to-end scene water-vapor estimate.

    Returns a (rows, cols) float32 H2OSTR field, super-pixel-median-smoothed and
    clipped to ``[max(WV_TEMP_CLIP_MIN, lut_min), lut_max]``.
    """
    bands, water_wl = get_water_wavelengths(wl_nm)
    solar_irr = np.asarray(solar_irr, dtype=np.float64)
    interp, _ = create_water_estimator(
        water_vapors,
        np.asarray(water_wl, dtype=np.float64),
        rte,
        solar_irr[bands],
        np.deg2rad(float(sza_deg)),
        bands,
    )

    water_rdn = np.asarray(rdn_full[..., bands], dtype=np.float32)
    raw = estimate_water_vapor(water_rdn, interp, np.asarray(water_wl, dtype=np.float64))
    smoothed = water_superpixel_nanmedian(raw, k=int(superpixel_size))

    wv_lut_min = float(np.nanmin(np.asarray(water_vapors, dtype=np.float64)))
    wv_lut_max = float(np.nanmax(np.asarray(water_vapors, dtype=np.float64)))
    wv_clip_min = max(wv_lut_min, float(WV_TEMP_CLIP_MIN))

    finite = np.isfinite(smoothed)
    return np.where(
        finite,
        np.clip(smoothed, wv_clip_min, wv_lut_max),
        np.nan,
    ).astype(np.float32)


# Fire endmember library

def load_fire_library_resampled(
    fire_lib_csv: str,
    wl_nm: np.ndarray,
    fwhm_nm: np.ndarray,
) -> np.ndarray:
    """Load endmember spectra from CSV and resample them onto the sensor grid.

    The trailing 9 bands are linearly extrapolated from the prior 41.
    """
    df = pd.read_csv(fire_lib_csv)
    lib_spectra = df.iloc[:, 18:].values
    lib_wl = np.arange(350, 2500.1, 1, dtype=np.float64)
    lib_spectra = resample_spectrum(lib_spectra, lib_wl, wl_nm, fwhm_nm).astype(np.float64)
    for s in range(lib_spectra.shape[0]):
        slope, intercept, *_ = linregress(wl_nm[-50:-9], lib_spectra[s, -50:-9])
        lib_spectra[s, -9:] = intercept + slope * wl_nm[-9:]
    return lib_spectra


# Mixture retrieval config

@dataclass
class MixtureRetrievalConfig:
    """Parameters for the two-component Gaussian temperature mixture retrieval."""
    Tmin: float = 200.0
    Tmax: float = 2400.0
    N_T: int = 240
    T_step1_init: float = 900.0

    mu_hot_min_T: float = 500.0
    mu_hot_max_T: float = 2100.0
    mu_smol_min_T: float = 450.0
    mu_smol_max_T: float = 850.0
    sigma_hot_min: float = 50.0
    sigma_hot_max: float = 400.0
    sigma_smol_min: float = 100.0
    sigma_smol_max: float = 400.0

    alpha1_min: float = 1e-4
    alpha1_max: float = 10.0
    alpha2_min: float = 0.0
    alpha2_max: float = 10.0
    eta_min: float = 0.0
    eta_max: float = 1.0

    eps: float = 1e-12

    def __post_init__(self):
        self.T_grid = np.linspace(float(self.Tmin), float(self.Tmax), int(self.N_T), dtype=np.float64)
        # Trapezoidal weights.
        x = self.T_grid
        w = np.zeros_like(x, dtype=np.float64)
        if x.size >= 2:
            dx = np.diff(x)
            w[0] = dx[0] * 0.5
            w[-1] = dx[-1] * 0.5
            if x.size > 2:
                w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        else:
            w[:] = 1.0
        self.w_T = w

    @property
    def mu_hot_min(self): return float(self.mu_hot_min_T)
    @property
    def mu_hot_max(self): return float(self.mu_hot_max_T)
    @property
    def mu_smol_min(self): return float(self.mu_smol_min_T)
    @property
    def mu_smol_max(self): return float(self.mu_smol_max_T)


# Affine state normalization helpers

def _normalize(x, lo, hi, eps=1e-12):
    """Map a physical state into [-1, 1] given per-element bounds."""
    x = jnp.asarray(x)
    lo = jnp.asarray(lo, dtype=x.dtype)
    hi = jnp.asarray(hi, dtype=x.dtype)
    half = jnp.maximum(0.5 * (hi - lo), jnp.asarray(float(eps), dtype=x.dtype))
    center = 0.5 * (hi + lo)
    return jnp.clip((x - center) / half, -1.0, 1.0)


def _denormalize(z, lo, hi, eps=1e-12):
    """Map a normalized state in [-1, 1] back to physical units."""
    z = jnp.asarray(z)
    lo = jnp.asarray(lo, dtype=z.dtype)
    hi = jnp.asarray(hi, dtype=z.dtype)
    half = jnp.maximum(0.5 * (hi - lo), jnp.asarray(float(eps), dtype=z.dtype))
    center = 0.5 * (hi + lo)
    return jnp.clip(center + jnp.clip(z, -1.0, 1.0) * half, lo, hi)


# Analytic moment-matched single Gaussian (saved output)

@jax.jit
def _single_gaussian_moments_jit(
    mu_hot, sigma_hot, mu_smol, sigma_smol,
    eta, alpha2, alpha_tuning, sigma_min_k,
):
    """Moment-match the two-component mixture to (mu, inflated sigma)."""
    mu_hot = jnp.asarray(mu_hot, dtype=jnp.float32)
    sigma_hot = jnp.maximum(jnp.asarray(sigma_hot, dtype=jnp.float32), 1e-6)
    mu_smol = jnp.asarray(mu_smol, dtype=jnp.float32)
    sigma_smol = jnp.maximum(jnp.asarray(sigma_smol, dtype=jnp.float32), 1e-6)
    eta = jnp.clip(jnp.asarray(eta, dtype=jnp.float32), 0.0, 1.0)
    alpha2 = jnp.maximum(jnp.asarray(alpha2, dtype=jnp.float32), 0.0)

    mu_single = eta * mu_hot + (1.0 - eta) * mu_smol
    var_single = (
        eta * sigma_hot ** 2
        + (1.0 - eta) * sigma_smol ** 2
        + eta * (1.0 - eta) * (mu_hot - mu_smol) ** 2
    )
    sigma_single = jnp.sqrt(jnp.maximum(var_single, sigma_min_k ** 2))
    inflation = 1.0 + alpha_tuning * alpha2
    sigma_post = jnp.maximum(sigma_single * inflation, sigma_min_k)
    return mu_single, sigma_post


def _warmup_single_gaussian():
    """Trigger the moment-matching JIT compile once, up front."""
    one = jnp.ones((1,), dtype=jnp.float32)
    out = _single_gaussian_moments_jit(
        1000.0 * one, 100.0 * one, 700.0 * one, 80.0 * one,
        0.5 * one, 1.0 * one,
        jnp.asarray(DEFAULT_ALPHA_TUNING, dtype=jnp.float32),
        jnp.asarray(DEFAULT_SIGMA_MIN_K, dtype=jnp.float32),
    )
    jax.block_until_ready(out)


# Main retrieval driver

def run_moment_matched_retrieval(
    rdn: np.ndarray,
    wv: np.ndarray,
    bm: np.ndarray,
    *,
    cfg: MixtureRetrievalConfig,
    rte: dict,
    solar_irr: np.ndarray,
    csza_val: float,
    wl_nm: np.ndarray,
    lib_spectra: np.ndarray,
    base_surface_1d: np.ndarray,
    base_hot_1d: np.ndarray,
    base_smolder_1d: np.ndarray,
    eta_swir2_window_1d: np.ndarray,
    eta_swir1_override_window_1d: np.ndarray,
    # Pre-fit eta bounds from saturation; see eta_bounds_from_saturation().
    eta_sat_frac_low: float = 0.02,
    eta_sat_frac_high: float = 0.08,
    eta_floor_swir2_only_sat: float = 0.75,
    eta_floor_swir1_and_swir2_sat: float = 0.85,
    eta_ceiling_low_sat: float = 1.0,
    # Legacy soft-ramp knobs, accepted but unused.
    eta_floor_swir2_ramp_lo: float = 0.10,
    eta_floor_swir2_ramp_hi: float = 0.50,
    eta_floor_swir1_ramp_lo: float = 0.05,
    eta_floor_swir1_ramp_hi: float = 0.30,
    eta_floor_gate_lo: float = 0.40,
    eta_floor_gate_hi: float = 0.60,
    eta_floor_swir2_only_max: float = 0.6,
    eta_floor_max: float = 0.0,
    skip_dark_pixels: bool = True,
    dark_thresh: float = 0.15,
    use_cold_prior: bool = False,
    mu0_T: float = 500.0,
    lambda_mu_cold: float = 1e-3,
    alpha_cold_thresh: float = 1e-3,
    hot_window_weight: float = 1.0,
    smolder_window_weight: float = 1.0,
    batch_size: int = 8000,
    step1_adam_steps: int = 30,
    step1_adam_lr: float = 0.03,
    step2_adam_steps: int = 100,
    step2_adam_lr: float = 0.015,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    adam_grad_clip: float = 5.0,
    wv_quant_step: float = 0.02,
    alpha_tuning: float = DEFAULT_ALPHA_TUNING,
    sigma_min_k: float = DEFAULT_SIGMA_MIN_K,
    qc_band_diff_filter: bool = True,
    qc_band_lo_nm: float = 1096.9,
    qc_band_hi_nm: float = 2249.4,
    qc_band_diff_max: float = 5.0,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Two-stage batched Optax Adam retrieval over a radiance cube.

    Stage 1 fits surface coefficients plus a one-temperature seed; stage 2 fits
    the two-component Gaussian mixture (mu_hot, sigma_hot, mu_smol, sigma_smol,
    eta, alpha2) with the surface fixed. A pixel is retrieved only if it passes
    the last-band darkness check and the finite-value checks on radiance and
    water vapor.

    Inputs
    ------
    rdn : (rows, cols, bands) float32 -- at-sensor radiance.
    wv  : (rows, cols)        float32 -- H2OSTR field (clipped/filled upstream).
    bm  : (rows, cols, *)     uint8   -- ENVI bandmask (packed bits or per-band byte).

    Returns a dict of (rows, cols) maps -- mu_map and sigma_map (the
    moment-matched single Gaussian, K), the underlying mixture state, rmse_map,
    success_map and the SWIR saturation fractions -- plus n_valid, n_qc_masked
    and elapsed.
    """
    rows, cols, bands = rdn.shape
    Npix = rows * cols
    bs = int(max(batch_size, 1))
    n_end = int(lib_spectra.shape[0])

    _warmup_single_gaussian()

    rdn_flat = np.asarray(rdn.reshape(Npix, bands), dtype=np.float32)
    wv_flat = np.asarray(wv.reshape(Npix), dtype=np.float32)
    bm_flat = np.asarray(bm.reshape(Npix, -1), dtype=np.uint8)

    sat_ok_all = decode_sat_ok(bm_flat, bands)
    surf_mask_all = sat_ok_all & np.asarray(base_surface_1d, dtype=bool)[None, :]
    hot_mask_all = sat_ok_all & np.asarray(base_hot_1d, dtype=bool)[None, :]
    smolder_mask_all = sat_ok_all & np.asarray(base_smolder_1d, dtype=bool)[None, :]

    # Eta saturation-driven bounds and per-pixel saturation fractions.
    sat_frac_swir2_all, sat_frac_swir1_all, eta_lo_all, eta_hi_all = eta_bounds_from_saturation(
        bm_flat=bm_flat,
        swir2_window_1d=eta_swir2_window_1d,
        swir1_override_window_1d=eta_swir1_override_window_1d,
        bands=bands,
        eta_min=cfg.eta_min,
        eta_max=cfg.eta_max,
        eta_sat_frac_low=eta_sat_frac_low,
        eta_sat_frac_high=eta_sat_frac_high,
        eta_floor_swir2_only_sat=eta_floor_swir2_only_sat,
        eta_floor_swir1_and_swir2_sat=eta_floor_swir1_and_swir2_sat,
        eta_ceiling_low_sat=eta_ceiling_low_sat,
        eta_floor_max=eta_floor_max,
    )

    if verbose and float(eta_floor_max) > 0.0:
        n_floor_active = int(np.count_nonzero(eta_lo_all > float(cfg.eta_min) + 1e-6))
        n_ceil_active = int(np.count_nonzero(eta_hi_all < float(cfg.eta_max) - 1e-6))
        if n_floor_active > 0 or n_ceil_active > 0:
            med = float(np.median(eta_lo_all[eta_lo_all > float(cfg.eta_min) + 1e-6])) if n_floor_active > 0 else 0.0
            mx = float(np.max(eta_lo_all))
            print(
                f"[moment_matched] pre-fit eta hard floor: floor active on "
                f"{n_floor_active}/{int(eta_lo_all.size)} pixels; "
                f"median(eta_lo|active)={med:.3f}; max(eta_lo)={mx:.3f}; "
                f"sat thresholds [low={eta_sat_frac_low:.3f}, high={eta_sat_frac_high:.3f}]; "
                f"floor levels (swir2_only={eta_floor_swir2_only_sat:.2f}, "
                f"swir1+swir2={eta_floor_swir1_and_swir2_sat:.2f}); "
                f"ceiling active on {n_ceil_active} pixels "
                f"(eta_ceiling_low_sat={eta_ceiling_low_sat:.2f})"
            )

    # Output maps.
    mu_map = np.full((rows, cols), np.nan, dtype=np.float32)
    sigma_map = np.full((rows, cols), np.nan, dtype=np.float32)
    mu_hot_map = np.full((rows, cols), np.nan, dtype=np.float32)
    sigma_hot_map = np.full((rows, cols), np.nan, dtype=np.float32)
    mu_smol_map = np.full((rows, cols), np.nan, dtype=np.float32)
    sigma_smol_map = np.full((rows, cols), np.nan, dtype=np.float32)
    eta_map = np.full((rows, cols), np.nan, dtype=np.float32)
    alpha2_map = np.full((rows, cols), np.nan, dtype=np.float32)
    rmse_map = np.full((rows, cols), np.nan, dtype=np.float32)
    success_map = np.zeros((rows, cols), dtype=np.uint8)
    swir1_sat_frac_map = np.full((rows, cols), np.nan, dtype=np.float32)
    swir2_sat_frac_map = np.full((rows, cols), np.nan, dtype=np.float32)

    # Constants for the JIT'd batched solvers.
    wl_j = jnp.asarray(wl_nm, dtype=jnp.float32)
    solar_j = jnp.asarray(solar_irr, dtype=jnp.float32)
    lib_j = jnp.asarray(lib_spectra, dtype=jnp.float32)
    T_grid_j = jnp.asarray(cfg.T_grid, dtype=jnp.float32)
    wT_j = jnp.asarray(cfg.w_T, dtype=jnp.float32)
    one_wl_j = jnp.ones_like(wl_j)
    planck_grid_j = jax.vmap(lambda T: emissive_radiance(one_wl_j, T, wl_j))(T_grid_j).astype(jnp.float32)
    mult_j = solar_j / jnp.asarray(float(csza_val), dtype=jnp.float32) / jnp.asarray(np.pi, dtype=jnp.float32)
    eps_j = jnp.asarray(float(cfg.eps), dtype=jnp.float32)

    mu0_T_j = jnp.asarray(float(mu0_T), dtype=jnp.float32)
    lambda_mu_cold_j = jnp.asarray(float(lambda_mu_cold), dtype=jnp.float32)
    alpha_cold_thresh_j = jnp.asarray(float(max(alpha_cold_thresh, cfg.eps)), dtype=jnp.float32)
    hot_w_j = jnp.asarray(float(hot_window_weight), dtype=jnp.float32)
    smol_w_j = jnp.asarray(float(smolder_window_weight), dtype=jnp.float32)

    T1_lo_j = jnp.asarray(float(cfg.mu_hot_min), dtype=jnp.float32)
    T1_hi_j = jnp.asarray(float(cfg.mu_hot_max), dtype=jnp.float32)
    alpha1_lo_j = jnp.asarray(float(cfg.alpha1_min), dtype=jnp.float32)
    alpha1_hi_j = jnp.asarray(float(cfg.alpha1_max), dtype=jnp.float32)

    # QC band-diff filter setup: nearest sensor band to each target wavelength.
    _wl_np = np.asarray(wl_nm, dtype=np.float64)
    qc_lo_idx = int(np.argmin(np.abs(_wl_np - float(qc_band_lo_nm))))
    qc_hi_idx = int(np.argmin(np.abs(_wl_np - float(qc_band_hi_nm))))
    qc_lo_actual_nm = float(_wl_np[qc_lo_idx])
    qc_hi_actual_nm = float(_wl_np[qc_hi_idx])
    qc_enable = bool(qc_band_diff_filter)

    # Disable QC if the nearest sensor band is too far from the target.
    qc_band_distance_warn_nm = 5.0
    d_lo = abs(qc_lo_actual_nm - float(qc_band_lo_nm))
    d_hi = abs(qc_hi_actual_nm - float(qc_band_hi_nm))
    if qc_enable and (d_lo > qc_band_distance_warn_nm or d_hi > qc_band_distance_warn_nm):
        if verbose:
            print(f"[qc] WARN: nearest sensor bands far from QC targets "
                  f"({qc_band_lo_nm:.1f} nm: {qc_lo_actual_nm:.2f}, dist={d_lo:.2f}; "
                  f"{qc_band_hi_nm:.1f} nm: {qc_hi_actual_nm:.2f}, dist={d_hi:.2f}); "
                  f"band-diff filter disabled.")
        qc_enable = False

    step1_lo = jnp.concatenate([
        jnp.zeros((n_end,), dtype=jnp.float32),
        jnp.asarray([float(cfg.alpha1_min)], dtype=jnp.float32),
        jnp.asarray([float(cfg.mu_hot_min)], dtype=jnp.float32),
    ])
    step1_hi = jnp.concatenate([
        jnp.ones((n_end,), dtype=jnp.float32),
        jnp.asarray([float(cfg.alpha1_max)], dtype=jnp.float32),
        jnp.asarray([float(cfg.mu_hot_max)], dtype=jnp.float32),
    ])

    # Forward-model fragments (jax).
    def _gaussian_pdf_batch(mu, sigma, mu_lo, mu_hi, sigma_lo, sigma_hi):
        mu = jnp.clip(mu, mu_lo, mu_hi)
        sigma = jnp.clip(sigma, sigma_lo, sigma_hi)
        z = (T_grid_j[None, :] - mu[:, None]) / jnp.maximum(sigma[:, None], eps_j)
        p = jnp.exp(-0.5 * z * z)
        Z = jnp.sum(wT_j[None, :] * p, axis=1, keepdims=True) + eps_j
        return p / Z

    def _baseline_solar(surf, rhoatm, sphalb, transm_total):
        rfl = surf @ lib_j
        denom = jnp.maximum(1.0 - sphalb * rfl, eps_j)
        return mult_j[None, :] * (rhoatm + transm_total * rfl / denom)

    def _project_simplex(v):
        v = jnp.clip(v, 0.0, 1.0)
        s = jnp.sum(v, axis=1, keepdims=True)
        fallback = jnp.concatenate([
            jnp.zeros((v.shape[0], n_end - 1), dtype=v.dtype),
            jnp.ones((v.shape[0], 1), dtype=v.dtype),
        ], axis=1)
        return jnp.where(s > eps_j, v / jnp.maximum(s, eps_j), fallback)

    def _step1_denorm(z):
        x = _denormalize(z, step1_lo, step1_hi, eps_j)
        surf = _project_simplex(x[:, :n_end])
        alpha1 = jnp.clip(x[:, n_end], alpha1_lo_j, alpha1_hi_j)
        T1 = jnp.clip(x[:, n_end + 1], T1_lo_j, T1_hi_j)
        return jnp.concatenate([surf, alpha1[:, None], T1[:, None]], axis=1)

    def _step1_norm(x_phys):
        x = jnp.concatenate([
            _project_simplex(x_phys[:, :n_end]),
            jnp.clip(x_phys[:, n_end:n_end + 1], alpha1_lo_j, alpha1_hi_j),
            jnp.clip(x_phys[:, n_end + 1:n_end + 2], T1_lo_j, T1_hi_j),
        ], axis=1)
        return _normalize(x, step1_lo, step1_hi, eps_j)

    def _step1_forward(z, rhoatm, sphalb, transm_up, transm_total):
        x = _step1_denorm(z)
        surf = x[:, :n_end]
        alpha1 = x[:, n_end]
        T1 = x[:, n_end + 1]
        baseline = _baseline_solar(surf, rhoatm, sphalb, transm_total)
        L1 = jax.vmap(lambda T: emissive_radiance(one_wl_j, T, wl_j))(T1) * transm_up
        fm = baseline + alpha1[:, None] * L1
        return fm, surf, T1, alpha1

    def _step1_loss(z, obs, surf_mask, rhoatm, sphalb, transm_up, transm_total, valid_f):
        fm, surf, T1, alpha1 = _step1_forward(z, rhoatm, sphalb, transm_up, transm_total)
        diff = jnp.where(surf_mask, obs - fm, jnp.asarray(0.0, dtype=jnp.float32))
        loss = jnp.sum(diff * diff, axis=1)
        surf_sum = jnp.sum(surf, axis=1)
        ok = (
            (valid_f > 0.0)
            & jnp.isfinite(surf_sum) & (jnp.abs(surf_sum - 1.0) <= 1e-3)
            & jnp.all(jnp.isfinite(fm), axis=1)
            & jnp.isfinite(T1) & jnp.isfinite(alpha1)
        )
        return jnp.sum(jnp.where(ok, loss, 0.0))

    def _step2_bounds(eta_lo, eta_hi):
        lo = jnp.stack([
            jnp.full_like(eta_lo, float(cfg.mu_hot_min)),
            jnp.full_like(eta_lo, float(cfg.sigma_hot_min)),
            jnp.full_like(eta_lo, float(cfg.mu_smol_min)),
            jnp.full_like(eta_lo, float(cfg.sigma_smol_min)),
            eta_lo,
            jnp.full_like(eta_lo, float(cfg.alpha2_min)),
        ], axis=1)
        hi = jnp.stack([
            jnp.full_like(eta_hi, float(cfg.mu_hot_max)),
            jnp.full_like(eta_hi, float(cfg.sigma_hot_max)),
            jnp.full_like(eta_hi, float(cfg.mu_smol_max)),
            jnp.full_like(eta_hi, float(cfg.sigma_smol_max)),
            eta_hi,
            jnp.full_like(eta_hi, float(cfg.alpha2_max)),
        ], axis=1)
        return lo, hi

    def _step2_denorm(z, eta_lo, eta_hi):
        lo, hi = _step2_bounds(eta_lo, eta_hi)
        x = _denormalize(z, lo, hi, eps_j)
        return jnp.stack([jnp.clip(x[:, i], lo[:, i], hi[:, i]) for i in range(6)], axis=1)

    def _step2_norm(x_phys, eta_lo, eta_hi):
        lo, hi = _step2_bounds(eta_lo, eta_hi)
        x = jnp.stack([jnp.clip(x_phys[:, i], lo[:, i], hi[:, i]) for i in range(6)], axis=1)
        return _normalize(x, lo, hi, eps_j)

    def _step2_forward(z, surf, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi):
        x = _step2_denorm(z, eta_lo, eta_hi)
        mu_hot, sig_hot, mu_smol, sig_smol, eta, alpha2 = (x[:, i] for i in range(6))
        baseline = _baseline_solar(surf, rhoatm, sphalb, transm_total)

        # Use the moment-matched single Gaussian as the temperature pdf, which
        # halves the pdf/Planck work and matches the saved summary.
        mu_single = eta * mu_hot + (1.0 - eta) * mu_smol
        var_single = (
            eta * sig_hot ** 2
            + (1.0 - eta) * sig_smol ** 2
            + eta * (1.0 - eta) * (mu_hot - mu_smol) ** 2
        )
        sigma_single = jnp.sqrt(jnp.maximum(var_single, 1e-6))

        # Normalization bounds: union of the per-component mu and sigma ranges.
        mu_lo = jnp.minimum(jnp.asarray(cfg.mu_hot_min, dtype=jnp.float32),
                            jnp.asarray(cfg.mu_smol_min, dtype=jnp.float32))
        mu_hi = jnp.maximum(jnp.asarray(cfg.mu_hot_max, dtype=jnp.float32),
                            jnp.asarray(cfg.mu_smol_max, dtype=jnp.float32))
        sig_lo = jnp.minimum(jnp.asarray(cfg.sigma_hot_min, dtype=jnp.float32),
                             jnp.asarray(cfg.sigma_smol_min, dtype=jnp.float32))
        sig_hi = jnp.maximum(jnp.asarray(cfg.sigma_hot_max, dtype=jnp.float32),
                             jnp.asarray(cfg.sigma_smol_max, dtype=jnp.float32))

        p_T = _gaussian_pdf_batch(mu_single, sigma_single, mu_lo, mu_hi, sig_lo, sig_hi)
        e_T = (p_T * wT_j[None, :]) @ planck_grid_j
        thermal = alpha2[:, None] * e_T * transm_up
        fm = baseline + thermal
        return fm, x

    def _step2_loss(z, obs, hot_mask, smol_mask, surf, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi, valid_f):
        fm, x = _step2_forward(z, surf, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi)
        diff = obs - fm
        loss = (
            hot_w_j * jnp.sum((diff * hot_mask.astype(jnp.float32)) ** 2, axis=1)
            + smol_w_j * jnp.sum((diff * smol_mask.astype(jnp.float32)) ** 2, axis=1)
        )
        if use_cold_prior:
            mu_hot = x[:, 0]
            mu_smol = x[:, 2]
            alpha_tot = jnp.maximum(x[:, 5], 0.0)
            gate = 1.0 / (1.0 + (alpha_tot / alpha_cold_thresh_j) ** 2)
            cold_pen = (mu_hot - mu0_T_j) ** 2 + (mu_smol - mu0_T_j) ** 2
            loss = loss + lambda_mu_cold_j * gate * cold_pen
        return jnp.sum(loss * valid_f)

    step1_vag = jax.jit(jax.value_and_grad(_step1_loss))
    step2_vag = jax.jit(jax.value_and_grad(_step2_loss))

    if float(adam_grad_clip) > 0:
        step1_opt = optax.chain(
            optax.clip_by_global_norm(float(adam_grad_clip)),
            optax.adam(float(step1_adam_lr), b1=adam_beta1, b2=adam_beta2, eps=adam_eps),
        )
    else:
        step1_opt = optax.adam(float(step1_adam_lr), b1=adam_beta1, b2=adam_beta2, eps=adam_eps)
    step2_opt = optax.adam(float(step2_adam_lr), b1=adam_beta1, b2=adam_beta2, eps=adam_eps)

    @jax.jit
    def _solve_stage1(obs, surf_mask, valid_f, rhoatm, sphalb, transm_up, transm_total):
        B = obs.shape[0]
        x0_surf = jnp.zeros((B, n_end), dtype=jnp.float32).at[:, -1].set(1.0)
        x0_alpha = jnp.full((B, 1), jnp.maximum(jnp.asarray(1.0, jnp.float32), alpha1_lo_j), dtype=jnp.float32)
        x0_T = jnp.full((B, 1), jnp.maximum(jnp.asarray(cfg.T_step1_init, jnp.float32), 400.0), dtype=jnp.float32)
        z = _step1_norm(jnp.concatenate([x0_surf, x0_alpha, x0_T], axis=1))
        opt_state = step1_opt.init(z)

        def body(i, carry):
            z_i, st_i = carry
            _, g = step1_vag(z_i, obs, surf_mask, rhoatm, sphalb, transm_up, transm_total, valid_f)
            up, st_i = step1_opt.update(g, st_i, z_i)
            z_i = jnp.clip(optax.apply_updates(z_i, up), -1.0, 1.0)
            return z_i, st_i

        z, _ = jax.lax.fori_loop(0, int(max(step1_adam_steps, 0)), body, (z, opt_state))

        fm, surf, T1, alpha1 = _step1_forward(z, rhoatm, sphalb, transm_up, transm_total)
        ok = (
            (valid_f > 0.0)
            & jnp.all(jnp.isfinite(z), axis=1)
            & jnp.all(jnp.isfinite(fm), axis=1)
            & jnp.all(jnp.isfinite(surf), axis=1)
            & jnp.isfinite(T1) & jnp.isfinite(alpha1)
        )
        return surf, T1, alpha1, ok

    @jax.jit
    def _solve_stage2(obs, hot_mask, smol_mask, surf, valid_f, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi, T1, alpha1):
        B = obs.shape[0]
        alpha2_init = jnp.maximum(alpha1, jnp.asarray(float(cfg.alpha2_min), dtype=jnp.float32))
        eta_init = jnp.maximum(jnp.full((B,), 0.7, dtype=jnp.float32), eta_lo)
        x0 = jnp.stack([
            jnp.maximum(T1, jnp.asarray(400.0, jnp.float32)),
            jnp.full((B,), 0.20 * (cfg.sigma_hot_max - cfg.sigma_hot_min) + cfg.sigma_hot_min, dtype=jnp.float32),
            jnp.full((B,), 700.0, dtype=jnp.float32),
            jnp.full((B,), 0.20 * (cfg.sigma_smol_max - cfg.sigma_smol_min) + cfg.sigma_smol_min, dtype=jnp.float32),
            eta_init,
            alpha2_init,
        ], axis=1)
        z = _step2_norm(x0, eta_lo, eta_hi)
        opt_state = step2_opt.init(z)

        def body(i, carry):
            z_i, st_i = carry
            _, g = step2_vag(z_i, obs, hot_mask, smol_mask, surf, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi, valid_f)
            up, st_i = step2_opt.update(g, st_i, z_i)
            z_i = jnp.clip(optax.apply_updates(z_i, up), -1.0, 1.0)
            return z_i, st_i

        z, _ = jax.lax.fori_loop(0, int(max(step2_adam_steps, 0)), body, (z, opt_state))

        fm, x_phys = _step2_forward(z, surf, rhoatm, sphalb, transm_up, transm_total, eta_lo, eta_hi)
        fit_union = jnp.logical_or(hot_mask, smol_mask).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(fit_union, axis=1), 1.0)
        rmse = jnp.sqrt(jnp.sum(((obs - fm) * fit_union) ** 2, axis=1) / denom)
        ok = (
            (valid_f > 0.0)
            & jnp.all(jnp.isfinite(z), axis=1)
            & jnp.all(jnp.isfinite(x_phys), axis=1)
            & jnp.all(jnp.isfinite(fm), axis=1)
            & jnp.isfinite(rmse)
        )
        return x_phys, rmse, ok

    # JIT warm-up at the configured batch size.
    warm_obs = jnp.zeros((bs, bands), dtype=jnp.float32)
    warm_mask = jnp.zeros((bs, bands), dtype=jnp.bool_)
    warm_valid = jnp.zeros((bs,), dtype=jnp.float32)
    warm_atm = jnp.zeros((bs, bands), dtype=jnp.float32)
    warm_eta = jnp.full((bs,), float(cfg.eta_min), dtype=jnp.float32)
    surf_w, T1_w, alpha1_w, _ = _solve_stage1(warm_obs, warm_mask, warm_valid, warm_atm, warm_atm, warm_atm, warm_atm)
    _ = _solve_stage2(warm_obs, warm_mask, warm_mask, surf_w, warm_valid, warm_atm, warm_atm, warm_atm, warm_atm, warm_eta, warm_eta + 1.0, T1_w, alpha1_w)

    # WV quantization cache.
    wv_quant_step = float(max(wv_quant_step, 1e-6))
    atm_cache: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _atm_for_key(key: float):
        """Cached (rhoatm, sphalb, transm_up, transm_total) for one H2OSTR value."""
        if key in atm_cache:
            return atm_cache[key]
        x_atm = np.asarray([key], dtype=np.float32)
        rho = np.asarray(rte["rhoatm"](x_atm), dtype=np.float32).reshape(-1)
        sph = np.asarray(rte["sphalb"](x_atm), dtype=np.float32).reshape(-1)
        tup = (
            np.asarray(rte["transm_up_dif"](x_atm), dtype=np.float32).reshape(-1)
            + np.asarray(rte["transm_up_dir"](x_atm), dtype=np.float32).reshape(-1)
        )
        ttot = (
            np.asarray(rte["transm_down_dif"](x_atm), dtype=np.float32).reshape(-1)
            + np.asarray(rte["transm_down_dir"](x_atm), dtype=np.float32).reshape(-1)
        ) * tup
        atm_cache[key] = (rho, sph, tup, ttot)
        return atm_cache[key]

    t0 = time.perf_counter()
    n_valid = 0
    n_qc_masked = 0   # pixels dropped by the band-diff filter

    for i0 in range(0, Npix, bs):
        i1 = min(Npix, i0 + bs)
        n = i1 - i0

        obs_b = rdn_flat[i0:i1]
        wv_b = wv_flat[i0:i1]
        sat_ok_b = sat_ok_all[i0:i1]
        surf_mask_b = surf_mask_all[i0:i1]
        hot_mask_b = hot_mask_all[i0:i1]
        smol_mask_b = smolder_mask_all[i0:i1]
        eta_lo_b = eta_lo_all[i0:i1]
        eta_hi_b = eta_hi_all[i0:i1]

        valid = np.isfinite(wv_b) & np.isfinite(obs_b).all(axis=1)
        if skip_dark_pixels:
            valid &= obs_b[:, -1] > float(dark_thresh)
        if qc_enable:
            # Drop non-fire-shaped pixels (SWIR1 brighter than SWIR2);
            # saturation in either band auto-passes.
            obs_lo = obs_b[:, qc_lo_idx]
            obs_hi = obs_b[:, qc_hi_idx]
            unsat_lo = sat_ok_b[:, qc_lo_idx]
            unsat_hi = sat_ok_b[:, qc_hi_idx]
            failed_qc = (
                unsat_lo & unsat_hi
                & np.isfinite(obs_lo) & np.isfinite(obs_hi)
                & ((obs_lo - obs_hi) > float(qc_band_diff_max))
            )
            n_qc_masked += int(np.count_nonzero(valid & failed_qc))
            valid &= ~failed_qc
        if not np.any(valid):
            continue

        obs_clean = np.nan_to_num(obs_b, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        wv_clean = np.nan_to_num(wv_b, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)

        # Pad to bs, the batch size the solvers were compiled at.
        obs_pad = np.zeros((bs, bands), dtype=np.float32); obs_pad[:n] = obs_clean
        surf_pad = np.zeros((bs, bands), dtype=bool); surf_pad[:n] = np.where(valid[:, None], surf_mask_b, False)
        hot_pad = np.zeros((bs, bands), dtype=bool); hot_pad[:n] = np.where(valid[:, None], hot_mask_b, False)
        smol_pad = np.zeros((bs, bands), dtype=bool); smol_pad[:n] = np.where(valid[:, None], smol_mask_b, False)
        valid_pad = np.zeros((bs,), dtype=np.float32); valid_pad[:n] = valid.astype(np.float32)
        eta_lo_pad = np.full((bs,), float(cfg.eta_min), dtype=np.float32); eta_lo_pad[:n] = eta_lo_b
        eta_hi_pad = np.full((bs,), float(cfg.eta_max), dtype=np.float32); eta_hi_pad[:n] = eta_hi_b

        # Per-pixel atmosphere via the WV quantization cache.
        rhoatm_pad = np.zeros((bs, bands), dtype=np.float32)
        sphalb_pad = np.zeros((bs, bands), dtype=np.float32)
        transm_up_pad = np.zeros((bs, bands), dtype=np.float32)
        transm_total_pad = np.zeros((bs, bands), dtype=np.float32)
        wv_q = np.round(wv_clean[valid] / wv_quant_step) * wv_quant_step
        wv_q = np.maximum(wv_q, np.float32(WV_TEMP_CLIP_MIN))
        for key in np.unique(wv_q):
            rho, sph, tup, ttot = _atm_for_key(float(np.float32(key)))
            sel_local = np.where(valid)[0][wv_q == key]
            rhoatm_pad[sel_local, :] = rho[None, :]
            sphalb_pad[sel_local, :] = sph[None, :]
            transm_up_pad[sel_local, :] = tup[None, :]
            transm_total_pad[sel_local, :] = ttot[None, :]

        surf_b_arr, T1_b_arr, alpha1_b_arr, ok1_b = [
            np.asarray(x)[:n] for x in _solve_stage1(
                jnp.asarray(obs_pad), jnp.asarray(surf_pad), jnp.asarray(valid_pad),
                jnp.asarray(rhoatm_pad), jnp.asarray(sphalb_pad),
                jnp.asarray(transm_up_pad), jnp.asarray(transm_total_pad),
            )
        ]

        x_phys, rmse_b, ok2_b = [
            np.asarray(x) if x.ndim == 0 else np.asarray(x)[:n]
            for x in _solve_stage2(
                jnp.asarray(obs_pad), jnp.asarray(hot_pad), jnp.asarray(smol_pad),
                jnp.asarray(np.pad(surf_b_arr, ((0, bs - n), (0, 0)))),
                jnp.asarray(valid_pad),
                jnp.asarray(rhoatm_pad), jnp.asarray(sphalb_pad),
                jnp.asarray(transm_up_pad), jnp.asarray(transm_total_pad),
                jnp.asarray(eta_lo_pad), jnp.asarray(eta_hi_pad),
                jnp.asarray(np.pad(T1_b_arr, (0, bs - n))),
                jnp.asarray(np.pad(alpha1_b_arr, (0, bs - n))),
            )
        ]

        ok_b = ok1_b.astype(bool) & ok2_b.astype(bool) & valid
        if not np.any(ok_b):
            continue

        idx = np.where(ok_b)[0]
        rr = (np.arange(i0, i1) // cols)[idx]
        cc = (np.arange(i0, i1) % cols)[idx]

        mu_hot = x_phys[idx, 0].astype(np.float32)
        sig_hot = x_phys[idx, 1].astype(np.float32)
        mu_smol = x_phys[idx, 2].astype(np.float32)
        sig_smol = x_phys[idx, 3].astype(np.float32)
        eta = np.clip(x_phys[idx, 4], eta_lo_b[idx], eta_hi_b[idx]).astype(np.float32)
        alpha2 = x_phys[idx, 5].astype(np.float32)

        # Per-pixel saturation fractions, saved as diagnostics.
        sat_swir2_b = sat_frac_swir2_all[i0:i1][idx]
        sat_swir1_b = sat_frac_swir1_all[i0:i1][idx]

        mu_single, sigma_post = _single_gaussian_moments_jit(
            jnp.asarray(mu_hot), jnp.asarray(sig_hot),
            jnp.asarray(mu_smol), jnp.asarray(sig_smol),
            jnp.asarray(eta), jnp.asarray(alpha2),
            jnp.asarray(float(alpha_tuning), dtype=jnp.float32),
            jnp.asarray(float(sigma_min_k), dtype=jnp.float32),
        )

        mu_map[rr, cc] = np.asarray(mu_single)
        sigma_map[rr, cc] = np.asarray(sigma_post)
        mu_hot_map[rr, cc] = mu_hot
        sigma_hot_map[rr, cc] = sig_hot
        mu_smol_map[rr, cc] = mu_smol
        sigma_smol_map[rr, cc] = sig_smol
        eta_map[rr, cc] = eta
        alpha2_map[rr, cc] = alpha2
        rmse_map[rr, cc] = rmse_b[idx].astype(np.float32)
        success_map[rr, cc] = 1
        swir1_sat_frac_map[rr, cc] = sat_swir1_b
        swir2_sat_frac_map[rr, cc] = sat_swir2_b
        n_valid += int(idx.size)

    elapsed = time.perf_counter() - t0
    if verbose:
        qc_str = f"; qc_masked={n_qc_masked}" if qc_enable else ""
        print(f"[moment_matched] valid pixels: {n_valid} / {Npix} in {elapsed:.2f}s; batch={bs}; backend={jax.default_backend()}{qc_str}")

    return dict(
        mu_map=mu_map,
        sigma_map=sigma_map,
        mu_hot_map=mu_hot_map,
        sigma_hot_map=sigma_hot_map,
        mu_smol_map=mu_smol_map,
        sigma_smol_map=sigma_smol_map,
        eta_map=eta_map,
        alpha2_map=alpha2_map,
        rmse_map=rmse_map,
        success_map=success_map,
        swir1_sat_frac_map=swir1_sat_frac_map,
        swir2_sat_frac_map=swir2_sat_frac_map,
        n_valid=n_valid,
        n_qc_masked=n_qc_masked,
        elapsed=elapsed,
    )