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
"""PCA surface-reflectance model and the spaceborne two-stage retrieval.

The airborne retrieval represents the background surface as a simplex mixture
of the fire endmember library. Here the surface is instead

    rfl = mean + s @ components,     s = prior_mean + z @ L.T,   L L^T = prior_cov

with ``mean``, ``components`` and the score prior taken from the npz written by
``specmix_charash_pca_compare_nonoise.ipynb`` (cover-class v2 specmix
simulation of soil, rock, pv, npv, snow+ice, water and char-ash). The solver
works on the whitened scores ``z`` (prior N(0, I)), fits ``n_pc`` of them per
pixel, and converts back to reflectance for output.

Other changes from ``retrieval.run_moment_matched_retrieval`` (reasons in the
``run_pca_fire_retrieval`` docstring): emission scale factors are fit as
log10 (a sub-pixel fire or flare covers ~1e-5 to 1e-2 of an EMIT pixel),
residuals are divided by a per-band noise estimate so the score prior has a
defined weight, stage 2 refits the surface scores, and both stages use a
batched Levenberg-Marquardt solver instead of Adam.

``include_solar=False`` drops the solar path entirely (night): stage 1 fits a
single temperature and scale, stage 2 the two-component mixture, and no
surface is retrieved.

The stage-2 mixture (moment-matched single Gaussian over the temperature grid)
and the moment-matching of the output are the same as the airborne retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import jax
import jax.numpy as jnp

from .retrieval import (
    DEFAULT_ALPHA_TUNING,
    DEFAULT_SIGMA_MIN_K,
    MixtureRetrievalConfig,
    WV_TEMP_CLIP_MIN,
    _single_gaussian_moments_jit,
    _warmup_single_gaussian,
)


# ------------------------------------------------------------------ basis

@dataclass
class PCASurfaceBasis:
    """A PCA reflectance basis resampled onto a sensor grid."""

    wl_nm: np.ndarray            # (bands,) sensor wavelengths
    mean: np.ndarray             # (bands,)
    components: np.ndarray       # (n_pc, bands), rows orthonormal on the source grid
    prior_mean: np.ndarray       # (n_pc,)
    prior_cov: np.ndarray        # (n_pc, n_pc)
    valid_1d: np.ndarray         # (bands,) True where the basis is defined
    source: str = ""
    approach: str = ""
    grid_match: str = ""

    @property
    def n_pc(self) -> int:
        return int(self.components.shape[0])

    @property
    def chol(self) -> np.ndarray:
        c = np.asarray(self.prior_cov, np.float64)
        jitter = 1e-9 * max(float(np.trace(c)) / max(c.shape[0], 1), 1e-12)
        return np.linalg.cholesky(c + jitter * np.eye(c.shape[0]))

    def scores_from_z(self, z: np.ndarray) -> np.ndarray:
        return self.prior_mean[None, :] + np.asarray(z) @ self.chol.T

    def reflectance(self, scores: np.ndarray) -> np.ndarray:
        r = self.mean[None, :] + np.asarray(scores) @ self.components
        r[:, ~self.valid_1d] = np.nan
        return r

    @classmethod
    def from_npz(cls, path: str, wl_nm: np.ndarray, n_pc: Optional[int] = None,
                 max_gap_factor: float = 2.5) -> "PCASurfaceBasis":
        """Load a ``pca_surface_*.npz`` and put it on ``wl_nm``.

        If the npz grid matches the sensor grid (same length, < 0.5 nm apart)
        it is used band for band; otherwise mean and components are linearly
        interpolated, and a sensor band is valid only if a good source band lies
        within ``max_gap_factor`` source spacings of it (so the water-band gaps
        the simulator dropped stay excluded rather than being bridged).
        """
        wl_nm = np.asarray(wl_nm, np.float64)
        with np.load(path, allow_pickle=False) as z:
            src_wl = np.asarray(z["wavelengths"], np.float64).ravel()
            if src_wl.size and np.nanmax(src_wl) < 20.0:
                src_wl = src_wl * 1000.0
            if not (np.isfinite(src_wl).all() and 300.0 <= src_wl.min() and src_wl.max() <= 2700.0):
                raise ValueError(
                    f"{path}: 'wavelengths' is not a wavelength grid in nm (range "
                    f"{np.nanmin(src_wl):.1f}-{np.nanmax(src_wl):.1f}). The notebook "
                    "falls back to band indices when specmix exposes no wavelengths; "
                    "rerun it with the wavelength mapping working.")
            mean = np.asarray(z["mean"], np.float64)
            comp = np.asarray(z["components"], np.float64)
            good = np.asarray(z["good_bands"], bool) if "good_bands" in z.files else np.ones(src_wl.size, bool)
            pm = np.asarray(z["prior_mean"], np.float64)
            pc = np.asarray(z["prior_cov"], np.float64)
            approach = str(z["approach"]) if "approach" in z.files else ""
        n_avail = comp.shape[0]
        n = n_avail if n_pc is None else int(n_pc)
        if n > n_avail:
            raise ValueError(f"{path} holds {n_avail} components; asked for {n}")
        comp, pm, pc = comp[:n], pm[:n], np.atleast_2d(pc)[:n, :n]
        good = good & np.isfinite(mean) & np.isfinite(comp).all(axis=0)

        near = np.abs(wl_nm[:, None] - src_wl[None, :]).argmin(axis=0)       # sensor band per source band
        exact = np.abs(wl_nm[near] - src_wl) < 0.5
        if src_wl.size == wl_nm.size and np.nanmax(np.abs(src_wl - wl_nm)) < 0.5:
            valid = good.copy()
            mean_t = np.where(valid, mean, 0.0)
            comp_t = np.where(valid[None, :], comp, 0.0)
            how = "same grid"
        elif exact.all() and np.unique(near).size == near.size:
            # source grid is a subset of the sensor bands (the specmix grid drops the water
            # bands and the spectral ends): map band for band, nothing interpolated
            valid = np.zeros(wl_nm.size, bool)
            valid[near[good]] = True
            mean_t = np.zeros(wl_nm.size)
            comp_t = np.zeros((comp.shape[0], wl_nm.size))
            mean_t[near[good]] = mean[good]
            comp_t[:, near[good]] = comp[:, good]
            how = f"subset of sensor grid ({int(good.sum())} of {wl_nm.size} bands)"
        else:
            sw = src_wl[good]
            o = np.argsort(sw)
            sw = sw[o]
            sm = mean[good][o]
            sc = comp[:, good][:, o]
            step = float(np.median(np.diff(sw))) if sw.size > 1 else 10.0
            near = np.abs(wl_nm[:, None] - sw[None, :]).min(axis=1)
            # bracketed on both sides by good bands within the gap tolerance
            left = np.array([np.any((sw <= w) & (w - sw <= max_gap_factor * step)) for w in wl_nm])
            right = np.array([np.any((sw >= w) & (sw - w <= max_gap_factor * step)) for w in wl_nm])
            valid = (near <= max_gap_factor * step) & left & right
            mean_t = np.where(valid, np.interp(wl_nm, sw, sm), 0.0)
            comp_t = np.stack([np.where(valid, np.interp(wl_nm, sw, c), 0.0) for c in sc])
            how = f"interpolated ({src_wl.size} -> {wl_nm.size} bands)"
        return cls(wl_nm=wl_nm, mean=mean_t, components=comp_t, prior_mean=pm,
                   prior_cov=pc, valid_1d=valid, source=str(path), approach=approach,
                   grid_match=how)


# ------------------------------------------------------------ noise model

def radiance_noise(obs: np.ndarray, noise_frac: float, noise_floor: float) -> np.ndarray:
    """Per-band 1-sigma radiance noise: a fractional term plus a floor."""
    obs = np.abs(np.nan_to_num(np.asarray(obs, np.float32)))
    return np.sqrt((float(noise_frac) * obs) ** 2 + float(noise_floor) ** 2).astype(np.float32)


# ------------------------------------------------------------- initial guess

_C2_NM_K = 1.4388e7


def ratio_temperature(L_hi: np.ndarray, L_lo: np.ndarray, wl_hi: float, wl_lo: float,
                      t_min: float = 500.0, t_max: float = 3000.0) -> np.ndarray:
    """Two-band Planck temperature from L(wl_hi)/L(wl_lo), wl_hi > wl_lo.

    Only meaningful with no reflected sunlight (night); used to seed stage 1.
    """
    T = np.arange(t_min, t_max + 0.1, 5.0)
    r = (wl_lo / wl_hi) ** 5 * np.expm1(_C2_NM_K / (wl_lo * T)) / np.expm1(_C2_NM_K / (wl_hi * T))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.asarray(L_hi, np.float64) / np.asarray(L_lo, np.float64)
    out = np.interp(np.where(np.isfinite(R), R, r[0]), r[::-1], T[::-1])
    return out.astype(np.float32)


# ---------------------------------------------------------------- Planck

# rt.emissive_radiance in closed form: K * wl^-5 / expm1(c2 / (wl T)), in
# uW/cm^2/sr/nm with wl in nm. Same constants, so the two agree to float32
# rounding.
_PLANCK_C2 = 14387690.0
_PLANCK_K = (1.88365e32 / np.pi) * 1.2398 * 1000.0 * 1.60218e-19 * 1e6


def planck_stable(T, wl_nm):
    """Overflow-safe Planck radiance (uW/cm^2/sr/nm), differentiable in T.

    ``rt.emissive_radiance`` evaluates c1/wl^4/(exp(x)-1); its forward-mode
    derivative forms (c1/wl^4)*exp(x), which overflows float32 at the shortest
    EMIT bands below ~1000 K, and its gradient is NaN for some bands at 500 K.
    Written as exp(-x)/(-expm1(-x)) the value and its derivative underflow to
    zero instead.
    """
    wl = jnp.asarray(wl_nm, jnp.float32)
    x = _PLANCK_C2 / (wl * jnp.asarray(T, jnp.float32))
    e = jnp.exp(-x)
    return jnp.float32(_PLANCK_K) * wl ** -5 * e / (-jnp.expm1(-x))


# ------------------------------------------------------------------- LM

def _lm_batch(resid, u0, data, n_iter: int, lam0: float = 1e-2, diag_floor: float = 1e-6):
    """Batched Levenberg-Marquardt.

    ``resid(u, *data_i) -> r`` is a single-pixel residual vector; it is vmapped
    over the leading axis of ``u0`` and of every array in ``data``. Each pixel
    keeps its own damping and only accepts steps that lower its cost.

    Damping is lam * max(diag(H), diag_floor * max(diag(H))): plain Marquardt
    scaling leaves a parameter with no influence (the smoulder component when
    eta -> 1) almost undamped, and LM then throws it far away, where it sits
    until eta moves and exposes it.
    """
    J_fn = jax.jacfwd(resid)

    def one(u, *d):
        def cost(v):
            r = resid(v, *d)
            return jnp.sum(r * r)

        def step(_, c):
            u, lam, cst = c
            r = resid(u, *d)
            J = J_fn(u, *d)
            H = J.T @ J
            g = J.T @ r
            dg = jnp.diag(H)
            D = jnp.maximum(dg, diag_floor * jnp.max(dg) + 1e-30)
            du = -jnp.linalg.solve(H + lam * jnp.diag(D), g)
            u_new = u + jnp.where(jnp.isfinite(du), du, 0.0)
            c_new = cost(u_new)
            better = jnp.isfinite(c_new) & (c_new < cst)
            u = jnp.where(better, u_new, u)
            cst = jnp.where(better, c_new, cst)
            lam = jnp.clip(jnp.where(better, lam * 0.3, lam * 10.0), 1e-9, 1e9)
            return u, lam, cst

        u, _, cst = jax.lax.fori_loop(0, int(n_iter), step, (u, jnp.asarray(lam0, u.dtype), cost(u)))
        return u, cst

    return jax.vmap(one)(u0, *data)


def _to_unit(x, lo, hi):
    f = jnp.clip((x - lo) / jnp.maximum(hi - lo, 1e-12), 1e-4, 1.0 - 1e-4)
    return jnp.log(f) - jnp.log1p(-f)


def _from_unit(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


# ----------------------------------------------------------------- solver

def run_pca_fire_retrieval(
    obs: np.ndarray,
    wv: np.ndarray,
    sat_ok: np.ndarray,
    *,
    cfg: MixtureRetrievalConfig,
    rte: dict,
    solar_irr: np.ndarray,
    mu0: np.ndarray,
    wl_nm: np.ndarray,
    basis: Optional[PCASurfaceBasis],
    surface_1d: np.ndarray,
    hot_1d: np.ndarray,
    smolder_1d: np.ndarray,
    eta_lo: np.ndarray,
    eta_hi: np.ndarray,
    include_solar: bool = True,
    solar_scaling: str = "cos",
    T1_init: Optional[np.ndarray] = None,
    log_alpha1_init: Optional[np.ndarray] = None,
    log_alpha_min: float = -7.0,
    log_alpha_max: float = 0.3,
    noise_frac: float = 0.01,
    noise_floor: float = 0.005,
    prior_weight: float = 1.0,
    rfl_bounds_weight: float = 1.0e4,
    eta_init: float = 0.999,
    refine_surface_stage2: bool = True,
    hot_window_weight: float = 1.0,
    smolder_window_weight: float = 1.0,
    batch_size: int = 2048,
    lm_iters_stage1: int = 40,
    lm_iters_stage2: int = 60,
    wv_quant_step: float = 0.02,
    alpha_tuning: float = DEFAULT_ALPHA_TUNING,
    sigma_min_k: float = DEFAULT_SIGMA_MIN_K,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Two-stage batched Levenberg-Marquardt retrieval over candidate pixels.

    Inputs are per pixel, not a cube: ``obs`` (n, bands) radiance in
    uW/cm^2/sr/nm, ``wv`` (n,) H2OSTR, ``sat_ok`` (n, bands) True where a band
    is unsaturated, ``mu0`` scalar or (n,) cos(SZA), ``eta_lo``/``eta_hi`` (n,).

    Stage 1 fits [z (n_pc, day only), log10 alpha1, T1]. Stage 2 fits
    (mu_hot, sigma_hot, mu_smol, sigma_smol, eta, log10 alpha2) and, on day
    runs with ``refine_surface_stage2``, refits the PC scores too, starting from
    the stage-1 values with mu_hot = T1, sigma_hot at its floor and
    eta = ``eta_init``, i.e. next to the stage-1 fit.

    Differences from the airborne Adam solver, checked on synthetic EMIT
    pixels drawn from this forward model (tests/test_pca_surface.py):

    * Levenberg-Marquardt instead of Adam: Adam needed ~600 steps for the 8
      scores plus T in stage 1 to converge; LM converges in a few tens of
      iterations per stage.
    * The moment-matched sigma is clamped at a bound that covers the
      between-component spread. Clamping it at max(sigma_hot_max,
      sigma_smol_max) as the airborne retrieval does zeroes the gradient on eta
      and both sigmas whenever the mixture is wider than that (any T1 above
      ~1600 K at the airborne stage-2 initial guess), freezing them.
    * Planck radiance comes from ``planck_stable``: the forward-mode derivative
      of ``rt.emissive_radiance`` overflows float32 at the shortest EMIT bands
      below ~1000 K, which turns the whole Jacobian non-finite.
    * Stage 2 refits the PC scores by default (``refine_surface_stage2``); on
      the synthetic tests this lowers the stage-2 chi-square slightly (e.g.
      0.27 -> 0.22) relative to holding the stage-1 surface fixed.
    * Every fit window drops bands where the PCA basis is undefined.

    E[T] is only as identifiable as the spectrum allows: on synthetic broad
    mixtures several (mu, sigma, eta) fit to the noise level, and day-time
    E[T] then tends toward the stage-1 single temperature with a reported
    sigma smaller than the actual error. Treat sigma as a lower bound there.

    Bounded parameters go through a sigmoid; the whitened scores carry an
    N(0, I) prior as extra residuals, and reflectance outside [0, 1] in the fit
    window is penalized the same way. Residuals are divided by
    ``radiance_noise``.

    ``solar_scaling``: "cos" uses solar * mu0 / pi, the standard ISOFIT
    convention and the one ``rt.algebraic_forward`` (the water estimator) uses;
    "inv_cos" reproduces the solar / mu0 / pi of the airborne retrieval.

    Returns (n,) arrays plus ``scores`` (n, n_pc) and ``rfl`` (n, bands) on
    day runs (NaN on night runs and failed pixels).
    """
    obs = np.asarray(obs, np.float32)
    n_pix, bands = obs.shape
    use_surf = bool(include_solar) and basis is not None
    if include_solar and basis is None:
        raise ValueError("include_solar=True needs a PCA basis")
    n_pc = basis.n_pc if use_surf else 0
    n_z2 = n_pc if (use_surf and refine_surface_stage2) else 0
    bs = int(min(max(batch_size, 1), max(64, 1 << int(np.ceil(np.log2(max(n_pix, 1)))))))

    _warmup_single_gaussian()

    mu0 = np.broadcast_to(np.asarray(mu0, np.float32), (n_pix,)).astype(np.float32)
    wv = np.asarray(wv, np.float32)
    sat_ok = np.asarray(sat_ok, bool)
    surf_m = sat_ok & np.asarray(surface_1d, bool)[None, :]
    hot_m = sat_ok & np.asarray(hot_1d, bool)[None, :]
    smol_m = sat_ok & np.asarray(smolder_1d, bool)[None, :]
    if use_surf:
        # Every window, not only the surface one: where the basis is undefined
        # (water-band gaps the simulator dropped) the modelled reflectance is 0,
        # and a thermal window that keeps those bands cannot be fit.
        for m_ in (surf_m, hot_m, smol_m):
            m_ &= basis.valid_1d[None, :]
    sig = radiance_noise(obs, noise_frac, noise_floor)

    if T1_init is None:
        T1_init = np.full(n_pix, float(cfg.T_step1_init), np.float32)
    if log_alpha1_init is None:
        log_alpha1_init = np.full(n_pix, -3.0, np.float32)
    T1_init = np.clip(np.asarray(T1_init, np.float32), cfg.mu_hot_min, cfg.mu_hot_max)
    log_alpha1_init = np.clip(np.asarray(log_alpha1_init, np.float32), log_alpha_min, log_alpha_max)
    eta_lo = np.asarray(eta_lo, np.float32)
    eta_hi = np.maximum(np.asarray(eta_hi, np.float32), eta_lo + 1e-3)

    # ---- constants
    f32 = jnp.float32
    wl_j = jnp.asarray(wl_nm, f32)
    solar_j = jnp.asarray(solar_irr, f32)
    T_grid_j = jnp.asarray(cfg.T_grid, f32)
    wT_j = jnp.asarray(cfg.w_T, f32)
    planck_grid_j = jax.vmap(lambda T: planck_stable(T, wl_j))(T_grid_j).astype(f32)
    eps = float(cfg.eps)
    sq_prior = float(np.sqrt(prior_weight))
    sq_bnd = float(np.sqrt(rfl_bounds_weight))
    sq_hot = float(np.sqrt(hot_window_weight))
    sq_smol = float(np.sqrt(smolder_window_weight))
    if use_surf:
        mean_j = jnp.asarray(basis.mean, f32)
        comp_j = jnp.asarray(basis.components, f32)
        pm_j = jnp.asarray(basis.prior_mean, f32)
        L_j = jnp.asarray(basis.chol, f32)
    la_lo, la_hi = float(log_alpha_min), float(log_alpha_max)
    T_lo, T_hi = float(cfg.mu_hot_min), float(cfg.mu_hot_max)

    mu_lo_u = float(min(cfg.mu_hot_min, cfg.mu_smol_min))
    mu_hi_u = float(max(cfg.mu_hot_max, cfg.mu_smol_max))
    sg_lo_u = float(min(cfg.sigma_hot_min, cfg.sigma_smol_min))
    sg_hi_u = float(np.sqrt(max(cfg.sigma_hot_max, cfg.sigma_smol_max) ** 2
                            + 0.25 * (mu_hi_u - mu_lo_u) ** 2))
    lo6 = np.array([cfg.mu_hot_min, cfg.sigma_hot_min, cfg.mu_smol_min, cfg.sigma_smol_min, 0.0, la_lo], np.float32)
    hi6 = np.array([cfg.mu_hot_max, cfg.sigma_hot_max, cfg.mu_smol_max, cfg.sigma_smol_max, 1.0, la_hi], np.float32)

    def _rfl(z):
        return mean_j + (pm_j + L_j @ z) @ comp_j

    def _base(rfl, rho, sph, ttot, mu0_i):
        mult = solar_j / jnp.maximum(mu0_i, 1e-3) / jnp.pi if solar_scaling == "inv_cos" \
            else solar_j * mu0_i / jnp.pi
        return mult * (rho + ttot * rfl / jnp.maximum(1.0 - sph * rfl, eps))

    def _mix_emission(p6):
        mu_h, sg_h, mu_s, sg_s, eta = p6[0], p6[1], p6[2], p6[3], p6[4]
        m = eta * mu_h + (1.0 - eta) * mu_s
        v = eta * sg_h ** 2 + (1.0 - eta) * sg_s ** 2 + eta * (1.0 - eta) * (mu_h - mu_s) ** 2
        m = jnp.clip(m, mu_lo_u, mu_hi_u)
        sd = jnp.clip(jnp.sqrt(jnp.maximum(v, 1e-6)), sg_lo_u, sg_hi_u)
        p = jnp.exp(-0.5 * ((T_grid_j - m) / sd) ** 2)
        p = p / (jnp.sum(wT_j * p) + eps)
        return (p * wT_j) @ planck_grid_j

    def _surface_terms(z, m, rfl):
        r_prior = sq_prior * z
        r_bnd = sq_bnd * jnp.where(m, jax.nn.relu(-rfl) + jax.nn.relu(rfl - 1.0), 0.0)
        return r_prior, r_bnd

    # ---- stage 1 residual (one pixel): u = [z (n_pc), unit(log10 a1), unit(T1)]
    def _res1(u, o, s, m, rho, sph, tup, ttot, mu0_i):
        la = _from_unit(u[n_pc], la_lo, la_hi)
        T = _from_unit(u[n_pc + 1], T_lo, T_hi)
        th = 10.0 ** la * planck_stable(T, wl_j) * tup
        if n_pc:
            z = u[:n_pc]
            rfl = _rfl(z)
            fm = _base(rfl, rho, sph, ttot, mu0_i) + th
            r = jnp.where(m, (o - fm) / s, 0.0)
            rp, rb = _surface_terms(z, m, rfl)
            return jnp.concatenate([r, rp, rb])
        return jnp.where(m, (o - th) / s, 0.0)

    # ---- stage 2 residual: u = [z (n_z2), unit(6 mixture params)]
    def _res2(u, o, s, mfit, hm, sm, base, rho, sph, tup, ttot, mu0_i, elo, ehi):
        lo = jnp.asarray(lo6).at[4].set(elo)
        hi = jnp.asarray(hi6).at[4].set(ehi)
        p6 = _from_unit(u[n_z2:], lo, hi)
        th = 10.0 ** p6[5] * _mix_emission(p6) * tup
        if n_z2:
            z = u[:n_z2]
            rfl = _rfl(z)
            fm = _base(rfl, rho, sph, ttot, mu0_i) + th
            r = jnp.where(mfit, (o - fm) / s, 0.0)
            rp, rb = _surface_terms(z, mfit, rfl)
            return jnp.concatenate([r, rp, rb])
        r = (o - (base + th)) / s
        return jnp.concatenate([sq_hot * jnp.where(hm, r, 0.0), sq_smol * jnp.where(sm, r, 0.0)])

    @jax.jit
    def _stage1(u0, o, s, m, rho, sph, tup, ttot, mu0_b):
        u, _ = _lm_batch(_res1, u0, (o, s, m, rho, sph, tup, ttot, mu0_b), lm_iters_stage1)
        la = _from_unit(u[:, n_pc], la_lo, la_hi)
        T1 = _from_unit(u[:, n_pc + 1], T_lo, T_hi)
        z = u[:, :n_pc]
        if n_pc:
            base = jax.vmap(lambda zz, a, b, c, d: _base(_rfl(zz), a, b, c, d))(z, rho, sph, ttot, mu0_b)
            th = (10.0 ** la)[:, None] * jax.vmap(lambda t: planck_stable(t, wl_j))(T1) * tup
            fm = base + th
        else:
            base = jnp.zeros_like(o)
            fm = (10.0 ** la)[:, None] * jax.vmap(lambda t: planck_stable(t, wl_j))(T1) * tup
        r = jnp.where(m, (o - fm) / s, 0.0)
        chi2 = jnp.sum(r * r, axis=1) / jnp.maximum(jnp.sum(m, axis=1), 1)
        ok = jnp.all(jnp.isfinite(u), axis=1) & jnp.all(jnp.isfinite(fm), axis=1)
        return z, 10.0 ** la, T1, base, chi2, ok

    @jax.jit
    def _stage2(u0, o, s, mfit, hm, sm, base, rho, sph, tup, ttot, mu0_b, elo, ehi):
        data = (o, s, mfit, hm, sm, base, rho, sph, tup, ttot, mu0_b, elo, ehi)
        u, _ = _lm_batch(_res2, u0, data, lm_iters_stage2)
        lo = jnp.broadcast_to(jnp.asarray(lo6), (u.shape[0], 6)).at[:, 4].set(elo)
        hi = jnp.broadcast_to(jnp.asarray(hi6), (u.shape[0], 6)).at[:, 4].set(ehi)
        p6 = _from_unit(u[:, n_z2:], lo, hi)
        if n_z2:
            z = u[:, :n_z2]
            base = jax.vmap(lambda zz, a, b, c, d: _base(_rfl(zz), a, b, c, d))(z, rho, sph, ttot, mu0_b)
        else:
            z = jnp.zeros((u.shape[0], 0), f32)
        fm = base + (10.0 ** p6[:, 5])[:, None] * jax.vmap(_mix_emission)(p6) * tup
        uu = jnp.logical_or(hm, sm)
        n_u = jnp.maximum(jnp.sum(uu, axis=1), 1)
        rmse = jnp.sqrt(jnp.sum(jnp.where(uu, o - fm, 0.0) ** 2, axis=1) / n_u)
        rr = jnp.where(uu, (o - fm) / s, 0.0)
        chi2 = jnp.sum(rr * rr, axis=1) / n_u
        ok = jnp.all(jnp.isfinite(u), axis=1) & jnp.isfinite(rmse)
        return p6, z, rmse, chi2, ok

    # ---- per-pixel atmosphere through a water-vapor quantization cache
    wq = float(max(wv_quant_step, 1e-6))
    cache: Dict[float, tuple] = {}

    def _atm(key: float):
        if key not in cache:
            xa = np.asarray([key], np.float32)
            g = lambda k: np.asarray(rte[k](xa), np.float32).reshape(-1)
            tup = g("transm_up_dif") + g("transm_up_dir")
            ttot = (g("transm_down_dif") + g("transm_down_dir")) * tup
            cache[key] = (g("rhoatm"), g("sphalb"), tup, ttot)
        return cache[key]

    # ---- outputs
    nan = lambda *sh: np.full(sh, np.nan, np.float32)
    out = dict(mu=nan(n_pix), sigma=nan(n_pix), mu_hot=nan(n_pix), sigma_hot=nan(n_pix),
               mu_smol=nan(n_pix), sigma_smol=nan(n_pix), eta=nan(n_pix), alpha2=nan(n_pix),
               T1=nan(n_pix), alpha1=nan(n_pix), rmse=nan(n_pix), chi2_stage1=nan(n_pix),
               chi2_stage2=nan(n_pix), success=np.zeros(n_pix, np.uint8),
               scores=nan(n_pix, n_pc), rfl=nan(n_pix, bands) if use_surf else None)

    t0 = time.perf_counter()
    for i0 in range(0, n_pix, bs):
        i1 = min(n_pix, i0 + bs)
        n = i1 - i0
        o = obs[i0:i1]
        valid = np.isfinite(o).all(axis=1) & np.isfinite(wv[i0:i1])
        if not valid.any():
            continue

        def pad(a, fill=0.0, dtype=np.float32):
            a = np.asarray(a)
            p = np.full((bs,) + a.shape[1:], fill, dtype=dtype)
            p[:n] = a
            return p

        rho = np.zeros((bs, bands), np.float32)
        sph = np.zeros_like(rho)
        tup = np.zeros_like(rho)
        ttot = np.zeros_like(rho)
        wvq = np.maximum(np.round(np.nan_to_num(wv[i0:i1], nan=1.0) / wq) * wq, WV_TEMP_CLIP_MIN)
        for key in np.unique(wvq):
            sel = np.flatnonzero(wvq == key)
            rho[sel], sph[sel], tup[sel], ttot[sel] = _atm(float(np.float32(key)))

        vm = valid[:, None]
        o_b = jnp.asarray(pad(np.nan_to_num(o)))
        s_b = jnp.asarray(pad(sig[i0:i1], 1.0))
        m1 = pad(surf_m[i0:i1] & vm, False, bool) if use_surf \
            else pad((hot_m[i0:i1] | smol_m[i0:i1]) & vm, False, bool)
        hm = pad(hot_m[i0:i1] & vm, False, bool)
        sm = pad(smol_m[i0:i1] & vm, False, bool)
        atm = [jnp.asarray(a) for a in (rho, sph, tup, ttot)]
        mu0_b = jnp.asarray(pad(mu0[i0:i1], 1.0))

        u1 = np.zeros((bs, n_pc + 2), np.float32)
        u1[:, n_pc] = np.asarray(_to_unit(jnp.asarray(pad(log_alpha1_init[i0:i1], -3.0)), la_lo, la_hi))
        u1[:, n_pc + 1] = np.asarray(_to_unit(jnp.asarray(pad(T1_init[i0:i1], cfg.T_step1_init)), T_lo, T_hi))
        z1, a1, T1, base, chi1, ok1 = _stage1(jnp.asarray(u1), o_b, s_b, jnp.asarray(m1), *atm, mu0_b)

        elo = pad(eta_lo[i0:i1], float(cfg.eta_min))
        ehi = pad(eta_hi[i0:i1], float(cfg.eta_max))
        T1n, a1n = np.asarray(T1), np.asarray(a1)
        x0 = np.stack([
            T1n,
            np.full(bs, cfg.sigma_hot_min * 1.05, np.float32),
            np.full(bs, 0.5 * (cfg.mu_smol_min + cfg.mu_smol_max), np.float32),
            np.full(bs, 0.2 * (cfg.sigma_smol_max - cfg.sigma_smol_min) + cfg.sigma_smol_min, np.float32),
            np.clip(np.full(bs, float(eta_init), np.float32), elo, ehi),
            np.log10(np.maximum(a1n, 10.0 ** la_lo)),
        ], axis=1)
        lo_b = np.broadcast_to(lo6, (bs, 6)).copy(); lo_b[:, 4] = elo
        hi_b = np.broadcast_to(hi6, (bs, 6)).copy(); hi_b[:, 4] = ehi
        u2 = np.asarray(_to_unit(jnp.asarray(x0), jnp.asarray(lo_b), jnp.asarray(hi_b)))
        if n_z2:
            u2 = np.concatenate([np.asarray(z1), u2], axis=1)
        mfit = jnp.asarray(m1 | hm | sm)
        p6, z2, rmse, chi2, ok2 = _stage2(jnp.asarray(u2), o_b, s_b, mfit, jnp.asarray(hm),
                                          jnp.asarray(sm), base, *atm, mu0_b,
                                          jnp.asarray(elo), jnp.asarray(ehi))

        ok = np.asarray(ok1)[:n] & np.asarray(ok2)[:n] & valid
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        g = i0 + idx
        xp = np.asarray(p6)[idx]
        alpha2 = (10.0 ** xp[:, 5]).astype(np.float32)
        m_, s_ = _single_gaussian_moments_jit(
            jnp.asarray(xp[:, 0]), jnp.asarray(xp[:, 1]), jnp.asarray(xp[:, 2]),
            jnp.asarray(xp[:, 3]), jnp.asarray(xp[:, 4]), jnp.asarray(alpha2),
            jnp.asarray(float(alpha_tuning), f32), jnp.asarray(float(sigma_min_k), f32))
        out["mu"][g] = np.asarray(m_)
        out["sigma"][g] = np.asarray(s_)
        out["mu_hot"][g], out["sigma_hot"][g] = xp[:, 0], xp[:, 1]
        out["mu_smol"][g], out["sigma_smol"][g] = xp[:, 2], xp[:, 3]
        out["eta"][g], out["alpha2"][g] = xp[:, 4], alpha2
        out["T1"][g] = T1n[idx]
        out["alpha1"][g] = a1n[idx]
        out["rmse"][g] = np.asarray(rmse)[idx]
        out["chi2_stage1"][g] = np.asarray(chi1)[idx]
        out["chi2_stage2"][g] = np.asarray(chi2)[idx]
        out["success"][g] = 1
        if use_surf:
            zf = np.asarray(z2)[idx] if n_z2 else np.asarray(z1)[idx]
            sc = basis.scores_from_z(zf)
            out["scores"][g] = sc
            out["rfl"][g] = basis.reflectance(sc)

    out["elapsed"] = time.perf_counter() - t0
    out["n_valid"] = int(out["success"].sum())
    if verbose:
        print(f"[pca_retrieval] {'day (PCA surface + thermal)' if use_surf else 'night (thermal only)'}: "
              f"{out['n_valid']}/{n_pix} pixels in {out['elapsed']:.2f}s, batch {bs}, "
              f"backend {jax.default_backend()}")
    return out
