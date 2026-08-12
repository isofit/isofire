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
#  Authors: william.r.keely@jpl.nasa.gov; philip.brodrick@jpl.nasa.gov
"""Radiative-transfer physics: 6S templating, spectral resampling, emissive
radiance, multipart 6S LUT engine, and forward-model helpers.
.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp


SIXS_TEMPLATE = """\
0 (User defined)
{solzen} {solaz} {viewzen} {viewaz} {month} {day}
8  (User defined H2O, O3)
{H2OSTR}, {O3}
{aermodel}
0
{AOT550}
-{elev:.2f} (target level)
-{alt:.2f} (sensor level)
-{H2OSTR}, -{O3}
{AOT550}
-2
{wlinf}
{wlsup}
0 Homogeneous surface
0 (no directional effects)
0
0
0
-1 No atm. corrections selected
"""


def parse_sixs(
    file: str,
    wl: np.ndarray,
    multipart_transmittance: bool = False,
    wl_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Parse a 6S output file into arrays used by LUT interpolation."""
    path, name = os.path.split(file)
    inpt = os.path.join(path, f"LUT_{name}.inp")

    with open(file, "r") as f:
        lines = f.readlines()
    with open(inpt, "r") as f:
        solzen = float(f.readlines()[1].strip().split()[0])
    coszen = np.cos(solzen / 360.0 * 2.0 * np.pi)

    data: Dict[str, list] = {}
    append = lambda key, val: data.setdefault(key, []).append(val)

    start = None
    start_multi = None
    tokens = None

    for end, line in enumerate(lines):
        if start is not None:
            tokens = re.findall(r"NaN|\d+\.?\d+", line.replace("******", "0.0"))
            if len(tokens) == 11:
                (w, gt, scad, scau, salb, rhoa, swl, step, sbor, dsol, toar) = tokens
                append("grid", float(w))
                append("sphalb", float(salb))
                append("rhoatm", float(rhoa))
                if not multipart_transmittance:
                    append("transm_down_dif", float(scau) * float(scad) * float(gt))
        elif line.startswith("*        trans  down   up"):
            start = end

        if start_multi and multipart_transmittance and tokens is not None and len(tokens) == 5:
            (w, tddir, tddif, tudir, tudif) = tokens
            append("transm_down_dir", float(tddir))
            append("transm_down_dif", float(tddif))
            append("transm_up_dir", float(tudir))
            append("transm_up_dif", float(tudif))
        elif line.startswith("*           down      down      up"):
            start_multi = end

        if start_multi and not multipart_transmittance:
            break
        if line.startswith("*                         integrated values of"):
            break

    if start is None:
        return {}
    if start_multi is None and multipart_transmittance:
        return {}

    out = {k: np.array(v) for k, v in data.items()}
    if wl_size > 0:
        out = {k: v[:wl_size] for k, v in out.items()}
    out["solzen"] = solzen
    out["coszen"] = coszen
    return out


def run_sixs(
    lut_dir: str,
    name: str,
    sixs_dict: dict,
    wavelengths: np.ndarray,
    sixs_dir: str,
    multipart_transmittance: bool = False,
) -> Dict[str, np.ndarray]:
    """Run 6S for one configuration and parse the output."""
    outp = os.path.join(lut_dir, name)
    inpt = os.path.join(lut_dir, f"LUT_{name}.inp")
    exe = os.path.join(sixs_dir, "sixsV2.1")

    with open(inpt, "w") as f:
        f.write(SIXS_TEMPLATE.format(**sixs_dict))

    """6S is executed directly rather than through a generated shell script.
    No shell is involved: the argument list is fixed and the redirections are
    real file handles, so nothing in ``lut_dir``, ``name`` or ``sixs_dir`` can
    be interpreted as a command (CWE-77). The .inp file is still written --
    parse_sixs reads the solar zenith back out of it -- but the intermediate
    LUT_<name>.sh is no longer needed."""
    if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        raise RuntimeError(
            f"6S binary not found or not executable: {exe}. Build it once with "
            f"`cd sixs/Sixs && make`, or point --sixs_dir at an existing build."
        )

    with open(inpt, "rb") as fin, open(outp, "wb") as fout:
        proc = subprocess.run(
            [exe], stdin=fin, stdout=fout, stderr=subprocess.PIPE, check=False,
        )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        print(f"[6s] WARNING: {exe} exited {proc.returncode} for {name}"
              + (f": {err}" if err else ""))

    return parse_sixs(outp, wavelengths, multipart_transmittance=multipart_transmittance)


def spectral_response_function(response_range: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normalized Gaussian SRF over a sampled wavelength grid."""
    u = (response_range - mu) / (abs(sigma) + 1e-12)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * (abs(sigma) + 1e-12))) * np.exp(-0.5 * u * u)
    return y / (y.sum() + 1e-12)


def resample_spectrum(x: np.ndarray, wl: np.ndarray, wl2: np.ndarray, fwhm2: np.ndarray) -> np.ndarray:
    """Resample a spectrum (or stack of spectra) onto (wl2, fwhm2)."""
    H = np.array([
        spectral_response_function(wl, wi, fwhmi / 2.355)
        for wi, fwhmi in zip(wl2, fwhm2)
    ])
    H[np.isnan(H)] = 0
    if x.ndim > 1:
        return np.dot(H, x.T).T
    return np.dot(H, x.reshape(-1, 1)).ravel()


def emissive_radiance(emissivity, T, wl_nm):
    """Surface emissive radiance in uW/cm^2/sr/nm at wavelengths in nm."""
    emissivity = jnp.asarray(emissivity, dtype=jnp.float32)
    T = jnp.asarray(T, dtype=jnp.float32)
    wl = jnp.asarray(wl_nm, dtype=jnp.float32)

    c_1 = 1.88365e32 / jnp.asarray(jnp.pi, dtype=jnp.float32)
    c_2 = 14387690.0
    J_per_eV = 1.60218e-19
    wl_um = wl / 1000.0

    ph = c_1 / (wl ** 4) / (jnp.exp(c_2 / wl / T) - 1.0) * emissivity
    eV = 1.2398 * ph / wl_um
    W = J_per_eV * eV
    return W * 1e6  # uW/cm^2/sr/nm


def _clear_dir(path: str) -> None:
    """Empty ``path`` (creating it if needed)."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except Exception:
                pass


def radiative_transfer_engine(
    lut_dir: str,
    points: Sequence[Sequence[float]],
    point_names: Sequence[str],
    sixs_dict: dict,
    wl_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    sixs_dir: str,
    multipart_transmittance: bool = True,
):
    """Build a 6S LUT and return JAX RegularGridInterpolators per field.

    With multipart_transmittance=True the returned dict has keys
    rhoatm, sphalb, transm_up_dir, transm_up_dif, transm_down_dir,
    transm_down_dif. Each interpolator takes a 1-D coordinate of length
    len(point_names). The LUT directory is cleared before being repopulated,
    so callers must use a per-process scratch path for concurrent jobs.
    """
    _clear_dir(lut_dir)

    jpoints = [jnp.asarray(p) for p in points]
    orig_shape = tuple(p.size for p in jpoints)
    grid = jnp.reshape(
        jnp.stack(jnp.meshgrid(*jpoints, indexing="ij")),
        (len(jpoints), -1),
    )
    N = int(np.prod(orig_shape))

    def _rg(arr):
        return jax.scipy.interpolate.RegularGridInterpolator(tuple(jpoints), arr)

    if not multipart_transmittance:
        sphalb = jnp.zeros((N, len(wl_nm)))
        transm = jnp.zeros_like(sphalb)
        rhoatm = jnp.zeros_like(sphalb)
        for p in range(N):
            ldict = sixs_dict.copy()
            for i, pn in enumerate(point_names):
                ldict[pn] = float(grid[i, p])
            lret = run_sixs(lut_dir, f"pt_{p}", ldict, wl_nm, sixs_dir, multipart_transmittance=False)
            sphalb = sphalb.at[p].set(resample_spectrum(lret["sphalb"], lret["grid"] * 1000, wl_nm, fwhm_nm))
            transm = transm.at[p].set(resample_spectrum(lret["transm_down_dif"], lret["grid"] * 1000, wl_nm, fwhm_nm))
            rhoatm = rhoatm.at[p].set(resample_spectrum(lret["rhoatm"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        new_shape = orig_shape + (len(wl_nm),)
        return {
            "sphalb": _rg(sphalb.reshape(new_shape)),
            "transm": _rg(transm.reshape(new_shape)),
            "rhoatm": _rg(rhoatm.reshape(new_shape)),
        }

    sphalb = jnp.zeros((N, len(wl_nm)))
    transm_up_dif = jnp.zeros((N, len(wl_nm)))
    transm_up_dir = jnp.zeros((N, len(wl_nm)))
    transm_down_dif = jnp.zeros((N, len(wl_nm)))
    transm_down_dir = jnp.zeros((N, len(wl_nm)))
    rhoatm = jnp.zeros((N, len(wl_nm)))

    last = None
    for p in range(N):
        ldict = sixs_dict.copy()
        for i, pn in enumerate(point_names):
            ldict[pn] = float(grid[i, p])
        lret = run_sixs(lut_dir, f"pt_{p}", ldict, wl_nm, sixs_dir, multipart_transmittance=True)
        last = lret
        sphalb = sphalb.at[p].set(resample_spectrum(lret["sphalb"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        transm_up_dif = transm_up_dif.at[p].set(resample_spectrum(lret["transm_up_dif"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        transm_up_dir = transm_up_dir.at[p].set(resample_spectrum(lret["transm_up_dir"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        transm_down_dif = transm_down_dif.at[p].set(resample_spectrum(lret["transm_down_dif"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        transm_down_dir = transm_down_dir.at[p].set(resample_spectrum(lret["transm_down_dir"], lret["grid"] * 1000, wl_nm, fwhm_nm))
        rhoatm = rhoatm.at[p].set(resample_spectrum(lret["rhoatm"], lret["grid"] * 1000, wl_nm, fwhm_nm))

    new_shape = orig_shape + (len(wl_nm),)
    return {
        "sphalb": _rg(sphalb.reshape(new_shape)),
        "transm_up_dif": _rg(transm_up_dif.reshape(new_shape)),
        "transm_up_dir": _rg(transm_up_dir.reshape(new_shape)),
        "transm_down_dif": _rg(transm_down_dif.reshape(new_shape)),
        "transm_down_dir": _rg(transm_down_dir.reshape(new_shape)),
        "rhoatm": _rg(rhoatm.reshape(new_shape)),
        "coszen": float(last["coszen"]) if last is not None else None,
    }


def rte_eval_1d(rte: dict, key: str, wv: float) -> np.ndarray:
    """Evaluate one atmosphere-dependent RT vector at a single H2OSTR value."""
    return np.asarray(
        rte[key](np.asarray([float(wv)], dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)


def combined_multipart_transmittance(rte: dict, wv: float) -> np.ndarray:
    """Total reflected-path transmittance (transm_down * transm_up).

    Falls back to the single-field 'transm' if a non-multipart engine is
    passed in.
    """
    try:
        td = rte_eval_1d(rte, "transm_down_dif", wv) + rte_eval_1d(rte, "transm_down_dir", wv)
        tu = rte_eval_1d(rte, "transm_up_dif", wv) + rte_eval_1d(rte, "transm_up_dir", wv)
        return td * tu
    except Exception:
        return rte_eval_1d(rte, "transm", wv)


def algebraic_forward(rhoatm, sphalb, transm, rfl, solar_irr, theta_s):
    """At-sensor radiance for a flat surface reflectance ``rfl`` (numpy)."""
    rho = transm / (1.0 / rfl - sphalb)
    return rho * solar_irr * np.cos(theta_s) / np.pi


def rdn_solar_from_surface(wv, x_surf, res, solar_irr_arr, csza_scalar, lib_spectra_arr):
    """Solar-path radiance from a surface mixture and 6S LUT terms (jax)."""
    wv = jnp.atleast_1d(wv).astype(jnp.float32)
    x_surf = jnp.asarray(x_surf, dtype=jnp.float32)
    lib = jnp.asarray(lib_spectra_arr, dtype=jnp.float32)

    rfl = lib.T @ x_surf
    mult = (solar_irr_arr * csza_scalar) / jnp.pi

    transm_up = res["transm_up_dif"](wv).flatten() + res["transm_up_dir"](wv).flatten()
    transm_down = res["transm_down_dif"](wv).flatten() + res["transm_down_dir"](wv).flatten()
    rhoatm = res["rhoatm"](wv).flatten()
    sphalb = res["sphalb"](wv).flatten()

    rdn_s = mult * (rhoatm + transm_down * transm_up * rfl / (1.0 - sphalb * rfl))
    return rdn_s, transm_up