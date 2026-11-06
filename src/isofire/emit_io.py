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
"""EMIT L1B readers for the spaceborne retrieval and the scene scanners.

Reads the EMIT SDS ops layout (ENVI pairs under ``<acq_root>/<YYYYMMDD>/<fid>/l1b/``)
and the LP DAAC netCDF products. Deliberately free of JAX so scanners can use
it from forked worker processes.

Only the bands and pixels that are asked for are read: a whole EMIT radiance
cube is ~1.8 GB, and the retrieval only ever needs a handful of bands over the
scene plus full spectra at the candidate pixels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:          # only needed for netCDF inputs
    h5py = None

FILL = -9999.0
FID_RE = re.compile(r"(emit\d{8}t\d{6})", re.IGNORECASE)

# Standard EMIT OBS band order, used when a file carries no band names.
OBS_INDEX_DEFAULT = dict(path_length=0, sensor_az=1, sensor_zen=2, sun_az=3,
                         sun_zen=4, phase=5, slope=6, aspect=7, cos_i=8,
                         utc_time=9, earth_sun_dist=10)
_OBS_NAME_KEYS = dict(path_length="path length", sensor_az="to-sensor azimuth",
                      sensor_zen="to-sensor zenith", sun_az="to-sun azimuth",
                      sun_zen="to-sun zenith", earth_sun_dist="earth-sun distance")

# ENVI file kinds in the ops layout: emit<date>t<time>_o<orbit>_s<scene>_l1b_<kind>_b<build>_v<ver>
_ENVI_KIND_RE = re.compile(r"_l1b_(rdn|obs|loc|glt|bandmask)_", re.IGNORECASE)
REQUIRED_KINDS = ("rdn", "obs", "loc")
OPTIONAL_KINDS = ("glt", "bandmask")


def fid_from_path(p) -> Optional[str]:
    m = FID_RE.search(str(p))
    return m.group(1).lower() if m else None


def fid_datetime(fid: str) -> Optional[datetime]:
    try:
        return datetime.strptime(fid[4:12] + fid[13:19], "%Y%m%d%H%M%S")
    except (ValueError, TypeError):
        return None


def _to_nm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v * 1000.0 if v.size and np.nanmax(v) < 20.0 else v


def _clean(a) -> np.ndarray:
    a = np.array(a, dtype=np.float32)
    a[a <= FILL + 1] = np.nan
    return a


# ---------------------------------------------------------------- ENVI headers

def read_envi_header(path) -> Dict[str, str]:
    txt = Path(path).read_text(errors="replace")
    out = {}
    for m in re.finditer(r"^[ \t]*([A-Za-z][^=\n]*?)[ \t]*=[ \t]*(\{.*?\}|[^\n]*)",
                         txt, re.M | re.S):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def _hlist(v, cast=str) -> list:
    if not v:
        return []
    v = v.strip().lstrip("{").rstrip("}")
    return [cast(x.strip()) for x in v.split(",") if x.strip()]


def _hdr_for(img: Path) -> Optional[Path]:
    for h in (img.with_suffix(".hdr"), Path(str(img) + ".hdr")):
        if h.exists():
            return h
    return None


def _img_for(hdr: Path) -> Optional[Path]:
    stem = hdr.with_suffix("")
    for c in (stem, stem.with_suffix(".img"), stem.with_suffix(".dat")):
        if c.exists() and c.is_file():
            return c
    return None


_ENVI_DT = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4",
            14: "i8", 15: "u8"}


class EnviCube:
    """numpy-memmap view of an ENVI cube with (row, col, band) accessors."""

    def __init__(self, img_or_hdr):
        p = Path(img_or_hdr)
        if p.suffix.lower() == ".hdr":
            self.hdr_path, self.img_path = p, _img_for(p)
        else:
            self.img_path, self.hdr_path = p, _hdr_for(p)
        if self.img_path is None or self.hdr_path is None:
            raise FileNotFoundError(f"ENVI pair incomplete for {img_or_hdr}")
        h = read_envi_header(self.hdr_path)
        self.header = h
        ns, nl, nb = (int(h[k]) for k in ("samples", "lines", "bands"))
        bo = ">" if int(h.get("byte order", "0")) == 1 else "<"
        dt = np.dtype(bo + _ENVI_DT[int(h["data type"])])
        self.interleave = h.get("interleave", "bsq").strip().lower()
        shape = {"bil": (nl, nb, ns), "bip": (nl, ns, nb), "bsq": (nb, nl, ns)}[self.interleave]
        self.mm = np.memmap(self.img_path, dtype=dt, mode="r",
                            offset=int(h.get("header offset", "0")), shape=shape)
        self.rows, self.cols, self.nb = nl, ns, nb
        self.band_names = _hlist(h.get("band names"))
        wl = np.array(_hlist(h.get("wavelength"), float), float)
        self.wl_nm = _to_nm(wl) if wl.size else wl
        fw = np.array(_hlist(h.get("fwhm"), float), float)
        if fw.size and self.wl_nm.size and np.nanmax(fw) < 0.1 and np.nanmax(self.wl_nm) > 100:
            fw = fw * 1000.0
        self.fwhm_nm = fw

    @property
    def shape(self):
        return self.rows, self.cols, self.nb

    def band(self, i: int, step: int = 1) -> np.ndarray:
        s = slice(None, None, step)
        if self.interleave == "bil":
            a = self.mm[s, i, s]
        elif self.interleave == "bip":
            a = self.mm[s, s, i]
        else:
            a = self.mm[i, s, s]
        return np.array(a)

    def bands(self, idx: Sequence[int], row_block: int = 256) -> np.ndarray:
        """(rows, cols, len(idx)) for the listed bands."""
        idx = [int(i) for i in idx]
        out = np.empty((self.rows, self.cols, len(idx)), dtype=np.float32)
        for r0 in range(0, self.rows, row_block):
            r1 = min(r0 + row_block, self.rows)
            if self.interleave == "bil":
                blk = np.asarray(self.mm[r0:r1][:, idx, :]).transpose(0, 2, 1)
            elif self.interleave == "bip":
                blk = np.asarray(self.mm[r0:r1][:, :, idx])
            else:
                blk = np.asarray(self.mm[idx][:, r0:r1, :]).transpose(1, 2, 0)
            out[r0:r1] = blk
        return out

    def pixels(self, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
        """(n, bands) full spectra at the given pixels."""
        rr = np.asarray(rr, int)
        cc = np.asarray(cc, int)
        if self.interleave == "bil":
            return np.asarray(self.mm[rr, :, cc])
        if self.interleave == "bip":
            return np.asarray(self.mm[rr, cc, :])
        return np.asarray(self.mm[:, rr, cc]).T


# --------------------------------------------------------------- netCDF cubes

class NcCube:
    """h5py view of an EMIT LP DAAC netCDF dataset, same accessors as EnviCube."""

    def __init__(self, path, dataset: str, wl_key: Optional[str] = None,
                 fwhm_key: Optional[str] = None, names_key: Optional[str] = None):
        if h5py is None:
            raise ImportError("h5py is needed to read EMIT netCDF inputs")
        self.path = str(path)
        self.dataset = dataset
        with h5py.File(self.path, "r") as f:
            ds = f[dataset]
            shp = ds.shape
            self.rows, self.cols = int(shp[0]), int(shp[1])
            self.nb = int(shp[2]) if len(shp) == 3 else 1
            self.wl_nm = _to_nm(f[wl_key][:]) if wl_key and wl_key in f else np.array([])
            self.fwhm_nm = np.asarray(f[fwhm_key][:], float) if fwhm_key and fwhm_key in f else np.array([])
            self.band_names = ([s.decode() if isinstance(s, bytes) else str(s) for s in f[names_key][:]]
                               if names_key and names_key in f else [])
            self.attrs = {k: f.attrs[k] for k in f.attrs}

    @property
    def shape(self):
        return self.rows, self.cols, self.nb

    def band(self, i: int, step: int = 1) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            ds = f[self.dataset]
            return np.asarray(ds[::step, ::step, int(i)] if ds.ndim == 3 else ds[::step, ::step])

    def bands(self, idx: Sequence[int], row_block: int = 256) -> np.ndarray:
        idx = [int(i) for i in idx]
        uniq = sorted(set(idx))
        out = np.empty((self.rows, self.cols, len(idx)), dtype=np.float32)
        with h5py.File(self.path, "r") as f:
            ds = f[self.dataset]
            for r0 in range(0, self.rows, row_block):
                r1 = min(r0 + row_block, self.rows)
                blk = np.asarray(ds[r0:r1, :, uniq])
                out[r0:r1] = blk[:, :, [uniq.index(i) for i in idx]]
        return out

    def pixels(self, rr, cc) -> np.ndarray:
        rr = np.asarray(rr, int)
        cc = np.asarray(cc, int)
        out = np.empty((rr.size, self.nb), dtype=np.float32)
        with h5py.File(self.path, "r") as f:
            ds = f[self.dataset]
            for r in np.unique(rr):
                sel = np.flatnonzero(rr == r)
                row = np.asarray(ds[int(r)])
                out[sel] = row[cc[sel]] if row.ndim == 2 else row[cc[sel], None]
        return out


class _NcLocCube:
    """LOC accessor over the 'location' group of an EMIT RAD netCDF (lon, lat, elev)."""

    band_names = ["Longitude", "Latitude", "Elevation"]

    def __init__(self, path):
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            self.rows, self.cols = f["location/lat"].shape
        self.nb = 3

    def band(self, i, step=1):
        key = ("location/lon", "location/lat", "location/elev")[int(i)]
        with h5py.File(self.path, "r") as f:
            return np.asarray(f[key][::step, ::step])

    def pixels(self, rr, cc):
        rr, cc = np.asarray(rr, int), np.asarray(cc, int)
        return np.stack([self.band(i)[rr, cc] for i in range(3)], axis=1)


# -------------------------------------------------------------- scene lookup

@dataclass
class EmitScene:
    fid: str
    fmt: str                         # "envi" or "nc"
    files: Dict[str, Path] = field(default_factory=dict)

    @property
    def datetime(self) -> Optional[datetime]:
        return fid_datetime(self.fid)

    def has(self, kind: str) -> bool:
        return kind in self.files

    # ---- cube accessors
    def radiance(self):
        if self.fmt == "envi":
            return EnviCube(self.files["rdn"])
        return NcCube(self.files["rdn"], "radiance",
                      "sensor_band_parameters/wavelengths", "sensor_band_parameters/fwhm")

    def obs(self):
        if self.fmt == "envi":
            return EnviCube(self.files["obs"])
        return NcCube(self.files["obs"], "obs", names_key="sensor_band_parameters/observation_bands")

    def loc(self):
        if self.fmt == "envi":
            return EnviCube(self.files["loc"])
        return _NcLocCube(self.files["loc"])

    def bandmask(self):
        if "bandmask" not in self.files:
            return None
        return EnviCube(self.files["bandmask"])


def _envi_kind(name: str) -> Optional[str]:
    m = _ENVI_KIND_RE.search(name)
    if m:
        return m.group(1).lower()
    n = name.upper()                 # looser fallback, as in the night-scan notebook
    if "BANDMASK" in n:
        return "bandmask"
    if "GLT" in n:
        return "glt"
    if "OBS" in n:
        return "obs"
    if "_LOC" in n or n.startswith("LOC"):
        return "loc"
    if ("RAD" in n or "RDN" in n) and not any(s in n for s in ("UNC", "MASK", "RFL")):
        return "rdn"
    return None


def _scan_dir(d: Path) -> Optional[EmitScene]:
    try:
        files = sorted(p for p in d.iterdir() if p.is_file())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None
    envi_, nc_ = {}, {}
    for p in files:
        suf = p.suffix.lower()
        if suf in (".nc", ".h5"):
            n = p.name.upper()
            if "L1B_RAD" in n:
                nc_.setdefault("rdn", p)
                nc_.setdefault("loc", p)        # location group lives in the RAD file
                nc_.setdefault("glt", p)
            elif "L1B_OBS" in n:
                nc_.setdefault("obs", p)
            continue
        if suf == ".hdr":
            continue
        k = _envi_kind(p.name)
        if k is not None and _hdr_for(p) is not None:
            envi_.setdefault(k, p)
    fid = None
    for p in files:
        fid = fid_from_path(p)
        if fid:
            break
    if fid is None:
        fid = fid_from_path(d) or d.name
    if all(k in envi_ for k in REQUIRED_KINDS):
        return EmitScene(fid, "envi", envi_)
    if all(k in nc_ for k in ("rdn", "obs")) and h5py is not None:
        return EmitScene(fid, "nc", nc_)
    return None


def find_emit_scene(path, acq_root: Optional[str] = None) -> EmitScene:
    """Resolve a scene from a scene dir, its l1b/ dir, a file inside it, or a bare fid.

    A bare fid (``emit20260925t075553``) is looked up under
    ``<acq_root>/<YYYYMMDD>/<fid>``.
    """
    p = Path(str(path)).expanduser()
    if not p.exists() and FID_RE.fullmatch(str(path).strip().lower() or ""):
        if not acq_root:
            raise FileNotFoundError(f"{path} looks like a fid; pass --acq_root to resolve it")
        fid = str(path).strip().lower()
        p = Path(acq_root) / fid[4:12] / fid
    if p.is_file():
        p = p.parent
    for d in (p / "l1b", p):
        sc = _scan_dir(d)
        if sc:
            return sc
    if p.is_dir():
        for d in sorted(x for x in p.iterdir() if x.is_dir() and x.name != "l1b"):
            sc = _scan_dir(d)
            if sc:
                return sc
    raise FileNotFoundError(f"no EMIT L1B RDN+OBS+LOC set found under {p}")


def list_scene_dirs(acq_root, start, end) -> List[Path]:
    """Scene folders between two dates (inclusive) in the ops acquisitions tree."""
    from datetime import timedelta
    root = Path(acq_root)
    out, d = [], start
    while d <= end:
        day = root / d.strftime("%Y%m%d")
        if day.is_dir():
            out += sorted(s for s in day.iterdir() if s.is_dir() and s.name.startswith("emit"))
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------- band helpers

def nearest_band(wl_nm: np.ndarray, target_nm: float) -> Tuple[int, float]:
    wl = np.asarray(wl_nm, dtype=np.float64)
    i = int(np.argmin(np.abs(wl - float(target_nm))))
    return i, float(wl[i])


def obs_band_index(obs_cube, key: str) -> int:
    names = [n.lower() for n in getattr(obs_cube, "band_names", []) or []]
    want = _OBS_NAME_KEYS.get(key)
    if names and want:
        hit = [i for i, n in enumerate(names) if want in n]
        if hit:
            return hit[0]
    return OBS_INDEX_DEFAULT[key]


def loc_band_index(loc_cube, key: str) -> int:
    names = [n.lower() for n in getattr(loc_cube, "band_names", []) or []]
    for i, n in enumerate(names):
        if key in n:
            return i
    return {"lon": 0, "lat": 1, "elev": 2}[key]


def scene_median_sza(scene: EmitScene, step: int = 8) -> float:
    ob = scene.obs()
    a = _clean(ob.band(obs_band_index(ob, "sun_zen"), step=step))
    return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")


def read_glt(scene: EmitScene):
    """(glt_samples, glt_lines, geotransform, wkt, nodata) or None.

    ENVI GLTs go through the airborne reader (rasterio + header fallbacks); the
    netCDF GLT lives in the RAD file's location group with a root geotransform.
    """
    if "glt" not in scene.files:
        return None
    if scene.fmt == "envi":
        from .airborne import _open_glt   # lazy: airborne imports jax
        hdr = _hdr_for(Path(scene.files["glt"]))
        return _open_glt(str(hdr))
    with h5py.File(scene.files["glt"], "r") as f:
        if "location/glt_x" not in f:
            return None
        gx = np.asarray(f["location/glt_x"][:], np.int64)
        gy = np.asarray(f["location/glt_y"][:], np.int64)
        gt = tuple(float(x) for x in np.asarray(f.attrs.get("geotransform")))
        wkt = f.attrs.get("spatial_ref", "")
        wkt = wkt.decode() if isinstance(wkt, bytes) else str(wkt)
    return gx, gy, gt, wkt, 0.0
