#!/usr/bin/env python
"""Day-time EMIT fire and flare scan: the day counterpart of the night-scan notebook.

Walks every EMIT scene in a date window, keeps day scenes (median SZA <= --day_sza_max),
and flags hot pixels with HFDI = (L2430 - L2061)/(L2430 + L2061) > --hfdi_min, OR a
scene-robust HFDI z-score > --hfdi_z (small flares over bright ground keep a negative
HFDI but stand far above the scene). Blobs get their size, peak HFDI / z, peak L2430,
an SNR against a ring, and lat/lon.

Optional seeding: --seed_csv night_detections.csv tags each day blob with the distance
to the nearest night detection within --seed_days, so day overpasses of fires already
confirmed at night (and the Permian flares) sort to the top.

Outputs in --out_dir: day_scene_summary.csv, day_detections.csv. Per-scene results are
cached by settings, so reruns only process new scenes.

    python scripts/emit_day_fire_scan.py --start 2026-08-01 --end 2026-11-04 \
        --out_dir day_fire_scan --seed_csv night_fire_scan/night_detections.csv
"""
import argparse, hashlib, json, os, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from isofire.emit_io import (_clean, find_emit_scene, fid_datetime, list_scene_dirs, loc_band_index,
                             nearest_band, scene_median_sza)

PERMIAN = (-104.8, -100.5, 30.3, 34.0)
A = None   # args, set in main (inherited by forked workers)


def robust_z(v):
    f = np.isfinite(v)
    if f.sum() < 100:
        return np.full(v.shape, np.nan, np.float32)
    med = np.nanmedian(v[f])
    mad = 1.4826 * np.nanmedian(np.abs(v[f] - med))
    return ((v - med) / max(mad, 1e-6)).astype(np.float32)


def ring_snr(band, rr, cc, ring=20, guard=3):
    r0, r1 = max(rr.min() - ring, 0), min(rr.max() + ring + 1, band.shape[0])
    c0, c1 = max(cc.min() - ring, 0), min(cc.max() + ring + 1, band.shape[1])
    box = band[r0:r1, c0:c1].copy()
    box[max(rr.min() - guard - r0, 0):rr.max() + guard + 1 - r0,
        max(cc.min() - guard - c0, 0):cc.max() + guard + 1 - c0] = np.nan
    if not np.isfinite(box).any():
        return float("nan")
    return float((np.nanmax(band[rr, cc]) - np.nanmedian(box)) / max(np.nanstd(box), 1e-9))


def analyse(path):
    path = Path(path)
    rec = dict(fid=path.name, path=str(path), utc=None, status="", sza=None, n_px=0, n_blobs=0, blobs=[])
    try:
        sc = find_emit_scene(path)
    except FileNotFoundError:
        rec["status"] = "missing L1B"
        return rec
    rec["fid"] = sc.fid
    dt = fid_datetime(sc.fid)
    rec["utc"] = dt.isoformat() if dt else None
    rec["sza"] = scene_median_sza(sc)
    if not np.isfinite(rec["sza"]):
        rec["status"] = "no SZA"
        return rec
    if rec["sza"] > A.day_sza_max:
        rec["status"] = "night/twilight"
        return rec
    rec["status"] = "day"
    rad = sc.radiance()
    lo, lwl = nearest_band(rad.wl_nm, A.hfdi_lo_nm)
    hi, hwl = nearest_band(rad.wl_nm, A.hfdi_hi_nm)
    B = _clean(rad.bands([lo, hi]))
    with np.errstate(invalid="ignore", divide="ignore"):
        hfdi = (B[..., 1] - B[..., 0]) / (B[..., 1] + B[..., 0])
    z = robust_z(hfdi)
    hot = np.isfinite(hfdi) & ((hfdi > A.hfdi_min) | (np.isfinite(z) & (z > A.hfdi_z)))
    rec["n_px"] = int(hot.sum())
    if not hot.any():
        return rec
    lab, _ = ndimage.label(hot, structure=np.ones((3, 3), int))
    loc = sc.loc()
    lat = _clean(loc.band(loc_band_index(loc, "lat")))
    lon = _clean(loc.band(loc_band_index(loc, "lon")))
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        r_, c_ = np.nonzero(lab[sl] == i)
        rr, cc = r_ + sl[0].start, c_ + sl[1].start
        if rr.size < A.min_blob_px:
            continue
        k = int(np.nanargmax(B[rr, cc, 1]))
        rec["blobs"].append(dict(n_px=int(rr.size), peak_rc=[int(rr[k]), int(cc[k])],
                                 hfdi_max=float(np.nanmax(hfdi[rr, cc])), z_max=float(np.nanmax(z[rr, cc])),
                                 peak_L2430=float(B[rr[k], cc[k], 1]), snr=ring_snr(B[..., 1], rr, cc),
                                 lat=float(np.nanmean(lat[rr, cc])), lon=float(np.nanmean(lon[rr, cc]))))
    rec["blobs"].sort(key=lambda b: -(b["snr"] if np.isfinite(b["snr"]) else -1))
    rec["blobs"] = rec["blobs"][:A.max_blobs]
    rec["n_blobs"] = len(rec["blobs"])
    return rec


def safe(path):
    try:
        return analyse(path)
    except Exception as err:
        return dict(fid=Path(path).name, path=str(path), status=f"error: {type(err).__name__}: {err}",
                    sza=None, n_px=0, n_blobs=0, blobs=[])


def main():
    global A
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    p.add_argument("--acq_root", default="/store/emit/ops/data/acquisitions")
    p.add_argument("--out_dir", default="day_fire_scan")
    p.add_argument("--day_sza_max", type=float, default=75.0)
    p.add_argument("--hfdi_lo_nm", type=float, default=2061.0)
    p.add_argument("--hfdi_hi_nm", type=float, default=2430.0)
    p.add_argument("--hfdi_min", type=float, default=0.0)
    p.add_argument("--hfdi_z", type=float, default=8.0)
    p.add_argument("--min_blob_px", type=int, default=1)
    p.add_argument("--max_blobs", type=int, default=500, help="per scene, highest SNR kept")
    p.add_argument("--min_snr", type=float, default=5.0)
    p.add_argument("--seed_csv", default=None, help="night_detections.csv from the night scan")
    p.add_argument("--seed_days", type=float, default=3.0)
    p.add_argument("--seed_km", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    A = p.parse_args()

    out = Path(A.out_dir)
    cfg = {k: v for k, v in vars(A).items() if k not in ("start", "end", "out_dir", "workers", "seed_csv",
                                                          "seed_days", "seed_km", "acq_root")}
    cache = out / ("cache_" + hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8])
    cache.mkdir(parents=True, exist_ok=True)
    scenes = list_scene_dirs(A.acq_root, A.start, A.end)
    done, todo = {}, []
    for s in scenes:
        cf = cache / f"{s.name}.json"
        (done.__setitem__(s.name, json.loads(cf.read_text())) if cf.exists() else todo.append(s))
    print(f"{len(scenes)} scene folders, {len(done)} cached, {len(todo)} to scan with {A.workers} workers")
    t0 = time.time()

    def store(r, k):
        done[Path(r["path"]).name] = r
        if r["status"] in ("day", "night/twilight", "no SZA"):
            (cache / f"{Path(r['path']).name}.json").write_text(json.dumps(r))
        if r["status"].startswith("error"):
            print("  ERROR", r["fid"], r["status"])
        elif r["n_blobs"]:
            print(f"  {r['fid']}  SZA {r['sza']:5.1f}  blobs {r['n_blobs']:4d}")
        if k % 1000 == 0:
            print(f"  ...{k}/{len(todo)} {(time.time() - t0) / 60:.1f} min")

    if A.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(A.workers, mp_context=mp.get_context("fork")) as ex:
            for k, f in enumerate(as_completed([ex.submit(safe, str(s)) for s in todo]), 1):
                store(f.result(), k)
    else:
        for k, s in enumerate(todo, 1):
            store(safe(str(s)), k)

    res = [done[s.name] for s in scenes if s.name in done]
    sdf = pd.DataFrame([{k: v for k, v in r.items() if k != "blobs"} for r in res])
    det = pd.DataFrame([dict(fid=r["fid"], utc=r["utc"], sza=r["sza"], **b) for r in res for b in r["blobs"]])
    if len(det):
        det["strong"] = det["snr"] >= A.min_snr
        det["permian"] = det["lon"].between(PERMIAN[0], PERMIAN[1]) & det["lat"].between(PERMIAN[2], PERMIAN[3])
        det["seed_km"] = np.nan
        if A.seed_csv:
            sd = pd.read_csv(A.seed_csv).dropna(subset=["lat", "lon"])
            st = pd.to_datetime(sd["utc"]).values
            dt_ = pd.to_datetime(det["utc"]).values
            for i in range(len(det)):
                near = np.abs((st - dt_[i]) / np.timedelta64(1, "D")) <= A.seed_days
                if near.any():
                    dlat = np.radians(sd["lat"].values[near] - det["lat"].iat[i])
                    dlon = np.radians(sd["lon"].values[near] - det["lon"].iat[i])
                    a_ = np.sin(dlat / 2) ** 2 + np.cos(np.radians(det["lat"].iat[i])) * \
                        np.cos(np.radians(sd["lat"].values[near])) * np.sin(dlon / 2) ** 2
                    det.iat[i, det.columns.get_loc("seed_km")] = float(6371 * 2 * np.arcsin(np.sqrt(a_)).min())
        det["seeded"] = det["seed_km"] <= A.seed_km
    sdf.to_csv(out / "day_scene_summary.csv", index=False)
    det.to_csv(out / "day_detections.csv", index=False)
    print(sdf["status"].value_counts().to_string())
    if len(det):
        top = (det[det["strong"]].groupby("fid")
               .agg(utc=("utc", "first"), lat=("lat", "mean"), lon=("lon", "mean"), n=("n_px", "size"),
                    max_snr=("snr", "max"), seeded=("seeded", "sum"), permian=("permian", "sum"))
               .sort_values(["seeded", "max_snr"], ascending=False))
        print(f"\n{len(top)} day scenes with a strong detection; top 40 (seeded by night detections first):")
        print(top.head(40).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"saved {out / 'day_scene_summary.csv'} and {out / 'day_detections.csv'}")


if __name__ == "__main__":
    main()
