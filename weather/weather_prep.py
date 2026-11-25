r"""Weather for the embedding model: daily rainfall, Tmax, Tmin from the IMD .grd folder -> one small file per season.

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python weather_prep.py --root D:\Manasvi\Embedding-Foundational-Model\GRD_DATA --periods 2025-04-01:2025-06-30

Writes into EMB_OUT (the season folder); <tag> = <first day>_to_<last day>, e.g. 2025-04-01_to_2025-06-30:
  weather_IMD_daily_values_<tag>.npz          dates, doy, rain[D, n_rain_cells], tmax[D, n_temp_cells], tmin[D, n_temp_cells]   (float32)
  weather_IMD_cell_map_<tag>.tif    on the 30 m grid of the stack, 2 bands (int16): 1 = rain cell id, 2 = temperature cell id,
                       -1 outside the Murshidabad polygon. A pixel's daily rain = rain[:, id].  (open it in QGIS)
  weather_IMD_cell_map_<tag>.csv    one row per cell used: id, variable, lon, lat, kind (centre inside / touching / filled)
  weather_IMD_cell_map_<tag>.png    picture of the cell ids

Rules (as agreed):
  * nearest neighbour: a 30 m pixel takes the value of the weather cell whose centre is nearest (no smoothing);
  * the Murshidabad polygon (the same mask as the stack, boundaries\murshidabad.geojson) decides which pixels get weather;
  * a cell with no data in the IMD grid (e.g. over Bangladesh, IMD only covers India) is replaced by the NEAREST
    VALID cell (not by 0: zero rain would mean "dry day"); a day that is still missing for a used cell
    is replaced by the mean of the other used cells of that day.
Values stay float32 in mm (rain) and degrees C (Tmax, Tmin): no scaling, no int16 limit.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: common.py
import argparse
import csv
import datetime as dt
import glob
import os

import numpy as np

import common as C
from grd_inspect import CONFIG, get_date, read_grd


def nearest_valid_fill(valid, lon_c, lat_c, rows, cols):
    """For each (row, col) cell that is not valid, the (row, col) of the nearest valid cell (degree distance)."""
    vr, vc = np.nonzero(valid)
    out = {}
    for r, c in zip(rows, cols):
        if valid[r, c]:
            out[(r, c)] = (r, c)
            continue
        d = (lon_c[vc] - lon_c[c]) ** 2 + (lat_c[vr] - lat_c[r]) ** 2
        k = int(np.argmin(d))
        out[(r, c)] = (int(vr[k]), int(vc[k]))
    return out


def day_list(periods):
    days = []
    for p in periods:
        a, b = p.split(":")
        d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        days += [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
    return sorted(set(days))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with RAIN, TMAX, TMIN")
    ap.add_argument("--periods", nargs="+", required=True, help="START:END ranges, same as stack_scenes.py")
    ap.add_argument("--block", type=int, default=256, help="rows per block")
    a = ap.parse_args()
    import rasterio
    from pyproj import Transformer
    g = C.get_grid()
    H, W, tr, pm = g["H"], g["W"], g["transform"], g["pmask"]
    days = day_list(a.periods)
    wtag = "%s_to_%s" % (days[0], days[-1])      # date range in every output file name
    os.makedirs(C.OUT, exist_ok=True)
    print("%d days: %s .. %s | output folder %s" % (len(days), days[0], days[-1], C.OUT))
    # 1. which weather cell does every 30 m pixel look at?
    tf = Transformer.from_crs(C.CRS, "EPSG:4326", always_xy=True)
    first = {}
    for var, cfg in CONFIG.items():
        files = {get_date(f, var): f for f in glob.glob(os.path.join(a.root, var, "**", "*.grd"), recursive=True)}
        missing = [d for d in days if d not in files]
        if missing:
            raise SystemExit("%s: %d days have no .grd file (first: %s)" % (var, len(missing), missing[0]))
        first[var] = files
    grids = {}      # var -> stack of daily arrays (D, nrows, ncols), NaN = nodata
    for var, cfg in CONFIG.items():
        arr = np.zeros((len(days), cfg["nrows"], cfg["ncols"]), np.float32)
        for k, d in enumerate(days):
            x, extra, n = read_grd(first[var][d], cfg)
            if x is None:
                raise SystemExit("%s %s: %d values, expected %d" % (var, d, n, cfg["ncols"] * cfg["nrows"]))
            arr[k] = x
        grids[var] = arr
        print("%s: read %d files, grid %d x %d" % (var, len(days), cfg["ncols"], cfg["nrows"]))
    kinds = {"rain": "RAIN", "temp": "TMAX"}      # TMAX and TMIN share the grid
    cellid = {k: np.full((H, W), -1, np.int32) for k in kinds}
    used = {k: {} for k in kinds}                 # (row, col) of the source grid -> compact id
    meta = {k: [] for k in kinds}
    # static validity: a cell is valid when it has data on every day
    valid = {k: np.isfinite(grids[v]).all(axis=0) for k, v in kinds.items()}
    for b0 in range(0, H, a.block):
        h = min(a.block, H - b0)
        jj, ii = np.meshgrid(np.arange(W), np.arange(b0, b0 + h))
        xs = tr.c + (jj + 0.5) * tr.a
        ys = tr.f + (ii + 0.5) * tr.e
        lon, lat = tf.transform(xs, ys)
        inside = pm[b0:b0 + h]
        for k, var in kinds.items():
            cfg = CONFIG[var]
            r = np.rint((lat - cfg["ymin"]) / cfg["res"]).astype(int)
            c = np.rint((lon - cfg["xmin"]) / cfg["res"]).astype(int)
            r, c = np.clip(r, 0, cfg["nrows"] - 1), np.clip(c, 0, cfg["ncols"] - 1)
            lon_c = cfg["xmin"] + np.arange(cfg["ncols"]) * cfg["res"]
            lat_c = cfg["ymin"] + np.arange(cfg["nrows"]) * cfg["res"]
            pairs = set(zip(r[inside].tolist(), c[inside].tolist()))
            bad = [(rr, cc) for rr, cc in pairs if not valid[k][rr, cc]]
            fill = nearest_valid_fill(valid[k], lon_c, lat_c, [p[0] for p in bad], [p[1] for p in bad]) if bad else {}
            for rr, cc in pairs:
                src = fill.get((rr, cc), (rr, cc))
                if src not in used[k]:
                    used[k][src] = len(used[k])
                    meta[k].append((len(used[k]) - 1, src, lon_c[src[1]], lat_c[src[0]],
                                    "filled from nearest valid cell" if (rr, cc) in fill else "own cell"))
            lut = {}
            for rr, cc in pairs:
                lut[(rr, cc)] = used[k][fill.get((rr, cc), (rr, cc))]
            blk = np.full((h, W), -1, np.int32)
            ri, cj = np.nonzero(inside)
            blk[ri, cj] = [lut[(r[x, y], c[x, y])] for x, y in zip(ri, cj)]
            cellid[k][b0:b0 + h] = blk
        print("  rows %d/%d" % (b0 + h, H), end="\r", flush=True)
    print()
    # 2. daily values of the used cells
    out = {}
    for k, vars_ in (("rain", ["RAIN"]), ("temp", ["TMAX", "TMIN"])):
        order = sorted(used[k].items(), key=lambda t: t[1])
        rr = np.array([s[0] for s, _ in order])
        cc = np.array([s[1] for s, _ in order])
        for var in vars_:
            v = grids[var][:, rr, cc].astype(np.float32)
            nanm = ~np.isfinite(v)
            if nanm.any():
                dm = np.nanmean(np.where(nanm, np.nan, v), axis=1, keepdims=True)
                v = np.where(nanm, np.broadcast_to(dm, v.shape), v).astype(np.float32)
                print("  %s: %d values were missing on single days and got that day's mean of the other cells" % (var, nanm.sum()))
            out[var.lower()] = v
    doy = np.array([d.timetuple().tm_yday for d in days], np.int16)
    date_int = np.array([int(d.strftime("%Y%m%d")) for d in days], np.int32)
    np.savez_compressed(os.path.join(C.OUT, "weather_IMD_daily_values_%s.npz" % wtag), dates=date_int, doy=doy, **out)
    with rasterio.open(os.path.join(C.OUT, "weather_IMD_cell_map_%s.tif" % wtag), "w", driver="GTiff", height=H, width=W, count=2,
                       dtype="int16", crs=C.CRS, transform=tr, nodata=-1, compress="deflate") as dst:
        dst.write(cellid["rain"].astype(np.int16), 1)
        dst.write(cellid["temp"].astype(np.int16), 2)
        dst.set_band_description(1, "rain cell id")
        dst.set_band_description(2, "temperature cell id")
    with open(os.path.join(C.OUT, "weather_IMD_cell_map_%s.csv" % wtag), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variable", "id", "grid_row", "grid_col", "lon", "lat", "kind", "pixels"])
        for k in kinds:
            cnt = np.bincount(cellid[k][cellid[k] >= 0])
            for i, src, lo, la, kind in meta[k]:
                w.writerow(["rain" if k == "rain" else "temperature", i, src[0], src[1], lo, la, kind, int(cnt[i])])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for x, (k, t) in zip(ax, (("rain", "rain cells (0.25 deg)"), ("temp", "temperature cells (0.5 deg)"))):
            im = np.ma.masked_less(cellid[k][::6, ::6], 0)
            x.imshow(im, cmap="tab20", interpolation="nearest")
            x.set_title("%s: %d cells" % (t, len(used[k])))
            x.axis("off")
        fig.savefig(os.path.join(C.OUT, "weather_IMD_cell_map_%s.png" % wtag), dpi=100, bbox_inches="tight")
    except Exception as e:
        print("picture skipped:", e)
    print("")
    print("RAIN: %d cells used (%d filled from a neighbour) | TEMPERATURE: %d cells (%d filled)" % (
        len(used["rain"]), sum(m[4] != "own cell" for m in meta["rain"]),
        len(used["temp"]), sum(m[4] != "own cell" for m in meta["temp"])))
    mon = {}
    for k_, d in enumerate(days):
        mon.setdefault(d.month, []).append(k_)
    for m, ks in sorted(mon.items()):
        print("  month %02d  rain total (district mean) %6.0f mm | Tmax mean %.1f C | Tmin mean %.1f C" % (
            m, out["rain"][ks].sum(0).mean(), out["tmax"][ks].mean(), out["tmin"][ks].mean()))
    print("saved weather_IMD_daily_values_%s.npz and weather_IMD_cell_map_%s.tif/.csv/.png in %s" % (wtag, wtag, C.OUT))


if __name__ == "__main__":
    main()