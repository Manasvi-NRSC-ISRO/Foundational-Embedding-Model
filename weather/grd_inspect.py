import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
r"""Inspect the IMD weather folder (daily .grd files) and write a report to send to Claude.

    python grd_inspect.py --root D:\Manasvi\Embedding-Foundational-Model\GRD_DATA
    python grd_inspect.py --root D:\GRD_DATA --aoi boundaries\murshidabad.geojson

Expected layout:  <root>\RAIN\...\*.grd   <root>\TMAX\...\*.grd   <root>\TMIN\...\*.grd   (any sub-folders)
Writes outputs\grd_inspect.txt  (change folder with --out). Only reads files; changes nothing.

For each variable it prints: number of files, first/last date, missing days per year, file sizes (is the extra
header value there?), nodata share, a small block of values around Murshidabad (orientation check), monthly values at
the Murshidabad cell (rain total, Tmax/Tmin mean: a sanity check that the grid is not flipped), and how many grid
cells lie inside the district polygon (cell centre inside / cell touched).
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import re

import numpy as np

CONFIG = {
    "RAIN": dict(ncols=135, nrows=129, xmin=66.5, ymin=6.5, res=0.25, nodata=-999.0),
    "TMAX": dict(ncols=61, nrows=61, xmin=67.5, ymin=7.5, res=0.5, nodata=99.9),
    "TMIN": dict(ncols=61, nrows=61, xmin=67.5, ymin=7.5, res=0.5, nodata=99.9),
}
LINES = []


def say(s=""):
    print(s)
    LINES.append(str(s))


def get_date(path, var):
    stem = os.path.splitext(os.path.basename(path))[0]
    if var == "RAIN":
        m = re.search(r"rain_ind0\.25_(\d{2})_(\d{2})_(\d{2})", stem, re.I)
        if m:
            yy, mm, dd = map(int, m.groups())
            return dt.date(2000 + yy, mm, dd)
    else:
        m = re.search(r"(?:max|min)(\d{2})(\d{2})(\d{4})", stem, re.I)
        if m:
            dd, mm, yyyy = map(int, m.groups())
            return dt.date(yyyy, mm, dd)
    return None


def read_grd(path, cfg):
    """Raw array exactly as stored: row 0 = southernmost (lat = ymin + row*res), col 0 = westernmost."""
    v = np.fromfile(path, dtype="<f4")
    n = cfg["ncols"] * cfg["nrows"]
    extra = None
    if v.size == n + 1:
        extra, v = float(v[0]), v[1:]
    if v.size != n:
        return None, extra, v.size
    a = v.reshape(cfg["nrows"], cfg["ncols"]).astype(np.float64)
    a[np.abs(a - cfg["nodata"]) < 0.01] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a, extra, v.size


def polygon_rings(path):
    gj = json.load(open(path, encoding="utf-8"))
    feats = gj["features"] if "features" in gj else [gj]
    rings = []
    for f in feats:
        g = f.get("geometry", f)
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            rings.append(np.array(p[0], float)[:, :2])
    return rings


def inside(rings, lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    out = np.zeros(lon.shape, bool)
    for r in rings:
        x, y = r[:, 0], r[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        c = np.zeros(lon.shape, bool)
        for xa, ya, xb, yb in zip(x, y, x2, y2):
            if ya == yb:
                continue
            cond = ((ya > lat) != (yb > lat)) & (lon < (xb - xa) * (lat - ya) / (yb - ya) + xa)
            c ^= cond
        out |= c
    return out


def ranges(dates):
    out, start, prev = [], None, None
    for d in dates:
        if start is None:
            start = prev = d
        elif (d - prev).days == 1:
            prev = d
        else:
            out.append((start, prev))
            start = prev = d
    if start:
        out.append((start, prev))
    return ["%s" % a if a == b else "%s..%s" % (a, b) for a, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains RAIN, TMAX, TMIN")
    ap.add_argument("--aoi", default="", help="murshidabad.geojson (lon/lat); default: searched automatically")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    aoi = a.aoi
    if not aoi:
        for p in ["murshidabad.geojson", os.path.join("boundaries", "murshidabad.geojson")] + \
                 glob.glob(os.path.join("**", "murshidabad.geojson"), recursive=True):
            if os.path.exists(p):
                aoi = p
                break
    rings = polygon_rings(aoi) if aoi else None
    if rings:
        allp = np.concatenate(rings)
        lon0, lat0 = allp[:, 0].mean(), allp[:, 1].mean()
        say("district polygon: %s | lon %.3f..%.3f, lat %.3f..%.3f (centre %.3f, %.3f)" % (
            aoi, allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max(), lon0, lat0))
        if allp[:, 0].max() > 180 or allp[:, 1].max() > 90:
            say("!! polygon does not look like lon/lat degrees: give --aoi in EPSG:4326")
    else:
        lon0, lat0 = 88.2, 24.2
        say("no polygon found: using the point lon 88.2, lat 24.2 as Murshidabad")
    for var, cfg in CONFIG.items():
        say("")
        say("=" * 72)
        say("%s   grid %d x %d, cell %.2f deg, lon from %.2f, lat from %.2f (cell centres)" % (
            var, cfg["ncols"], cfg["nrows"], cfg["res"], cfg["xmin"], cfg["ymin"]))
        files = sorted(glob.glob(os.path.join(a.root, var, "**", "*.grd"), recursive=True))
        if not files:
            say("  no .grd files under %s" % os.path.join(a.root, var))
            continue
        dated, bad = {}, []
        for f in files:
            d = get_date(f, var)
            (dated.__setitem__(d, f) if d else bad.append(f))
        days = sorted(dated)
        say("files: %d | date parsed: %d | name not understood: %d" % (len(files), len(dated), len(bad)))
        if bad:
            say("  e.g. %s" % ", ".join(os.path.basename(b) for b in bad[:3]))
        if not days:
            continue
        say("dates: %s .. %s" % (days[0], days[-1]))
        yrs = collections.Counter(d.year for d in days)
        for y in sorted(yrs):
            full = [dt.date(y, 1, 1) + dt.timedelta(k) for k in range(366 if y % 4 == 0 else 365)]
            last = min(full[-1], days[-1]) if y == days[-1].year else full[-1]
            miss = [d for d in full if d <= last and d not in dated]
            say("  %d: %d days present, %d missing%s" % (y, yrs[y], len(miss),
                (" -> " + ", ".join(ranges(miss)[:12])) if miss else ""))
        sizes = collections.Counter(os.path.getsize(f) for f in files)
        exp = 4 * cfg["ncols"] * cfg["nrows"]
        say("file sizes (bytes): %s | expected %d (%d with 1 extra value)" % (dict(sizes.most_common(4)), exp, exp + 4))
        # grid cells vs polygon
        iy = lambda lat: np.rint((lat - cfg["ymin"]) / cfg["res"]).astype(int)
        ix = lambda lon: np.rint((lon - cfg["xmin"]) / cfg["res"]).astype(int)
        cell_lon = cfg["xmin"] + np.arange(cfg["ncols"]) * cfg["res"]
        cell_lat = cfg["ymin"] + np.arange(cfg["nrows"]) * cfg["res"]
        ci, cj = ix(lon0), iy(lat0)
        say("Murshidabad centre falls in cell col %d, row %d (cell centre lon %.3f, lat %.3f)" % (
            ci, cj, cell_lon[ci], cell_lat[cj]))
        centre_cells, touch_cells = set(), set()
        if rings:
            LON, LAT = np.meshgrid(cell_lon, cell_lat)
            near = (np.abs(LON - lon0) < 2) & (np.abs(LAT - lat0) < 2)
            ins = inside(rings, LON, LAT) & near
            centre_cells = set(zip(*np.nonzero(ins)))
            gx, gy = np.meshgrid(np.arange(allp[:, 0].min(), allp[:, 0].max(), 0.01),
                                 np.arange(allp[:, 1].min(), allp[:, 1].max(), 0.01))
            m = inside(rings, gx, gy)
            touch_cells = set(zip(iy(gy[m]), ix(gx[m])))
            say("grid cells whose CENTRE is inside the district: %d | cells that TOUCH the district: %d "
                "(nearest-neighbour sampling uses the touched cells)" % (len(centre_cells), len(touch_cells)))
        # sample file
        mid = days[len(days) // 2]
        arr, extra, nval = read_grd(dated[mid], cfg)
        say("sample file %s (%s)" % (os.path.basename(dated[mid]), mid))
        if arr is None:
            say("  !! unexpected number of values: %d" % nval)
            continue
        say("  extra first value in file: %s | valid cells %.1f %% | min %.2f mean %.2f max %.2f" % (
            extra, 100 * np.isfinite(arr).mean(), np.nanmin(arr), np.nanmean(arr), np.nanmax(arr)))
        if touch_cells:
            vals = [arr[r, c] for r, c in touch_cells]
            nn = sum(not np.isfinite(v) for v in vals)
            say("  cells touching the district with NODATA in this file: %d of %d" % (nn, len(touch_cells)))
        r0, c0 = max(cj - 2, 0), max(ci - 2, 0)
        blk = arr[r0:cj + 3, c0:ci + 3][::-1]
        say("  values around Murshidabad (north at top, west at left; nan = nodata):")
        for row in blk:
            say("    " + " ".join("%7.2f" % v for v in row))
        # monthly sanity table for the latest full year
        yfull = [y for y in sorted(yrs) if yrs[y] >= 300]
        if yfull:
            y = yfull[-1]
            mon = collections.defaultdict(list)
            for d in days:
                if d.year == y:
                    aa, _, _ = read_grd(dated[d], cfg)
                    if aa is not None:
                        mon[d.month].append(aa[cj, ci])
            tag = "total mm" if var == "RAIN" else "mean deg C"
            say("monthly at the Murshidabad cell, %d (%s): %s" % (y, tag, "  ".join(
                "%02d:%.0f" % (m, (np.nansum(v) if var == "RAIN" else np.nanmean(v))) for m, v in sorted(mon.items()))))
            if var == "RAIN":
                say("  (Murshidabad usually gets ~1200-1500 mm a year, most in June-September)")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "grd_inspect.txt"), "w", encoding="utf-8").write("\n".join(LINES) + "\n")
    say("")
    say("saved %s" % os.path.join(a.out, "grd_inspect.txt"))


if __name__ == "__main__":
    main()

