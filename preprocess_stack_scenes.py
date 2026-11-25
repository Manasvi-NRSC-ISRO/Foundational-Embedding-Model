"""Stack EVERY scene of EVERY sensor into ONE GeoTIFF (one band per scene band), clipped to the district polygon.

This is step 1 of the pipeline. It reads the raw Bhoonidhi zips (bhoonidhi_year/...) directly, reprojects every
scene onto the common 30 m grid (see common.get_grid) and writes them all, one after another, as bands of one file.

Band order: all AWiFS scenes (4 bands each, by date), then all LISS-III scenes (4 bands each),
then all EOS-04 scenes (HH, HV). Total bands = 4*awifs + 4*liss3 + 2*eos04 scenes.

    python preprocess_stack_scenes.py                                  # every zip in bhoonidhi_year
    python preprocess_stack_scenes.py --periods 2025-04-01:2025-06-30  # ONE SEASON only (several START:END pairs allowed)
    python preprocess_stack_scenes.py --limit 2                        # test: first 2 scenes -> satellite_stack_<tag>_test.tif

One folder per season:  PowerShell  $env:EMB_OUT = "outputs\2025_Apr_Jun"   (default: outputs).
Everything (stack, masks, samples, models, embeddings) then goes into that folder.

Optional scene cache (--cache): every scene is also stored once as scene_cache/<sensor>_<date>_<scene>.tif (~40 MB
instead of ~330 MB per zip) and later runs read the cache instead of the zip. Zips are NEVER deleted by this script.

Every band has a description  sensor|date|band|scene  and tags (sensor, date, doy, scene, band, unit, orbit).
Sidecar tables: <out>/satellite_stack_<tag>_bands.csv (per band), <out>/satellite_scene_table_<tag>.csv (per scene, % of district covered).
Data type int16, nodata -32768.  Optical: DN (reflectance = DN * 0.0001).  SAR: gamma0 dB * 100.

The cloud mask is NOT applied here; it is a separate step (cloudmask_scenes.py) so the stack stays untouched.
"""
import argparse
import csv
import glob
import os
import time
import zipfile
from collections import defaultdict

import numpy as np

import common as C

# ----------------------------------------------------------------------------------------------
# PART 1 - READING ONE SCENE OUT OF A ZIP (formerly in preprocess.py)
# ----------------------------------------------------------------------------------------------


def sar_power(dn, k=C.SAR_K_DB):
    """SAR digital numbers (DN) -> linear gamma0 power.

    gamma0[dB] = 10*log10(DN^2) - K   =>   linear power = DN^2 * 10^(-K/10).
    We average scenes in LINEAR power (averaging dB values would be statistically wrong) and convert to dB later.
    """
    return (dn.astype(np.float32) ** 2) * np.float32(10.0 ** (-k / 10.0))


def find_member(zf, *needles, exclude=()):
    """Name of the first .tif inside the open zip `zf` whose UPPER-CASE name contains ALL `needles`
    and none of the `exclude` words. Returns None if there is no such file."""
    for n in zf.namelist():
        u = n.upper()
        if u.endswith((".TIF", ".TIFF")) and all(x in u for x in needles) \
                and not any(x in u for x in exclude):
            return n
    return None


def _warp(src_path, grid, resampling, dtype, src_nodata=None, nbands=1):
    """Reproject band(s) of a raster onto the common grid.

    src_path   : file to read (may be a /vsizip/ path pointing inside a zip)
    resampling : how to resample (bilinear for reflectance)
    src_nodata : source value to treat as nodata (it becomes 0 in the output)
    Returns an array (nbands, H, W) on the common grid; pixels with no source data are 0.
    """
    import rasterio
    from rasterio.warp import reproject
    dst = np.zeros((nbands, grid["H"], grid["W"]), dtype)
    with rasterio.open(src_path) as ds:
        for b in range(nbands):
            reproject(rasterio.band(ds, b + 1), dst[b], dst_transform=grid["transform"],
                      dst_crs=C.CRS, resampling=resampling, src_nodata=src_nodata,
                      dst_nodata=0, num_threads=2)
    return dst


def read_optical_scene(zpath, grid):
    """Read one LISS-III / AWiFS zip -> (4, H, W) float32 DN on the common grid (0 = nodata), or None on failure.

    The four bands are green, red, NIR, SWIR (C.OPT_BANDS). Bilinear resampling is used because
    these are continuous reflectance values (the sensor pixel size differs from 30 m).
    """
    from rasterio.enums import Resampling
    bands = []
    with zipfile.ZipFile(zpath) as zf:
        names = [find_member(zf, b) for b in C.OPT_BANDS]   # file name of each band inside the zip
    if None in names:
        print("   !! missing band in", os.path.basename(zpath), names)
        return None
    for n in names:
        # /vsizip/ lets GDAL read straight from inside the zip without unpacking it
        src = "/vsizip/%s/%s" % (os.path.abspath(zpath).replace("\\", "/"), n)
        bands.append(_warp(src, grid, Resampling.bilinear, np.float32, src_nodata=0)[0])
    return np.stack(bands)


def read_sar_scene(zpath, grid):
    """Read one EOS-04 SAR zip -> (power HH, weight HH, power HV, weight HV, pass flag), or None on failure.

    power  : linear gamma0 power averaged from the native 18 m pixels into each 30 m pixel
             (only the valid native pixels contribute; shadow/layover/nodata are zeroed first)
    weight : fraction of the 30 m pixel that was covered by valid native pixels (0..1). Dividing power by weight
             gives the mean power of the valid part; preprocess_stack_scenes keeps a pixel only if weight >= 0.5.
    flag   : "A" ascending or "D" descending orbit pass (read from the product metadata), "" if unknown.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    ap = os.path.abspath(zpath).replace("\\", "/")
    with zipfile.ZipFile(zpath) as zf:
        hh = find_member(zf, "HH", exclude=("MASK", "LIA", "AREA"))
        hv = find_member(zf, "HV", exclude=("MASK", "LIA", "AREA"))
        mk = find_member(zf, "MASK")
        meta = [n for n in zf.namelist() if n.endswith("BAND_META.txt")] or \
               [n for n in zf.namelist() if n.lower().endswith("product.xml")]
        txt = zf.read(meta[0]).decode("utf-8", "ignore").upper() if meta else ""
    if hh is None or hv is None:
        print("   !! HH/HV not found in", os.path.basename(zpath))
        return None
    mask_src = "/vsizip/%s/%s" % (ap, mk) if mk else None

    def prep(member):
        """Linear power with invalid pixels set to 0, at native resolution, plus a 0/1 valid map."""
        with rasterio.open("/vsizip/%s/%s" % (ap, member)) as ds:
            dn = ds.read(1)
            prof = (ds.crs, ds.transform)
        good = dn > 0                                                  # DN 0 = nodata
        if mask_src:                                                   # official mask: keep valid, drop shadow/layover
            with rasterio.open(mask_src) as dm:
                m = dm.read(1)
            if m.shape == dn.shape and (m & C.SAR_VALID_BIT).any():
                good &= ((m & C.SAR_VALID_BIT) > 0) & ((m & C.SAR_BAD_BITS) == 0)
        p = np.where(good, sar_power(dn), 0).astype(np.float32)
        return p, good.astype(np.float32), prof

    outs = []
    for member in (hh, hv):
        p, g, (crs, tr) = prep(member)
        res = []
        for arr in (p, g):                                             # average both the power and the valid map
            dst = np.zeros((grid["H"], grid["W"]), np.float32)
            reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=grid["transform"],
                      dst_crs=C.CRS, resampling=Resampling.average, num_threads=2)
            res.append(dst)
        outs.append(res)
    flag = "D" if "NODE=DESCENDING" in txt else ("A" if "NODE=ASCENDING" in txt else "")
    return outs[0][0], outs[0][1], outs[1][0], outs[1][1], flag


def zips_by_date(sensor, d0, d1):
    """Valid zips of one sensor, grouped by acquisition date.

    Returns {"YYYY-MM-DD": [zip paths]} sorted by date. d0 / d1 are optional "YYYY-MM-DD" limits (None = no limit).
    Files that are not real zip archives (for example half-downloaded) are skipped with a message.
    """
    by = defaultdict(list)
    for z in sorted(glob.glob(C.SENSORS[sensor]["pattern"])):
        d = C.scene_date(z)
        if d is None:
            continue
        s = d.isoformat()
        if (d0 and s < d0) or (d1 and s > d1):
            continue
        try:
            if not zipfile.is_zipfile(z):
                print("skip (not a valid zip):", z)
                continue
        except OSError:
            continue
        by[s].append(z)
    return dict(sorted(by.items()))


# ----------------------------------------------------------------------------------------------
# PART 2 - THE SCENE CACHE AND THE STACK
# ----------------------------------------------------------------------------------------------
ORDER = ["awifs", "liss3", "eos04"]    # order of the sensors inside the stack (AWiFS first, then LISS-III, then SAR)
CACHE = "scene_cache"                  # optional folder with one small tif per scene (see --cache)


def cache_path(s, d, scene):
    """File name of the cache entry of one scene: scene_cache/<sensor>_<date>_<scene>.tif"""
    return os.path.join(CACHE, "%s_%s_%s.tif" % (s, d, scene))


def write_cache(path, arr, s, d, scene, orbit, g):
    """Save one finished scene (bands, H, W int16) as a small compressed tif so later runs can skip the zip.

    It is written to a .tmp file first and renamed at the end, so an interrupted run never leaves a broken cache file.
    """
    import rasterio
    os.makedirs(CACHE, exist_ok=True)
    prof = dict(driver="GTiff", height=g["H"], width=g["W"], count=arr.shape[0], dtype="int16", crs=C.CRS,
                transform=g["transform"], nodata=-32768, compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    tmp = path + ".tmp"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        dst.update_tags(sensor=s, date=d, scene=scene, orbit=orbit, status="ok")
    os.replace(tmp, path)


def read_cache(path, nb, g):
    """Read a cache entry -> (array, orbit), or None if the file is missing, has the wrong band count or the wrong grid."""
    import rasterio
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as ds:
            if ds.count != nb or (ds.height, ds.width) != (g["H"], g["W"]):
                return None
            return ds.read(), ds.tags().get("orbit", "")
    except Exception:
        return None


def in_periods(d, periods):
    """True if the date string d ('YYYY-MM-DD') lies inside any (start, end) pair of `periods` (None = all dates)."""
    return periods is None or any(a <= d <= b for a, b in periods)


def collect_items(sensors, periods=None):
    """List of scenes to stack: [(sensor, date, scene, zip_path_or_None)], sorted by sensor order, date, scene.

    Scenes come from the zips AND from the cache folder (a cached scene whose zip was deleted still counts;
    its zip entry is then None). `periods` restricts the dates (see in_periods).
    """
    found = {}
    for s in sensors:
        for d, zips in zips_by_date(s, None, None).items():
            if not in_periods(d, periods):
                continue
            for z in sorted(zips):
                found[(s, d, os.path.splitext(os.path.basename(z))[0])] = z
    if os.path.isdir(CACHE):
        for f in os.listdir(CACHE):
            if f.endswith(".tif"):
                parts = f[:-4].split("_", 2)
                if len(parts) == 3 and parts[0] in sensors and in_periods(parts[1], periods):
                    found.setdefault((parts[0], parts[1], parts[2]), None)
    return [(s, d, sc, z) for (s, d, sc), z in sorted(found.items(),
            key=lambda kv: (sensors.index(kv[0][0]), kv[0][1], kv[0][2]))]


def main():
    """Command line entry point: find the scenes, build the stack band by band and write the two csv tables."""
    import rasterio
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (test)")
    ap.add_argument("--from", dest="d0", help="first date to include, YYYY-MM-DD (e.g. 2025-01-01)")
    ap.add_argument("--to", dest="d1", help="last date to include, YYYY-MM-DD (e.g. 2025-12-31)")
    ap.add_argument("--periods", nargs="+", metavar="START:END",
                    help="only these date ranges, e.g. 2024-04-01:2024-06-30 2025-04-01:2025-06-30 2025-11-01:2025-12-31")
    ap.add_argument("--cache", action="store_true", help="also keep a per-scene cache in scene_cache/ (uses disk)")
    ap.add_argument("--cache_only", action="store_true", help="only fill scene_cache/, do not write the big stack")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # --- decide output file, which sensors and which dates ---
    sensors = list(a.sensors)
    periods = [tuple(p.split(":")) for p in a.periods] if a.periods else None
    if a.d0 or a.d1:
        periods = (periods or []) + [(a.d0 or "0000-00-00", a.d1 or "9999-99-99")]
    # --- list the scenes that will go into the stack ---
    items = collect_items(sensors, periods)
    if a.limit:
        items = items[:a.limit]
    # Self-explanatory output names: the date range of the scenes is part of the file name (see common.py).
    #   satellite_stack_<first>_to_<last>.tif, ..._bands.csv, satellite_scene_table_<first>_to_<last>.csv
    tag = C.date_tag([it[1] for it in items]) if items else "empty"   # it[1] = scene date (see collect_items)
    out = a.out or os.path.join(C.OUT, "satellite_stack_%s%s.tif" % (tag, "_test" if a.limit else ""))
    # helpers: sensor kind ("optical"/"sar") and the band names of a scene of that sensor
    kind = lambda s: C.SENSORS[s]["kind"]
    names_of = lambda s: C.OPT_BANDS if kind(s) == "optical" else ["HH_dB", "HV_dB"]
    nb = sum(len(names_of(s)) for s, _, _, _ in items)   # total number of bands in the final stack
    g = C.get_grid()                           # common 30 m grid and district mask (pm)
    H, W, pm = g["H"], g["W"], g["pmask"]
    n_cached = sum(os.path.exists(cache_path(s, d, sc)) for s, d, sc, _ in items)
    print("%d scenes (%d already cached) -> %d bands, %d x %d px" % (len(items), n_cached, nb, W, H))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def get_scene(s, d, scene, z):
        """Get one scene on the common grid: (array (bands,H,W) int16, status text, orbit flag, source).

        Source order: the cache if it has the scene, else the zip (then optionally written to the cache).
        Optical : values are DN (reflectance*10000); a pixel is nodata (-32768) unless ALL four bands have data.
        SAR     : gamma0 in dB*100, only where at least half of the 30 m pixel was covered by valid SAR pixels.
        Everything outside the district polygon is set to nodata so the stack only holds district pixels.
        Errors never stop the run: the scene is filled with nodata and the status says what went wrong.
        """
        names = names_of(s)
        cp = cache_path(s, d, scene)
        r = read_cache(cp, len(names), g)
        if r is not None:
            return r[0], "ok", r[1], "cache"
        if z is None:
            return np.full((len(names), H, W), -32768, np.int16), "failed: no zip and no cache", "", "none"
        arr = np.full((len(names), H, W), -32768, np.int16)
        status, orbit = "ok", ""
        try:
            if kind(s) == "optical":
                sc = read_optical_scene(z, g)
                if sc is None:
                    status = "failed"
                else:
                    ok = (sc > 0).all(axis=0)              # a pixel needs data in all four bands
                    arr[:] = np.where(ok, np.minimum(np.rint(sc), 32767), -32768)
            else:
                r = read_sar_scene(z, g)
                if r is None:
                    status = "failed"
                else:
                    orbit = r[4]
                    for k, (p, gg) in enumerate(((r[0], r[1]), (r[2], r[3]))):
                        good = gg >= 0.5                       # keep pixels where >= 50 % was valid SAR data
                        # Mean linear power of the valid part of the 30 m pixel (power / weight) converted to dB.
                        # np.maximum(gg, 1e-6) avoids a division by zero where no valid SAR pixel exists (those pixels are
                        # dropped anyway by the 'good' test above); np.maximum(..., 1e-8) keeps log10 away from log10(0).
                        # (A previous version had the "* np.log10(...)" part inside a comment, so db was always 10.0 and every
                        # SAR pixel became +10 dB. This is the correct formula.)
                        db = 10.0 * np.log10(np.maximum(p / np.maximum(gg, 1e-6), 1e-8))
                        arr[k] = np.where(good, np.clip(np.round(db * 100), -3000, 1500), -32768)
        except Exception as e:
            status = "error: %r" % (e,)
        arr[:, ~pm] = -32768                           # clip to the district polygon
        if status == "ok":
            if a.cache or a.cache_only:
                write_cache(cp, arr, s, d, scene, orbit, g)
        return arr, status, orbit, "zip"

    # --- create the output file (written to a .tmp name and renamed at the end) ---
    prof = dict(driver="GTiff", height=H, width=W, count=nb, dtype="int16", crs=C.CRS,
                transform=g["transform"], nodata=-32768, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, interleave="band", BIGTIFF="YES")
    table, scenes_tab, b, t0 = [], [], 0, time.time()
    tmp = out + ".tmp"
    dst = None
    if not a.cache_only:
        dst = rasterio.open(tmp, "w", **prof)
        dst.update_tags(layout="sensor order: " + ",".join(sensors), crs=C.CRS, resolution_m=str(C.RES),
                        note="int16 nodata -32768; optical DN (x0.0001=reflectance); SAR gamma0 dB*100")
    try:
        # --- main loop: one scene at a time, written straight into its bands (keeps memory low) ---
        for i, (s, d, scene, z) in enumerate(items):
            names = names_of(s)
            arr, status, orbit, src = get_scene(s, d, scene, z)
            # share of the district covered by this scene (any band valid) - shows partial swaths and failed scenes
            cover = 100.0 * (arr != -32768).any(axis=0).sum() / pm.sum()   # % of the district covered by this scene
            if dst is not None:
                dst.write(arr, indexes=list(range(b + 1, b + len(names) + 1)))
                for k, n in enumerate(names):          # label every band so QGIS shows sensor|date|band|scene
                    bi = b + 1 + k
                    dst.set_band_description(bi, "%s|%s|%s|%s" % (s, d, n, scene))
                    dst.update_tags(bi, sensor=s, date=d, doy=str(C.doy_of(d)), scene=scene, band=n,
                                    unit=("DN" if kind(s) == "optical" else "gamma0 dB*100"), orbit=orbit,
                                    aoi_cover_pct="%.1f" % cover)
                    table.append([bi, s, d, C.doy_of(d), scene, n, status, orbit, "%.1f" % cover])
                scenes_tab.append([s, d, C.doy_of(d), scene, b + 1, b + len(names), "%.1f" % cover, status, orbit])
            b += len(names)
            print("[%d/%d] %s %s %s  covers %.1f%% of AOI  (%s, %.0fs)" % (
                i + 1, len(items), s, d, status, cover, src, time.time() - t0), flush=True)
    finally:
        if dst is not None:
            dst.close()
    if a.cache_only:
        print("cache filled in %s/ (no stack written)" % CACHE)
        return
    os.replace(tmp, out)                       # the stack is complete: give it its final name
    stem = out[:-4]
    # --- sidecar tables: one row per band, and one row per scene (first/last band of that scene in the stack) ---
    with open(stem + "_bands.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "sensor", "date", "doy", "scene", "band_name", "status", "orbit", "aoi_cover_pct"])
        w.writerows(table)
    cov = os.path.join(os.path.dirname(out), "satellite_scene_table_%s%s.csv" % (tag, "_test" if a.limit else ""))
    with open(cov, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "date", "doy", "scene", "first_band", "last_band", "aoi_cover_pct", "status", "orbit"])
        w.writerows(scenes_tab)
    print("saved", out, "and its _bands.csv and", cov)


if __name__ == "__main__":
    main()