"""Cloud mask for every optical scene, computed directly on the single stack file (satellite_stack_<tag>.tif).

This is step 2 of the pipeline (after preprocess_stack_scenes.py).

    python cloudmask_scenes.py

Output: <season folder>/satellite_cloudmask_<tag>.tif  (uint8, ONE band per scene, in the same order as the rows of
        satellite_scene_table_<tag>.csv with status ok: band n of the mask belongs to scene n of that table)
        values: 0 no data, 1 clear, 2 cloud, 255 outside the district
        (SAR scenes: 1 where there is data, 0 where not; no cloud test is done for radar)
        <season folder>/satellite_cloudmask_<tag>_stats.csv  (sensor, date, scene, mask_band, valid / cloud / clear %)

The rule (no cloud product exists for this data, so brightness is used):
  Every optical SCENE is tested on its own (scenes of the same date are NOT averaged), and the "normal" brightness
  of a pixel is its median over all scenes of its own sensor. A valid pixel is cloud if
    (a) the pixel has at least MIN_OBS valid scenes and its visible reflectance is T_BRIGHT above its own median
        and brighter than VIS_MIN, OR
    (b) the pixel has fewer than MIN_OBS valid scenes (short seasons) and its visible reflectance is above VIS_ABS, OR
    (c) any band is saturated (DN >= SAT_DN), OR
    (d) its visible reflectance is above VIS_STRONG whatever the median is.
  Before (a), scenes that are uniformly brighter than normal (haze, calibration) get their brightness offset removed,
  up to SHIFT_CAP, so that only real bright spots are called cloud. Cloud is then grown by DILATE pixels to catch
  thin edges.
"""
import csv
import os

import numpy as np

import common as C

# ----------------------------------------------------------------------------------------------
# THRESHOLDS (reflectance units, 0..1; visible = mean of green and red)
# ----------------------------------------------------------------------------------------------
T_BRIGHT = 0.07         # a pixel this much brighter than its own median is cloud (rule a)
VIS_MIN = 0.15          # ...but only if it is also at least this bright in absolute terms (rule a)
VIS_ABS = 0.30          # absolute limit used when there are too few scenes for a median (rule b)
MIN_OBS = 5             # minimum number of valid scenes for the median to be trusted
SAT_DN = 10000          # DN of 10000 = reflectance 1.0 = saturated sensor (rule c)
DILATE = 1              # cloud is grown by this many pixels in every direction
SHIFT_CAP = 0.10        # max scene-wide brightening that is NOT counted as cloud
VIS_STRONG = 0.40       # visible reflectance above this is cloud whatever the median is (rule d)
BLOCK = 256             # number of image rows processed at a time (keeps memory low)

OUT_TIF = os.path.join(C.OUT, "satellite_cloudmask_%s.tif" % C.TAG if C.TAG else "satellite_cloudmask.tif")      # satellite_cloudmask_<tag>.tif
OUT_CSV = OUT_TIF[:-4] + "_stats.csv"


def cloud_flags(refl_g, refl_r, valid, sat, med, nobs):
    """Apply rules (a), (b), (c) to arrays of the same shape.

    refl_g, refl_r : green / red reflectance of ONE scene
    valid          : True where the scene has data
    sat            : True where any band is saturated
    med            : the pixel's median visible reflectance over the sensor's scenes (plus the scene's offset)
    nobs           : number of valid scenes at the pixel
    Returns a bool array, True = cloud.
    """
    vis = 0.5 * (refl_g + refl_r)                                     # visible brightness
    rel = (nobs >= MIN_OBS) & ((vis - med) > T_BRIGHT) & (vis > VIS_MIN)   # rule (a): brighter than usual
    ab = (nobs < MIN_OBS) & (vis > VIS_ABS)                           # rule (b): too few scenes, absolute limit
    return valid & (rel | ab | sat)                                   # rule (c): saturated


def dilate(m, n=1):
    """Grow a boolean mask by n pixels in all 8 directions (3x3 square), without needing scipy."""
    for _ in range(n):
        p = np.pad(m, 1)                                              # pad so the edges can be shifted safely
        m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] | p[1:-1, :-2] | p[1:-1, 1:-1] |
             p[1:-1, 2:] | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return m


def scene_shifts(src, rows, opt, pm, step=8):
    """Scene-wide brightness offset of every optical scene vs the median of its sensor.
    (src: open stack file, rows: scene table, opt: sensor -> scene indices, pm: district mask, step: down-sampling)

    A scene that is uniformly brighter (haze, calibration, dry fields) would otherwise be flagged
    cloud everywhere. Computed on an 8x down-sampled copy; capped at SHIFT_CAP so that real overcast
    scenes still show as cloud.
    """
    h, w = src.height // step, src.width // step
    inside = pm[: h * step: step, : w * step: step][:h, :w]
    shifts = {}
    for sensor, idx in opt.items():
        vis = np.full((len(idx), h, w), np.nan, np.float32)
        for k, i in enumerate(idx):
            f, l = int(rows[i]["first_band"]), int(rows[i]["last_band"])
            a = src.read(list(range(f, l + 1)), out_shape=(l - f + 1, h, w)).astype(np.float32)
            v = (a > 0).all(axis=0) & inside
            vis[k] = np.where(v, 0.5 * (a[0] + a[1]) * 1e-4, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(vis, axis=0)
        for k, i in enumerate(idx):
            d = (vis[k] - med)[~np.isnan(vis[k] - med)]
            shifts[i] = float(np.clip(np.median(d), 0.0, SHIFT_CAP)) if len(d) > 500 else 0.0
    return shifts


def main():
    """Build the mask file block by block (rows 256 at a time) and write the per-scene statistics csv."""
    import rasterio
    from rasterio.windows import Window
    rows = C.load_scene_table()
    g = C.get_grid()
    H, W, pm = g["H"], g["W"], g["pmask"]
    n = len(rows)                               # number of scenes = number of mask bands
    opt = {}                                    # sensor -> [indices into rows]
    for i, r in enumerate(rows):
        if C.SENSORS[r["sensor"]]["kind"] == "optical":
            opt.setdefault(r["sensor"], []).append(i)
    prof = dict(driver="GTiff", height=H, width=W, count=n, dtype="uint8", crs=C.CRS,
                transform=g["transform"], compress="deflate", tiled=True, blockxsize=256,
                blockysize=256, BIGTIFF="IF_SAFER")
    tmp = OUT_TIF + ".tmp"
    # counters per scene for the stats csv: valid, cloud and clear pixel counts inside the district
    cnt = np.zeros((n, 3), np.int64)            # valid, cloud, clear pixel counts inside the district
    with rasterio.open(C.SCENES_TIF) as src, rasterio.open(tmp, "w", **prof) as dst:
        if (src.height, src.width) != (H, W):
            raise SystemExit("satellite_stack_<tag>.tif is on a different grid. Re-run preprocess_stack_scenes.py.")
        shifts = scene_shifts(src, rows, opt, pm)     # scene-wide haze/brightness offsets (see function)
        for i, sh in sorted(shifts.items()):
            if sh > 0.02:
                print("  scene-wide brightening %.3f removed: %s %s" % (sh, rows[i]["sensor"], rows[i]["date"]))
        for r0 in range(0, H, BLOCK):
            h = min(BLOCK, H - r0)
            a0, a1 = max(r0 - DILATE, 0), min(r0 + h + DILATE, H)      # read a little more for dilation
            win = Window(0, a0, W, a1 - a0)
            hh = a1 - a0
            M = np.zeros((n, hh, W), np.uint8)          # mask block for ALL scenes
            for sensor, idx in opt.items():             # the median is taken per sensor
                T = len(idx)
                vis = np.full((T, hh, W), np.nan, np.float32)
                val = np.zeros((T, hh, W), bool)
                sat = np.zeros((T, hh, W), bool)
                for k, i in enumerate(idx):
                    f, l = int(rows[i]["first_band"]), int(rows[i]["last_band"])
                    a = src.read(list(range(f, l + 1)), window=win)
                    val[k] = (a > 0).all(axis=0)
                    sat[k] = (a >= SAT_DN).any(axis=0)
                    vis[k] = np.where(val[k], 0.5 * (a[0] + a[1]) * 1e-4, np.nan)
                nobs = val.sum(0)                        # valid scenes per pixel (for rule a/b)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    med = np.nan_to_num(np.nanmedian(vis, axis=0), nan=1.0)
                for k, i in enumerate(idx):              # test every scene of this sensor on its own
                    vv = np.nan_to_num(vis[k], nan=0.0)
                    c = cloud_flags(vv, vv, val[k], sat[k], med + shifts[i], nobs)
                    c |= val[k] & (vv > VIS_STRONG)      # rule (d): very bright is always cloud
                    if DILATE:
                        c = dilate(c, DILATE) & val[k]
                    m = np.zeros((hh, W), np.uint8)       # 0 = nodata
                    m[val[k]] = 1                         # 1 = clear
                    m[c] = 2                              # 2 = cloud
                    M[i] = m
            for i, r in enumerate(rows):                  # SAR scenes: just "has data" (1) or not (0)
                if C.SENSORS[r["sensor"]]["kind"] == "sar":
                    f = int(r["first_band"])
                    a = src.read(f, window=win)
                    M[i] = (a != C.SAR_NODATA).astype(np.uint8)
            o = r0 - a0                                   # drop the extra rows that were read only for dilation
            M = M[:, o:o + h]
            inside = pm[r0:r0 + h]
            M[:, ~inside] = 255                           # outside the district polygon
            dst.write(M, window=Window(0, r0, W, h))
            for i in range(n):
                m = M[i][inside]
                cnt[i, 0] += (m > 0).sum()
                cnt[i, 1] += (m == 2).sum()
                cnt[i, 2] += (m == 1).sum()
            print("  rows %d-%d / %d" % (r0, r0 + h, H), end="\r", flush=True)
        for i, r in enumerate(rows):
            dst.set_band_description(i + 1, "%s|%s|mask|%s" % (r["sensor"], r["date"], r["scene"]))
            dst.update_tags(i + 1, sensor=r["sensor"], date=r["date"], scene=r["scene"],
                            values="0 nodata, 1 clear, 2 cloud, 255 outside")
    os.replace(tmp, OUT_TIF)                              # finished: give the file its final name
    npx = pm.sum()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mask_band", "sensor", "date", "scene", "valid_pct", "cloud_pct", "clear_pct"])
        for i, r in enumerate(rows):
            w.writerow([i + 1, r["sensor"], r["date"], r["scene"]] +
                       [round(100 * cnt[i, j] / npx, 1) for j in range(3)])
    print("\nwrote", OUT_TIF, "and", OUT_CSV)


if __name__ == "__main__":
    main()