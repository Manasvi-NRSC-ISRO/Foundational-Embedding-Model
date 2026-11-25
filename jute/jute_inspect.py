r"""Look at the jute map and put it on our 30 m grid (needed once; the same file serves every season).

    python jute/jute_inspect.py --tif D:\path\to\jute_map.tif                    # 1) just look: values, CRS, overlap
    python jute/jute_inspect.py --tif D:\path\to\jute_map.tif --jute_values 1    # 2) after you know which value = jute
    add --zero_is_class when the map has nodata=0 but 0 means "not jute" (then 0 becomes the class 'other')

Writes into outputs\  (change with --out):
  jute_labels_30m.tif   int16 on the SAME grid as every embedding (-1 = no label / outside district)
  jute_inspect.txt      everything that is printed  -> send this file to Claude
  jute_labels.png       small preview
Fine labels (e.g. 10 m) are reduced to 30 m by majority vote.

NEW: --pure P (for example --pure 0.9) also writes
  jute_fraction_30m.tif        float32: share of jute inside every 30 m pixel (0 = no jute, 1 = all jute) = "purity" map
  jute_labels_30m_pure.tif     same labels as jute_labels_30m.tif, but MIXED pixels (jute share between 1-P and P)
                               are set to -1 (= no label). Pure pixels keep their label.
Why: the map is 10 m, our pixels are 30 m, so a 30 m pixel holds 9 map pixels. At field edges and in small fields
the pixel is part jute, part other: its label is a coin toss and no model can get it right. Run any jute script with
--labels outputs\...\jute_labels_30m_pure.tif to train AND test only on pixels whose label is clear.
Assumes jute = value 1 and every other class = 0 in the source map (true for jute_2025.tif); needs --jute_values 1.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: common.py, model.py
import argparse
import os

import numpy as np

import common as C

LINES = []


def say(s=""):
    print(s)
    LINES.append(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--jute_values", default="", help="label value(s) that mean jute, e.g. 1 or 1,2")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--zero_is_class", action="store_true",
                    help="the file marks 0 as nodata but 0 is a real class (= not jute): keep it")
    ap.add_argument("--pure", type=float, default=0.0,
                    help="e.g. 0.9: also write the jute-share map and a label file without MIXED 30 m pixels "
                         "(a pixel is kept only if its jute share is >= P or <= 1-P). 0 = off")
    a = ap.parse_args()
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds

    os.makedirs(a.out, exist_ok=True)
    g = C.get_grid()
    H, W = g["H"], g["W"]
    with rasterio.open(a.tif) as ds:
        say("FILE      %s" % a.tif)
        say("size      %d x %d px, %d band(s), dtype %s" % (ds.width, ds.height, ds.count, ds.dtypes[0]))
        say("CRS       %s" % ds.crs)
        say("pixel     %s x %s (CRS units)" % (abs(ds.res[0]), abs(ds.res[1])))
        say("bounds    %s" % (tuple(round(v, 4) for v in ds.bounds),))
        say("nodata    %s%s" % (ds.nodata, "" if ds.nodata is not None else "   (none set: 0 will count as a class)"))
        try:
            cm = ds.colormap(1)
            say("colour table (value: rgba) first 15: %s" % dict(list(cm.items())[:15]))
        except Exception:
            pass
        say("descr     %s" % (ds.descriptions,))
        # overlap with our grid (bbox of the district)
        x0, y1 = g["transform"].c, g["transform"].f
        ours = (x0, y1 - H * C.RES, x0 + W * C.RES, y1)
        b = transform_bounds(ds.crs, C.CRS, *ds.bounds)
        ox = max(0, min(b[2], ours[2]) - max(b[0], ours[0]))
        oy = max(0, min(b[3], ours[3]) - max(b[1], ours[1]))
        say("overlap   %.1f%% of the district bounding box lies inside the jute map" % (
            100 * ox * oy / ((ours[2] - ours[0]) * (ours[3] - ours[1]))))
        # values in the source (reduced copy, nearest)
        scale = max(ds.width, ds.height) / 3000.0
        shp = (max(int(ds.height / max(scale, 1)), 1), max(int(ds.width / max(scale, 1)), 1))
        arr = ds.read(1, out_shape=shp, resampling=Resampling.nearest)
        vals, cnt = np.unique(arr, return_counts=True)
        say("")
        say("VALUES in the source map (reduced copy):")
        for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])[:30]:
            say("  value %-8s %10d px  %5.1f %%" % (v, c, 100 * c / arr.size))
        # onto our grid
        dst = np.full((H, W), -1, np.int16)
        # rasterio falls back to the file's own nodata when src_nodata is None, so use an unused value (255) instead
        src_nd = 255 if a.zero_is_class else ds.nodata
        say("nodata used for the 30 m labels: %s" % src_nd)
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=g["transform"], dst_crs=C.CRS,
                  resampling=Resampling.mode, src_nodata=src_nd, dst_nodata=-1)
        frac = None
        if a.pure > 0:
            # jute share per 30 m pixel: the average of the 0/1 source pixels that fall into it (area weighted by GDAL).
            # Source pixels with the 'unused' nodata value (255) are ignored by the average, as in the mode step above.
            frac = np.full((H, W), -1, np.float32)
            reproject(source=rasterio.band(ds, 1), destination=frac, dst_transform=g["transform"], dst_crs=C.CRS,
                      resampling=Resampling.average, src_nodata=src_nd, dst_nodata=-1)
    dst[~g["pmask"]] = -1
    inside = int(g["pmask"].sum())
    lab = dst >= 0
    say("")
    say("ON OUR 30 m GRID (district polygon, %d px = %.0f ha):" % (inside, inside * 0.09))
    say("  labelled pixels: %d (%.1f %% of the district)" % (lab.sum(), 100 * lab.sum() / inside))
    vals, cnt = np.unique(dst[lab], return_counts=True)
    for v, c in zip(vals, cnt):
        say("  value %-6s %9d px  %9.0f ha  %5.1f %% of labelled" % (v, c, c * 0.09, 100 * c / lab.sum()))
    jv = [int(v) for v in a.jute_values.split(",") if v.strip() != ""]
    if jv:
        j = np.isin(dst, jv)
        say("  JUTE (values %s): %d px = %.0f ha = %.1f %% of the labelled area" % (
            jv, j.sum(), j.sum() * 0.09, 100 * j.sum() / max(lab.sum(), 1)))
    out = os.path.join(a.out, "jute_labels_30m.tif")
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="int16", crs=C.CRS,
                transform=g["transform"], nodata=-1, compress="deflate")
    with rasterio.open(out, "w", **prof) as d:
        d.write(dst, 1)
        d.update_tags(source=a.tif, jute_values=a.jute_values)
    say("saved %s" % out)
    if frac is not None:
        frac[~g["pmask"]] = -1
        ok = frac >= 0
        mixed = ok & (frac > 1 - a.pure) & (frac < a.pure)          # neither clearly jute nor clearly other
        pure_lab = np.where(mixed | ~ok, -1, np.where(frac >= a.pure, 1, 0)).astype(np.int16)
        pure_lab[dst < 0] = -1
        n_ok = int(ok.sum())
        say("")
        say("PURITY (threshold %.2f): a 30 m pixel is PURE if its jute share is >= %.2f (jute) or <= %.2f (other)" % (
            a.pure, a.pure, 1 - a.pure))
        say("  mixed pixels (no clear label): %d = %.1f %% of the district" % (mixed.sum(), 100 * mixed.sum() / max(n_ok, 1)))
        say("  pure jute : %d px   pure other: %d px   jute share among the pure pixels: %.1f %%" % (
            (pure_lab == 1).sum(), (pure_lab == 0).sum(), 100 * (pure_lab == 1).sum() / max((pure_lab >= 0).sum(), 1)))
        jt = frac[ok & (dst == 1)]
        say("  jute pixels (majority vote) whose share is below 0.9: %.1f %%" % (100 * (jt < 0.9).mean() if jt.size else 0))
        prof_f = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=C.CRS,
                      transform=g["transform"], nodata=-1, compress="deflate")
        with rasterio.open(os.path.join(a.out, "jute_fraction_30m.tif"), "w", **prof_f) as d:
            d.write(frac, 1)
        with rasterio.open(os.path.join(a.out, "jute_labels_30m_pure.tif"), "w", **prof) as d:
            d.write(pure_lab, 1)
            d.update_tags(source=a.tif, jute_values=a.jute_values, pure=str(a.pure))
        say("saved jute_fraction_30m.tif and jute_labels_30m_pure.tif in %s" % a.out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = 4
    sm = dst[::s, ::s].astype(float)
    sm[sm < 0] = np.nan
    if jv:
        sm = np.where(np.isnan(sm), np.nan, np.isin(dst[::s, ::s], jv).astype(float))
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(sm, cmap="RdYlGn_r" if jv else "tab20", interpolation="nearest")
    ax.set_title("labels on the 30 m grid" + (" (red = jute)" if jv else ""))
    ax.axis("off")
    fig.savefig(os.path.join(a.out, "jute_labels.png"), dpi=100, bbox_inches="tight")
    open(os.path.join(a.out, "jute_inspect.txt"), "w", encoding="utf-8").write("\n".join(LINES) + "\n")
    say("saved %s and jute_labels.png" % os.path.join(a.out, "jute_inspect.txt"))


if __name__ == "__main__":
    main()