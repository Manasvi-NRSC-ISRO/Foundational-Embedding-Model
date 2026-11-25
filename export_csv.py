"""Embeddings -> CSV (sample or a small area). Whole district would be ~6 GB of text.

    python export_csv.py                          # 100000 random pixels
    python export_csv.py --n 20000
    python export_csv.py --bbox 88.0 24.0 88.2 24.2 --n 0    # every pixel in a lon/lat box
    python export_csv.py --n 0                    # EVERYTHING (huge)

Columns: row, col, x, y, lon, lat, e000 ... e127   (e = float32 between 0 and 1)
"""
import argparse
import csv
import os

import numpy as np

import common as C


def main():
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="random sample size, 0 = all")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--out", default=os.path.join(C.OUT, "embeddings_sample.csv"))
    a = ap.parse_args()
    src = C.emb_path("barlow")
    to_ll = Transformer.from_crs(C.CRS, "EPSG:4326", always_xy=True)
    with rasterio.open(src) as ds:
        H, W, tr = ds.height, ds.width, ds.transform
        valid = ds.read(1) >= 0
        if a.bbox:
            to_utm = Transformer.from_crs("EPSG:4326", C.CRS, always_xy=True)
            x0, y0 = to_utm.transform(a.bbox[0], a.bbox[1])
            x1, y1 = to_utm.transform(a.bbox[2], a.bbox[3])
            c0, r1 = [int(v) for v in (~tr) * (min(x0, x1), min(y0, y1))]
            c1, r0 = [int(v) for v in (~tr) * (max(x0, x1), max(y0, y1))]
            box = np.zeros_like(valid)
            box[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = True
            valid &= box
        rows, cols = np.nonzero(valid)
        if a.n and a.n < len(rows):
            idx = np.sort(np.random.default_rng(0).choice(len(rows), a.n, replace=False))
            rows, cols = rows[idx], cols[idx]
        print("writing %d pixels -> %s" % (len(rows), a.out))
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "col", "x", "y", "lon", "lat"] + ["e%03d" % i for i in range(128)])
            for r0 in range(0, H, 128):
                h = min(128, H - r0)
                s = (rows >= r0) & (rows < r0 + h)
                if not s.any():
                    continue
                rr, cc = rows[s], cols[s]
                arr = ds.read(window=Window(0, r0, W, h))
                vals = arr[:, rr - r0, cc].T
                x = tr.c + (cc + .5) * C.RES
                y = tr.f - (rr + .5) * C.RES
                lon, lat = to_ll.transform(x, y)
                for i in range(len(rr)):
                    w.writerow([rr[i], cc[i], "%.0f" % x[i], "%.0f" % y[i], "%.5f" % lon[i],
                                "%.5f" % lat[i]] + ["%.3f" % v for v in vals[i]])
    print("done")


if __name__ == "__main__":
    main()