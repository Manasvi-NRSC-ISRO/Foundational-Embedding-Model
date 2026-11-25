"""Embedding tif -> numpy arrays (only pixels that have an embedding).

    python export_embeddings.py

outputs/embeddings_float32.npy      (N, 128) float32   embedding between 0 and 1
outputs/embeddings_rowcol.npy    (N, 2)   int32     [row, col] of every embedding on the 30 m grid
outputs/embeddings_meta.json     crs, transform, grid size, how to get x/y:
                                 x = x0 + (col + 0.5) * 30 ;  y = y0 - (row + 0.5) * 30
Load:  E = np.load("outputs/embeddings_float32.npy", mmap_mode="r");  rc = np.load("outputs/embeddings_rowcol.npy")
"""
import json
import os

import numpy as np

import common as C


def main():
    import rasterio
    from rasterio.windows import Window
    src = C.emb_path("barlow")
    with rasterio.open(src) as ds:
        H, W, tr = ds.height, ds.width, ds.transform
        valid = ds.read(1) >= 0
        rows, cols = np.nonzero(valid)
        N = len(rows)
        print("pixels with embedding:", N)
        E = np.lib.format.open_memmap(os.path.join(C.OUT, "embeddings_float32.npy"), mode="w+",
                                      dtype=np.float32, shape=(N, 128))
        k = 0
        for r0 in range(0, H, 128):
            h = min(128, H - r0)
            m = valid[r0:r0 + h]
            n = int(m.sum())
            if n:
                a = ds.read(window=Window(0, r0, W, h))        # (128, h, W)
                E[k:k + n] = a[:, m].T
                k += n
        E.flush()
    np.save(os.path.join(C.OUT, "embeddings_rowcol.npy"), np.stack([rows, cols], 1).astype(np.int32))
    json.dump(dict(crs=C.CRS, resolution_m=C.RES, height=H, width=W, x0=tr.c, y0=tr.f, n_pixels=N,
                   scale="float32 between 0 and 1", unit_vector="(u * 8 - 4) / sqrt(128)"),
              open(os.path.join(C.OUT, "embeddings_meta.json"), "w"), indent=1)
    print("saved outputs/embeddings_float32.npy, embeddings_rowcol.npy, embeddings_meta.json")


if __name__ == "__main__":
    main()