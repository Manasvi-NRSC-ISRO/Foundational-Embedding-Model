"""Run the trained model on every AOI pixel -> ONE GeoTIFF (128 bands, float32 between 0 and 1) + PCA preview.

    python embed_to_tif.py                       # whole district
    python embed_to_tif.py --rows 400 --test     # first 400 rows only, quick check
    python embed_to_tif.py --tta 2               # average with 2 random-subset passes

Output: outputs/embedding_30m.tif   float32, 128 bands, values between 0 and 1, nodata -1
        fixed for every season:  u = (z + 4) / 8,  z = unit vector * sqrt(128)  (clipped to -4..4)
        outputs/embedding_pca.png / embedding_pca.tif   (3 PCA bands as RGB)
"""
import argparse
import os
import time

import numpy as np
import torch

import common as C
from pixel_encoder import PixelEncoder


def quantise(z):
    """(n,128) raw model output -> float32 between 0 and 1. z is L2-normalised, scaled by sqrt(128), mapped (z+4)/8."""
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8) * np.sqrt(z.shape[1])
    return C.z_to_u(z)


def embed_pixels(enc, ck, X, V, tta=0, chunk=4096):
    # chunk = pixels sent through the transformer at once. The attention memory grows with chunk x tokens x tokens, so a
    # small chunk (4096) keeps RAM low; 20000 needed 1.3 to 2.7 GB per layer and failed with "not enough memory".
    mean, std = torch.from_numpy(ck["mean"]), torch.from_numpy(ck["std"])
    doy = torch.from_numpy(ck["doy"])
    sid = ck["sid"]
    pad = torch.from_numpy(~C.feature_mask(sid))[None]                  # columns a sensor does not have (weather exists)
    out = np.zeros((len(X), 128), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = (torch.from_numpy(X[i:i + chunk]).float() - mean) / std
            x = torch.where(pad, torch.zeros_like(x), x)
            v = torch.from_numpy(V[i:i + chunk])
            z = enc(x, doy, v)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            for _ in range(tta):
                z2 = enc(x, doy, v & (torch.rand(v.shape) < 0.8))
                z = z + z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
            out[i:i + chunk] = z.numpy()
    return out


def pca_preview(path, out_png, out_tif, step=4):
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(path) as ds:
        h, w = ds.height // step, ds.width // step
        a = ds.read(out_shape=(ds.count, h, w)).astype(np.float32)
        tr = ds.transform * ds.transform.scale(step, step)
        crs = ds.crs
    ok = C.emb_valid(a)
    a = np.where(a < 0, 0, a)               # nodata (-1) must not take part in the PCA
    flat = a.reshape(a.shape[0], -1).T
    sel = np.flatnonzero(ok.ravel())
    if len(sel) < 10:
        print("PCA: no valid pixels")
        return
    sub = flat[np.random.default_rng(0).choice(sel, min(50000, len(sel)), replace=False)]
    mu = sub.mean(0)
    _, S, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    pcs = (flat[sel] - mu) @ Vt[:3].T
    lo, hi = np.percentile(pcs, 2, axis=0), np.percentile(pcs, 98, axis=0)
    rgb = np.zeros((h * w, 3), np.float32)
    rgb[sel] = np.clip((pcs - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = rgb.reshape(h, w, 3)
    plt.figure(figsize=(8, 8 * h / w + 0.5))
    plt.imshow(rgb)
    plt.title("Embedding PCA (3 comps, %.0f%% variance)" % (100 * (S[:3] ** 2).sum() / (S ** 2).sum()))
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    with rasterio.open(out_tif, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                       crs=crs, transform=tr, compress="deflate") as dst:
        dst.write((rgb.transpose(2, 0, 1) * 255).astype(np.uint8))
    print("PCA preview ->", out_png)


def main():
    import rasterio
    from rasterio.windows import Window
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.OUT, "model.pt"))
    ap.add_argument("--kind", default="barlow", choices=["barlow", "dae", "conv"],
                    help="which model the embedding comes from; names the file embedding_<kind>_<tag>.tif")
    ap.add_argument("--out", default=None, help="output tif (default: embedding_<kind>_<tag>.tif in the season folder)")
    ap.add_argument("--block", type=int, default=32, help="rows per block (lower = less RAM)")
    ap.add_argument("--rows", type=int, default=0, help="only the first N rows")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="write float-free tiny output name")
    a = ap.parse_args()
    a.out = a.out or C.emb_path(a.kind)
    ck = torch.load(a.model, weights_only=False)
    enc = PixelEncoder(ck["sid"])
    enc.load_state_dict(ck["model"])
    enc.eval()
    g = C.get_grid()
    H, W = (a.rows or g["H"]), g["W"]
    out = a.out.replace(".tif", "_test.tif") if a.test else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prof = dict(driver="GTiff", height=H, width=W, count=128, dtype="float32", crs=C.CRS,
                transform=g["transform"], nodata=C.EMB_NODATA, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="YES")
    t0 = time.time()
    # The file is written under a temporary name and renamed only when ALL rows are done. If the run fails half way
    # (for example out of memory) no half-empty embedding tif is left behind that a later step could read by mistake.
    tmp = out + ".tmp"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.update_tags(scale="float32 between 0 and 1 ; z = u * 8 - 4 ; unit vector = z / sqrt(128) ; -1 = nodata", model=a.model,
                        sensors=",".join(C.SENSORS), source=C.MODEL_TIF)
        for r0 in range(0, H, a.block):
            h = min(a.block, H - r0)
            sel = np.arange(h * W)
            X, V, _, _, _ = C.read_all_scenes(Window(0, r0, W, h), sel)
            if X.shape[1] != len(ck["doy"]):
                raise SystemExit("Model was trained on %d observations but the stack has %d. "
                                 "Re-run build_samples.py and barlow_twins.py." % (len(ck["doy"]), X.shape[1]))
            inside = g["pmask"][r0:r0 + h].ravel() & (V.sum(1) >= 1)
            q = np.full((h * W, 128), C.EMB_NODATA, np.float32)
            if inside.any():
                z = embed_pixels(enc, ck, X[inside], V[inside], a.tta)
                q[inside] = quantise(z)
            dst.write(q.T.reshape(128, h, W), window=Window(0, r0, W, h))
            print("rows %d/%d  (%.0fs)" % (r0 + h, H, time.time() - t0), end="\r", flush=True)
    os.replace(tmp, out)                                         # complete: give it its final name
    print("\nsaved", out)
    stem = os.path.splitext(out)[0]                              # embedding_<kind>_<tag>.tif -> embedding_<kind>_<tag>_pca.*
    pca_preview(out, stem + "_pca.png", stem + "_pca.tif")


if __name__ == "__main__":
    main()