"""Convolutional autoencoder on top of the pixel embeddings (spatial context).

Stage 1 (already done): every pixel -> 128-D embedding from its own time series (embedding_dae_30m.tif).
Stage 2 (this file):    a small CNN looks at the 5x5 neighbourhood of embeddings and produces a NEW
                        128-D embedding of the centre pixel, so a pixel also knows about its surroundings.

    python convolutional_autoencoder.py train --epochs 30                       # needs outputs/embedding_dae_30m.tif
    python convolutional_autoencoder.py apply                                   # -> outputs/embedding_conv_30m.tif
    python convolutional_autoencoder.py train --epochs 1 --patches 3000         # smoke test

Options: --emb (input embedding tif), --out (output tif), --model (weights).

How it learns (denoising convolutional autoencoder):
  input  : 5x5 patch of embeddings, with ~25 % of the neighbour pixels randomly erased + a little noise
  encoder: Conv3x3 -> GELU -> Conv3x3   (5x5 -> 3x3 -> 1x1)  =  128-D code of the centre pixel
  decoder: code -> the full clean 5x5 patch  (error counted only on pixels that really exist)
Only the ENCODER is used to write the final map; the whole map is processed convolutionally.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

P = 5            # patch size
D = 128          # embedding size


class ConvAE(nn.Module):
    def __init__(self, d=D, h=192):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(d, h, 3), nn.GELU(), nn.Conv2d(h, d, 3))   # valid convs
        self.dec = nn.Sequential(nn.Linear(d, 2 * h), nn.GELU(), nn.Linear(2 * h, d * P * P))

    def forward(self, x):                                   # x (B,d,5,5)
        z = self.enc(x).flatten(1)                          # (B,d)
        zf = F.normalize(z, dim=1) * math.sqrt(D)
        return self.dec(zf).view(-1, D, P, P), zf


def sample_patches(path, n, seed=0, strips=120, rows=64):
    """Random 5x5 patches (float16 0..1, nodata -1, (n,128,5,5)) from the embedding tif; centre must be valid."""
    import rasterio
    from rasterio.windows import Window
    rng = np.random.default_rng(seed)
    out = []
    per = int(math.ceil(n / strips))
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        for s in range(strips):
            r0 = int(rng.integers(0, max(H - rows - 4, 1)))
            a = ds.read(window=Window(0, r0, W, rows + 4))                   # (128, rows+4, W) float32
            valid = C.emb_valid(a)
            cand = np.argwhere(valid[2:-2, 2:-2])
            if len(cand) == 0:
                continue
            pick = cand[rng.choice(len(cand), min(per, len(cand)), replace=False)]
            for r, c in pick:                                                # (r,c) are centre - 2 offsets
                p = a[:, r:r + P, c:c + P]
                if C.emb_valid(p).mean() >= 0.5:
                    out.append(p.astype(np.float16))      # half precision keeps the patch set small
            print("  reading strips %d/%d  patches %d" % (s + 1, strips, len(out)), end="\r", flush=True)
    print()
    return np.stack(out)[:n]


def train(a):
    torch.manual_seed(0)
    P8 = torch.from_numpy(sample_patches(a.emb, a.patches))
    n = len(P8)
    nv = max(n // 20, 1)                                    # 5 % held out
    Pv, Pt = P8[:nv], P8[nv:]
    model = ConvAE()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, len(Pt) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    print("patches: %d train, %d held-out;  params %.2fM" % (
        len(Pt), nv, sum(p.numel() for p in model.parameters()) / 1e6))

    def prep(b):
        valid = (b >= 0).any(1)                                              # (B,5,5) pixel has an embedding
        x = (b.float() * (2 * C.Z_LIM) - C.Z_LIM) * valid[:, None]           # 0..1 -> z scale, nodata pixels -> 0
        return x, valid

    def loss_fn(x, valid, noisy):
        keep = valid.clone()
        if noisy:
            keep &= torch.rand(valid.shape) > 0.25
            keep[:, P // 2, P // 2] = valid[:, P // 2, P // 2]
            xin = (x + 0.1 * torch.randn_like(x)) * keep[:, None]
        else:
            xin = x
        rec, _ = model(xin)
        err = (rec - x).pow(2).mean(1)
        return (err * valid).sum() / valid.sum().clamp(min=1)

    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Pt))
        tot, nb = 0.0, 0
        model.train()
        for i in range(0, len(Pt) - a.batch + 1, a.batch):
            x, valid = prep(Pt[perm[i:i + a.batch]])
            loss = loss_fn(x, valid, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            xv, vv = prep(Pv)
            lv = loss_fn(xv, vv, False).item()
        print("epoch %2d/%d  train loss %.4f  held-out loss %.4f  (%.0fs)" % (
            ep + 1, a.epochs, tot / max(nb, 1), lv, time.time() - t0), flush=True)
        torch.save(dict(model=model.state_dict(), emb=a.emb), a.model)
    print("saved", a.model)


def apply(a):
    import rasterio
    from rasterio.windows import Window
    ck = torch.load(a.model, weights_only=False)
    model = ConvAE()
    model.load_state_dict(ck["model"])
    model.eval()
    t0 = time.time()
    with rasterio.open(a.emb) as ds:
        H, W = ds.height, ds.width
        prof = ds.profile.copy()
        prof.update(BIGTIFF="YES", compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(a.out, "w", **prof) as dst:
            dst.update_tags(scale="float32 between 0 and 1 ; z = u * 8 - 4 ; -1 = nodata",
                            model=a.model, source=a.emb, note="conv autoencoder over 5x5 neighbourhood")
            for r0 in range(0, H, a.block):
                h = min(a.block, H - r0)
                lo, hi = max(r0 - 2, 0), min(r0 + h + 2, H)
                arr = ds.read(window=Window(0, lo, W, hi - lo))                     # (128, rows, W) float32
                full = np.full((D, h + 4, W + 4), C.EMB_NODATA, np.float32)
                full[:, lo - (r0 - 2): lo - (r0 - 2) + arr.shape[1], 2:W + 2] = arr
                valid = C.emb_valid(full)
                x = torch.from_numpy(C.u_to_z(full)) * torch.from_numpy(valid)[None]
                with torch.no_grad():
                    z = model.enc(x[None])[0]                                       # (128, h, W)
                    z = F.normalize(z, dim=0) * math.sqrt(D)
                q = C.z_to_u(z.numpy())
                centre = valid[2:-2, 2:-2]
                q[:, ~centre] = C.EMB_NODATA
                dst.write(q, window=Window(0, r0, W, h))
                print("rows %d/%d  (%.0fs)" % (r0 + h, H, time.time() - t0), end="\r", flush=True)
    print("\nsaved", a.out)
    from embed_to_tif import pca_preview
    stem = os.path.splitext(a.out)[0].replace("_30m", "")
    pca_preview(a.out, stem + "_pca.png", stem + "_pca.tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "apply"])
    ap.add_argument("--emb", default=C.emb_path("dae"))
    ap.add_argument("--out", default=C.emb_path("conv"))
    ap.add_argument("--model", default=os.path.join(C.OUT, "model_conv.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patches", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--block", type=int, default=64, help="rows per block when applying")
    a = ap.parse_args()
    (train if a.mode == "train" else apply)(a)


if __name__ == "__main__":
    main()