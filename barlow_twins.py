"""Self-supervised training (Barlow Twins on two random temporal subsets of each pixel).

Every token has 9 input numbers: 5 satellite features + 4 weather features (rain 7 d, rain 14 d, Tmax 7 d, Tmin 7 d of the
pixel's IMD cell, see common.py). Weather dropout: in each of the two views the 4 weather columns of a random half of the
pixels are hidden, so the embedding must agree whether or not weather is visible and cannot use it as a location code.

    python barlow_twins.py                # 20 epochs
    python barlow_twins.py --epochs 1 --max 2000   # smoke test
"""
import argparse
import os
import time

import numpy as np
import torch

import common as C
from pixel_encoder import PixelEncoder, Projector, barlow_twins


def views(valid, keep_lo=0.4, keep_hi=0.7):
    """Two independent random subsets of the valid tokens."""
    B, T = valid.shape
    out = []
    for _ in range(2):
        p = torch.empty(B, 1).uniform_(keep_lo, keep_hi)
        out.append(valid & (torch.rand(B, T) < p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max", type=int, default=0, help="use only first N samples")
    a = ap.parse_args()
    torch.manual_seed(0)
    s = np.load(os.path.join(C.OUT, "samples.npz"))
    X, V = s["X"], s["V"]
    if a.max:
        X, V = X[: a.max], V[: a.max]
    doy, sid = s["doy"], s["sid"]
    nsens = int(sid.max()) + 1
    mean = np.zeros((nsens, C.N_FEAT), np.float32)       # per sensor and feature (5 satellite + 4 weather)
    std = np.ones((nsens, C.N_FEAT), np.float32)
    for k in range(nsens):
        cols = np.flatnonzero(sid == k)
        xv = X[:, cols][V[:, cols]].astype(np.float32)
        if len(xv):
            mean[k], std[k] = xv.mean(0), xv.std(0) + 1e-3
    nmean, nstd = mean[sid][None], std[sid][None]            # (1,T,5)
    Xt = torch.from_numpy(X)
    Vt = torch.from_numpy(V)
    mean_t, std_t = torch.from_numpy(nmean), torch.from_numpy(nstd)
    doy_t = torch.from_numpy(doy)
    # columns a sensor does not have (SAR has no 4th/5th feature) have std 1e-3 and mean 0: neutralise them
    pad_cols = torch.from_numpy(~C.feature_mask(sid))[None]                 # (1,T,9)
    enc = PixelEncoder(sid)
    prj = Projector()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(prj.parameters()), lr=a.lr, weight_decay=1e-4)
    n = len(Xt)
    steps = a.epochs * max(1, n // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    print("samples %d, tokens %d, params %.2fM" % (n, Xt.shape[1],
          sum(p.numel() for p in enc.parameters()) / 1e6))
    os.makedirs(C.OUT, exist_ok=True)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            x = (Xt[b].float() - mean_t) / std_t
            x = torch.where(pad_cols, torch.zeros_like(x), x)
            v1, v2 = views(Vt[b])
            xs = []
            for _ in range(2):                                           # weather dropout, independently per view
                w_on = (torch.rand(len(b), 1, 1) > 0.5).float()
                xv = x.clone()
                xv[..., C.N_SAT_FEAT:] *= w_on
                xs.append(xv)
            loss = barlow_twins(prj(enc(xs[0], doy_t, v1)), prj(enc(xs[1], doy_t, v2)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        print("epoch %2d/%d  loss %.2f  (%.0fs)" % (ep + 1, a.epochs, tot / max(nb, 1), time.time() - t0),
              flush=True)
        torch.save(dict(model=enc.state_dict(), sid=sid, doy=doy, mean=nmean, std=nstd,
                        names=s["names"]), os.path.join(C.OUT, "model.pt"))
    print("saved", os.path.join(C.OUT, "model.pt"))


if __name__ == "__main__":
    main()