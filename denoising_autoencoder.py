"""Denoising autoencoder on the same samples (alternative to Barlow Twins in barlow_twins.py).

    python denoising_autoencoder.py                         # 20 epochs -> outputs/model_dae.pt
    python denoising_autoencoder.py --epochs 1 --max 2000   # smoke test
    python embed_to_tif.py --model outputs/model_dae.pt --out outputs/embedding_dae_30m.tif

Idea: the encoder (same PixelEncoder as before) sees only a random 40-70 % of a pixel's valid
observations, plus a little Gaussian noise. A decoder gets the 128-D embedding and the date/sensor
of EVERY observation and must rebuild all valid observations (also the hidden ones).
So the embedding has to hold the whole time series, and the decoder can also fill gaps.
Loss = weighted MSE on valid tokens (hidden tokens count x2). Printed 'hidden' = MSE on the
observations the encoder did NOT see (a gap-filling score, lower is better).
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
from pixel_encoder import PixelEncoder


class Decoder(nn.Module):
    """embedding (B,128) + (date, sensor) of each token -> its 5 features (B,T,5)."""

    def __init__(self, sid, d=64, zdim=128, feats=5):
        super().__init__()
        self.register_buffer("sid", torch.as_tensor(sid, dtype=torch.long))
        self.z = nn.Linear(zdim, d)
        self.doy = nn.Linear(4, d)
        self.sens = nn.Embedding(int(self.sid.max()) + 1, d)
        self.net = nn.Sequential(nn.GELU(), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d),
                                 nn.GELU(), nn.Linear(2 * d, feats))

    def forward(self, z, doy):
        ang = 2 * math.pi * doy / 365.25
        t_emb = torch.stack([torch.sin(ang), torch.cos(ang), torch.sin(2 * ang), torch.cos(2 * ang)], -1)
        q = self.doy(t_emb) + self.sens(self.sid)                       # (T,d)
        return self.net(self.z(z)[:, None] + q[None])                   # (B,T,5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.1, help="std of Gaussian noise on the input")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.OUT, "model_dae.pt"))
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
    nmean, nstd = mean[sid][None], std[sid][None]
    Xt, Vt = torch.from_numpy(X), torch.from_numpy(V)
    mean_t, std_t, doy_t = torch.from_numpy(nmean), torch.from_numpy(nstd), torch.from_numpy(doy)
    pad_cols = torch.from_numpy(~C.feature_mask(sid))[None]                 # (1,T,9) columns a sensor does not have
    real = torch.from_numpy(C.feature_mask(sid, targets_only=True)).float()[None]   # columns to rebuild: weather NOT included
    enc, dec = PixelEncoder(sid), Decoder(sid, feats=C.N_FEAT)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    n = len(Xt)
    steps = a.epochs * max(1, n // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    print("samples %d, tokens %d, encoder %.2fM + decoder %.2fM params" % (
        n, Xt.shape[1], sum(p.numel() for p in enc.parameters()) / 1e6,
        sum(p.numel() for p in dec.parameters()) / 1e6))
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = hid_sum = hid_n = 0.0
        nb = 0
        for i in range(0, n - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            x = (Xt[b].float() - mean_t) / std_t
            x = torch.where(pad_cols, torch.zeros_like(x), x)
            v = Vt[b]
            p = torch.empty(len(b), 1).uniform_(0.4, 0.7)
            keep = v & (torch.rand(v.shape) < p)
            x_in = (x + a.noise * torch.randn_like(x)) * keep[..., None]
            # weather dropout: for a random half of the pixels the 4 weather columns are hidden completely, so the
            # encoder learns to work with and without weather (and cannot lean on it as a location code)
            x_in[..., C.N_SAT_FEAT:] *= (torch.rand(len(b), 1, 1) > 0.5).float()
            z = enc(x_in, doy_t, keep)
            z = F.normalize(z, dim=1) * math.sqrt(z.shape[1])
            rec = dec(z, doy_t)
            err = (rec - x).pow(2) * real                                # (B,T,5)
            w = v.float() + (v & ~keep).float()                          # hidden valid tokens count x2
            loss = (err.sum(-1) * w).sum() / (w.sum() * real.sum(-1).mean() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
            hm = (v & ~keep)
            hid_sum += (err.sum(-1) * hm).sum().item()
            hid_n += (hm.float().sum() * real.sum(-1).mean()).item()
        print("epoch %2d/%d  loss %.4f  hidden-token MSE %.4f  (%.0fs)" % (
            ep + 1, a.epochs, tot / max(nb, 1), hid_sum / max(hid_n, 1), time.time() - t0), flush=True)
        torch.save(dict(model=enc.state_dict(), decoder=dec.state_dict(), sid=sid, doy=doy,
                        mean=nmean, std=nstd, names=s["names"]), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()