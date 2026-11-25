"""Observation-set transformer (TESSERA-style): a pixel's time series from all sensors -> 128-D.

Every token has 9 input features: 5 satellite slots + 4 weather slots (see common.N_FEAT)."""
import math

import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    def __init__(self, sid, d=64, out=128, heads=4, layers=2, feats=9):
        super().__init__()
        self.register_buffer("sid", torch.as_tensor(sid, dtype=torch.long))
        self.nsens = int(self.sid.max()) + 1
        self.idx = [torch.nonzero(self.sid == s).flatten() for s in range(self.nsens)]
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(feats, d), nn.GELU(), nn.Linear(d, d)) for _ in range(self.nsens)])
        self.doy = nn.Linear(4, d)
        self.sens = nn.Embedding(self.nsens, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out)

    def forward(self, x, doy, use):
        """x (B,T,F) normalised; doy (T,) ; use (B,T) bool = token is valid AND selected."""
        B, T, _ = x.shape
        tok = x.new_zeros(B, T, self.cls.shape[-1])
        for s in range(self.nsens):
            ii = self.idx[s].to(x.device)
            if len(ii):
                tok[:, ii] = self.proj[s](x[:, ii])
        ang = 2 * math.pi * doy / 365.25
        t_emb = torch.stack([torch.sin(ang), torch.cos(ang), torch.sin(2 * ang), torch.cos(2 * ang)], -1)
        tok = tok + self.doy(t_emb)[None] + self.sens(self.sid)[None]
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~use], 1)
        h = self.enc(tok, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0]))


class Projector(nn.Module):
    def __init__(self, d=128, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, h))

    def forward(self, z):
        return self.net(z)


def barlow_twins(a, b, lam=5e-3):
    a = (a - a.mean(0)) / (a.std(0) + 1e-5)
    b = (b - b.mean(0)) / (b.std(0) + 1e-5)
    c = a.T @ b / a.shape[0]
    on = (torch.diagonal(c) - 1).pow(2).sum()
    off = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on + lam * off