r"""Fine-tune the pre-trained pixel encoder on the jute map (Apr-Jun only):  raw observations -> encoder -> jute / not jute.

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python jute\jute_finetune.py --seeds 0                     # first test (about 10-15 min)
    python jute\jute_finetune.py --seeds 0 1 2                 # 3 different block splits -> mean +- std
    python jute\jute_finetune.py --seeds 0 --scratch           # same network, but NOT pre-trained (shows what pre-training is worth)

WHY.  jute_classify.py puts a small classifier on top of the FROZEN 128-D embedding: whatever the embedding left out is lost.
Here the whole encoder (the same PixelEncoder that made embedding_barlow_*.tif, weights from model.pt) is trained a little
further with the jute labels, so it can bring forward exactly the details of the time series (dates, bands, weather) that
separate jute from other crops. This is the usual way a foundation model is adapted to a task.

HOW.
  * The split is IDENTICAL to jute_classify.py (same 3 km blocks, same seeds, same sampling code), so the numbers can be put
    next to the numbers of jute_results.txt:  --seeds 0 here = seed 0 there (Barlow mlp+prior: jute F1 0.765).
  * Training pixels: --n_per_class jute and --n_per_class not-jute pixels from TRAIN blocks (balanced). Test pixels:
    --n_test pixels from TEST blocks with the real class shares. Train and test blocks never touch each other.
  * Every training step randomly hides 15 % of the valid observations of each pixel (so the network cannot lean on one
    single date) and uses a small learning rate for the encoder (it is pre-trained) and a larger one for the new output layer.
  * Two scores per run, as in jute_classify.py:  plain, and "+prior" (the output is corrected for the real class shares,
    because training used equal class sizes).
Output: jute_finetune_results.txt in the season folder, and model_jute_finetune.pt (the fine-tuned network of the first seed).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# This script lives in the jute\ sub-folder; common.py, pixel_encoder.py (one folder up) and jute_classify.py (same folder)
# are found by adding both folders to the import path.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C
from pixel_encoder import PixelEncoder
from jute_classify import block_split, pick, confusion, scores


class JuteNet(nn.Module):
    """PixelEncoder (128-D output) followed by one linear layer with 2 outputs (not jute, jute)."""

    def __init__(self, ck, pretrained=True):
        super().__init__()
        self.enc = PixelEncoder(ck["sid"])
        if pretrained:
            self.enc.load_state_dict(ck["model"])                 # weights learned by Barlow Twins on the whole district
        self.head = nn.Linear(128, 2)

    def forward(self, x, doy, v):
        z = self.enc(x, doy, v)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * np.sqrt(128.0)   # unit vector scaled like the embedding files
        return self.head(z)


def load_pixels(rows, cols, W, chunk=40000):
    """All observations of the given pixels (rows must be sorted): X (n,T,9) float16, V (n,T) bool = observation is valid.
    Read in chunks of `chunk` pixels so that the memory stays small."""
    flat = rows.astype(np.int64) * W + cols
    Xs, Vs = [], []
    for i in range(0, len(flat), chunk):
        X, V, _, _, _ = C.read_all_scenes(None, flat[i:i + chunk])
        Xs.append(X)
        Vs.append(V)
    return np.concatenate(Xs), np.concatenate(Vs)


def prepare(X, ck):
    """Normalise like embed_to_tif.py does: (value - mean) / std per observation and feature, and zero the feature
    columns that a sensor does not have. Returns a float32 tensor (n,T,9)."""
    mean, std = torch.from_numpy(ck["mean"]), torch.from_numpy(ck["std"])
    pad = torch.from_numpy(~C.feature_mask(ck["sid"]))[None]
    x = (torch.from_numpy(X).float() - mean) / std
    return torch.where(pad, torch.zeros_like(x), x)


def predict_logits(net, x, v, doy, chunk=1024):
    """Network output (n,2) for all pixels, in chunks (the attention memory grows with chunk x tokens x tokens).
    A small chunk (1024) keeps the RAM low, so that this step also runs when other programs hold much memory."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            out.append(net(x[i:i + chunk], doy, v[i:i + chunk]).numpy())
    return np.concatenate(out)


def train_net(net, x, v, y, doy, epochs, lr_enc, lr_head, batch=256, hide=0.15, seed=0):
    """Fine-tune: cross-entropy on balanced pixels. Each step hides `hide` of the valid observations at random."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": lr_enc},
                             {"params": net.head.parameters(), "lr": lr_head}], weight_decay=1e-4)
    yt = torch.from_numpy(y.astype(np.int64))
    n = len(x)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            vb = v[b] & (torch.rand(v[b].shape) > hide)             # random observation dropout (augmentation)
            vb = vb | (~vb.any(1, keepdim=True) & v[b])             # never leave a pixel without any observation
            loss = nn.functional.cross_entropy(net(x[b], doy, vb), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print("   epoch %2d/%d  train loss %.4f  (%.0f s)" % (ep + 1, epochs, tot / n, time.time() - t0), flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("outputs", "jute_labels_30m.tif"))
    ap.add_argument("--model", default=os.path.join(C.OUT, "model.pt"), help="pre-trained Barlow Twins encoder")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="block splits, same meaning as in jute_classify.py")
    ap.add_argument("--n_per_class", type=int, default=20000, help="training pixels per class (20000 = as in jute_classify.py)")
    ap.add_argument("--n_test", type=int, default=100000)
    ap.add_argument("--block", type=int, default=100, help="block size in pixels (100 = 3 km)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr_enc", type=float, default=3e-4, help="learning rate of the pre-trained encoder")
    ap.add_argument("--lr_head", type=float, default=2e-3, help="learning rate of the new output layer")
    ap.add_argument("--scratch", action="store_true", help="do NOT use the pre-trained weights (random start)")
    a = ap.parse_args()

    import rasterio
    g = C.get_grid()
    with rasterio.open(a.labels) as ds:
        L = ds.read(1)
    if L.shape != (g["H"], g["W"]):
        raise SystemExit("labels are not on the 30 m grid: run jute_inspect.py first")
    y_map = np.where(L >= 0, (L == 1).astype(np.int16), -1)        # 1 = jute, 0 = other, -1 = no label
    rr, cc = np.nonzero(y_map >= 0)
    yy = y_map[rr, cc].astype(np.int64)
    ck = torch.load(a.model, weights_only=False)
    doy = torch.from_numpy(ck["doy"])
    lr_enc = 2e-3 if a.scratch else a.lr_enc                       # a random start needs a larger step
    name = "scratch (not pre-trained)" if a.scratch else "fine-tuned Barlow encoder"

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("JUTE FINE-TUNING   folder %s   %s   seeds %s   %d epochs, %d+%d training pixels" % (
        C.OUT, name, a.seeds, a.epochs, a.n_per_class, a.n_per_class))
    res = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        is_test = block_split(rr, cc, a.block, seed)                # same blocks as jute_classify.py
        sel = []
        for k in (0, 1):
            tr = np.flatnonzero((yy == k) & ~is_test)
            sel.append(pick(rng, tr, a.n_per_class))
        te = pick(rng, np.flatnonzero(is_test), a.n_test)
        tr_idx = np.concatenate(sel)
        idx = np.concatenate([tr_idx, te])
        istr = np.concatenate([np.ones(len(tr_idx), bool), np.zeros(len(te), bool)])
        order = np.argsort(rr[idx], kind="stable")                  # the reader needs rows in increasing order
        idx, istr = idx[order], istr[order]
        t0 = time.time()
        X, V = load_pixels(rr[idx], cc[idx], g["W"])
        if X.shape[1] != len(ck["doy"]):
            raise SystemExit("model was trained on %d observations but the stack has %d" % (len(ck["doy"]), X.shape[1]))
        ok = V.sum(1) >= 1                                          # pixels with at least one valid observation
        y = yy[idx]
        x = prepare(X, ck)
        del X                                                       # the float16 copy is not needed any more
        v = torch.from_numpy(V)
        itr, ite = np.flatnonzero(istr & ok), np.flatnonzero(~istr & ok)
        say()
        say("##### seed %d: train %d px, test %d px on UNSEEN blocks (jute share of the test pixels %.1f %%)  [read %.0f s]" % (
            seed, len(itr), len(ite), 100 * y[ite].mean(), time.time() - t0))
        net = JuteNet(ck, pretrained=not a.scratch)
        t0 = time.time()
        train_net(net, x[itr], v[itr], y[itr], doy, a.epochs, lr_enc, a.lr_head, seed=seed)
        z = predict_logits(net, x[ite], v[ite], doy)
        # prior correction: log(real class share in the train blocks / share in the balanced training sample, 0.5)
        nat = np.bincount(yy[~is_test], minlength=2) / max((~is_test).sum(), 1)
        adj = (np.log(np.maximum(nat, 1e-6)) - np.log(0.5)).astype(np.float32)
        for var, ad in (("finetune", 0.0), ("finetune+prior", adj)):
            s = scores(confusion(y[ite], (z + ad).argmax(1), 2))
            say("  %-15s acc %.3f | bal.acc %.3f | macro-F1 %.3f | jute P %.3f R %.3f F1 %.3f IoU %.3f" % (
                var, s["acc"], s["bal_acc"], s["macro_f1"], s["prec"][1], s["rec"][1], s["f1"][1], s["iou"][1]))
            res.append(dict(var=var, f1m=s["macro_f1"], P=s["prec"][1], R=s["rec"][1], F=s["f1"][1], I=s["iou"][1]))
        say("  (%.0f s)" % (time.time() - t0))
        # the result file is rewritten after EVERY seed, so a crash in a later seed does not lose the finished ones
        open(os.path.join(C.OUT, "jute_finetune_results%s.txt" % ("_scratch" if a.scratch else "")), "w",
             encoding="utf-8").write("\n".join(lines) + "\n")
        if seed == a.seeds[0]:
            torch.save(dict(model=net.state_dict(), sid=ck["sid"], doy=ck["doy"], mean=ck["mean"], std=ck["std"]),
                       os.path.join(C.OUT, "model_jute_finetune%s.pt" % ("_scratch" if a.scratch else "")))

    say()
    say("SUMMARY over %d seed(s): mean +- std   (compare: jute_results.txt, barlow mlp+prior jute F1 0.750 +- 0.010, macro-F1 0.852)" % len(a.seeds))
    say("%-16s %-16s %-16s %-16s %-16s" % ("variant", "macro-F1", "jute precision", "jute recall", "jute F1"))
    for var in ("finetune", "finetune+prior"):
        rs = [r for r in res if r["var"] == var]
        say("%-16s " % var + " ".join("%.3f +- %.3f " % (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
                                       for k in ("f1m", "P", "R", "F")))
    out = os.path.join(C.OUT, "jute_finetune_results%s.txt" % ("_scratch" if a.scratch else ""))
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()