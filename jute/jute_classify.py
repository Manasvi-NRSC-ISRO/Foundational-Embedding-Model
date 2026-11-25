r"""Does an embedding know where jute is?  Train a small classifier on the embedding, test it on places it never saw.

    python jute\jute_classify.py --jute_values 1                # jute vs everything else (binary)
    python jute_classify.py                                      # every label value is one class (multi-class)
    python jute_classify.py --jute_values 1 --predict            # also write a jute map per embedding
    python jute_classify.py --jute_values 1 --seeds 0 1 2        # 3 different splits -> mean +- std (is the ranking real?)
    python jute_classify.py --jute_values 1 --both outputs\2025_Apr_Jun outputs\2025_Nov_Dec   # + both seasons together
    python jute_classify.py --jute_values 1 --context 2         # embedding averaged over the 5x5 pixels around each pixel
    python jute_classify.py --jute_values 1 --mix               # also 'embedding + raw features together' (e.g. barlow+raw)
    python jute_classify.py --jute_values 1 --n_per_class 100000 --epochs 60   # more training pixels, longer training

Run it per season:   $env:EMB_OUT="outputs\2025_Apr_Jun"   (jute is sown Mar-May, so Apr-Jun is the important one)
Needs outputs\jute_labels_30m.tif from jute_inspect.py (change with --labels).
Compares every embedding found in EMB_OUT: barlow, dae, conv = embedding_<kind>_<tag>.tif  (or give --emb name=path ...),
and a RAW baseline ("raw": monthly mean reflectance/NDVI/SAR taken straight from the stack, no learning). If an embedding
is not clearly better than raw, it adds nothing. --both D1 D2 adds the same pixels from two seasons side by side.
Variants per classifier: plain, and "+prior" (training uses equal class sizes; +prior corrects the output for the real
class shares, which usually raises precision for a rare class like jute).

Fair test: the district is cut into square blocks (--block px); ~30 % of the BLOCKS are test blocks. Train and test pixels
are never neighbours, so the classifier cannot cheat by memorising a field. Two classifiers per embedding:
linear (a straight line in 128-D: shows how well the embedding itself separates classes) and mlp (small network).
Output: jute_results.txt / .csv / .png in EMB_OUT  -> send jute_results.txt to Claude.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

# This script lives in the jute\ sub-folder; common.py is one folder up, so that folder is added to the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

# The three embedding kinds. Their files are called embedding_<kind>_<tag>.tif (tag = first date _to_ last date of the
# season), so they are looked up by pattern in the season folder instead of using a fixed name.
KINDS = ["barlow", "dae", "conv"]


def emb_file(folder, kind):
    """Newest embedding_<kind>_<date>_to_<date>.tif in `folder`, or None if that embedding does not exist yet.
    Only the REAL embedding is accepted: the name must end right after the second date (so side products such as
    embedding_<kind>_<tag>_pca.tif or _rgb.tif, which have only 3 bands, are skipped) and the file must have 128 bands."""
    import re
    import rasterio
    pat = re.compile(r"^embedding_%s_\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}\.tif$" % re.escape(kind))
    hits = [f for f in glob.glob(os.path.join(folder, "embedding_%s_20*_to_*.tif" % kind))
            if pat.match(os.path.basename(f))]
    hits = sorted(hits, key=os.path.getmtime)
    for f in reversed(hits):                                   # newest first
        with rasterio.open(f) as ds:
            if ds.count == 128:
                return f
    return None


def raw_stack_file(folder):
    """The raw satellite stack satellite_stack_<tag>.tif of a season folder (the 'filled' one is not matched), or None."""
    hits = sorted(glob.glob(os.path.join(folder, "satellite_stack_20*_to_*.tif")), key=os.path.getmtime)
    return hits[-1] if hits else None


# ---------------------------------------------------------------- numpy helpers (no torch needed)
def block_split(rows, cols, block, seed, test_frac=0.3):
    """True for pixels lying in a test block (blocks are chosen at random)."""
    bid = (rows // block).astype(np.int64) * 100000 + (cols // block)
    ub, inv = np.unique(bid, return_inverse=True)
    rng = np.random.default_rng(seed)
    is_test_block = rng.random(len(ub)) < test_frac
    return is_test_block[inv]


def confusion(y, p, K):
    return np.bincount(y.astype(np.int64) * K + p, minlength=K * K).reshape(K, K)


def scores(cm):
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pred = cm.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    iou = np.divide(tp, sup + pred - tp, out=np.zeros_like(tp), where=(sup + pred - tp) > 0)
    present = sup > 0
    return dict(acc=tp.sum() / max(cm.sum(), 1), bal_acc=rec[present].mean(), macro_f1=f1[present].mean(),
                prec=prec, rec=rec, f1=f1, iou=iou, support=sup)


def pick(rng, idx, n):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


MONTH_END = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def monthly_features(X, V, doy, sid):
    """Hand-made baseline from raw observations: for every calendar month present, the mean of each valid optical
    value (green, red, NIR, SWIR, NDVI) and of each SAR value (HH dB, HV dB). Months without valid data = NaN."""
    Xf = X.astype(np.float32)
    month = np.searchsorted(MONTH_END, doy.astype(int) - 1, side="right") + 1
    cols = []
    for m in sorted(set(month.tolist())):
        for sens, nf in (((0, 1), 5), ((2,), 2)):
            tok = np.flatnonzero((month == m) & np.isin(sid, sens))
            if len(tok) == 0:
                continue
            v = V[:, tok, None]
            den = v.sum(1)                                                     # (n,1)
            num = (Xf[:, tok, :nf] * v).sum(1)                                 # (n,nf)
            cols.append(np.where(den > 0, num / np.maximum(den, 1), np.nan))
    return np.concatenate(cols, 1).astype(np.float32)


def prior_adjust(y_train_pool, y_sample, K):
    """log(real class share / share in the (balanced) training sample), added to the logits."""
    nat = np.bincount(y_pool_clip(y_train_pool, K), minlength=K) / max(len(y_train_pool), 1)
    smp = np.bincount(y_sample, minlength=K) / max(len(y_sample), 1)
    return (np.log(np.maximum(nat, 1e-6)) - np.log(np.maximum(smp, 1e-6))).astype(np.float32)


def y_pool_clip(y, K):
    return np.clip(y, 0, K - 1)


def fill_nan(Xtr, Xte):
    mu = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0) if np.isnan(Xtr).any() else Xtr.mean(0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    for A in (Xtr, Xte):
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = np.broadcast_to(mu, A.shape)[bad]
    return Xtr, Xte


# ---------------------------------------------------------------- reading embeddings
# Radius (in pixels) of the neighbourhood over which the embedding is averaged before it is given to a classifier.
# 0 = the embedding of the pixel itself (default). 1 = mean of the 3x3 pixels around it, 2 = 5x5, 3 = 7x7 (about 200 m).
# Set from the command line (--context). Jute fields are patches several pixels wide, and the 30 m label itself is a vote
# of 10 m pixels, so averaging the neighbours removes single-pixel noise and often raises the score.
CTX = 0


def box_sum(x, r):
    """Sum of x over a (2r+1) x (2r+1) window around every pixel, for x of shape (..., h, w).
    Done with cumulative sums (one pass, independent of r). Outside the array counts as 0. x must be float64 because a
    float32 cumulative sum over a whole strip loses the small differences that this subtraction needs."""
    k = 2 * r + 1
    pad = [(0, 0)] * (x.ndim - 2)
    c = np.pad(x, pad + [(r, r), (r, r)]).cumsum(-2).cumsum(-1)
    c = np.pad(c, pad + [(1, 0), (1, 0)])
    return c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]


def read_features(path, rows, cols, strip=256):
    """128-D vectors (n,128) float32 between 0 and 1 at the given pixels; rows must be sorted. ok=False where nodata.
    With CTX > 0 every vector is the MEAN of the valid embeddings in the (2*CTX+1)^2 window around the pixel (nodata
    pixels are left out of the mean); ok still refers to the centre pixel itself."""
    import rasterio
    from rasterio.windows import Window
    out = np.zeros((len(rows), 128), np.float32)
    ok = np.zeros(len(rows), bool)
    if CTX > 0:
        strip = min(strip, 32)                      # the neighbourhood sums need float64 copies: keep the strip small
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        for r0 in range(0, H, strip):
            lo, hi = np.searchsorted(rows, [r0, r0 + strip])
            if hi == lo:
                continue
            if CTX == 0:
                a = ds.read(window=Window(0, r0, W, min(strip, H - r0)))            # (128,h,W) float32
                v = a[:, rows[lo:hi] - r0, cols[lo:hi]].T                           # (m,128)
                out[lo:hi] = np.where(v < 0, 0, v)        # nodata (-1) -> 0, flagged by ok
                ok[lo:hi] = (v >= 0).any(1)
            else:
                ra, rb = max(0, r0 - CTX), min(H, r0 + strip + CTX)                 # strip plus CTX rows above and below
                a = ds.read(window=Window(0, ra, W, rb - ra))                       # (128,h',W)
                val = (a >= 0).any(0)                                               # pixel has an embedding
                s = box_sum(np.where(a < 0, 0, a).astype(np.float64), CTX)          # sum of the valid vectors around
                n = box_sum(val.astype(np.float64), CTX)                            # how many valid pixels around
                mean = (s / np.maximum(n, 1)).astype(np.float32)
                rr_, cc_ = rows[lo:hi] - ra, cols[lo:hi]
                out[lo:hi] = mean[:, rr_, cc_].T
                ok[lo:hi] = val[rr_, cc_]
    return out, ok


def raw_features(folder, rows, cols):
    """Monthly means from the raw season stack in `folder` (satellite_stack_<tag>.tif + scene table + cloud masks) at the
    given pixels. common.py takes its file paths from module variables, so they are pointed at `folder` for this call
    (common.read_all_scenes also finds the weather files through C.OUT) and restored afterwards."""
    g = C.get_grid()
    keep = (C.OUT, C.SCENES_TIF, C.SCENES_CSV, C.MASKS_TIF, C.TAG)
    stack = raw_stack_file(folder)
    tag = os.path.basename(stack)[len("satellite_stack_"):-4]          # '2025-04-14_to_2025-06-30'
    C.OUT = folder
    C.TAG = tag
    C.SCENES_TIF = stack
    C.SCENES_CSV = os.path.join(folder, "satellite_scene_table_%s.csv" % tag)
    C.MASKS_TIF = os.path.join(folder, "satellite_cloudmask_%s.tif" % tag)
    try:
        X, V, doy, sid, _ = C.read_all_scenes(None, rows.astype(np.int64) * g["W"] + cols)
    finally:
        C.OUT, C.SCENES_TIF, C.SCENES_CSV, C.MASKS_TIF, C.TAG = keep
    return monthly_features(X, V, doy, sid)


def load_spec(spec, rows, cols):
    """spec = ('emb', [tif, ...]) -> 128-D vectors side by side;  ('raw', [folder, ...]) -> monthly means;
    ('embraw', [tif, folder]) -> the embedding AND the raw monthly means of the same season in one feature vector."""
    kind, items = spec
    if kind == "embraw":                          # items = [embedding tif, season folder]: 128 + 21 features side by side
        f1, ok = read_features(items[0], rows, cols)
        return np.concatenate([f1, raw_features(items[1], rows, cols)], 1), ok
    parts, ok = [], np.ones(len(rows), bool)
    for it in items:
        if kind == "emb":
            f, o = read_features(it, rows, cols)
            ok &= o
        else:
            f = raw_features(it, rows, cols)
        parts.append(f)
    return np.concatenate(parts, 1), ok


# ---------------------------------------------------------------- classifiers (torch)
def make_model(kind, K, d=128):
    import torch.nn as nn
    if kind == "linear":
        return nn.Linear(d, K)
    return nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(),
                         nn.Linear(128, K))


def fit(kind, X, y, K, epochs):
    import torch
    torch.manual_seed(0)
    m = make_model(kind, K, X.shape[1])
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    opt = torch.optim.AdamW(m.parameters(), lr=0.01 if kind == "linear" else 2e-3, weight_decay=1e-4)
    bs = 2048
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(m(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def logits(m, X):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 100000):
            out.append(m(torch.from_numpy(X[i:i + 100000])).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def predict(m, X, adj=None):
    z = logits(m, X)
    if adj is not None:
        z = z + adj
    return z.argmax(1)


def write_map(path, emb_path, model, mu, sd, to_value, grid, adj=None, strip=128):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(emb_path) as ds:
        prof = dict(driver="GTiff", height=ds.height, width=ds.width, count=1, dtype="uint8", crs=ds.crs,
                    transform=ds.transform, nodata=255, compress="deflate")
        with rasterio.open(path, "w", **prof) as dst:
            for r0 in range(0, ds.height, strip):
                h = min(strip, ds.height - r0)
                a = ds.read(window=Window(0, r0, ds.width, h))
                v = a.reshape(128, -1).T
                valid = (v >= 0).any(1)
                x = (np.where(v < 0, 0, v) - mu) / sd
                p = np.full(len(v), 255, np.uint8)
                if valid.any():
                    p[valid] = to_value[predict(model, x[valid], adj)].astype(np.uint8)
                p = p.reshape(h, ds.width)
                p[~grid["pmask"][r0:r0 + h]] = 255
                dst.write(p, 1, window=Window(0, r0, ds.width, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("outputs", "jute_labels_30m.tif"))
    ap.add_argument("--jute_values", default="", help="label value(s) = jute; empty = multi-class on all values")
    ap.add_argument("--emb", nargs="*", help="name=path ...  (default: the 3 embeddings in EMB_OUT)")
    ap.add_argument("--both", nargs=2, metavar=("DIR1", "DIR2"),
                    help="two season folders: also test their embeddings (and raw features) side by side")
    ap.add_argument("--no_raw", action="store_true", help="skip the raw-data baseline")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="repeat with these seeds (different splits)")
    ap.add_argument("--n_per_class", type=int, default=20000, help="training pixels per class")
    ap.add_argument("--n_test", type=int, default=100000, help="test pixels (natural class shares)")
    ap.add_argument("--block", type=int, default=100, help="size of a split block in pixels (100 = 3 km)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min_class", type=int, default=200, help="drop classes with fewer train pixels")
    ap.add_argument("--context", type=int, default=0,
                    help="average the embedding over the (2R+1)x(2R+1) pixels around each pixel (0 = off, 1 = 3x3, 2 = 5x5)")
    ap.add_argument("--mix", action="store_true",
                    help="also test 'embedding + raw features together' (e.g. barlow+raw)")
    ap.add_argument("--predict", action="store_true", help="also write jute_pred_<name>.tif (best classifier, 1 seed)")
    a = ap.parse_args()
    global CTX
    CTX = a.context                                  # read_features() takes the neighbourhood size from here
    if a.predict and a.context:
        print("note: --predict writes maps from the single-pixel embedding, so --context is ignored for the maps")
    import rasterio
    g = C.get_grid()
    with rasterio.open(a.labels) as ds:
        L = ds.read(1)
    if L.shape != (g["H"], g["W"]):
        raise SystemExit("labels are not on the 30 m grid: run jute_inspect.py first")
    jv = [int(v) for v in a.jute_values.split(",") if v.strip() != ""]
    lab = L >= 0
    if jv:
        names = ["other", "jute"]
        y_map = np.where(lab, np.isin(L, jv).astype(np.int16), -1)
        to_value = np.array([0, 1])
    else:
        classes = np.unique(L[lab])
        names = ["value %d" % c for c in classes]
        remap = np.full(int(L.max()) + 2, -1, np.int16)
        remap[classes] = np.arange(len(classes))
        y_map = np.where(lab, remap[np.clip(L, 0, None)], -1)
        to_value = classes
    K = len(names)
    rr, cc = np.nonzero(y_map >= 0)
    yy = y_map[rr, cc]
    # feature sets to compare
    specs = []                                     # (name, (kind, [items]))
    if a.emb:
        for e in a.emb:
            n, p = e.split("=", 1)
            specs.append((n, ("emb", p.split("+"))))
    else:
        for n in KINDS:
            f = emb_file(C.OUT, n)
            if f:
                specs.append((n, ("emb", [f])))
    if not a.no_raw and raw_stack_file(C.OUT):
        specs.append(("raw", ("raw", [C.OUT])))
    if a.mix and raw_stack_file(C.OUT):               # embedding + raw features of the same season, one vector
        for n, sp in list(specs):
            if sp[0] == "emb" and len(sp[1]) == 1:
                specs.append((n + "+raw", ("embraw", [sp[1][0], C.OUT])))
    if a.both:
        d1, d2 = a.both
        for n in KINDS:
            p1, p2 = emb_file(d1, n), emb_file(d2, n)
            if p1 and p2:
                specs.append((n + "_both", ("emb", [p1, p2])))
        if not a.no_raw:
            specs.append(("raw_both", ("raw", [d1, d2])))
    if not specs:
        raise SystemExit("no embedding found in %s" % C.OUT)
    lines, res = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("JUTE TEST   folder %s   labels %s   seeds %s   context radius %d px" % (C.OUT, a.labels, a.seeds, a.context))
    say("feature sets: %s" % ", ".join(n for n, _ in specs))
    best_for_map = {}
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        is_test = block_split(rr, cc, a.block, seed)
        sel, kept = [], []
        for k in range(K):
            tr = np.flatnonzero((yy == k) & ~is_test)
            if len(tr) < a.min_class:
                say("!! class %s has only %d training pixels: dropped" % (names[k], len(tr)))
                continue
            sel.append(pick(rng, tr, a.n_per_class))
            kept.append(k)
        te = pick(rng, np.flatnonzero(is_test), a.n_test)
        tr_idx = np.concatenate(sel)
        idx = np.concatenate([tr_idx, te])
        istr = np.concatenate([np.ones(len(tr_idx), bool), np.zeros(len(te), bool)])
        order = np.argsort(rr[idx], kind="stable")
        idx, istr = idx[order], istr[order]
        rows, cols, y = rr[idx], cc[idx], yy[idx]
        keepm = np.isin(y, kept)
        rows, cols, y, istr = rows[keepm], cols[keepm], y[keepm], istr[keepm]
        newid = np.full(K, -1)
        newid[kept] = np.arange(len(kept))
        y = newid[y]
        K2 = len(kept)
        cnames = [names[k] for k in kept]
        cvals = to_value[kept]
        pool = newid[yy[~is_test]]
        pool = pool[pool >= 0]
        adj = prior_adjust(pool, y[istr], K2)
        say("")
        say("##### seed %d: classes %s | train %d px (blocks of %d px = %.1f km), test %d px on UNSEEN blocks" % (
            seed, ", ".join(cnames), istr.sum(), a.block, a.block * 0.03, (~istr).sum()))
        say("test class shares: %s | answering the biggest class always gives accuracy %.1f %%" % (
            ", ".join("%s %.1f%%" % (cnames[k], 100 * (y[~istr] == k).mean()) for k in range(K2)),
            100 * np.bincount(y[~istr], minlength=K2).max() / max((~istr).sum(), 1)))
        for name, spec in specs:
            t0 = time.time()
            X, ok = load_spec(spec, rows, cols)
            Xtr, ytr = X[istr & ok].copy(), y[istr & ok]
            Xte, yte = X[~istr & ok].copy(), y[~istr & ok]
            Xtr, Xte = fill_nan(Xtr, Xte)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr, Xte = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
            say("=== %-11s %3d features | train %d, test %d" % (name, X.shape[1], len(Xtr), len(Xte)))
            for kind in ("linear", "mlp"):
                m = fit(kind, Xtr, ytr, K2, a.epochs if kind == "mlp" else a.epochs * 3)
                for var, ad in ((kind, None), (kind + "+prior", adj)):
                    s = scores(confusion(yte, predict(m, Xte, ad), K2))
                    jt = ""
                    if K2 == 2:
                        jt = " | jute P %.3f R %.3f F1 %.3f IoU %.3f" % (s["prec"][1], s["rec"][1], s["f1"][1], s["iou"][1])
                    say("  %-11s acc %.3f | bal.acc %.3f | macro-F1 %.3f%s" % (var, s["acc"], s["bal_acc"], s["macro_f1"], jt))
                    res.append(dict(seed=seed, name=name, clf=var, acc=s["acc"], bal=s["bal_acc"], f1m=s["macro_f1"],
                                    P=s["prec"][-1], R=s["rec"][-1], F=s["f1"][-1], I=s["iou"][-1]))
                if len(spec[1]) == 1 and spec[0] == "emb" and X.shape[1] == 128:
                    f1 = scores(confusion(yte, predict(m, Xte, adj), K2))["macro_f1"]
                    if name not in best_for_map or f1 > best_for_map[name][0]:
                        best_for_map[name] = (f1, m, mu, sd, spec[1][0], kind, adj)
            say("  (%.0f s)" % (time.time() - t0))
    # ------------------------------------------------ summary over seeds
    say("")
    say("SUMMARY over %d seed(s): mean +- std.  macro-F1 = average of the classes' F1 (higher = better)." % len(a.seeds))
    hdr = "%-11s %-13s %-15s" % ("features", "classifier", "macro-F1")
    if K == 2:
        hdr += " %-15s %-15s %-15s" % ("jute precision", "jute recall", "jute F1")
    say(hdr)
    rows_csv = []
    keys = []
    for r in res:
        if (r["name"], r["clf"]) not in keys:
            keys.append((r["name"], r["clf"]))
    table = []
    for n, c in keys:
        rs = [r for r in res if r["name"] == n and r["clf"] == c]
        mean = lambda k: np.mean([r[k] for r in rs])
        std = lambda k: np.std([r[k] for r in rs])
        table.append((mean("f1m"), n, c, rs))
        line = "%-11s %-13s %.3f +- %.3f " % (n, c, mean("f1m"), std("f1m"))
        if K == 2:
            line += " %.3f +- %.3f  %.3f +- %.3f  %.3f +- %.3f" % (mean("P"), std("P"), mean("R"), std("R"), mean("F"), std("F"))
        say(line)
        rows_csv.append([n, c, len(rs), mean("acc"), mean("bal"), mean("f1m"), std("f1m"), mean("P"), mean("R"), mean("F"), mean("I")])
    say("")
    say("RANKING by macro-F1 (best classifier variant of each feature set):")
    seen = set()
    for f, n, c, _ in sorted(table, key=lambda t: -t[0]):
        if n not in seen:
            seen.add(n)
            say("  %-11s %.3f  (%s)" % (n, f, c))
    open(os.path.join(C.OUT, "jute_results.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(C.OUT, "jute_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["features", "classifier", "n_seeds", "accuracy", "balanced_acc", "macro_f1", "macro_f1_std",
                    "last_class_precision", "last_class_recall", "last_class_f1", "last_class_iou"])
        w.writerows(rows_csv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nm = []
    for n, _ in specs:
        nm.append(n)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(nm)), 4))
    xs = np.arange(len(nm))
    for j, c in enumerate(("linear", "mlp", "mlp+prior")):
        v = [np.mean([r["f1m"] for r in res if r["name"] == n and r["clf"] == c]) for n in nm]
        ax.bar(xs + (j - 1) * 0.27, v, 0.27, label=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(nm, rotation=20)
    ax.set_ylabel("macro-F1 (unseen blocks)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(os.path.join(C.OUT, "jute_results.png"), dpi=110, bbox_inches="tight")
    say("saved jute_results.txt, .csv, .png in %s" % C.OUT)
    if a.predict:
        for n, (f, m, mu, sd, path, kind, ad) in best_for_map.items():
            out = os.path.join(C.OUT, "jute_pred_%s.tif" % n)
            write_map(out, path, m, mu.astype(np.float32), sd.astype(np.float32), cvals, g, ad)
            print("saved", out, "(values %s; 255 = nodata)" % list(cvals))


if __name__ == "__main__":
    main()