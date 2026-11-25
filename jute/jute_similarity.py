r"""Jute vs not-jute by SIMILARITY ONLY (no classifier is trained): cosine similarity to a jute "fingerprint".

    $env:EMB_OUT = "outputs\2025_Apr_Jun"          # jute is sown Mar-May, so Apr-Jun is the important season
    python jute\jute_similarity.py                  # the 3 embeddings of the season folder (barlow, dae, conv)
    python jute\jute_similarity.py --emb barlow=outputs\2025_Apr_Jun\embedding_barlow_....tif   # one given file
    python jute\jute_similarity.py --both outputs\2025_Apr_Jun outputs\2025_Nov_Dec            # both seasons side by side
    python jute\jute_similarity.py --seeds 0 1 2    # 3 different train/test block splits -> mean +- std

Needs outputs\jute_labels_30m.tif made once by jute_inspect.py (1 = jute, 0 = other, -1 = no label / outside district).

IDEA, step by step (the same for every embedding)
  1. The district is cut into square blocks (--block pixels, 100 = 3 km). About 30 % of the blocks are TEST blocks,
     the rest are TRAIN blocks. Train and test pixels are never neighbours, so a good score cannot come from
     memorising a field that touches the training pixels.
  2. FINGERPRINTS: from TRAIN blocks only, the mean embedding of jute pixels (p_jute) and of all other pixels
     (p_other). Every embedding vector is first turned into a unit vector, so a dot product IS the cosine similarity.
  3. Every pixel gets two similarities: s_jute = cos(pixel, p_jute) and s_other = cos(pixel, p_other), and the
     difference  d = s_jute - s_other  (d > 0: looks more like jute than like the rest).
  4. CLASSIFICATION (two rules, both reported):
       "nearest"   : jute when d > 0 (no tuning at all);
       "tuned"     : jute when d > t, where t is the value that gives the best F1 on a validation sample taken from
                     the TRAIN blocks (never from the test blocks).
  5. SCORE on the TEST blocks only (natural class shares, so the many non-jute pixels count like in reality):
       ROC-AUC (how well d ranks jute above non-jute, independent of any threshold), precision, recall, F1 and IoU of
       jute, balanced accuracy, and the 2x2 confusion matrix.
  6. SIMILARITY MATRIX: a heat map of the cosine similarity between 300 jute and 300 non-jute test pixels (sorted by
     class). A clear dark/bright block pattern = the embedding separates the classes. Also the 2x2 table of the mean
     cosine similarity inside jute, between jute and other, and inside other.

OUTPUT (in the season folder EMB_OUT)
  jute_similarity_results.txt / .csv     all numbers (send the .txt to Claude)
  jute_similarity_<name>.png             similarity matrix + histogram of d (jute vs other) + ROC curve
  jute_similarity_compare.png            one bar chart: AUC / F1 / IoU of every embedding side by side
  jute_simmap_<name>.tif                 float32 map of d for the WHOLE district (-9999 = nodata); the "similarity map"
  jute_simpred_<name>.tif                uint8 map: 1 = jute, 0 = not jute (rule "tuned"), 255 = nodata
  (the two maps are for looking in QGIS; the honest numbers are the TEST-block numbers in the .txt)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

# This script lives in the jute\ sub-folder; common.py and jute_classify.py helpers are found by adding it to the path.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C
# Re-used from the classifier script: block split, embedding file lookup, reading of 128-D vectors at pixels, spec loader.
from jute_classify import KINDS, emb_file, block_split, pick, read_features, load_spec, confusion, scores


# ---------------------------------------------------------------- small numpy helpers (no torch, no rasterio)
def unit_blocks(U, n_parts):
    """Stored values u (0..1) of n_parts embeddings placed side by side -> unit vectors.
    Every 128-column part is decoded with z = u*8 - 4 and normalised to length 1 on its own; when two seasons are used
    together the parts are then joined and normalised again, so the result still has length 1.
    A dot product of two such vectors is their cosine similarity."""
    parts = []
    for k in range(n_parts):
        z = U[:, k * 128:(k + 1) * 128] * 8.0 - 4.0
        parts.append(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9))
    z = np.concatenate(parts, 1)
    return (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def fingerprint(Z):
    """Mean direction of a set of unit vectors: their average, scaled back to length 1."""
    m = Z.mean(0)
    return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


def roc_auc(score, y):
    """Area under the ROC curve = probability that a random jute pixel gets a higher score than a random non-jute pixel
    (rank formula, ties get the mean rank). 0.5 = no skill, 1.0 = perfect."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]                                  # tied scores share the mean of their ranks
    start = 0
    while start < len(s_sorted):
        end = start
        while end + 1 < len(s_sorted) and s_sorted[end + 1] == s_sorted[start]:
            end += 1
        if end > start:
            ranks[order[start:end + 1]] = (start + end + 2) / 2.0
        start = end + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_points(score, y, n=200):
    """(false positive rate, true positive rate) at n thresholds, for the ROC picture."""
    y = y.astype(bool)
    th = np.unique(np.percentile(score, np.linspace(0, 100, n)))[::-1]
    tpr = np.array([(score[y] > t).mean() for t in th])
    fpr = np.array([(score[~y] > t).mean() for t in th])
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]


def best_threshold(d, y, n=300):
    """Threshold on d with the best F1 of jute (class 1), searched over n percentiles of d. Used on VALIDATION pixels."""
    y = y.astype(bool)
    best_t, best_f = 0.0, -1.0
    for t in np.unique(np.percentile(d, np.linspace(1, 99, n))):
        p = d > t
        tp = float((p & y).sum())
        f = 2 * tp / max(p.sum() + y.sum(), 1)               # F1 = 2TP / (predicted + actual positives)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def rule_scores(d, y, t):
    """Jute scores of the rule 'jute when d > t' on pixels with true labels y (0/1)."""
    p = (d > t).astype(np.int64)
    cm = confusion(y.astype(np.int64), p, 2)
    s = scores(cm)
    return dict(prec=float(s["prec"][1]), rec=float(s["rec"][1]), f1=float(s["f1"][1]), iou=float(s["iou"][1]),
                bal_acc=float(s["bal_acc"]), acc=float(s["acc"]), cm=cm)


def evaluate(Ztr, ytr, Zval, yval, Zte, yte):
    """The whole similarity method on already prepared unit vectors. Returns a dict of numbers and the arrays for plots."""
    p_j = fingerprint(Ztr[ytr == 1])                         # jute fingerprint (train blocks only)
    p_o = fingerprint(Ztr[ytr == 0])                         # fingerprint of everything else
    d_val = Zval @ p_j - Zval @ p_o                          # similarity difference on validation pixels
    t, f_val = best_threshold(d_val, yval)                   # threshold tuned on validation pixels
    s_j, s_o = Zte @ p_j, Zte @ p_o
    d = s_j - s_o
    return dict(p_j=p_j, p_o=p_o, t=t, f_val=f_val, d=d, s_j=s_j, s_o=s_o,
                auc=roc_auc(d, yte), nearest=rule_scores(d, yte, 0.0), tuned=rule_scores(d, yte, t),
                auc_sj=roc_auc(s_j, yte))


def similarity_matrix(Z, y, n_each, rng):
    """Cosine similarity between n_each jute and n_each non-jute test pixels, jute first, non-jute second.
    Returns the (2n, 2n) matrix and the 2x2 table of mean similarities [[jute-jute, jute-other],[other-jute, other-other]]."""
    j = rng.choice(np.flatnonzero(y == 1), min(n_each, int((y == 1).sum())), replace=False)
    o = rng.choice(np.flatnonzero(y == 0), min(n_each, int((y == 0).sum())), replace=False)
    Zs = np.concatenate([Z[j], Z[o]])
    M = Zs @ Zs.T
    nj = len(j)
    tab = np.array([[M[:nj, :nj].mean(), M[:nj, nj:].mean()], [M[nj:, :nj].mean(), M[nj:, nj:].mean()]])
    return M, nj, tab


# ---------------------------------------------------------------- whole-district maps
def write_maps(paths, p_j, p_o, t, map_path, pred_path, strip=32):
    """Reads the embedding file(s) strip by strip and writes d = cos(pixel, p_jute) - cos(pixel, p_other) for EVERY pixel of
    the district (float32, -9999 = nodata) and the 0/1 jute map of the rule 'd > t' (uint8, 255 = nodata).
    strip = image rows per block. 32 rows x 3161 columns x 128 bands is only about 50 MB per array, so this also runs when
    another heavy job (e.g. an embedding run) is using most of the RAM; a larger strip is faster but needs more memory."""
    import rasterio
    from rasterio.windows import Window
    dss = [rasterio.open(p) for p in paths]
    try:
        H, W = dss[0].height, dss[0].width
        prof = dss[0].profile.copy()
        pf = dict(prof, count=1, dtype="float32", nodata=-9999.0, compress="deflate", tiled=True, blockxsize=256,
                  blockysize=256, BIGTIFF="IF_SAFER")
        pu = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True, blockxsize=256,
                  blockysize=256, BIGTIFF="IF_SAFER")
        for k in ("photometric", "interleave"):
            pf.pop(k, None)
            pu.pop(k, None)
        with rasterio.open(map_path, "w", **pf) as fo, rasterio.open(pred_path, "w", **pu) as fp:
            for r0 in range(0, H, strip):
                h = min(strip, H - r0)
                win = Window(0, r0, W, h)
                A = [ds.read(window=win) for ds in dss]                        # each (128, h, W) float32
                ok = np.all([(a >= 0).any(0) for a in A], axis=0)              # pixel has an embedding in every file
                U = np.concatenate([a.reshape(128, -1).T for a in A], 1)       # (h*W, 128*n_parts)
                Z = unit_blocks(U, len(A))
                d = (Z @ p_j - Z @ p_o).reshape(h, W)
                fo.write(np.where(ok, d, -9999.0).astype(np.float32), 1, window=win)
                fp.write(np.where(ok, (d > t), 255).astype(np.uint8), 1, window=win)
                del A, U, Z, d                                                  # free the memory before the next strip
                print("  map rows %d/%d" % (r0 + h, H), end="\r", flush=True)
        print()
    finally:
        for ds in dss:
            ds.close()


# ---------------------------------------------------------------- pictures
def picture(path, name, M, nj, res, yte):
    """3 panels: similarity matrix, histogram of d for jute and other, ROC curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    im = ax[0].imshow(M, cmap="viridis", vmin=-0.2, vmax=1.0)
    ax[0].axhline(nj - 0.5, color="w", lw=1)
    ax[0].axvline(nj - 0.5, color="w", lw=1)
    ax[0].set_title("%s: cosine similarity matrix\n(first block = jute pixels, second = other)" % name, fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    d = res["d"]
    bins = np.linspace(np.percentile(d, 0.5), np.percentile(d, 99.5), 60)
    ax[1].hist(d[yte == 0], bins=bins, alpha=0.6, density=True, label="other (not jute)")
    ax[1].hist(d[yte == 1], bins=bins, alpha=0.6, density=True, label="jute")
    ax[1].axvline(0, color="k", lw=1, ls=":", label="nearest rule (d=0)")
    ax[1].axvline(res["t"], color="r", lw=1, label="tuned threshold")
    ax[1].set_xlabel("d = cos(pixel, jute fingerprint) - cos(pixel, other fingerprint)")
    ax[1].set_title("Test blocks: how far apart are the two classes?", fontsize=10)
    ax[1].legend(fontsize=8)
    fpr, tpr = roc_points(d, yte)
    ax[2].plot(fpr, tpr, lw=2)
    ax[2].plot([0, 1], [0, 1], "k:", lw=1)
    ax[2].set_xlabel("false positive rate")
    ax[2].set_ylabel("true positive rate")
    ax[2].set_title("ROC curve, AUC = %.3f" % res["auc"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compare_picture(path, rows):
    """Bar chart: AUC, F1 and IoU (rule 'tuned') of every embedding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4))
    for i, (key, lab) in enumerate((("auc", "ROC-AUC"), ("f1", "F1 (jute)"), ("iou", "IoU (jute)"))):
        ax.bar(x + (i - 1) * 0.26, [r[key] for r in rows], 0.26, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Jute by cosine similarity, TEST blocks only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("outputs", "jute_labels_30m.tif"))
    ap.add_argument("--emb", nargs="*", help="name=path ...  (join two seasons with +; default: the 3 embeddings in EMB_OUT)")
    ap.add_argument("--both", nargs=2, metavar=("DIR1", "DIR2"), help="also use the embeddings of two season folders side by side")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="repeat with these seeds (different block splits)")
    ap.add_argument("--n_proto", type=int, default=20000, help="max pixels per class used for a fingerprint")
    ap.add_argument("--n_val", type=int, default=50000, help="validation pixels (train blocks) to tune the threshold")
    ap.add_argument("--n_test", type=int, default=100000, help="test pixels (test blocks, natural class shares)")
    ap.add_argument("--block", type=int, default=100, help="block size in pixels (100 = 3 km)")
    ap.add_argument("--context", type=int, default=0,
                    help="average the embedding over the (2R+1)x(2R+1) pixels around each pixel (0 = off, 1 = 3x3, 2 = 5x5)")
    ap.add_argument("--no_maps", action="store_true", help="skip the whole-district maps (they take a few minutes each)")
    a = ap.parse_args()
    import jute_classify as JC
    JC.CTX = a.context                       # read_features() (used by load_spec) averages the neighbourhood when > 0
    if a.context:
        a.no_maps = True                     # the whole-district maps use the single-pixel embedding: skipped here
        print("context radius %d px: the maps are skipped (they would not use the neighbourhood mean)" % a.context)

    import rasterio
    g = C.get_grid()
    with rasterio.open(a.labels) as ds:
        L = ds.read(1)
    if L.shape != (g["H"], g["W"]):
        raise SystemExit("labels are not on the 30 m grid: run jute_inspect.py first")
    y_map = np.where(L >= 0, (L == 1).astype(np.int16), -1)     # 1 = jute, 0 = other, -1 = no label
    rr, cc = np.nonzero(y_map >= 0)
    yy = y_map[rr, cc].astype(np.int64)
    print("labelled pixels %d, jute share %.2f %%" % (len(yy), 100 * yy.mean()))

    # feature sets: (name, [tif, ...])
    specs = []
    if a.emb:
        for e in a.emb:
            n, p = e.split("=", 1)
            specs.append((n, p.split("+")))
    else:
        for n in KINDS:
            f = emb_file(C.OUT, n)
            if f:
                specs.append((n, [f]))
    if a.both:
        for n in KINDS:
            p1, p2 = emb_file(a.both[0], n), emb_file(a.both[1], n)
            if p1 and p2:
                specs.append((n + "_both_seasons", [p1, p2]))
    if not specs:
        raise SystemExit("no embedding found in %s" % C.OUT)

    lines, table = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("JUTE BY SIMILARITY   folder %s   labels %s   seeds %s   context radius %d px" % (C.OUT, a.labels, a.seeds, a.context))
    say("jute share of the labelled pixels: %.2f %%   (a map that says 'not jute' everywhere would be %.1f %% accurate)"
        % (100 * yy.mean(), 100 * (1 - yy.mean())))

    for name, paths in specs:
        runs = []
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            is_test = block_split(rr, cc, a.block, seed)
            tr_pool = np.flatnonzero(~is_test)
            # fingerprints from TRAIN blocks: up to n_proto pixels of each class
            j_idx = pick(rng, tr_pool[yy[tr_pool] == 1], a.n_proto)
            o_idx = pick(rng, tr_pool[yy[tr_pool] == 0], a.n_proto)
            # validation sample (natural shares, TRAIN blocks, not the same pixels as the fingerprints) for the threshold
            rest = np.setdiff1d(tr_pool, np.concatenate([j_idx, o_idx]))
            v_idx = pick(rng, rest, a.n_val)
            te_idx = pick(rng, np.flatnonzero(is_test), a.n_test)
            idx = np.concatenate([j_idx, o_idx, v_idx, te_idx])
            order = np.argsort(rr[idx], kind="stable")            # read_features needs rows sorted
            U, ok = load_spec(("emb", paths), rr[idx][order], cc[idx][order])
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            U, ok = U[inv], ok[inv]                                # back to the order of idx
            Z = unit_blocks(U, len(paths))
            n1, n2, n3 = len(j_idx), len(o_idx), len(v_idx)
            cut = np.cumsum([n1, n2, n3])
            Ztr = np.concatenate([Z[:cut[1]]])
            ytr = np.r_[np.ones(n1, np.int64), np.zeros(n2, np.int64)]
            keep_tr = ok[:cut[1]]
            Zv, yv, okv = Z[cut[1]:cut[2]], yy[idx][cut[1]:cut[2]], ok[cut[1]:cut[2]]
            Zt, yt, okt = Z[cut[2]:], yy[idx][cut[2]:], ok[cut[2]:]
            res = evaluate(Ztr[keep_tr], ytr[keep_tr], Zv[okv], yv[okv], Zt[okt], yt[okt])
            res["yt"], res["Zt"], res["yv"] = yt[okt], Zt[okt], yv[okv]
            runs.append(res)

        # ---- numbers of this embedding (mean +- std over the seeds)
        def ms(get):
            v = np.array([get(r) for r in runs], float)
            return v.mean(), v.std()
        say()
        say("=== %s   (%s)" % (name, ", ".join(os.path.basename(p) for p in paths)))
        auc_m, auc_s = ms(lambda r: r["auc"])
        say("ROC-AUC of d (threshold free)  %.3f +- %.3f      (AUC of s_jute alone %.3f)" % (auc_m, auc_s, ms(lambda r: r["auc_sj"])[0]))
        row = dict(name=name, auc=auc_m, auc_std=auc_s)
        for rule in ("nearest", "tuned"):
            vals = {k: ms(lambda r, k=k: r[rule][k]) for k in ("prec", "rec", "f1", "iou", "bal_acc", "acc")}
            say("rule %-8s jute: precision %.3f  recall %.3f  F1 %.3f  IoU %.3f   | balanced acc %.3f  overall acc %.3f   (+- F1 %.3f)"
                % (rule, vals["prec"][0], vals["rec"][0], vals["f1"][0], vals["iou"][0], vals["bal_acc"][0], vals["acc"][0], vals["f1"][1]))
            for k, (m, s) in vals.items():
                row["%s_%s" % (rule, k)] = m
        row.update(f1=row["tuned_f1"], iou=row["tuned_iou"])
        say("tuned threshold t (mean over seeds) %.4f" % np.mean([r["t"] for r in runs]))
        cm = runs[0]["tuned"]["cm"]
        say("confusion matrix, rule tuned, seed %d   (rows = map: other, jute ; columns = predicted: other, jute)" % a.seeds[0])
        say("   %8d %8d" % (cm[0, 0], cm[0, 1]))
        say("   %8d %8d" % (cm[1, 0], cm[1, 1]))
        r0 = runs[0]
        Mx, nj, tab = similarity_matrix(r0["Zt"], r0["yt"], 300, np.random.default_rng(0))
        say("mean cosine similarity   jute-jute %.3f | jute-other %.3f | other-other %.3f"
            % (tab[0, 0], tab[0, 1], tab[1, 1]))
        say("  (jute-jute clearly above jute-other = the embedding keeps the classes apart)")
        picture(os.path.join(C.OUT, "jute_similarity_%s.png" % name), name, Mx, nj, r0, r0["yt"])
        table.append(row)

        # ---- maps of the whole district with the fingerprints of seed 0
        if not a.no_maps:
            t0 = time.time()
            write_maps(paths, r0["p_j"], r0["p_o"], r0["t"],
                       os.path.join(C.OUT, "jute_simmap_%s.tif" % name), os.path.join(C.OUT, "jute_simpred_%s.tif" % name))
            say("maps written: jute_simmap_%s.tif, jute_simpred_%s.tif  (%.0f s)" % (name, name, time.time() - t0))

    say()
    say("SUMMARY (rule 'tuned', TEST blocks)")
    say("%-24s %7s %7s %7s %7s %7s" % ("embedding", "AUC", "prec", "recall", "F1", "IoU"))
    for r in table:
        say("%-24s %7.3f %7.3f %7.3f %7.3f %7.3f" % (r["name"], r["auc"], r["tuned_prec"], r["tuned_rec"], r["tuned_f1"], r["tuned_iou"]))
    with open(os.path.join(C.OUT, "jute_similarity_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(C.OUT, "jute_similarity_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    compare_picture(os.path.join(C.OUT, "jute_similarity_compare.png"), table)
    print("saved jute_similarity_results.txt / .csv / _compare.png in", C.OUT)


if __name__ == "__main__":
    main()