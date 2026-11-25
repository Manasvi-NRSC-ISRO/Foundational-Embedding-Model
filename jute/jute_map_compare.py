r"""Picture of the jute classification: the label map, the map predicted from an embedding, and where they differ.

    $env:EMB_OUT="outputs\2025_Apr_Jun"
    python jute_classify.py --jute_values 1 --seeds 0 --no_raw --predict      # writes jute_pred_<name>.tif
    python jute_map_compare.py                                                # -> jute_compare_<name>.png

Green = jute in both, red = predicted jute but the map says not (false alarm), blue = jute in the map but missed.
Note: the model was trained on a few 3 km blocks; the picture shows the whole district, so numbers in jute_results.txt
(unseen blocks only) are the honest score, the picture is for looking.

NEW (17 Nov): besides the picture the script now COUNTS PIXELS and writes jute_pixel_counts_<name>.txt / .csv:
for the whole district, for the TRAIN blocks and for the TEST blocks separately it gives the number of pixels, the true
jute pixels found (TP), the false alarms (FP, predicted jute but the map says not), the misses (FN, jute in the map but not
predicted), the true "not jute" (TN) and precision, recall, F1, IoU and accuracy. The numbers of pixels used to build the
fingerprints / to tune the threshold / to test are printed as well. Which blocks are train and which are test is found with
the SAME function as in jute_similarity.py / jute_classify.py (block_split), so --split_labels must be the label file that
was given to jute_similarity.py as --labels and --seed the seed of the maps (0, the first seed of the run):
    # new maps (fingerprints learned on the CLEAN labels), counted against the FULL reference map and the CLEAN labels
    python jute\jute_map_compare.py --labels outputs\jute_labels_30m.tif --split_labels outputs\inspect_check\jute_labels_30m_pure.tif
    # old maps of 11 Nov (all-pixel labels), moved to old_maps\ ; only the counts, no picture
    python jute\jute_map_compare.py --pred outputs\2025_Apr_Jun\old_maps\jute_simpred_barlow.tif --counts_only

NEW (18 Nov): --sweep shows what happens when the decision threshold of the similarity method is moved. jute_similarity.py
writes, next to jute_simpred_<name>.tif (jute where d > t), the score map jute_simmap_<name>.tif (d = cos to the jute
fingerprint - cos to the other fingerprint, for every pixel). With --sweep the script reads that score map and counts, for a
list of thresholds, the true jute (TP), the false alarms (FP) and the misses (FN) of the WHOLE district against --labels, and
it also gives the threshold that is needed to find 60, 70, 80 and 90 % of the jute (recall target) with the precision that
this costs. A lower threshold finds more jute (fewer misses) but gives more false alarms. Nothing is trained and no other file
is overwritten (only jute_threshold_sweep_<name>.txt / .csv are written):
    python jute\jute_map_compare.py --labels outputs\jute_labels_30m.tif --pred outputs\2025_Apr_Jun\jute_simpred_barlow.tif --sweep --tuned_t 0.0462
"""
import argparse
import glob
import os
import sys

import numpy as np

# jute\ sub-folder: common.py is one folder up, so that folder is added to the import path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


def test_block_flags(Ls, block, seed):
    """For EVERY pixel of the raster: +1 = the pixel lies in a TEST block, 0 = in a TRAIN block, -1 = in a block without any
    labelled pixel (cannot be classified as train or test). The blocks and the random choice of the test blocks are exactly
    those of jute_similarity.py / jute_classify.py: block_split() is called with the coordinates of the labelled pixels of the
    label file that was used for the training (Ls >= 0), the same seed and the same block size, so the train/test blocks are
    identical to the ones of the run that produced the maps. Returns an int8 array of the shape of Ls."""
    from jute_classify import block_split              # the very same function (jute\ is the folder of this script)
    rr, cc = np.nonzero(Ls >= 0)
    is_test_l = block_split(rr, cc, block, seed)       # True for labelled pixels in a test block
    bid_l = (rr // block).astype(np.int64) * 100000 + (cc // block)
    ub, first = np.unique(bid_l, return_index=True)    # one entry per block that holds at least one labelled pixel
    test_of_block = is_test_l[first]                   # is that block a test block?
    H, W = Ls.shape
    R, Cc = np.indices((H, W))
    bid_all = (R // block).astype(np.int64) * 100000 + (Cc // block)
    idx = np.clip(np.searchsorted(ub, bid_all), 0, len(ub) - 1)
    found = ub[idx] == bid_all
    flag = np.where(found, np.where(test_of_block[idx], 1, 0), -1).astype(np.int8)
    return flag, int((~test_of_block).sum()), int(test_of_block.sum())


def count_block(P, L, sel):
    """Pixel counts of the prediction P (1 = jute, 0 = not jute, 255 = no data) against the labels L (1 = jute, 0 = other,
    -1 = no label) inside the pixels selected by the boolean array `sel`. FP = false alarms, FN = misses."""
    ok = sel & (P != 255) & (L >= 0)
    pr, lab = (P == 1), (L == 1)
    tp = int((ok & pr & lab).sum())
    fp = int((ok & pr & ~lab).sum())
    fn = int((ok & ~pr & lab).sum())
    tn = int((ok & ~pr & ~lab).sum())
    n = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(pixels=n, jute_in_map=tp + fn, TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec,
                F1=(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
                IoU=tp / max(tp + fp + fn, 1), accuracy=(tp + tn) / max(n, 1))


def write_counts(P, Lcmp, Lsplit, block, seed, name, label_note, out_base, n_proto=20000, n_val=50000, n_test=100000):
    """Counts the pixels for the whole district, the train blocks and the test blocks, against the comparison labels Lcmp
    (and, when it is a different file, also against Lsplit, the labels the model was trained with), prints the table and
    writes it to <out_base>.txt and <out_base>.csv."""
    import csv
    flag, n_tr_blocks, n_te_blocks = test_block_flags(Lsplit, block, seed)
    scopes = [("whole district", np.ones(P.shape, bool)), ("TRAIN blocks", flag == 0), ("TEST blocks", flag == 1)]
    label_sets = [("labels of the comparison: " + label_note[0], Lcmp)]
    if Lsplit is not Lcmp:
        label_sets.append(("labels of the training: " + label_note[1], Lsplit))
    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("PIXEL COUNTS of %s   (block %d px = %.1f km, seed %d)" % (name, block, block * 0.03, seed))
    say("blocks: %d train blocks, %d test blocks (test blocks are never used for fingerprints, threshold or classifier)"
        % (n_tr_blocks, n_te_blocks))
    # numbers of pixels that the run itself used (the same sampling rules as in jute_similarity.py)
    lt = Lsplit
    n_j = int(((lt == 1) & (flag == 0)).sum())
    n_o = int(((lt == 0) & (flag == 0)).sum())
    say("labelled pixels (training labels): train blocks %d (jute %d, other %d), test blocks %d"
        % (n_j + n_o, n_j, n_o, int(((lt >= 0) & (flag == 1)).sum())))
    say("used by the similarity method: fingerprint = up to %d jute + %d other pixels of the train blocks (here %d + %d), "
        "threshold tuned on up to %d further train pixels, score on up to %d test pixels"
        % (n_proto, n_proto, min(n_proto, n_j), min(n_proto, n_o), n_val, n_test))
    say("(the whole-district counts below use EVERY pixel, not a sample)")
    for lname, Lx in label_sets:
        say()
        say(lname)
        say("%-15s %10s %9s %9s %9s %10s %8s %7s %7s %7s %8s" % (
            "region", "pixels", "jute(map)", "TP", "FP=false", "FN=missed", "TN", "prec", "recall", "F1", "IoU"))
        for sname, sel in scopes:
            r = count_block(P, Lx, sel)
            say("%-15s %10d %9d %9d %9d %10d %8d %7.3f %7.3f %7.3f %8.3f" % (
                sname, r["pixels"], r["jute_in_map"], r["TP"], r["FP"], r["FN"], r["TN"], r["precision"], r["recall"],
                r["F1"], r["IoU"]))
            rows.append(dict(labels=lname, region=sname, **r))
    with open(out_base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out_base + ".txt /.csv")


def threshold_sweep(D, Lc, out_base, tuned_t=None, bin_w=0.0005):
    """D = score map (float32, -9999 = no data), Lc = labels as 1 = jute, 0 = other, -1 = no label (both full resolution).
    The scores of the jute pixels and of the other pixels are put in two histograms with very narrow bins (0.0005), so that the
    counts for ANY threshold follow from a cumulative sum (exact up to the bin width) without a loop over the 13 million pixels.
    A pixel is predicted as jute when d > t, as in jute_similarity.py. Prints the table and writes it to out_base.txt / .csv."""
    import csv
    ok = (D > -9000) & (Lc >= 0)
    lab = Lc == 1
    edges = np.arange(-2.0, 2.0 + bin_w, bin_w)                 # d lies between -2 and 2 (difference of two cosines)
    hj, _ = np.histogram(D[ok & lab], bins=edges)               # jute pixels
    ho, _ = np.histogram(D[ok & ~lab], bins=edges)              # other pixels
    tp_c = np.cumsum(hj[::-1])[::-1]                            # tp_c[i] = jute pixels with d >= edges[i]
    fp_c = np.cumsum(ho[::-1])[::-1]
    nj = int(hj.sum())
    lines, rows = [], []

    def say(x=""):
        print(x)
        lines.append(x)

    def at(t):
        i = int(np.clip(round((t - edges[0]) / bin_w), 0, len(edges) - 2))
        tp, fp = int(tp_c[i]), int(fp_c[i])
        fn = nj - tp
        pr, rc = tp / max(tp + fp, 1), tp / max(nj, 1)
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
        return dict(threshold=round(float(edges[i]), 4), TP=tp, FP=fp, FN=fn, precision=pr, recall=rc, F1=f1,
                    jute_missed_pct=100.0 * fn / max(nj, 1))

    say("THRESHOLD SWEEP of the whole district (jute pixels in the reference map: %d)" % nj)
    say("lower threshold = more jute found (fewer misses) but more false alarms")
    say("%-10s %10s %10s %10s %8s %8s %7s %12s" % ("threshold", "TP", "FP=false", "FN=missed", "prec", "recall", "F1", "jute missed"))
    grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    if tuned_t is not None:
        grid = sorted(set(grid + [round(float(tuned_t), 4)]))
    for t in grid:
        r = at(t)
        mark = "   <- threshold of the similarity run (best F1 on validation pixels)" if tuned_t is not None and abs(t - tuned_t) < 1e-9 else ""
        say("%-10.4f %10d %10d %10d %8.3f %8.3f %7.3f %11.1f%%%s" % (r["threshold"], r["TP"], r["FP"], r["FN"], r["precision"],
                                                                     r["recall"], r["F1"], r["jute_missed_pct"], mark))
        rows.append(dict(kind="grid", **r))
    say()
    say("What the threshold has to be to find a given share of the jute (and what it costs):")
    say("%-14s %10s %10s %10s %10s %8s %7s" % ("recall target", "threshold", "TP", "FP=false", "FN=missed", "prec", "F1"))
    rec = tp_c / max(nj, 1)
    for target in (0.6, 0.7, 0.8, 0.9):
        idx = np.flatnonzero(rec >= target)
        if len(idx) == 0:
            continue
        i = int(idx[-1])                                         # the highest threshold that still reaches the target
        r = at(edges[i])
        say("%-14s %10.4f %10d %10d %10d %8.3f %7.3f" % ("%.0f %%" % (100 * target), r["threshold"], r["TP"], r["FP"], r["FN"],
                                                         r["precision"], r["F1"]))
        rows.append(dict(kind="recall %.0f%%" % (100 * target), **r))
    with open(out_base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out_base + ".txt /.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("outputs", "jute_labels_30m.tif"))
    ap.add_argument("--step", type=int, default=4, help="show every Nth pixel")
    ap.add_argument("--split_labels", default="",
                    help="label file that was given to jute_similarity.py / jute_classify.py as --labels when the maps were "
                         "made (it decides which blocks are train and which are test); empty = same as --labels")
    ap.add_argument("--seed", type=int, default=0, help="seed of the block split of the maps (first seed of the run)")
    ap.add_argument("--block", type=int, default=100, help="block size in pixels (100 = 3 km), as in the run")
    ap.add_argument("--pred", default="", help="count / draw only this prediction tif (e.g. a map in old_maps\\)")
    ap.add_argument("--counts_only", action="store_true", help="write the pixel counts only, no picture")
    ap.add_argument("--sweep", action="store_true",
                    help="threshold sweep: reads jute_simmap_<name>.tif (next to --pred) and counts TP / FP / FN for many thresholds")
    ap.add_argument("--tuned_t", type=float, default=None, help="threshold printed by jute_similarity.py (marked in the sweep table)")
    a = ap.parse_args()
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # maps of the trained classifier (jute_pred_<name>.tif, from jute_classify.py --predict) AND the maps of the similarity
    # method (jute_simpred_<name>.tif, from jute_similarity.py); the similarity ones get "sim_" in front of their name
    preds = sorted(glob.glob(os.path.join(C.OUT, "jute_pred_*.tif"))) + \
        sorted(glob.glob(os.path.join(C.OUT, "jute_simpred_*.tif")))
    if a.pred:
        preds = [a.pred]                      # one given map only (for example one moved to old_maps\\)
    if not preds:
        raise SystemExit("no jute_pred_*.tif or jute_simpred_*.tif in %s: run jute_similarity.py (or jute_classify.py --predict) first" % C.OUT)
    s = a.step
    with rasterio.open(a.labels) as ds:
        Lfull = ds.read(1)                    # full resolution: the pixel counts use every pixel
    if a.split_labels and os.path.abspath(a.split_labels) != os.path.abspath(a.labels):
        with rasterio.open(a.split_labels) as ds:
            Lsplit = ds.read(1)
    else:
        Lsplit = Lfull                        # the maps were made with the same labels as the comparison
    L = Lfull[::s, ::s]
    lab = L == 1
    for p in preds:
        base = os.path.basename(p)[:-4]
        name = ("sim_" + base[len("jute_simpred_"):]) if base.startswith("jute_simpred_") else base[len("jute_pred_"):]
        if a.sweep:
            # score map of the same run: jute_simpred_<name>.tif -> jute_simmap_<name>.tif (same folder)
            mp = os.path.join(os.path.dirname(p), os.path.basename(p).replace("jute_simpred_", "jute_simmap_"))
            if not os.path.exists(mp):
                raise SystemExit("score map %s not found (made by jute_similarity.py without --no_maps)" % mp)
            with rasterio.open(mp) as ds:
                D = ds.read(1)
            Lc0 = np.where(Lfull >= 0, (Lfull == 1).astype(np.int16), -1)
            threshold_sweep(D, Lc0, os.path.join(os.path.dirname(p), "jute_threshold_sweep_%s" % name), a.tuned_t)
            continue                          # the sweep needs no counts and no picture
        with rasterio.open(p) as ds:
            Pfull = ds.read(1)
        # ---- pixel counts: whole district / train blocks / test blocks, all pixels (no thinning)
        # labels are turned into 1 = jute, 0 = other, -1 = no label, as in jute_similarity.py
        def to01(X):
            return np.where(X >= 0, (X == 1).astype(np.int16), -1)
        Lc = to01(Lfull)
        Ls = Lc if Lsplit is Lfull else to01(Lsplit)      # the same object when both label files are the same file
        write_counts(Pfull, Lc, Ls, a.block, a.seed, name,
                     (os.path.basename(a.labels), os.path.basename(a.split_labels or a.labels)),
                     os.path.join(C.OUT, "jute_pixel_counts_%s" % name))
        if a.counts_only:
            continue
        P = Pfull[::s, ::s]
        valid = (P != 255) & (L >= 0)
        pr = (P == 1) & valid
        tp, fp, fn = pr & lab, pr & ~lab, ~pr & lab & valid
        img = np.ones(P.shape + (3,), np.float32)
        img[~valid] = 0.85
        img[valid & ~lab & ~pr] = (1, 1, 1)
        img[tp] = (0.1, 0.65, 0.2)
        img[fp] = (0.9, 0.15, 0.15)
        img[fn] = (0.15, 0.35, 0.9)
        two = np.ones(P.shape + (3,), np.float32)
        two[~valid] = 0.85
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        two[valid & lab] = (0.1, 0.65, 0.2)
        ax[0].imshow(two)
        ax[0].set_title("jute map (label)")
        two2 = np.ones(P.shape + (3,), np.float32)
        two2[~valid] = 0.85
        two2[valid & pr] = (0.1, 0.65, 0.2)
        ax[1].imshow(two2)
        ax[1].set_title("predicted from the %s embedding" % name)
        ax[2].imshow(img)
        ax[2].set_title("green = right jute | red = false alarm | blue = missed")
        for x in ax:
            x.axis("off")
        out = os.path.join(C.OUT, "jute_compare_%s.png" % name)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved", out, " (jute pixels: label %d, predicted %d, both %d)" % (lab[valid].sum(), pr.sum(), tp.sum()))


if __name__ == "__main__":
    main()