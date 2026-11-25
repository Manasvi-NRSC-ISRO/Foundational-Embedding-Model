"""Can the ORIGINAL observations be rebuilt from the stored 128-D embedding?  (MAE, MSE, RMSE, NRMSE, R2, bias)

    python reconstruct_from_embedding.py --kind barlow                 # Apr-Jun folder in $env:EMB_OUT
    python reconstruct_from_embedding.py --kind barlow --gap           # + tests B (50 % hidden) and C (time hole)
    python reconstruct_from_embedding.py --kind dae --gap
    python reconstruct_from_embedding.py --kind conv                   # conv-AE map (spatial) -> test A only
    python reconstruct_from_embedding.py --kind barlow --epochs 2 --strips 4 2 --per_strip 200   # quick smoke test

WHY A NEW SCRIPT.  eval_dae.py only works for the denoising AE because it needs the AE's own decoder.  The Barlow Twins model
has NO decoder (it learns by comparing two views), and the conv AE writes an embedding of a 5x5 neighbourhood, not of one
pixel's time series.  To compare the three on the SAME footing this script does the following for every kind:

  1. Take the embedding that is really STORED in the tif (the thing a user receives), converted back to z with u_to_z.
  2. Train ONE small decoder (the same `Decoder` class as denoising_autoencoder.py) that maps  stored embedding + (date, sensor)
     -> the 5 features of that scene.  The decoder is trained on pixels of "training strips" and the encoder is frozen.
     (A frozen-encoder decoder is the standard way to measure how much information an embedding keeps: if a small decoder can
     rebuild the series, the information is inside the 128 numbers.)
  3. Score the decoder on pixels of other, spatially separate "test strips" that the encoder never saw in its own training
     (pixels listed in samples.npz are excluded).  Everything is reported in REAL units (reflectance, NDVI, dB).

TESTS
  A  RECONSTRUCTION  embedding = the stored one (the encoder saw ALL valid observations), decoder rebuilds all of them.
  B  GAP FILLING     (OPTIONAL, only with --gap; sir: not required)  the encoder is run again on a random 50 % of the observations; the decoder has to
                     rebuild the other, hidden 50 %.  Scored only on the hidden observations.
  C  TIME HOLE       (OPTIONAL, only with --gap) all observations inside one continuous window of days (--hole, default 20 days, in the middle of
                     the season) are hidden; the decoder has to rebuild exactly that window.  Harder than B because the
                     neighbouring dates are missing too.
  For the DAE the AE's own trained decoder is scored as well ("native", test A), so you can compare it with the generic one.

BASELINES printed next to every score (a score is only meaningful against them)
  scene mean : for every scene and feature the average of the TRAINING pixels (no pixel information at all)
  pixel mean : the average of the pixel's own visible observations of the same sensor and feature (for test A this uses the
               target itself, so it is an optimistic reference; for B and C it only uses the visible observations)

METRICS (t = real, p = rebuilt, over all valid scored observations of one sensor/feature)
  MAE   mean |t-p|          MSE  mean (t-p)^2          RMSE  sqrt(MSE)          bias  mean(p-t)
  NRMSE RMSE divided by a scale so that features with different units are comparable. Two versions are written:
        nrmse_range = RMSE / (p99 - p1 of t)   (default, robust range)   nrmse_std = RMSE / std(t)
        (sir's exact definition is still to be confirmed; both are in the csv.)
  R2    1 - sum (t-p)^2 / sum (t-mean(t))^2   (1 = perfect, 0 = as good as predicting the mean, <0 = worse)

OUTPUT (inside the season folder = $env:EMB_OUT):
  reconstruction_<kind>.txt   readable table          reconstruction_<kind>.csv   same numbers for Excel / docs
  reconstruction_<kind>_overall.csv  scores of ALL bands together (per sensor and all sensors)
  reconstruction_<kind>_scenes.csv   scene-wise scores (one row per scene)
  reconstruction_<kind>_*.png   real vs rebuilt NDVI time series of 4 unseen pixels + scatter plots (test A)
  reconstruction_decoder_<kind>.pt   the trained decoder (use --reuse to skip training and only score)
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

import common as C
from denoising_autoencoder import Decoder

FEAT = {0: ["Green", "Red", "NIR", "SWIR", "NDVI"], 1: ["Green", "Red", "NIR", "SWIR", "NDVI"], 2: ["HH dB", "HV dB"]}
SENSOR_NAME = {0: "LISS-III", 1: "AWiFS", 2: "EOS-04 SAR"}
STRIP = 32          # rows per strip


# --------------------------------------------------------------------------------------------------------------------
# 1) choosing pixels: whole strips of rows, so that train and test pixels are far apart in space
# --------------------------------------------------------------------------------------------------------------------
def choose_strips(g, n_train, n_test, seed):
    """Random, non-overlapping STRIP-row blocks that contain district pixels. First n_train -> train, next n_test -> test."""
    rng = np.random.default_rng(seed)
    H = g["pmask"].shape[0]
    starts = np.arange(0, H - STRIP, STRIP)
    starts = [s for s in starts if g["pmask"][s:s + STRIP].sum() > 2000]      # skip blocks with almost no district
    rng.shuffle(starts)
    return sorted(starts[:n_train]), sorted(starts[n_train:n_train + n_test])


def read_strip_pixels(path, g, starts, per_strip, exclude, seed):
    """For every strip: pick up to per_strip random district pixels (not in `exclude`) with a valid stored embedding.
    Returns flat pixel indices (sorted) and their stored embeddings E (n,128) as float32 in 0..1."""
    import rasterio
    from rasterio.windows import Window
    rng = np.random.default_rng(seed)
    W = g["pmask"].shape[1]
    idx_all, emb_all = [], []
    with rasterio.open(path) as ds:
        for s in starts:
            a = ds.read(window=Window(0, s, W, STRIP))                      # (128, STRIP, W)
            ok = g["pmask"][s:s + STRIP] & C.emb_valid(a)                    # inside district and has an embedding
            rr, cc = np.nonzero(ok)
            flat = (rr + s).astype(np.int64) * W + cc
            if len(exclude):
                keep = ~np.isin(flat, exclude)
                rr, cc, flat = rr[keep], cc[keep], flat[keep]
            if len(flat) == 0:
                continue
            pick = rng.choice(len(flat), min(per_strip, len(flat)), replace=False)
            idx_all.append(flat[pick])
            emb_all.append(a[:, rr[pick], cc[pick]].T.astype(np.float32))    # (n,128)
    idx = np.concatenate(idx_all)
    emb = np.concatenate(emb_all)
    o = np.argsort(idx)
    return idx[o], emb[o]


def load_set(path, g, starts, per_strip, exclude, seed, label):
    """Pixels of the strips with BOTH a stored embedding and >= 10 valid observations -> (idx, E, X, V, doy, sid)."""
    t0 = time.time()
    idx, E = read_strip_pixels(path, g, starts, per_strip, exclude, seed)
    X, V, doy, sid, names = C.read_all_scenes(None, idx)
    keep = V.sum(1) >= 10
    print("%s: %d pixels (%d dropped, < 10 valid observations)  [%.0fs]" % (label, keep.sum(), (~keep).sum(), time.time() - t0),
          flush=True)
    return idx[keep], E[keep], X[keep].astype(np.float32), V[keep], doy, sid, names


# --------------------------------------------------------------------------------------------------------------------
# 2) metrics
# --------------------------------------------------------------------------------------------------------------------
def metrics(t, p):
    """t real, p rebuilt (1-D arrays) -> dict of MAE, MSE, RMSE, NRMSE (two versions), R2, bias."""
    e = p - t
    mse = float((e ** 2).mean())
    rmse = math.sqrt(mse)
    lo, hi = np.percentile(t, [1, 99])
    sd = float(t.std())
    return dict(n=len(t), mae=float(np.abs(e).mean()), mse=mse, rmse=rmse,
                nrmse_range=rmse / max(hi - lo, 1e-9), nrmse_std=rmse / max(sd, 1e-9),
                r2=1 - float((e ** 2).sum()) / (float(((t - t.mean()) ** 2).sum()) + 1e-12), bias=float(e.mean()))


def pixel_mean_baseline(X, V, vis, sid):
    """For every pixel / sensor / feature: mean of its own VISIBLE valid observations (nan where there are none)."""
    out = np.full(X.shape[:2] + (5,), np.nan, np.float32)
    for s in np.unique(sid):
        cols = np.flatnonzero(sid == s)
        m = (V & vis)[:, cols]                                             # (n,Ts)
        cnt = m.sum(1)
        for f in range(5):
            val = (X[:, cols, f] * m).sum(1) / np.maximum(cnt, 1)
            out[:, cols, f] = np.where(cnt > 0, val, np.nan)[:, None]
    return out


# --------------------------------------------------------------------------------------------------------------------
# 3) the decoder on stored embeddings
# --------------------------------------------------------------------------------------------------------------------
def train_decoder(Etr, Xtr, Vtr, sid, doy, mean_t, std_t, fm, epochs, batch, lr):
    """Train Decoder on stored embeddings Etr (n,128 in 0..1). Loss = masked MSE in standardised units (valid obs only)."""
    torch.manual_seed(0)
    n = len(Etr)
    nv = max(n // 20, 1)                                                   # 5 % of the training pixels held out
    Z = torch.from_numpy(C.u_to_z(Etr))
    Y = torch.from_numpy(((Xtr[:, :, :5] - mean_t) / std_t) * fm[None])
    M = torch.from_numpy(Vtr[:, :, None] & fm[None])
    doy_t = torch.from_numpy(doy)
    dec = Decoder(sid)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, (n - nv) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n - nv) + nv
        dec.train()
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - batch + 1, batch):
            b = perm[i:i + batch]
            err = ((dec(Z[b], doy_t) - Y[b]) ** 2) * M[b]
            loss = err.sum() / M[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        dec.eval()
        with torch.no_grad():
            err = ((dec(Z[:nv], doy_t) - Y[:nv]) ** 2) * M[:nv]
            lv = (err.sum() / M[:nv].sum().clamp(min=1)).item()
        print("decoder epoch %2d/%d  train %.4f  held-out %.4f  (%.0fs)" % (ep + 1, epochs, tot / max(nb, 1), lv, time.time() - t0),
              flush=True)
    return dec


def decode(dec, Z, doy, mean_t, std_t, chunk=2000):
    """z (n,128) on the unit-variance scale -> real units (n,T,5)."""
    out = []
    doy_t = torch.from_numpy(doy)
    with torch.no_grad():
        for i in range(0, len(Z), chunk):
            out.append(dec(torch.from_numpy(Z[i:i + chunk]), doy_t).numpy())
    # the DAE's own decoder has 9 outputs (5 satellite + 4 weather): only the first 5 are reconstruction targets
    return np.concatenate(out)[..., :5] * std_t + mean_t



# --------------------------------------------------------------------------------------------------------------------
# 4) extra pictures
# --------------------------------------------------------------------------------------------------------------------
def band_and_scene_plots(rows, scene_rows, A_NAME, base):
    """reconstruction_<kind>_bands.png : R2 of every band vs the scene-mean baseline (test A, generic decoder).
       reconstruction_<kind>_scenes.png: macro R2 of every scene against its day of year, one colour per sensor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = [r for r in rows if r["test"] == A_NAME and r["decoder"].startswith("generic")]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(rr))
    ax.bar(x - 0.2, [r["r2"] for r in rr], 0.4, label="rebuilt from the embedding")
    ax.bar(x + 0.2, [r["scene_mean_r2"] for r in rr], 0.4, label="baseline: scene mean (no pixel information)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (r["sensor"].split()[0], r["feature"]) for r in rr], fontsize=7)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_ylabel("R2 (1 = perfect)")
    ax.set_title("Reconstruction quality per band (unseen pixels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base + "_bands.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for sname, col in (("LISS-III", "tab:green"), ("AWiFS", "tab:blue"), ("EOS-04 SAR", "tab:red")):
        q = [r for r in scene_rows if r["sensor"] == sname]
        ax.scatter([r["doy"] for r in q], [r["r2_macro"] for r in q], s=22, c=col, label=sname)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("day of year of the scene")
    ax.set_ylabel("R2 averaged over the bands of the scene")
    ax.set_title("Reconstruction quality per scene")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base + "_scenes.png", dpi=120)
    plt.close(fig)


def spatial_maps(emb_path, g, dec, sid, doy, mean_t, std_t, tr_starts, names, base, S=256):
    """One SxS patch of the district (not overlapping the decoder-training strips), ALL bands of ONE scene per sensor.
    Writes reconstruction_<kind>_maps_<sensor>.png for LISS-III, AWiFS and EOS-04 SAR. In every picture:
      one row per band (LISS-III / AWiFS: Green, Red, NIR, SWIR, NDVI;  SAR: HH, HV)
      columns = real | rebuilt from the embedding | error (rebuilt - real, with the MAE of that band in the title)
      optical sensors get one extra row: false colour (NIR, Red, Green as R, G, B), real | rebuilt | mean |error| of the 3 bands.
    The scene shown for a sensor is the one with the most valid pixels inside the patch (clear sky). Pixels without an
    observation are white. The full scene names are printed to the console."""
    import rasterio
    from rasterio.windows import Window
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = g["pmask"].shape
    best, bf = None, -1.0
    for r0 in range(0, H - S, 128):
        if any(s < r0 + S and s + STRIP > r0 for s in tr_starts):          # keep away from the decoder's training pixels
            continue
        for c0 in range(0, W - S, 128):
            fr = float(g["pmask"][r0:r0 + S, c0:c0 + S].mean())
            if fr > bf:
                best, bf = (r0, c0), fr
    r0, c0 = best
    print("map patch rows %d-%d cols %d-%d (%.0f %% inside the district)" % (r0, r0 + S, c0, c0 + S, 100 * bf), flush=True)
    win = Window(c0, r0, S, S)
    with rasterio.open(emb_path) as ds:
        e = ds.read(window=win)                                             # (128,S,S)
    ok = g["pmask"][r0:r0 + S, c0:c0 + S] & C.emb_valid(e)
    sel = np.flatnonzero(ok.ravel())
    E = e.reshape(128, -1)[:, sel].T
    X, V, _, _, _ = C.read_all_scenes(win, sel)
    rec = decode(dec, C.u_to_z(E), doy, mean_t, std_t)

    def to_img(vals, valid):
        """values of the selected pixels -> SxS image, NaN where the pixel has no valid observation"""
        im = np.full(S * S, np.nan, np.float32)
        im[sel[valid]] = vals[valid]
        return im.reshape(S, S)

    for s, fname_list in FEAT.items():
        cols = np.flatnonzero(sid == s)
        t = cols[np.argmax(V[:, cols].sum(0))]                              # clearest scene of this sensor in the patch
        v = V[:, t]
        optical = s < 2
        n_rows = len(fname_list) + (1 if optical else 0)
        fig, axs = plt.subplots(n_rows, 3, figsize=(12, 3.6 * n_rows), constrained_layout=True)
        print("%s: scene %s  day %d  %d valid px" % (SENSOR_NAME[s], names[t] if names is not None else t, doy[t], v.sum()))
        imgs = {}
        for row, fname in enumerate(fname_list):
            real, reb = to_img(X[:, t, row], v), to_img(rec[:, t, row], v)
            imgs[fname] = (real, reb)
            lo, hi = np.nanpercentile(real, [2, 98])
            err = reb - real
            em = max(np.nanpercentile(np.abs(err), 98), 1e-6)
            cm = "gray" if not optical else ("RdYlGn" if fname == "NDVI" else "viridis")
            for col, (img, kw, ttl) in enumerate(((real, dict(cmap=cm, vmin=lo, vmax=hi), "real"),
                                                  (reb, dict(cmap=cm, vmin=lo, vmax=hi), "rebuilt"),
                                                  (err, dict(cmap="coolwarm", vmin=-em, vmax=em), "error (rebuilt - real)"))):
                im = axs[row, col].imshow(img, **kw)
                extra = "   MAE %.3f  R2 %.2f" % (np.nanmean(np.abs(err)), 1 - np.nansum(err ** 2) / max(np.nansum((real - np.nanmean(real)) ** 2), 1e-12)) if col == 2 else ""
                axs[row, col].set_title("%s | %s%s" % (fname, ttl, extra), fontsize=9)
                axs[row, col].axis("off")
                fig.colorbar(im, ax=axs[row, col], fraction=0.046)
        if optical:                                                         # false colour: NIR, Red, Green -> R, G, B
            row = n_rows - 1
            order = ("NIR", "Red", "Green")
            lohi = [np.nanpercentile(imgs[b][0], [2, 98]) for b in order]   # stretch of the REAL image, used for both

            def comp(k):
                ch = [np.clip((imgs[b][k] - lohi[i][0]) / max(lohi[i][1] - lohi[i][0], 1e-9), 0, 1) for i, b in enumerate(order)]
                rgb = np.stack(ch, -1)
                rgb[np.isnan(rgb).any(-1)] = 1.0                            # white where no observation
                return rgb
            axs[row, 0].imshow(comp(0)); axs[row, 0].set_title("false colour (NIR,Red,Green) | real", fontsize=9)
            axs[row, 1].imshow(comp(1)); axs[row, 1].set_title("false colour | rebuilt", fontsize=9)
            import warnings
            warnings.filterwarnings('ignore', message='Mean of empty slice')   # white (no-data) pixels are all-NaN
            em3 = np.nanmean([np.abs(imgs[b][1] - imgs[b][0]) for b in order], axis=0)
            im = axs[row, 2].imshow(em3, cmap="magma", vmin=0, vmax=np.nanpercentile(em3, 98))
            axs[row, 2].set_title("mean |error| of the 3 bands", fontsize=9)
            fig.colorbar(im, ax=axs[row, 2], fraction=0.046)
            for c in range(3):
                axs[row, c].axis("off")
        fig.suptitle("%s   scene day %d   %d valid pixels in the patch   (rows %d-%d, cols %d-%d)" % (
            SENSOR_NAME[s], doy[t], v.sum(), r0, r0 + S, c0, c0 + S), fontsize=10)
        fig.savefig(base + "_maps_%s.png" % SENSOR_NAME[s].split()[0].replace("-", ""), dpi=100)
        plt.close(fig)


def fullmap(emb_path, g, dec, sid, doy, mean_t, std_t, pick, names, base, B=96, step=8):
    """Reconstruction map of the WHOLE district for one scene per sensor (the clearest one, chosen from the test pixels).
    For LISS-III, AWiFS and EOS-04 SAR it writes
      reconstruction_<kind>_fullmap_<sensor>.tif : float32 GeoTIFF on the 30 m grid, bands = [rebuilt x F | real x F | error x F]
                                                   (F = 5 features for optical, 2 for SAR; -9999 = no observation / outside district)
      reconstruction_<kind>_fullmap_<sensor>.png : the same, every 8th pixel, one row per band: real | rebuilt | error
      reconstruction_<kind>_fullmap_scores.txt   : R2, MAE, RMSE, bias over ALL valid pixels of that scene of the district
    The district is processed in blocks of B rows (B must be a multiple of `step`): embedding block -> observations of the
    block (read_all_scenes) -> decoder -> the chosen scene only. The decoder is the one trained on the strips, so the
    pixels of the training strips are part of the map (about 6 % of the rows); their share is small and is not removed."""
    import rasterio
    from rasterio.windows import Window
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NOD = -9999.0
    H, W = g["pmask"].shape
    with rasterio.open(emb_path) as ds:
        prof = ds.profile.copy()
    prof.update(count=1, dtype="float32", nodata=NOD, compress="deflate", predictor=3, tiled=True, blockxsize=256,
                blockysize=256, BIGTIFF="YES")
    tag = {0: "LISSIII", 1: "AWiFS", 2: "EOS04"}
    nfeat = {s: len(FEAT[s]) for s in FEAT}
    dst, small, acc = {}, {}, {}
    sh, sw = (H + step - 1) // step, (W + step - 1) // step
    for s, t in pick.items():
        f = nfeat[s]
        p = dict(prof)
        p["count"] = 3 * f
        dst[s] = rasterio.open("%s_fullmap_%s.tif" % (base, tag[s]), "w", **p)
        dst[s].update_tags(scene=names[t] if names is not None else str(t), doy=str(doy[t]),
                           bands="rebuilt x%d, real x%d, error x%d: %s" % (f, f, f, ",".join(FEAT[s])))
        small[s] = np.full((3 * f, sh, sw), np.nan, np.float32)
        acc[s] = np.zeros((f, 6))                                           # n, sum t, sum t^2, sum err^2, sum |err|, sum err
        print("%s: scene %s (day %d)" % (SENSOR_NAME[s], names[t] if names is not None else t, doy[t]), flush=True)
    t0 = time.time()
    with rasterio.open(emb_path) as ds:
        for r0 in range(0, H, B):
            h = min(B, H - r0)
            win = Window(0, r0, W, h)
            e = ds.read(window=win)
            ok = g["pmask"][r0:r0 + h] & C.emb_valid(e)
            sel = np.flatnonzero(ok.ravel())
            blk = {s: np.full((3 * nfeat[s], h * W), NOD, np.float32) for s in pick}
            if len(sel):
                E = e.reshape(128, -1)[:, sel].T
                X, V, _, _, _ = C.read_all_scenes(win, sel)
                rec = decode(dec, C.u_to_z(E), doy, mean_t, std_t)
                for s, t in pick.items():
                    f = nfeat[s]
                    v = V[:, t]
                    px = sel[v]
                    for k in range(f):
                        real, reb = X[v, t, k].astype(np.float32), rec[v, t, k]
                        blk[s][k, px], blk[s][f + k, px], blk[s][2 * f + k, px] = reb, real, reb - real
                        err = reb - real
                        a = acc[s][k]
                        a += [len(real), real.sum(), (real.astype(np.float64) ** 2).sum(), (err.astype(np.float64) ** 2).sum(),
                              np.abs(err).sum(), err.sum()]
            for s in pick:
                b3 = blk[s].reshape(-1, h, W)
                dst[s].write(b3, window=win)
                sm = b3[:, ::step, ::step].copy()
                sm[sm == NOD] = np.nan
                small[s][:, r0 // step: r0 // step + sm.shape[1], :sm.shape[2]] = sm
            print("fullmap rows %d/%d (%.0fs)" % (r0 + h, H, time.time() - t0), end="\r", flush=True)
    print()
    for d in dst.values():
        d.close()
    lines = ["Whole-district reconstruction of ONE scene per sensor (all valid pixels of the district; decoder trained on strips)"]
    for s, t in pick.items():
        f = nfeat[s]
        lines += ["", "%s  scene %s  day %d" % (SENSOR_NAME[s], names[t] if names is not None else t, doy[t]),
                  "%-8s %10s %8s %9s %9s %9s" % ("band", "n pixels", "R2", "MAE", "RMSE", "bias")]
        for k, fname in enumerate(FEAT[s]):
            n, st, stt, sse, sae, sbias = acc[s][k]
            if n < 2:
                continue
            r2 = 1 - sse / max(stt - st * st / n, 1e-12)
            lines.append("%-8s %10d %8.3f %9.4f %9.4f %9s" % (fname, n, r2, sae / n, math.sqrt(sse / n), "%.4f" % (sbias / n)))
        # picture
        sm = small[s]
        fig, axs = plt.subplots(f, 3, figsize=(12, 3.2 * f), constrained_layout=True)
        for k, fname in enumerate(FEAT[s]):
            real, reb, err = sm[f + k], sm[k], sm[2 * f + k]
            lo, hi = np.nanpercentile(real, [2, 98])
            em = max(np.nanpercentile(np.abs(err), 98), 1e-9)
            cm = "gray" if s == 2 else ("RdYlGn" if fname == "NDVI" else "viridis")
            for c, (img, kw, ttl) in enumerate(((real, dict(cmap=cm, vmin=lo, vmax=hi), "real"),
                                                (reb, dict(cmap=cm, vmin=lo, vmax=hi), "rebuilt"),
                                                (err, dict(cmap="coolwarm", vmin=-em, vmax=em), "error (rebuilt - real)"))):
                im = axs[k, c].imshow(img, **kw)
                axs[k, c].set_title("%s | %s" % (fname, ttl), fontsize=9)
                axs[k, c].axis("off")
                fig.colorbar(im, ax=axs[k, c], fraction=0.046)
        fig.suptitle("%s  scene day %d, whole district (every %dth pixel)" % (SENSOR_NAME[s], doy[t], step), fontsize=10)
        fig.savefig("%s_fullmap_%s.png" % (base, tag[s]), dpi=100)
        plt.close(fig)
    txt = "\n".join(lines)
    print(txt)
    open(base + "_fullmap_scores.txt", "w", encoding="utf-8").write(txt + "\n")


# --------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["barlow", "dae", "conv"], default="barlow", help="which embedding tif to test")
    ap.add_argument("--emb", default=None, help="embedding tif (default: the one of --kind in the season folder)")
    ap.add_argument("--model", default=None, help="encoder checkpoint for --gap (default model.pt / model_dae.pt)")
    ap.add_argument("--strips", type=int, nargs=2, default=[30, 10], metavar=("TRAIN", "TEST"),
                    help="number of 32-row strips for decoder training and for testing")
    ap.add_argument("--per_strip", type=int, default=700, help="pixels taken from every strip")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gap", action="store_true", help="also tests B (50 %% hidden) and C (time hole); needs the encoder")
    ap.add_argument("--hole", type=float, default=20.0, help="length in days of the hidden window of test C")
    ap.add_argument("--maps", action="store_true", help="also draw the spatial real/rebuilt/error maps of one patch (reads the stack again, ~2 min)")
    ap.add_argument("--fullmap", action="store_true", help="also write the reconstruction map of the WHOLE district (one scene per sensor; GeoTIFF + png; ~15-25 min)")
    ap.add_argument("--reuse", action="store_true", help="load the saved decoder instead of training again")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    emb_path = a.emb or C.emb_path(a.kind)
    dec_path = os.path.join(C.OUT, "reconstruction_decoder_%s.pt" % a.kind)
    g = C.get_grid()
    used = np.load(os.path.join(C.OUT, "samples.npz"))["idx"]               # pixels the encoder was trained on

    # ---- pixels: spatially separate train / test strips; test pixels were never used to train the encoder -----------------
    tr_starts, te_starts = choose_strips(g, a.strips[0], a.strips[1], a.seed)
    itr, Etr, Xtr, Vtr, doy, sid, _ = load_set(emb_path, g, tr_starts, a.per_strip, np.array([], np.int64), a.seed, "train set")
    ite, Ete, Xte, Vte, _, _, names_all = load_set(emb_path, g, te_starts, a.per_strip, used, a.seed + 1, "test set (unseen)")
    fm = C.feature_mask(sid, targets_only=True)[:, :5]                      # (T,5) features that are reconstructed

    # ---- normalisation per sensor and feature, from the TRAINING pixels only ----------------------------------------------
    ms = np.zeros((3, 5), np.float32)
    ss = np.ones((3, 5), np.float32)
    for s in range(3):
        cols = np.flatnonzero(sid == s)
        for f in range(5):
            if not fm[cols[0], f]:
                continue
            v = Xtr[:, cols, f][Vtr[:, cols]]
            ms[s, f], ss[s, f] = v.mean(), max(v.std(), 1e-6)
    mean_t, std_t = ms[sid], ss[sid]                                        # (T,5)

    # scene-mean baseline (per scene and feature, from the training pixels)
    scene_mean = np.zeros((len(sid), 5), np.float32)
    for t in range(len(sid)):
        v = Vtr[:, t]
        if v.sum() > 5:
            scene_mean[t] = Xtr[v, t, :5].mean(0)
        else:
            scene_mean[t] = ms[sid[t]]

    # ---- decoder ------------------------------------------------------------------------------------------------------------
    if a.reuse and os.path.exists(dec_path):
        dec = Decoder(sid)
        dec.load_state_dict(torch.load(dec_path, weights_only=False))
        dec.eval()
        print("decoder loaded from", dec_path)
    else:
        dec = train_decoder(Etr, Xtr, Vtr, sid, doy, mean_t, std_t, fm, a.epochs, a.batch, a.lr)
        torch.save(dec.state_dict(), dec_path)

    # ---- test scenarios: name -> dict(z = embedding the decoder gets, vis = visible mask, score = mask of scored obs) ---------
    # Test A is the one that is required. B and C (--gap) are optional extras and are OFF by default.
    A_NAME = "A reconstruction (stored embedding, all obs)"
    scen = {A_NAME: dict(z=C.u_to_z(Ete), vis=Vte, score=Vte)}
    native = {}                      # scenario name -> the AE's own trained decoder (DAE only)
    native_norm = None               # (mean, std) the AE's own decoder was trained with
    if a.kind == "dae":
        # The DAE has its own decoder inside model_dae.pt. It is scored on the SAME stored embedding as the generic decoder, so
        # the two can be compared. Its last layer has 9 outputs (5 satellite + 4 weather features); only the first 5 are used.
        try:
            mp = a.model or os.path.join(C.OUT, "model_dae.pt")
            ck = torch.load(mp, weights_only=False)
            nf = ck["decoder"]["net.5.weight"].shape[0]                      # number of output features of the saved decoder
            nd = Decoder(ck["sid"], feats=nf)
            nd.load_state_dict(ck["decoder"])
            nd.eval()
            native[A_NAME] = nd
            native_norm = (np.asarray(ck["mean"])[..., :5], np.asarray(ck["std"])[..., :5])
        except Exception as ex:                                              # the native comparison is optional
            print("native DAE decoder not scored:", ex)
    if a.gap:
        if a.kind == "conv":
            print("test B/C skipped: the conv AE embeds a 5x5 NEIGHBOURHOOD of embeddings, it has no per-pixel time-series encoder.")
        else:
            from embed_to_tif import embed_pixels
            from pixel_encoder import PixelEncoder
            mp = a.model or os.path.join(C.OUT, "model.pt" if a.kind == "barlow" else "model_dae.pt")
            ck = torch.load(mp, weights_only=False)
            enc = PixelEncoder(ck["sid"])
            enc.load_state_dict(ck["model"])
            enc.eval()
            if Xte.shape[1] != len(ck["doy"]):
                raise SystemExit("Model has %d observations per pixel but the stack has %d." % (len(ck["doy"]), Xte.shape[1]))
            rng = np.random.default_rng(0)
            visB = Vte & (rng.random(Vte.shape) < 0.5)                     # B: random half visible
            lo = doy.min() + 0.5 * (doy.max() - doy.min()) - a.hole / 2   # C: window of `hole` days in the middle of the season
            inhole = (doy >= lo) & (doy <= lo + a.hole)
            visC = Vte & ~inhole[None]
            print("test C hides days %.0f .. %.0f  (%d scenes)" % (lo, lo + a.hole, inhole.sum()))
            for name, vis in (("B gap filling (50% hidden)", visB), ("C time hole (%g days hidden)" % a.hole, visC)):
                z = embed_pixels(enc, ck, Xte.astype(np.float16), vis) * math.sqrt(128)   # unit vector x sqrt(128) = stored scale
                scen[name] = dict(z=np.clip(z, -C.Z_LIM, C.Z_LIM), vis=vis, score=Vte & ~vis)

    # ---- score ---------------------------------------------------------------------------------------------------------------
    # Per band (sensor x feature): MAE, MSE, RMSE, NRMSE, R2, bias in real units.  On top of that three summaries are made:
    #   per sensor "ALL bands" and "ALL sensors" =
    #     R2 macro / NRMSE macro : plain average of the per-band values (every band counts equally)
    #     *_z pooled             : every band is first standardised with the TRAINING mean/std of that sensor+feature, so that
    #                              reflectance, NDVI and dB are on one scale; MAE_z, MSE_z, RMSE_z, R2_z are then computed over
    #                              all bands together (R2_z = 1 - SSE / SST, SST taken around each band's own mean).
    #   and a scene-wise table (one row per scene) in reconstruction_<kind>_scenes.csv.
    rows, lines, overall, recs = [], [], [], {}
    sums_head = "%-24s %8s %7s %7s %8s %8s %8s %8s %8s | %9s"

    def acc_new():
        return dict(sse=0.0, sst=0.0, sae=0.0, n=0, bsse=0.0, bsst=0.0, r2=[], nr=[], ns=[], b1r2=[], b2r2=[])

    def acc_add(ac, tz, pz, q, b1z, b1r2, b2r2):
        ac["sse"] += float(((tz - pz) ** 2).sum())
        ac["sst"] += float(((tz - tz.mean()) ** 2).sum())
        ac["sae"] += float(np.abs(tz - pz).sum())
        ac["bsse"] += float(((tz - b1z) ** 2).sum())
        ac["n"] += len(tz)
        ac["r2"].append(q["r2"]); ac["nr"].append(q["nrmse_range"]); ac["ns"].append(q["nrmse_std"])
        ac["b1r2"].append(b1r2); ac["b2r2"].append(b2r2)

    def acc_row(ac):
        n = max(ac["n"], 1)
        return dict(n=ac["n"], r2_macro=float(np.mean(ac["r2"])), r2_z=1 - ac["sse"] / max(ac["sst"], 1e-12),
                    mae_z=ac["sae"] / n, mse_z=ac["sse"] / n, rmse_z=math.sqrt(ac["sse"] / n),
                    nrmse_range_macro=float(np.mean(ac["nr"])), nrmse_std_macro=float(np.mean(ac["ns"])),
                    scene_mean_r2_macro=float(np.mean(ac["b1r2"])), scene_mean_r2_z=1 - ac["bsse"] / max(ac["sst"], 1e-12),
                    pixel_mean_r2_macro=float(np.nanmean(ac["b2r2"])))

    for name, sc in scen.items():
        rec = decode(dec, sc["z"], doy, mean_t, std_t)
        recs[name] = rec
        variants = [("generic decoder on stored embedding", rec)]
        if name in native and native_norm is not None:
            variants.append(("native DAE decoder", decode(native[name], sc["z"], doy, native_norm[0], native_norm[1])))
        pm = pixel_mean_baseline(Xte, Vte, sc["vis"], sid)
        for vname, r in variants:
            lines += ["", "%s  |  %s  |  %s" % (a.kind, name, vname),
                      "%-11s %-6s %8s %7s %8s %8s %8s %8s %8s %8s | %8s %8s %8s %8s" % (
                          "sensor", "feat", "n", "R2", "MAE", "MSE", "RMSE", "NRMSE_r", "NRMSE_s", "bias",
                          "sceneR2", "sceneMAE", "pixR2", "pixMAE")]
            acs = {s: acc_new() for s in FEAT}
            acall = acc_new()
            for s, feats in FEAT.items():
                cols = np.flatnonzero(sid == s)
                m = sc["score"][:, cols]
                for f, fname in enumerate(feats):
                    t = Xte[:, cols, f][m]
                    if len(t) < 50:
                        continue
                    pr = r[:, cols, f][m]
                    q = metrics(t, pr)
                    b1p = np.broadcast_to(scene_mean[cols, f][None], m.shape)[m]
                    b1 = metrics(t, b1p)
                    pb = pm[:, cols, f][m]
                    okp = ~np.isnan(pb)
                    b2 = metrics(t[okp], pb[okp]) if okp.sum() > 50 else dict(r2=float("nan"), mae=float("nan"))
                    lines.append("%-11s %-6s %8d %7.3f %8.4f %8.5f %8.4f %8.3f %8.3f %8.4f | %8.3f %8.4f %8.3f %8.4f" % (
                        SENSOR_NAME[s], fname, q["n"], q["r2"], q["mae"], q["mse"], q["rmse"], q["nrmse_range"],
                        q["nrmse_std"], q["bias"], b1["r2"], b1["mae"], b2["r2"], b2["mae"]))
                    rows.append(dict(kind=a.kind, test=name, decoder=vname, sensor=SENSOR_NAME[s], feature=fname, **q,
                                     scene_mean_r2=b1["r2"], scene_mean_mae=b1["mae"], pixel_mean_r2=b2["r2"],
                                     pixel_mean_mae=b2["mae"]))
                    z = lambda v: (v - ms[s, f]) / ss[s, f]                  # standardise with the training mean / std
                    for ac in (acs[s], acall):
                        acc_add(ac, z(t), z(pr), q, z(b1p), b1["r2"], b2["r2"])
            lines += ["", "OVERALL (all bands)  |  %s  |  %s" % (name, vname),
                      sums_head % ("", "n", "R2mac", "R2_z", "MAE_z", "MSE_z", "RMSE_z", "NRMSEr", "NRMSEs", "sceneR2_z") +
                      "   (R2mac/NRMSE = mean of the bands; *_z = all bands pooled after standardising)"]
            for label, ac in [(SENSOR_NAME[s] + " all bands", acs[s]) for s in FEAT] + [("ALL SENSORS, ALL BANDS", acall)]:
                if ac["n"] == 0:
                    continue
                o = acc_row(ac)
                lines.append(sums_head % (label, o["n"], "%.3f" % o["r2_macro"], "%.3f" % o["r2_z"], "%.3f" % o["mae_z"],
                                          "%.3f" % o["mse_z"], "%.3f" % o["rmse_z"], "%.3f" % o["nrmse_range_macro"],
                                          "%.3f" % o["nrmse_std_macro"], "%.3f" % o["scene_mean_r2_z"]))
                overall.append(dict(kind=a.kind, test=name, decoder=vname, scope=label, **o))

    # ---- scene-wise table (test A, generic decoder): one row per scene ---------------------------------------------------------
    scene_rows = []
    rA = recs[A_NAME]
    for t in range(len(sid)):
        v = Vte[:, t]
        if v.sum() < 50:
            continue
        s = int(sid[t])
        zt, zp, r2f = [], [], {}
        for f, fname in enumerate(FEAT[s]):
            tz = (Xte[v, t, f] - ms[s, f]) / ss[s, f]
            pz = (rA[v, t, f] - ms[s, f]) / ss[s, f]
            zt.append(tz); zp.append(pz)
            r2f[fname] = 1 - float(((tz - pz) ** 2).sum()) / (float(((tz - tz.mean()) ** 2).sum()) + 1e-12)
        e = np.concatenate(zt) - np.concatenate(zp)
        row = dict(scene=t, name=(names_all[t] if names_all is not None else ""), sensor=SENSOR_NAME[s], doy=float(doy[t]),
                   n_pixels=int(v.sum()), mae_z=float(np.abs(e).mean()), rmse_z=float(math.sqrt((e ** 2).mean())),
                   r2_macro=float(np.mean(list(r2f.values()))))
        for fname in ("Green", "Red", "NIR", "SWIR", "NDVI", "HH dB", "HV dB"):
            row["r2_" + fname] = r2f.get(fname, float("nan"))
        scene_rows.append(row)

    head = ["Reconstruction from the embedding  |  kind %s  |  season folder %s" % (a.kind, C.OUT),
            "decoder trained on %d pixels (%d strips), scored on %d unseen pixels (%d strips)" % (
                len(Etr), a.strips[0], len(Ete), a.strips[1]),
            "units: reflectance (0-1), NDVI (-1..1), dB. NRMSE_r = RMSE/(p99-p1), NRMSE_s = RMSE/std. sceneR2/pixR2 = baselines.",
            "scene-wise scores: reconstruction_%s_scenes.csv   overall scores: reconstruction_%s_overall.csv" % (a.kind, a.kind)]
    txt = "\n".join(head + lines)
    print("\n" + txt)
    base = os.path.join(C.OUT, "reconstruction_%s" % a.kind)
    open(base + ".txt", "w", encoding="utf-8").write(txt + "\n")
    for suffix, data in ((".csv", rows), ("_overall.csv", overall), ("_scenes.csv", scene_rows)):
        with open(base + suffix, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    band_and_scene_plots(rows, scene_rows, A_NAME, base)
    if a.fullmap:
        pick = {}                                                           # clearest scene of every sensor in the test pixels
        for s in range(3):
            cols = np.flatnonzero(sid == s)
            pick[s] = cols[np.argmax(Vte[:, cols].sum(0))]
        fullmap(emb_path, g, dec, sid, doy, mean_t, std_t, pick, names_all, base)
    if a.maps:
        spatial_maps(emb_path, g, dec, sid, doy, mean_t, std_t, tr_starts, names_all, base)

    # ---- pictures: NDVI time series of 4 unseen pixels + scatter plots of test A -----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    opt = np.flatnonzero(sid < 2)
    order = opt[np.argsort(doy[opt])]
    good = np.flatnonzero(Vte[:, order].sum(1) >= 25)[:4]
    names = list(recs)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, p in zip(axs.ravel(), good):
        v = Vte[p, order]
        d = doy[order]
        ax.plot(d[v], Xte[p, order, 4][v], "o", c="k", ms=4, label="real")
        ax.plot(d[v], recs[names[0]][p, order, 4][v], "-", c="tab:orange", lw=1, label="rebuilt, test A")
        for nm, col in zip(names[1:], ("tab:red", "tab:blue")):
            h = scen[nm]["score"][p, order]
            ax.plot(d[h], recs[nm][p, order, 4][h], "x", c=col, ms=7, label="rebuilt, " + nm.split(" (")[0])
        ax.set_title("unseen pixel %d" % ite[p])
        ax.set_xlabel("day of year")
        ax.set_ylabel("NDVI")
    if axs[0, 0].get_legend_handles_labels()[0]:
        axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(base + "_timeseries.png", dpi=110)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (s, f, fname) in zip(axs, ((0, 1, "LISS-III Red"), (0, 2, "LISS-III NIR"), (1, 4, "AWiFS NDVI"), (2, 0, "SAR HH dB"))):
        cols = np.flatnonzero(sid == s)
        m = Vte[:, cols]
        t = Xte[:, cols, f][m]
        p = recs[names[0]][:, cols, f][m]
        sel = np.random.default_rng(0).choice(len(t), min(20000, len(t)), replace=False)
        ax.hexbin(t[sel], p[sel], gridsize=45, bins="log", cmap="viridis", mincnt=1)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "r-", lw=0.8)
        ax.set_title("%s  R2 %.3f" % (fname, metrics(t, p)["r2"]))
        ax.set_xlabel("real")
        ax.set_ylabel("rebuilt")
    fig.tight_layout()
    fig.savefig(base + "_scatter.png", dpi=110)
    print("\nsaved %s .txt, .csv, _overall.csv, _scenes.csv, _timeseries.png, _scatter.png, _bands.png, _scenes.png%s" % (os.path.basename(base), ", _maps_LISSIII.png, _maps_AWiFS.png, _maps_EOS04.png" if a.maps else ""))


if __name__ == "__main__":
    main()