r"""Sanity checks for an embedding tif (no GPU / no torch needed, only numpy + rasterio).

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python validate_embeddings.py --kind barlow        # or dae / conv ; or give --emb <file.tif>

Everything is computed on a coarse sample of the district (every --step-th row/column), so it runs in about a minute.
Each check prints PASS / WARN / FAIL with the number behind it. What the checks mean:

 1. FORMAT      128 bands, float32, nodata -1, same grid (size, CRS, origin) as the satellite stack.
 2. NODATA      nodata (-1) must be exactly the pixels outside the district polygon, nothing inside it.
 3. RANGE       valid values lie in 0..1; only a tiny share may sit on the clip limits (0 or 1).
 4. DEAD BANDS  no band may be (almost) constant; a constant band carries no information.
 5. UNIT NORM   decoding z = u*8-4 and dividing by sqrt(128) must give vectors of length about 1.
 6. DIMENSION   how many principal components explain 90 % of the variance (1 = collapsed model, 128 = pure noise).
 7. SPATIAL     cosine similarity of neighbouring pixels must be clearly higher than that of random pixel pairs
                (a real embedding is spatially coherent; a random one is not).
 8. SPECTRAL    pairs of pixels with similar season-mean reflectance must be more similar in embedding space:
                correlation between embedding cosine and (negative) satellite-feature distance must be positive.

NEW (18 Nov) - two optional checks that do NOT need any jute label (they test the embedding as a general-purpose
representation, not one application). Both read the raw satellite scenes of a small sample of pixels (12 strips of 8 rows
spread over the district, 2000 pixels per strip), so they need the stack / cloud-mask / weather files of the season, and the
first one also needs the trained model (model.pt in the season folder, or --model):

 9. ROBUSTNESS  (--robust)  The pixel's embedding is made again from FEWER observations and compared (cosine) with the
                embedding from ALL its observations: 75 %, 50 % and 25 % of the valid scenes kept at random (= more
                cloud), and whole sensors removed (no SAR; only AWiFS; only LISS-III). A good general embedding changes
                little. The number is compared with the cosine of two RANDOM different pixels (the floor): the line also
                prints 'kept' = (cos - random) / (1 - random), 1 = unchanged, 0 = as different as a random pixel.
10. PROBE       (--probe)   Several different tasks on the SAME embedding, with labels made from the satellite data itself
                (no reference map): season-mean NDVI, season-maximum NDVI, share of clear optical scenes that look like
                water (NDWI > 0), mean SWIR reflectance, mean SAR HH. A ridge regression (linear, closed form) is learned
                on 8 strips and tested on 4 other strips (different places). R2 = 1 perfect, 0 = no better than the mean.
                Note: these targets are made from the same inputs the model saw, so this shows that the embedding KEEPS
                this information in a simple, linear-readable form; it is not an independent ground truth.

    python validate_embeddings.py --kind barlow --robust --probe
"""
import argparse
import os

import numpy as np

import common as C

OK, WARN, BAD = "PASS", "WARN", "FAIL"


def report(level, name, text):
    """Print one result line."""
    print("[%s] %-11s %s" % (level, name, text))


def cos_sim(a, b):
    """Row-wise cosine similarity of two (n, 128) arrays."""
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return (a * b).sum(1) / (na * nb)


def sample_strips(g, n_strips=12, rows=8, per_strip=2000, seed=0):
    """Read the raw observations of a small, spread-out sample of pixels.

    The district is cut into n_strips horizontal strips of `rows` rows, spread evenly from the first to the last row that
    holds district pixels. In every strip `per_strip` random pixels INSIDE the district polygon are taken. For these pixels all
    scenes of all sensors are read with common.read_all_scenes (same cloud mask / fill / weather logic as in training).

    Returns a dict with
        X (n,T,9) float16, V (n,T) bool, doy (T,), sid (T,)   the observation sets
        strip (n,)  strip number of each pixel (used to split train / test places in the probe)
        row, col (n,) pixel position in the grid (to read the stored embedding at the same pixels)
    """
    from rasterio.windows import Window
    rng = np.random.default_rng(seed)
    W = g["W"]
    inside_rows = np.flatnonzero(g["pmask"].any(axis=1))
    r0s = np.linspace(inside_rows.min(), inside_rows.max() - rows, n_strips).astype(int)
    out = dict(X=[], V=[], strip=[], row=[], col=[])
    doy = sid = None
    for k, r0 in enumerate(r0s):
        flat = np.flatnonzero(g["pmask"][r0:r0 + rows].ravel())         # polygon pixels of this strip (flat index in the strip)
        if len(flat) == 0:
            continue
        sel = np.sort(rng.choice(flat, min(per_strip, len(flat)), replace=False))
        X, V, doy, sid, _ = C.read_all_scenes(Window(0, int(r0), W, rows), sel)
        keep = V.sum(1) >= 1                                           # pixels with no valid observation have no embedding
        out["X"].append(X[keep])
        out["V"].append(V[keep])
        out["strip"].append(np.full(int(keep.sum()), k))
        out["row"].append((r0 + sel // W)[keep])
        out["col"].append((sel % W)[keep])
        print("  strip %d/%d read (%d pixels)" % (k + 1, len(r0s), int(keep.sum())), end="\r", flush=True)
    print()
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["doy"], res["sid"] = doy, sid
    return res


def robustness(embed_fn, X, V, sid, rng):
    """Check 9. embed_fn(X, V) -> (n,128) UNIT vectors (the trained encoder). Returns a list of (label, mean cosine, 5th pct,
    kept, n pixels) rows; also the cosine floor of random pixel pairs."""
    z0 = embed_fn(X, V)                                                # reference: all valid observations
    perm = rng.permutation(len(z0))
    floor = float((z0 * z0[perm]).sum(1).mean())                       # cosine of two unrelated pixels
    tests = []
    for frac in (0.75, 0.50, 0.25):                                    # random scenes hidden per pixel (like more cloud)
        tests.append(("%d %% of the valid scenes kept" % round(100 * frac), V & (rng.random(V.shape) < frac)))
    names = {0: "LISS-III only (no AWiFS, no SAR)", 1: "AWiFS only (no LISS-III, no SAR)"}
    tests.append(("SAR removed (optical only)", V & (sid != 2)[None, :]))
    for k, nm in names.items():
        if (sid == k).any():
            tests.append((nm, V & (sid == k)[None, :]))
    rows = []
    for label, Vs in tests:
        ok = Vs.sum(1) >= 1                                            # a pixel with nothing left cannot be embedded
        if ok.sum() < 100:
            continue
        z = embed_fn(X[ok], Vs[ok])
        c = (z0[ok] * z).sum(1)
        rows.append((label, float(c.mean()), float(np.percentile(c, 5)), float((c.mean() - floor) / (1 - floor)), int(ok.sum())))
    return rows, floor


def probe_targets(X, V, sid):
    """Labels for the probe, made from the satellite observations of each pixel (no reference map). Returns {name: (values, ok)}
    where ok marks the pixels for which the target exists (e.g. needs >= 3 clear optical scenes)."""
    Xf = X.astype(np.float32)
    opt = V & (sid < 2)[None, :]                                       # valid optical observations (LISS-III, AWiFS)
    n_opt = opt.sum(1)
    safe = np.maximum(n_opt, 1)
    ndvi = Xf[:, :, 4]
    green, nir, swir = Xf[:, :, 0], Xf[:, :, 2], Xf[:, :, 3]
    ndwi = (green - nir) / (green + nir + 1e-6)                        # > 0 = water-like
    t = {}
    ok_o = n_opt >= 3
    t["NDVI season mean"] = ((ndvi * opt).sum(1) / safe, ok_o)
    t["NDVI season max"] = (np.where(opt, ndvi, -9.0).max(1), ok_o)
    t["water-like share (NDWI>0)"] = (((ndwi > 0) & opt).sum(1) / safe, ok_o)
    t["SWIR season mean"] = ((swir * opt).sum(1) / safe, ok_o)
    sar = V & (sid == 2)[None, :]
    if (sid == 2).any():
        t["SAR HH mean (dB)"] = ((Xf[:, :, 0] * sar).sum(1) / np.maximum(sar.sum(1), 1), sar.sum(1) >= 2)
    return t


def ridge_r2(Ztr, ytr, Zte, yte, lam=1.0):
    """Linear (ridge) regression in closed form on standardised features.

    Returns (r2_test, r2_train, r_test):
      r2_test   R2 on the test pixels (1 = perfect, 0 = as good as always answering the TRAIN mean, below 0 = worse than that)
      r2_train  R2 on the training pixels (if this is high and r2_test is low, the fit does not carry over to other places)
      r_test    Pearson correlation between prediction and truth on the test pixels. It does not care about a shift or a
                different scale of the target between places, so r_test high with r2_test negative means the embedding
                ORDERS the pixels correctly but the relation differs from one area to the other (a calibration problem,
                not a missing-information problem).
    """
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A, B = (Ztr - mu) / sd, (Zte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) * 1e-3, A.T @ (ytr - ym))
    ptr, pte = A @ w + ym, B @ w + ym

    def r2(y, p):
        return float(1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))

    r = float(np.corrcoef(pte, yte)[0, 1]) if yte.std() > 0 and pte.std() > 0 else float("nan")
    return r2(yte, pte), r2(ytr, ptr), r


def main():
    import rasterio
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="barlow", choices=["barlow", "dae", "conv"])
    ap.add_argument("--emb", default=None, help="embedding tif (default: the one of --kind in the season folder)")
    ap.add_argument("--step", type=int, default=8, help="sampling step in pixels (rows and columns)")
    ap.add_argument("--pairs", type=int, default=20000, help="number of random pixel pairs")
    ap.add_argument("--robust", action="store_true", help="check 9: embedding from fewer scenes / fewer sensors (needs torch + model)")
    ap.add_argument("--probe", action="store_true", help="check 10: several linear tasks on the embedding, no reference map")
    ap.add_argument("--model", default=None, help="model.pt for --robust (default: model.pt in the season folder)")
    ap.add_argument("--strips", type=int, default=12, help="strips of raw data for --robust / --probe")
    ap.add_argument("--per_strip", type=int, default=2000, help="pixels per strip for --robust / --probe")
    a = ap.parse_args()
    emb_path = a.emb or C.emb_path(a.kind)
    rng = np.random.default_rng(0)
    g = C.get_grid()
    inside = g["pmask"]                                              # (H, W) True inside the district polygon

    with rasterio.open(emb_path) as ds:
        # ---- 1. FORMAT -------------------------------------------------------------------------------------------
        ok = ds.count == 128 and ds.dtypes[0] == "float32" and ds.nodata == -1 and (ds.height, ds.width) == (g["H"], g["W"])
        report(OK if ok else BAD, "FORMAT", "bands %d, dtype %s, nodata %s, size %dx%d, crs %s"
               % (ds.count, ds.dtypes[0], ds.nodata, ds.width, ds.height, ds.crs))
        oh, ow = -(-ds.height // a.step), -(-ds.width // a.step)    # size of the decimated sample grid
        ds_h, ds_w = ds.height, ds.width
        E = ds.read(out_shape=(ds.count, oh, ow)).astype(np.float32)  # (128, oh, ow) nearest-neighbour sample
    # rasterio's decimated read (out_shape) takes, for output cell k, the source pixel at floor((k + 0.5) * scale), i.e. the
    # CENTRE of each block, not the first pixel of the block. The polygon mask must be sampled at exactly the same rows and
    # columns, otherwise pixels on the district border are compared with the wrong mask pixel and the NODATA check gives
    # a false alarm (about 0.5 % of the sample, all on the border).
    ri = np.minimum(((np.arange(oh) + 0.5) * ds_h / oh).astype(int), ds_h - 1)
    ci = np.minimum(((np.arange(ow) + 0.5) * ds_w / ow).astype(int), ds_w - 1)
    inside_s = inside[np.ix_(ri, ci)]
    valid = E[0] != -1                                              # sampled pixels that carry an embedding

    # ---- 2. NODATA -----------------------------------------------------------------------------------------------
    miss_in = int((inside_s & ~valid).sum())                        # polygon pixels without embedding (bad)
    extra_out = int((~inside_s & valid).sum())                      # embedding outside polygon (bad)
    lvl = OK if miss_in == 0 and extra_out == 0 else BAD
    report(lvl, "NODATA", "inside polygon without embedding: %d, outside polygon with embedding: %d (of %d sample pixels)"
           % (miss_in, extra_out, inside_s.sum()))

    U = E[:, valid].T                                               # (n, 128) stored values u in 0..1
    # ---- 3. RANGE ------------------------------------------------------------------------------------------------
    clip_share = float(((U <= 0.0) | (U >= 1.0)).mean())
    in_range = U.min() >= 0.0 and U.max() <= 1.0
    report(OK if in_range and clip_share < 0.01 else WARN, "RANGE", "min %.3f max %.3f, share of values on the clip limits %.4f%%"
           % (U.min(), U.max(), 100 * clip_share))

    # ---- 4. DEAD BANDS -------------------------------------------------------------------------------------------
    sd = U.std(0)
    dead = int((sd < 0.005).sum())
    report(OK if dead == 0 else BAD, "DEAD BANDS", "bands with std < 0.005: %d (std min %.4f, median %.4f, max %.4f)"
           % (dead, sd.min(), np.median(sd), sd.max()))

    # ---- 5. UNIT NORM --------------------------------------------------------------------------------------------
    Z = (U * 8.0 - 4.0) / np.sqrt(128.0)                            # decoded unit vectors (clipping lowers the length a bit)
    nrm = np.linalg.norm(Z, axis=1)
    report(OK if 0.9 < np.median(nrm) < 1.1 else WARN, "UNIT NORM", "vector length: median %.3f, 1st pct %.3f, 99th pct %.3f"
           % (np.median(nrm), np.percentile(nrm, 1), np.percentile(nrm, 99)))

    # ---- 6. DIMENSION --------------------------------------------------------------------------------------------
    Zc = Z - Z.mean(0)
    ev = np.linalg.svd(Zc[rng.choice(len(Zc), min(len(Zc), 20000), replace=False)], compute_uv=False) ** 2
    cum = np.cumsum(ev) / ev.sum()
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    lvl = OK if 5 <= k90 <= 100 else WARN
    report(lvl, "DIMENSION", "components for 90%% of the variance: %d of 128 (first component %.1f%%)" % (k90, 100 * ev[0] / ev.sum()))

    # ---- 7. SPATIAL COHERENCE ------------------------------------------------------------------------------------
    both = valid[:, :-1] & valid[:, 1:]                             # pixel and its right neighbour (a.step px apart)
    A = E[:, :, :-1][:, both].T
    B = E[:, :, 1:][:, both].T
    sel = rng.choice(len(A), min(len(A), a.pairs), replace=False)
    near = cos_sim(A[sel] * 8 - 4, B[sel] * 8 - 4)
    i, j = rng.integers(0, len(U), a.pairs), rng.integers(0, len(U), a.pairs)
    far = cos_sim(U[i] * 8 - 4, U[j] * 8 - 4)
    lvl = OK if near.mean() - far.mean() > 0.1 else BAD
    report(lvl, "SPATIAL", "cosine of neighbours (%d px apart) %.3f vs random pairs %.3f" % (a.step, near.mean(), far.mean()))

    # ---- 8. SPECTRAL AGREEMENT -----------------------------------------------------------------------------------
    try:
        with rasterio.open(C.FILLED_TIF) as sd_:
            keys = [tuple(d.split("|")[:1] + d.split("|")[2:3]) for d in sd_.descriptions]
            acc, cnt = {}, {}
            for b in range(1, sd_.count + 1):                       # season mean per (sensor, band) on the sample grid
                x = sd_.read(b, out_shape=(oh, ow)).astype(np.float32)
                x[x == -32768] = np.nan
                acc[keys[b - 1]] = acc.get(keys[b - 1], 0) + np.nan_to_num(x)
                cnt[keys[b - 1]] = cnt.get(keys[b - 1], 0) + 1
        F = np.stack([acc[k] / cnt[k] for k in sorted(acc)], 0)[:, valid].T / 10000.0   # (n, n_sensor_bands)
        d = np.linalg.norm(F[i] - F[j], axis=1)                     # satellite-feature distance of the random pairs
        r = float(np.corrcoef(far, -d)[0, 1])
        report(OK if r > 0.2 else WARN, "SPECTRAL", "corr(embedding cosine, -spectral distance) = %.3f  (>0.2 expected)" % r)
    except Exception as e:                                          # filled stack missing etc.: not fatal
        report(WARN, "SPECTRAL", "skipped (%s)" % e)

    if not (a.robust or a.probe):
        return

    # ---- raw observations of a small sample of pixels (shared by checks 9 and 10) -------------------------------------
    print("reading the raw scenes of %d strips x %d pixels ..." % (a.strips, a.per_strip))
    S = sample_strips(g, a.strips, 8, a.per_strip)
    X, V, sid = S["X"], S["V"], S["sid"]

    # ---- 9. ROBUSTNESS ----------------------------------------------------------------------------------------------------
    if a.robust:
        import torch
        from embed_to_tif import embed_pixels
        from pixel_encoder import PixelEncoder
        ck = torch.load(a.model or os.path.join(C.OUT, "model.pt"), weights_only=False)
        enc = PixelEncoder(ck["sid"])
        enc.load_state_dict(ck["model"])
        enc.eval()
        if X.shape[1] != len(ck["doy"]):
            raise SystemExit("Model has %d observations per pixel but the stack has %d." % (len(ck["doy"]), X.shape[1]))
        rows, floor = robustness(lambda x, v: embed_pixels(enc, ck, x, v), X, V, sid, rng)
        print("ROBUSTNESS: cosine between the embedding from ALL valid scenes and the embedding from fewer; "
              "two random pixels have cosine %.3f (the floor)" % floor)
        for label, m, p5, kept, n in rows:
            report(OK if kept >= 0.7 else WARN, "ROBUST", "%-36s cos mean %.3f (5th pct %.3f)  kept %.2f   [%d px]"
                   % (label, m, p5, kept, n))

    # ---- 10. PROBE ------------------------------------------------------------------------------------------------------------
    if a.probe:
        from rasterio.windows import Window
        Zs = np.full((len(X), 128), np.nan, np.float32)                # stored embedding at the sampled pixels
        with rasterio.open(emb_path) as ds:
            for k in np.unique(S["strip"]):
                m = np.flatnonzero(S["strip"] == k)
                r0 = int(S["row"][m].min())
                blk = ds.read(window=Window(0, r0, ds.width, int(S["row"][m].max()) - r0 + 1)).astype(np.float32)
                Zs[m] = blk[:, S["row"][m] - r0, S["col"][m]].T
        has = Zs[:, 0] >= 0                                            # -1 = no embedding at this pixel
        Zs = Zs * 8.0 - 4.0
        test = (S["strip"] % 3 == 2)                                   # every third strip = test places (unseen areas)
        print("PROBE: ridge regression on the 128 embedding bands, learned on %d strips, tested on %d other strips"
              % (len(np.unique(S["strip"][~test])), len(np.unique(S["strip"][test]))))
        for name, (y, okt) in probe_targets(X, V, sid).items():
            tr, te = has & okt & ~test, has & okt & test
            if tr.sum() < 200 or te.sum() < 200:
                report(WARN, "PROBE", "%-28s skipped (too few pixels)" % name)
                continue
            r2, r2_tr, r_te = ridge_r2(Zs[tr], y[tr], Zs[te], y[te])
            # extra numbers to understand a low score: R2 on the training strips, the correlation on the test strips, and the
            # mean and spread of the target on the training and on the test strips (a different mean or a tiny spread in the test
            # strips makes R2 very negative even if the ordering of the pixels is right).
            report(OK if r2 >= 0.5 else WARN, "PROBE", "%-28s R2 on unseen strips %.3f   [train %d / test %d px]"
                   % (name, r2, tr.sum(), te.sum()))
            print("             %-28s   R2 train strips %.3f | correlation on test strips %.3f | target mean train %.3f test %.3f"
                  " | std train %.3f test %.3f" % ("", r2_tr, r_te, y[tr].mean(), y[te].mean(), y[tr].std(), y[te].std()))


if __name__ == "__main__":
    main()