"""Random training pixels -> outputs/samples.npz   (read from the single stacked file outputs/stack_all_scenes.tif).

    python build_samples.py                 # 100000 pixels
    python build_samples.py --n 5000        # quick test
"""
import argparse
import os

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_valid", type=int, default=3, help="min valid observations per pixel")
    a = ap.parse_args()
    g = C.get_grid()
    rng = np.random.default_rng(a.seed)
    idx = np.flatnonzero(g["pmask"].ravel())
    idx = np.sort(rng.choice(idx, size=min(a.n * 2, len(idx)), replace=False))  # 2x: some get dropped
    X, V, doy, sid, names = C.read_all_scenes(None, idx)
    keep = V.sum(1) >= a.min_valid
    print("pixels with >= %d valid obs: %d / %d" % (a.min_valid, keep.sum(), len(idx)))
    keep = np.flatnonzero(keep)[: a.n]
    if len(keep) == 0:
        raise SystemExit("No usable pixels. Check the stacks / masks.")
    os.makedirs(C.OUT, exist_ok=True)
    np.savez_compressed(os.path.join(C.OUT, "samples.npz"), X=X[keep], V=V[keep], doy=doy,
                        sid=sid, names=np.array(names), idx=idx[keep])
    print("saved %d samples, %d tokens each (%s) -> %s/samples.npz" % (
        len(keep), X.shape[1], ", ".join("%s=%d" % (k, (sid == i).sum())
                                          for i, k in enumerate(C.SENSORS)), C.OUT))


if __name__ == "__main__":
    main()
