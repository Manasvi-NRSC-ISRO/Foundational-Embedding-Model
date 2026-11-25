"""Picture to check the cloud mask: for the cloudiest scenes, show the image next to its mask.

    python mask_check.py            # the 4 scenes with the most cloud
    python mask_check.py --n 6

Output: outputs/mask_check.png   left = false-colour image (NIR, Red, Green), right = mask
        (green = clear, white = cloud, black = no data, grey = outside the district).
If the left image is really white / hazy where the right one says cloud, the mask is right.
"""
import argparse
import csv
import os

import numpy as np

import common as C


def stretch(a):
    out = np.zeros(a.shape, np.float32)
    ok = a > 0
    if ok.sum() < 10:
        return out
    lo, hi = np.percentile(a[ok], (2, 98))
    out = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    out[~ok] = 0
    return out


def main():
    import rasterio
    from rasterio.enums import Resampling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--step", type=int, default=6)
    a = ap.parse_args()
    stats = list(csv.DictReader(open(os.path.join(C.OUT, "stack_all_masks_stats.csv"))))
    scenes = {(r["scene"]): r for r in C.load_scene_table()}
    stats = [s for s in stats if s["sensor"] in ("awifs", "liss3") and s["scene"] in scenes]
    stats.sort(key=lambda s: -float(s["cloud_pct"]))
    stats = stats[: a.n]
    cmap = np.array([[0, 0, 0], [0.2, 0.7, 0.3], [1, 1, 1], [0.5, 0.5, 0.5]], np.float32)
    fig, axs = plt.subplots(len(stats), 2, figsize=(8, 5 * len(stats)))
    axs = np.atleast_2d(axs)
    with rasterio.open(C.SCENES_TIF) as sc, rasterio.open(C.MASKS_TIF) as mk:
        h, w = sc.height // a.step, sc.width // a.step
        for i, s in enumerate(stats):
            f = int(scenes[s["scene"]]["first_band"])
            img = sc.read([f, f + 1, f + 2], out_shape=(3, h, w), resampling=Resampling.average
                          ).astype(np.float32)
            m = mk.read(int(s["mask_band"]), out_shape=(h, w), resampling=Resampling.nearest)
            rgb = np.stack([stretch(img[2]), stretch(img[1]), stretch(img[0])], -1)
            mi = np.where(m == 255, 3, np.clip(m, 0, 2))
            axs[i, 0].imshow(rgb)
            axs[i, 0].set_title("%s %s  image" % (s["sensor"], s["date"]))
            axs[i, 1].imshow(cmap[mi])
            axs[i, 1].set_title("mask (cloud %s%%, band %s)" % (s["cloud_pct"], s["mask_band"]))
            for ax in axs[i]:
                ax.axis("off")
    out = os.path.join(C.OUT, "mask_check.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
