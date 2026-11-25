r"""One pass over an EXISTING satellite stack: fill no-data inside the district and bring every value to 0..10000.

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python fill_rescale_stack.py

Input : the raw stack of the season folder, satellite_stack_<tag>.tif (found by pattern; int16, nodata -32768).
Output: satellite_stack_filled_<tag>.tif in the same folder (same bands, same order, same band names/tags,
        so the scene table and the cloud masks of the raw stack stay valid for it).
The raw stack is NEVER changed (this pass can be re-run with other rules at any time).

What it does, per band
  1. FILL.  For every (sensor, band name), e.g. AWiFS-NIR, LISS-III-red, EOS-04 HH, EOS-04 HV, the mean of ALL valid
     values of that band over ALL scenes of the season is computed (sampled on a coarse grid of the district:
     every 4th pixel in rows and columns, which is plenty for a mean). Every pixel INSIDE the district polygon
     that has no data in a scene gets this mean. Pixels outside the polygon stay nodata (-32768).
  2. RANGE.  Optical bands are clipped to 0..10000 (reflectance x 10000, as before).
     SAR (stored as dB x 100, range -3000..+1500 in this data) is rescaled to the SAME 0..10000 range:
         stored = round((dB - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN) * 10000),   SAR_DB_MIN = -30, SAR_DB_MAX = +15
     so -30 dB -> 0 and +15 dB -> 10000 (common.py has the constants; the reader inverts the formula).
  Data type stays int16.

IMPORTANT: a filled pixel is NOT real data. The cloud mask of that scene still says 0 (no data) there, and
common.read_all_scenes() treats mask != 1 as invalid, so the model never sees the filled numbers as observations.
They are only there so that no pixel of the district shows "no data" when you click it in QGIS.
Tags written to the file document all of this (sar_encoding, fill_rule, the fill value of every band).
"""
import argparse
import json
import os
import time

import numpy as np

import common as C

STEP = 4          # the means are computed from every STEP-th row and column (1/16 of the pixels)


def band_key(desc):
    """'eos04|2025-04-17|HH_dB|<scene>' -> ('eos04', 'HH_dB'): the sensor and the band name of a stack band."""
    p = desc.split("|")
    return p[0], p[2]


def main():
    import rasterio
    from rasterio.windows import Window
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=C.SCENES_TIF, help="raw stack (default: the one of the season folder)")
    ap.add_argument("--out", default=None, help="output tif (default: satellite_stack_filled_<tag>.tif)")
    ap.add_argument("--block", type=int, default=64, help="image rows per block (lower = less memory)")
    a = ap.parse_args()
    g = C.get_grid()
    out = a.out or (os.path.join(C.OUT, "satellite_stack_filled_%s.tif" % C.TAG) if C.TAG else
                    os.path.join(C.OUT, "satellite_stack_filled.tif"))
    NODATA = C.SAR_NODATA                                           # -32768

    with rasterio.open(a.src) as src:
        if (src.height, src.width) != (g["H"], g["W"]):
            raise SystemExit("stack is not on the common grid")
        keys = [band_key(d) for d in src.descriptions]              # (sensor, band name) of every band
        sensors_bands = sorted(set(keys))

        # ---- step 1: season-wide mean of every (sensor, band), from valid values only -------------------
        sums = {k: 0.0 for k in sensors_bands}
        cnts = {k: 0 for k in sensors_bands}
        inside_small = g["pmask"][::STEP, ::STEP]                   # polygon on the same coarse grid
        t0 = time.time()
        for b in range(1, src.count + 1):
            k = keys[b - 1]
            x = src.read(b, out_shape=(inside_small.shape[0], inside_small.shape[1]))   # decimated read
            ok = inside_small & (x != NODATA)
            if C.SENSORS[k[0]]["kind"] == "optical":
                ok &= x > 0                                         # DN 0 = nodata for optical
            sums[k] += float(x[ok].sum(dtype=np.float64))
            cnts[k] += int(ok.sum())
            print("  means: band %d/%d" % (b, src.count), end="\r", flush=True)
        fill = {}
        for k in sensors_bands:
            if cnts[k] == 0:
                raise SystemExit("no valid value at all for %s %s" % k)
            m = sums[k] / cnts[k]                                   # mean in STORED units (DN or dB x 100)
            if C.SENSORS[k[0]]["kind"] == "sar":                    # SAR: convert to the 0..10000 scale
                m = (m / 100.0 - C.SAR_DB_MIN) / (C.SAR_DB_MAX - C.SAR_DB_MIN) * 10000.0
            fill[k] = int(np.clip(round(m), 0, 10000))
        print("\nfill value (0..10000) per band:")
        for k in sensors_bands:
            print("  %-6s %-8s %6d   (from %d valid sample pixels)" % (k[0], k[1], fill[k], cnts[k]))
        fill_vec = np.array([fill[k] for k in keys], np.int16)      # one fill value per stack band
        is_sar = np.array([C.SENSORS[k[0]]["kind"] == "sar" for k in keys])
        is_opt = ~is_sar

        # ---- step 2: rewrite the stack block by block -----------------------------------------------------
        prof = src.profile.copy()
        prof.update(dtype="int16", nodata=NODATA, compress="deflate", predictor=2, tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF="YES", interleave="band")
        tmp = out + ".tmp"
        with rasterio.open(tmp, "w", **prof) as dst:
            for r0 in range(0, src.height, a.block):
                h = min(a.block, src.height - r0)
                win = Window(0, r0, src.width, h)
                x = src.read(window=win)                            # (bands, h, W) int16, raw values
                inside = g["pmask"][r0:r0 + h]                      # (h, W) district polygon
                bad = (x == NODATA)                                 # SAR nodata and any nodata-coded optical
                bad[is_opt] |= (x[is_opt] <= 0)                     # optical: 0 (or less) = nodata
                v = x.astype(np.float32)
                v[is_sar] = np.round((v[is_sar] / 100.0 - C.SAR_DB_MIN) / (C.SAR_DB_MAX - C.SAR_DB_MIN) * 10000.0)
                v = np.clip(v, 0, 10000)                            # everything now lives in 0..10000
                y = np.where(bad, fill_vec[:, None, None], v.astype(np.int16)).astype(np.int16)
                y[:, ~inside] = NODATA                              # outside the district polygon: nodata
                dst.write(y, window=win)
                print("  rewrite: rows %d/%d" % (r0 + h, src.height), end="\r", flush=True)
            dst.update_tags(**src.tags())
            for b in range(1, src.count + 1):                       # keep the band names and band tags
                dst.set_band_description(b, src.descriptions[b - 1])
                dst.update_tags(b, **src.tags(b))
            dst.update_tags(
                sar_encoding="scaled_0_10000",
                sar_db_range="%g..%g dB -> 0..10000" % (C.SAR_DB_MIN, C.SAR_DB_MAX),
                optical_encoding="reflectance x 10000, clipped 0..10000",
                fill_rule="nodata inside the district = season-wide mean of all valid values of that sensor band; "
                          "filled pixels stay 'no data' (0) in satellite_cloudmask so the model ignores them",
                fill_values=json.dumps({"%s|%s" % k: v for k, v in fill.items()}))
    os.replace(tmp, out)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
