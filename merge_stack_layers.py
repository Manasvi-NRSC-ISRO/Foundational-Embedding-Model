r"""Merge the satellite stack with other data layers (for example the weather) into ONE final tif.

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python merge_stack_layers.py                                   # satellite stack + the weather tif of the season
    python merge_stack_layers.py --layers outputs\2025_Apr_Jun\weather_IMD_2025-04-01_to_2025-06-30.tif
    python merge_stack_layers.py --layers a.tif b.tif --out outputs\2025_Apr_Jun\satellite_weather_stack_<tag>.tif

Output (default): <season folder>\satellite_weather_stack_<tag>.tif and satellite_weather_stack_<tag>_bands.csv. Band order:
    AWiFS scenes -> LISS-III scenes -> EOS-04 scenes (exactly the bands of satellite_stack_<tag>.tif)
    -> then the bands of every layer file, in the order they are given (weather: rain, Tmax, Tmin, one band per day).

Why a separate script: preprocess_stack_scenes.py only builds the satellite stack and stays the same when more satellite
data is added; the masks, samples and models only need satellite_stack_<tag>.tif. This script never changes that file, it
only READS it, and writes a new one. Run it whenever you want everything in one tif.

Rules a layer file has to follow (weather_stack.py does):  the same grid as the stack, data type int16, nodata -32768,
floats stored as integers with a known scale, and bands described  sensor|date|band|source  with tags
(sensor, date, doy, band, unit). Any new data source can be added by writing such a tif and passing it to --layers.
"""
import argparse
import csv
import glob
import os

import common as C


def main():
    import rasterio
    from rasterio.windows import Window

    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=C.MODEL_TIF, help="the satellite stack (default: the season's filled stack satellite_stack_filled_<tag>.tif, or the raw one if the fill pass was not run)")
    ap.add_argument("--layers", nargs="*", help="layer tifs to append, in this order "
                    "(default: weather_IMD_*_to_*.tif of the season folder)")
    ap.add_argument("--out", default=None, help="output tif (default: satellite_weather_stack_<tag>.tif next to the stack)")
    ap.add_argument("--block", type=int, default=128, help="image rows copied at a time (keeps memory low)")
    a = ap.parse_args()

    layers = a.layers or sorted(glob.glob(os.path.join(C.OUT, "weather_IMD_*_to_*.tif")))
    if not layers:
        raise SystemExit("No layer tif found in %s. Run weather/weather_stack.py first or pass --layers." % C.OUT)
    paths = [a.stack] + layers
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit("not found: " + p)
    out = a.out or os.path.join(os.path.dirname(a.stack), "satellite_weather_stack_%s.tif" % C.TAG if C.TAG else "satellite_weather_stack.tif")

    srcs = [rasterio.open(p) for p in paths]
    try:
        ref = srcs[0]
        # every layer must sit exactly on the stack's grid and be int16, otherwise bands would not line up
        for p, s in zip(paths, srcs):
            if (s.height, s.width) != (ref.height, ref.width) or s.transform != ref.transform:
                raise SystemExit("%s is not on the same grid as the stack. Rebuild it." % p)
            if s.dtypes[0] != "int16":
                raise SystemExit("%s is %s; layers must be int16 (scaled integers, nodata -32768)." % (p, s.dtypes[0]))
        counts = [s.count for s in srcs]
        prof = ref.profile.copy()
        prof.update(count=sum(counts), BIGTIFF="YES", compress="deflate", tiled=True, blockxsize=256,
                    blockysize=256, interleave="band")
        tmp = out + ".tmp"                                   # written under a temporary name, renamed at the end
        layer_rows = []                                      # rows for the bands table (layer bands only)
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.update_tags(**ref.tags())
            dst.update_tags(layout="satellite scenes (see satellite_stack_<tag>_bands.csv), then: " +
                                   ", ".join(os.path.basename(p) for p in layers))
            offset = 0                                       # number of bands already written
            for si, s in enumerate(srcs):
                for r0 in range(0, ref.height, a.block):     # copy this file block by block
                    h = min(a.block, ref.height - r0)
                    win = Window(0, r0, ref.width, h)
                    dst.write(s.read(window=win), indexes=list(range(offset + 1, offset + s.count + 1)), window=win)
                for b in range(1, s.count + 1):              # keep the band names and tags of the source file
                    dst.set_band_description(offset + b, s.descriptions[b - 1])
                    t = s.tags(b)
                    dst.update_tags(offset + b, **t)
                    if si > 0:
                        layer_rows.append([offset + b, t.get("sensor", "layer"), t.get("date", ""), t.get("doy", ""),
                                           os.path.splitext(os.path.basename(paths[si]))[0], t.get("band", ""),
                                           "ok", "", "100.0"])
                offset += s.count
                print("  copied %s (%d bands)" % (os.path.basename(paths[si]), s.count), flush=True)
    finally:
        for s in srcs:
            s.close()
    os.replace(tmp, out)

    # bands table: the satellite rows of satellite_stack_<tag>_bands.csv, then one row per layer band
    sat_csv = a.stack[:-4] + "_bands.csv"
    if not os.path.exists(sat_csv):          # the filled stack has the same bands as the raw one: use its table
        sat_csv = C.SCENES_TIF[:-4] + "_bands.csv"
    rows = []
    if os.path.exists(sat_csv):
        with open(sat_csv, newline="") as fh:
            rows = list(csv.reader(fh))
    with open(out[:-4] + "_bands.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows + layer_rows)
    print("saved %s: %d bands (%d satellite + %d from %d layer file(s))" % (
        out, sum(counts), counts[0], sum(counts[1:]), len(layers)))


if __name__ == "__main__":
    main()