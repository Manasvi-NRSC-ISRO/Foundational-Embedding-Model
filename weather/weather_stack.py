r"""Weather as ONE separate stack tif on the 30 m grid, with the same band logic as the satellite stack.

    $env:EMB_OUT = "outputs\2025_Apr_Jun"
    python weather\weather_stack.py

Needs, in the season folder (EMB_OUT), the two files made by weather\weather_prep.py:
    weather_IMD_daily_values_<tag>.npz         daily rain / Tmax / Tmin of every weather cell (float32, mm and degrees C)
    weather_IMD_cell_map_<tag>.tif   which weather cell every 30 m pixel belongs to (band 1 rain cell, band 2 temperature cell)

Writes <season folder>\weather_IMD_<first day>_to_<last day>.tif, for example
    outputs\2025_Apr_Jun\weather_IMD_2025-04-01_to_2025-06-30.tif      (one file, int16, nodata -32768):
    first the RAINFALL bands (one band per day), then Tmax (one band per day), then Tmin (one band per day).
    Rain is stored as mm x 10 (0.1 mm steps), temperatures as degrees C x 100 (0.01 degree steps): divide by the
    scale to get the real value. Outside the district polygon the value is -32768.
    Every band is described  weather|<date>|<rain/tmax/tmin>|IMD  and tagged (sensor, date, doy, band, unit, scale),
    so QGIS shows the day and the variable when you change band.

The name says what and when (data source IMD, first and last day), so several seasons or years can never be mixed up.
To put it together with the satellite stack run merge_stack_layers.py (it takes any number of such layer tifs, so other
data can be merged in the same way later; without it the rest of the pipeline works unchanged).

The weather of one pixel only takes 16 (rain) / 6 (temperature) different values over the whole district because the
IMD cells are 26-55 km wide; the compression of the tif therefore makes this file small.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: common.py

import numpy as np

import common as C

RAIN_SCALE = 10.0      # stored rain = mm * 10
TEMP_SCALE = 100.0     # stored temperature = degC * 100
NODATA = -32768        # int16 nodata, the same as the satellite stack


def expand(ids, table):
    """Look up the daily values of the weather cell of every pixel.

    ids   : (h, w) integer weather-cell id of every pixel (-1 = outside the district)
    table : (D, n_cells) daily value of every weather cell, already scaled to int16
    Returns (D, h, w) int16 with the pixel's cell values, and NODATA where ids == -1.
    """
    out = np.full((table.shape[0],) + ids.shape, NODATA, np.int16)
    inside = ids >= 0
    out[:, inside] = table[:, ids[inside]]           # fancy indexing: pick the column of each pixel's cell
    return out


def main():
    import rasterio
    from rasterio.windows import Window
    import glob
    npz = sorted(glob.glob(os.path.join(C.OUT, "weather_IMD_daily_values_*.npz")), key=os.path.getmtime)
    if not npz:
        raise SystemExit("run weather_prep.py first (no weather_IMD_daily_values_*.npz in %s)" % C.OUT)
    cells_tif = npz[-1].replace("daily_values", "cell_map")[:-4] + ".tif"       # the cell map of the same date range
    z = np.load(npz[-1])
    dates = [int(d) for d in z["dates"]]
    days = ["%04d-%02d-%02d" % (d // 10000, d // 100 % 100, d % 100) for d in dates]    # YYYY-MM-DD
    D = len(days)
    # (name, unit text, scale, band of weather_cells.tif holding the cell id, daily table [D, n_cells])
    layers = [("rain", "mm x10", RAIN_SCALE, 1, z["rain"]),
              ("tmax", "degC x100", TEMP_SCALE, 2, z["tmax"]),
              ("tmin", "degC x100", TEMP_SCALE, 2, z["tmin"])]
    # float -> scaled integers once (tables are tiny: D x number of cells)
    scaled = [np.clip(np.round(t * sc), -32767, 32767).astype(np.int16) for _, _, sc, _, t in layers]
    out = os.path.join(C.OUT, "weather_IMD_%s_to_%s.tif" % (days[0], days[-1]))     # self-explanatory file name
    tmp = out + ".tmp"
    with rasterio.open(cells_tif) as cells:
        H, W = cells.height, cells.width
        prof = dict(driver="GTiff", height=H, width=W, count=3 * D, dtype="int16", crs=cells.crs,
                    transform=cells.transform, nodata=NODATA, compress="deflate", predictor=2, tiled=True,
                    blockxsize=256, blockysize=256, interleave="band", BIGTIFF="YES")
        with rasterio.open(tmp, "w", **prof) as dst:
            for r0 in range(0, H, 256):                              # 256 image rows at a time (low memory)
                h = min(256, H - r0)
                win = Window(0, r0, W, h)
                for k, (name, unit, sc, cband, _) in enumerate(layers):
                    ids = cells.read(cband, window=win).astype(np.int64)
                    first = k * D + 1                                # rain: bands 1..D, tmax next D, tmin last D
                    dst.write(expand(ids, scaled[k]), indexes=list(range(first, first + D)), window=win)
                print("  rows %d/%d" % (r0 + h, H), end="\r", flush=True)
            dst.update_tags(layout="rain (%d days), tmax (%d days), tmin (%d days)" % (D, D, D),
                            rain_scale="mm x 10", temperature_scale="degC x 100", source="IMD gridded daily data")
            for k, (name, unit, sc, cband, _) in enumerate(layers):  # label every band: weather|date|variable|IMD
                for j, d in enumerate(days):
                    b = k * D + j + 1
                    dst.set_band_description(b, "weather|%s|%s|IMD" % (d, name))
                    dst.update_tags(b, sensor="weather", date=d, doy=str(C.doy_of(d)), band=name, unit=unit,
                                    scale=str(int(sc)))
    os.replace(tmp, out)
    print("\nsaved %s: %d bands (rain, tmax, tmin x %d days), int16, nodata %d" % (out, 3 * D, D, NODATA))


if __name__ == "__main__":
    main()