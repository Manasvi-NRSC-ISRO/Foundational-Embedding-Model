"""Everything known about ONE pixel: every observation (sensor, date, status, values) and its embedding.

    python pixel_info.py --x 755361 --y 2632158        # UTM 45N metres (the coordinate QGIS shows in its status bar)
    python pixel_info.py --lon 88.25 --lat 24.10       # or longitude / latitude

It reads the season folder (EMB_OUT): the scene stack (satellite_stack_<tag>.tif, raw values), the scene table
(satellite_scene_table_<tag>.csv), the cloud masks (satellite_cloudmask_<tag>.tif) and any embedding tifs that exist there.
It prints a table, and saves  <season folder>/pixel_<x>_<y>.csv  and  <season folder>/pixel_<x>_<y>.png
(a time-series plot of NDVI for the optical sensors and of HH / HV for the radar).

Status of an observation:
    clear   optical scene, data present, not cloud
    CLOUD   optical scene flagged as cloud by cloudmask_scenes.py
    valid   optical scene without a mask file, or a SAR scene with data
    nodata  the scene does not cover this pixel
Only clear / valid observations are used by the model.
"""
import argparse
import csv
import os

import numpy as np

import common as C

# meaning of the values stored in the cloud-mask file
STATUS = {0: "nodata", 1: "clear", 2: "CLOUD"}


def main():
    import rasterio
    from rasterio.windows import Window

    # ------------------------------------------------------------------------------------------
    # 1. Which pixel? Convert whatever the user gave (UTM metres or lon/lat) to a row and a column of the grid.
    # ------------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, help="UTM 45N easting in metres")
    ap.add_argument("--y", type=float, help="UTM 45N northing in metres")
    ap.add_argument("--lon", type=float, help="longitude in degrees (alternative to --x/--y)")
    ap.add_argument("--lat", type=float, help="latitude in degrees (alternative to --x/--y)")
    a = ap.parse_args()
    if a.lon is not None and a.lat is not None:
        from pyproj import Transformer
        a.x, a.y = Transformer.from_crs("EPSG:4326", C.CRS, always_xy=True).transform(a.lon, a.lat)
    if a.x is None or a.y is None:
        raise SystemExit("give --x --y (UTM metres) or --lon --lat")
    g = C.get_grid()
    col, row = [int(np.floor(v)) for v in (~g["transform"]) * (a.x, a.y)]   # inverse transform: metres -> pixel
    if not (0 <= col < g["W"] and 0 <= row < g["H"]):
        raise SystemExit("point is outside the grid")
    print("pixel col %d row %d | centre x %.0f y %.0f | inside district: %s" % (
        col, row, g["transform"].c + (col + .5) * C.RES, g["transform"].f - (row + .5) * C.RES,
        bool(g["pmask"][row, col])))
    win = Window(col, row, 1, 1)                                          # a window of exactly one pixel

    # ------------------------------------------------------------------------------------------
    # 2. Read the pixel from every scene of the stack (scene table row t <-> bands first_band..last_band
    #    of the stack <-> band t+1 of the mask file).
    # ------------------------------------------------------------------------------------------
    table = C.load_scene_table()
    mk = rasterio.open(C.MASKS_TIF) if os.path.exists(C.MASKS_TIF) else None
    if mk is not None and mk.count != len(table):
        mk.close()
        mk = None
        print("!! mask file does not match the scene table (re-run cloudmask_scenes.py): statuses use data only")
    rows = []
    with rasterio.open(C.SCENES_TIF) as ds:      # always the RAW stack: shows what the satellites really measured
        for t, sc in enumerate(table):
            sensor, d = sc["sensor"], sc["date"]
            f, l = int(sc["first_band"]), int(sc["last_band"])
            v = ds.read(list(range(f, l + 1)), window=win)[:, 0, 0]       # this pixel's values in this scene
            r = dict(sensor=sensor, date=d, doy=C.doy_of(d), scene=sc["scene"])
            if C.SENSORS[sensor]["kind"] == "optical":
                st = "valid"
                if mk is not None:
                    st = STATUS.get(int(mk.read(t + 1, window=win)[0, 0]), "outside")
                if (v <= 0).any():                                         # a missing band means no data
                    st = "nodata"
                refl = v.astype(float) * 1e-4                              # DN -> reflectance
                r.update(status=st, green=refl[0], red=refl[1], nir=refl[2], swir=refl[3],
                         ndvi=(refl[2] - refl[1]) / (refl[2] + refl[1] + 1e-9),
                         hh_db="", hv_db="", orbit="")
            else:
                ok = (v != C.SAR_NODATA).all()
                r.update(status="valid" if ok else "nodata", green="", red="", nir="", swir="", ndvi="",
                         hh_db=v[0] / 100.0 if ok else "", hv_db=v[1] / 100.0 if ok else "",   # stored dB*100 -> dB
                         orbit={"A": "ascending", "D": "descending"}.get(sc["orbit"], ""))
            rows.append(r)
    if mk is not None:
        mk.close()
    rows.sort(key=lambda r: (r["date"], r["sensor"]))

    # ------------------------------------------------------------------------------------------
    # 3. Print the table and a usable-observation count per sensor.
    # ------------------------------------------------------------------------------------------
    hdr = ["date", "doy", "sensor", "scene", "status", "green", "red", "nir", "swir", "ndvi",
           "hh_db", "hv_db", "orbit"]
    print("\n%-10s %3s %-6s %-7s %6s %6s %6s %6s %6s %7s %7s %s" % (
        "date", "doy", "sensor", "status", "green", "red", "nir", "swir", "ndvi", "hh_db", "hv_db", "orbit"))
    fmt = lambda x: ("%.3f" % x) if isinstance(x, float) else str(x)
    for r in rows:
        print("%-10s %3d %-6s %-7s %6s %6s %6s %6s %6s %7s %7s %s" % (
            r["date"], r["doy"], r["sensor"], r["status"], fmt(r["green"]), fmt(r["red"]),
            fmt(r["nir"]), fmt(r["swir"]), fmt(r["ndvi"]), fmt(r["hh_db"]), fmt(r["hv_db"]), r["orbit"]))
    use = [r for r in rows if r["status"] in ("clear", "valid")]
    print("\nusable observations: %d of %d  (liss3 %d, awifs %d, eos04 %d)" % (
        len(use), len(rows), *[sum(r["sensor"] == s for r in use) for s in ("liss3", "awifs", "eos04")]))

    # ------------------------------------------------------------------------------------------
    # 4. The embeddings of this pixel (first 8 of 128 numbers, on the 0-1 scale) for every embedding tif present.
    # ------------------------------------------------------------------------------------------
    for name in [os.path.basename(C.emb_path(k)) for k in ("barlow", "dae", "conv")]:
        ep = os.path.join(C.OUT, name)
        if not os.path.exists(ep):
            continue
        with rasterio.open(ep) as ds:
            q = ds.read(window=win)[:, 0, 0]
        if (q < 0).all():
            print("%s: nodata here" % name)
        else:
            print("%s (first 8 of 128, 0-1 scale): %s" % (name, np.round(q[:8], 3)))

    # ------------------------------------------------------------------------------------------
    # 5. Save the csv and the time-series plot.
    # ------------------------------------------------------------------------------------------
    os.makedirs(C.OUT, exist_ok=True)
    tag = "pixel_%d_%d" % (a.x, a.y)
    with open(os.path.join(C.OUT, tag + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    col_s = {"liss3": "tab:blue", "awifs": "tab:green"}
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for s in ("liss3", "awifs"):                                           # top panel: NDVI of the optical sensors
        for ok, mkr, lab in ((True, "o", "clear"), (False, "x", "cloud/nodata")):
            pts = [(datetime.strptime(r["date"], "%Y-%m-%d"), r["ndvi"]) for r in rows
                   if r["sensor"] == s and (r["status"] in ("clear", "valid")) == ok and r["ndvi"] != ""]
            if pts:
                ax[0].scatter(*zip(*pts), c=col_s[s], marker=mkr, alpha=1 if ok else .35,
                              label="%s %s" % (s, lab))
    ax[0].set_ylabel("NDVI")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title("pixel col %d row %d (x %.0f, y %.0f)" % (col, row, a.x, a.y))
    for key, c in (("hh_db", "tab:orange"), ("hv_db", "tab:red")):         # bottom panel: SAR backscatter
        pts = [(datetime.strptime(r["date"], "%Y-%m-%d"), r[key]) for r in rows
               if r["sensor"] == "eos04" and r[key] != ""]
        if pts:
            ax[1].scatter(*zip(*pts), c=c, label="eos04 " + key[:2].upper())
    ax[1].set_ylabel("SAR gamma0 (dB)")
    ax[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(os.path.join(C.OUT, tag + ".png"), dpi=150, bbox_inches="tight")
    print("saved %s.csv and %s.png in %s" % (tag, tag, C.OUT))


if __name__ == "__main__":
    main()