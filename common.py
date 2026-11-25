"""Shared settings and helper functions used by EVERY script of the Murshidabad embedding pipeline.

Nothing in this file is run on its own; the other scripts do `import common as C`.

What lives here
---------------
1. Global settings      : map projection (CRS), pixel size (RES), district polygon file (AOI),
                          output folder (OUT), SAR calibration constants, band names.
2. Sensor table         : SENSORS - where the zips of each satellite sensor are and what kind of sensor it is.
3. The common grid      : get_grid() - the ONE 30 m pixel grid (height, width, affine transform, district mask)
                          that every stack, mask, weather map and embedding in this project is aligned to.
4. Reading observations : load_scene_table() and read_all_scenes() - read the per-pixel observation sets
                          (the "tokens" the model sees) out of the big stack file made by preprocess_stack_scenes.py.
5. Embedding codec      : z_to_u(), u_to_z(), emb_valid() - embeddings are stored as float32 between 0 and 1.

Pipeline order (run from the repo folder, every path in the scripts is relative to it):
    preprocess_stack_scenes.py -> cloudmask_scenes.py -> build_samples.py -> train*.py -> embed_to_tif.py
"""
import glob
import os
import re
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------------------------
# Windows fix: on some installs rasterio cannot find its projection database (proj.db) and every
# reprojection fails. pyproj ships a copy, so we point the PROJ environment variables to it.
# This MUST happen before rasterio is imported anywhere, which is why it is at the top of common.py.
# ----------------------------------------------------------------------------------------------
try:
    import pyproj
    _pd = pyproj.datadir.get_data_dir()
    if os.path.exists(os.path.join(_pd, "proj.db")):
        os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _pd
except Exception:
    pass  # if pyproj is missing or has no proj.db we simply keep the defaults

# ----------------------------------------------------------------------------------------------
# 1. GLOBAL SETTINGS
# ----------------------------------------------------------------------------------------------
CRS = "EPSG:32645"      # UTM zone 45N: a metric projection, the common grid for every sensor
RES = 30.0              # pixel size in metres (LISS-III is 23.5 m, AWiFS 56 m, SAR 18 m -> all resampled to 30 m)


def _find_aoi():
    """Find the district polygon (AOI = area of interest) file.

    Looks in the current folder, then in boundaries/, then anywhere below the current folder.
    If nothing is found the plain file name is returned so that the later error message is clear.
    """
    for p in ["murshidabad.geojson", os.path.join("boundaries", "murshidabad.geojson")]:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join("**", "murshidabad.geojson"), recursive=True)
    return hits[0] if hits else "murshidabad.geojson"


AOI = _find_aoi()

# Every season has its own output folder. Select it in PowerShell with
#     $env:EMB_OUT = "outputs\2025_Apr_Jun"
# Stack, masks, samples, models, embeddings and weather of that season all go into / are read from it.
OUT = os.environ.get("EMB_OUT", "outputs")

# EOS-04 MRS L2B SAR calibration (from BAND_META.txt: Calibration_Constant_Gamma0 = 68.255, Beta0 = 67.140).
# gamma0 in dB = 10*log10(DN^2) - K. A constant K only shifts all values by the same amount
# (the model normalises that away), but we keep the official value so dB numbers are physical.
SAR_K_DB = 68.255
SAR_VALID_BIT = 128                     # bit of the SAR mask.tif that marks a valid pixel
SAR_BAD_BITS = 64 | 16                  # bits for radar shadow (64) and layover (16): such pixels are dropped
SAR_NODATA = -32768                     # the SAR part of the stack stores gamma0 dB * 100 as int16; this is "no data"

# The four optical bands we use, as named inside the Bhoonidhi zips (same names for LISS-III and AWiFS):
# BAND2 = green, BAND3 = red, BAND4 = near infrared (NIR), BAND5 = short-wave infrared (SWIR).
OPT_BANDS = ["BAND2", "BAND3", "BAND4", "BAND5"]

# ----------------------------------------------------------------------------------------------
# 2. SENSOR TABLE
# "kind"    : optical (4 reflectance bands) or sar (HH and HV backscatter).
# "pattern" : where the downloaded zips of that sensor are (bhoonidhi_year/<product folder>/*.zip).
# The ORDER of this dictionary matters: it defines the sensor id used by the model (0 liss3, 1 awifs, 2 eos04).
# ----------------------------------------------------------------------------------------------
# Folder that holds the downloaded zips (one sub-folder per Bhoonidhi collection, as made by bhoonidhi_download.py).
# By default it is "bhoonidhi_year", exactly as before. For another download (for example the Jul-Aug 2025 one) select its
# folder in PowerShell BEFORE running stack_scenes.py / cloudmask_scenes.py / the embedding scripts:
#     $env:EMB_DATA = "bhoonidhi_JulAug"
# The season output folder is still chosen with EMB_OUT (see above); the two variables are independent of each other.
_DATA = os.environ.get("EMB_DATA", "bhoonidhi_year")
SENSORS = {
    "liss3": dict(kind="optical", pattern=os.path.join(_DATA, "ResourceSat-2*_LISS3_BOA", "*.zip")),
    "awifs": dict(kind="optical", pattern=os.path.join(_DATA, "ResourceSat-2*_AWIFS_BOA", "*.zip")),
    "eos04": dict(kind="sar", pattern=os.path.join(_DATA, "EOS-04_SAR-MRS_L2B", "*.zip")),
}
SENS_ID = {s: i for i, s in enumerate(SENSORS)}   # sensor name -> integer id (embedding index inside the model)

# Bhoonidhi scene names contain the acquisition date like "04NOV2025": 2 digits, 3 capital letters, year 20xx.
DATE_RE = re.compile(r"\d{2}[A-Z]{3}20\d{2}")


def scene_date(name):
    """Acquisition date (datetime.date) from a Bhoonidhi scene / zip name, e.g. R2304NOV2025... -> 2025-11-04.

    Returns None if the name contains no date.
    """
    m = DATE_RE.search(os.path.basename(name).upper())
    return datetime.strptime(m.group(0), "%d%b%Y").date() if m else None


def doy_of(date_str):
    """Day of year (1..366) of a 'YYYY-MM-DD' string. The model uses it (as sin/cos) to know the season."""
    return datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday


# ----------------------------------------------------------------------------------------------
# 3. THE COMMON GRID
# ----------------------------------------------------------------------------------------------
_GRID = {}   # cache: the grid is built once per run, then reused


def get_grid():
    """The one 30 m grid that everything lives on.

    It is the bounding box of the district polygon in EPSG:32645, snapped outwards to multiples of 30 m.
    Returns a dict with
        H, W       : number of rows / columns of pixels
        transform  : rasterio affine transform (pixel -> UTM metres), north-up
        pmask      : (H, W) bool array, True for pixels INSIDE the district polygon
                     (the polygon is burnt into the grid, so everything outside it is later treated as nodata).
    """
    if _GRID:
        return _GRID
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    poly = gpd.read_file(AOI).to_crs(CRS)                 # district polygon in metres
    minx, miny, maxx, maxy = poly.total_bounds
    x0 = np.floor(minx / RES) * RES                        # snap west edge down to a multiple of 30 m
    y1 = np.ceil(maxy / RES) * RES                         # snap north edge up to a multiple of 30 m
    W = int(np.ceil((maxx - x0) / RES))
    H = int(np.ceil((y1 - miny) / RES))
    tr = from_origin(x0, y1, RES, RES)                     # top-left corner + pixel size
    pmask = rasterize([(g, 1) for g in poly.geometry], out_shape=(H, W), transform=tr,
                      dtype="uint8").astype(bool)          # 1 inside the polygon, 0 outside
    _GRID.update(H=H, W=W, transform=tr, pmask=pmask)
    return _GRID


# ----------------------------------------------------------------------------------------------
# 4. READING OBSERVATIONS FROM THE SCENE STACK
# preprocess_stack_scenes.py writes ONE file with every scene of every sensor as bands.
# Each scene is one "observation" (token) of a pixel: (date, sensor, 5 feature numbers).
# ----------------------------------------------------------------------------------------------
# ---- self-explanatory file names -------------------------------------------------------------
# Every file of a season carries the date range of its data in its name, <tag> = <first day>_to_<last day>:
#     satellite_stack_<tag>.tif          all satellite scenes as bands (int16, raw: real nodata -32768)
#     satellite_stack_<tag>_bands.csv    one row per band
#     satellite_scene_table_<tag>.csv    one row per scene (first / last band of the scene in the stack)
#     satellite_cloudmask_<tag>.tif      one cloud-mask band per scene (+ _stats.csv)
#     satellite_stack_filled_<tag>.tif   the stack after fill_rescale_stack.py: nodata filled, values 0..10000
#     embedding_<kind>_<tag>.tif         kind = barlow / dae / conv
# Scripts never hard-code a date range: they look for the file by pattern in the season folder (OUT). If only the
# OLD names (stack_all_scenes.tif ...) exist they are used instead, so nothing breaks before the rename.
def find_file(pattern, legacy):
    """Newest file of OUT matching `pattern` (a glob); if there is none the legacy path (may not exist)."""
    hits = sorted(glob.glob(os.path.join(OUT, pattern)), key=os.path.getmtime)
    return hits[-1] if hits else os.path.join(OUT, legacy)


def date_tag(dates):
    """'2025-04-03', '2025-06-28' ... -> '2025-04-01_to_2025-06-30' style tag from the first and last date."""
    d = sorted(dates)
    return "%s_to_%s" % (d[0], d[-1])


SCENES_TIF = find_file("satellite_stack_20*_to_*.tif", "stack_all_scenes.tif")          # raw stack (int16)
FILLED_TIF = find_file("satellite_stack_filled_20*_to_*.tif", "satellite_stack_filled.tif")  # after the fill pass
TAG = (re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2})", os.path.basename(SCENES_TIF)) or [None, None])[1]
SCENES_CSV = os.path.join(OUT, "satellite_scene_table_%s.csv" % TAG) if TAG else os.path.join(OUT, "scene_coverage.csv")
MASKS_TIF = os.path.join(OUT, "satellite_cloudmask_%s.tif" % TAG) if TAG else os.path.join(OUT, "stack_all_masks.tif")
if TAG and not os.path.exists(SCENES_CSV):
    SCENES_CSV = os.path.join(OUT, "scene_coverage.csv")
if TAG and not os.path.exists(MASKS_TIF) and os.path.exists(os.path.join(OUT, "stack_all_masks.tif")):
    MASKS_TIF = os.path.join(OUT, "stack_all_masks.tif")
# the file the MODEL reads: the filled stack when it exists (the fill pass was run), otherwise the raw stack
MODEL_TIF = FILLED_TIF if os.path.exists(FILLED_TIF) else SCENES_TIF

# value encoding of the filled stack (see fill_rescale_stack.py): SAR dB -> integer 0..10000 over this dB range
SAR_DB_MIN, SAR_DB_MAX = -30.0, 15.0


def emb_path(kind):
    """Embedding file of this season: embedding_<kind>_<tag>.tif (kind = barlow / dae / conv)."""
    return os.path.join(OUT, "embedding_%s_%s.tif" % (kind, TAG) if TAG else "embedding_%s_30m.tif" % kind)


def optical_features(refl):
    """Add NDVI to optical reflectance.

    refl: (..., 4) array with reflectance of [green, red, NIR, SWIR] (0..1).
    Returns (..., 5): the same four bands plus NDVI = (NIR - red) / (NIR + red), the usual vegetation index.
    The 1e-6 only avoids division by zero.
    """
    red, nir = refl[..., 1], refl[..., 2]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.concatenate([refl, ndvi[..., None]], axis=-1)


def load_scene_table():
    """Rows (as dicts) of scene_coverage.csv for the scenes that were stacked successfully (status == ok).

    The order of the rows is the order of the scenes in the stack AND of the bands of the mask file.
    """
    import csv
    if not (os.path.exists(SCENES_TIF) and os.path.exists(SCENES_CSV)):
        raise SystemExit("Run preprocess_stack_scenes.py first (needs %s and %s)." % (SCENES_TIF, SCENES_CSV))
    with open(SCENES_CSV) as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "ok"]


# ----------------------------------------------------------------------------------------------
# WEATHER AS EXTRA INPUT FEATURES OF EVERY OBSERVATION
# Every token (one satellite scene at one pixel) gets 4 extra numbers describing the weather of the 14 days that end on
# the scene date, at the pixel's IMD cell (made by weather/weather_prep.py, which also writes the cell map):
#     feature 5: rain sum of the last 7 days (mm)        feature 6: rain sum of the last 14 days (mm)
#     feature 7: mean Tmax of the last 7 days (degC)     feature 8: mean Tmin of the last 7 days (degC)
# If the season starts less than 7/14 days before the scene, the days that exist are used (the weather files start on the
# first day of the season). The weather columns are INPUT ONLY: the decoder is never asked to rebuild them (the IMD cells
# are 26-55 km wide, so weather also works as a location code; it must not be what the embedding is made of), and
# during training the weather columns of a random half of the pixels are hidden, so the embedding also works without it.
# ----------------------------------------------------------------------------------------------
N_FEAT = 9          # 5 satellite feature slots + 4 weather slots per token
N_SAT_FEAT = 5


def feature_mask(sid, targets_only=False):
    """(T, N_FEAT) bool: which feature columns exist for each scene's sensor.

    Optical scenes: columns 0-4 (green, red, NIR, SWIR, NDVI).  SAR scenes: columns 0-2 (HH dB, HV dB, ascending flag).
    All scenes also have the 4 weather columns 5-8, unless targets_only=True: then only the columns the decoder has
    to reconstruct are True (weather excluded).
    """
    m = np.zeros((len(sid), N_FEAT), bool)
    for t, k in enumerate(sid):
        m[t, :(5 if k < 2 else 3)] = True
        if not targets_only:
            m[t, N_SAT_FEAT:] = True
    return m


def _weather_tables(rows):
    """Per-scene weather feature tables for every IMD cell, or None if the weather files do not exist.

    Returns (rain_ids_tif_path, rain7[T,nr], rain14[T,nr], tmax7[T,nt], tmin7[T,nt]).
    """
    maps = sorted(glob.glob(os.path.join(OUT, "weather_IMD_cell_map_*.tif")), key=os.path.getmtime)
    if not maps:
        return None
    npz = maps[-1].replace("cell_map", "daily_values")[:-4] + ".npz"
    if not os.path.exists(npz):
        return None
    z = np.load(npz)
    dd = [int(d) for d in z["dates"]]
    days = [datetime(d // 10000, d // 100 % 100, d % 100) for d in dd]
    T = len(rows)
    nr, nt = z["rain"].shape[1], z["tmax"].shape[1]
    r7 = np.zeros((T, nr), np.float32); r14 = np.zeros((T, nr), np.float32)
    tx = np.zeros((T, nt), np.float32); tn = np.zeros((T, nt), np.float32)
    for t, r in enumerate(rows):
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        last = max([i for i, x in enumerate(days) if x <= d], default=None)   # last weather day on or before the scene
        if last is None:
            continue                                                          # scene before the weather starts: zeros
        r7[t] = z["rain"][max(0, last - 6):last + 1].sum(0)
        r14[t] = z["rain"][max(0, last - 13):last + 1].sum(0)
        tx[t] = z["tmax"][max(0, last - 6):last + 1].mean(0)
        tn[t] = z["tmin"][max(0, last - 6):last + 1].mean(0)
    return maps[-1], r7, r14, tx, tn


def read_all_scenes(window=None, sel=None):
    """Every scene of every sensor, for the selected pixels -> one observation set per pixel.

    window : rasterio Window to read (None = the whole district).
    sel    : flat indices (row * width + col, inside that window) of the pixels wanted.

    Optical scenes give 5 features: reflectance green, red, NIR, SWIR and NDVI. A pixel is INVALID in a
    scene if any band is nodata (0) or the cloud mask says it is not clear (mask != 1).
    SAR scenes give 3 features: HH dB, HV dB and a flag (1 = ascending pass, 0 = descending); the last
    two feature slots stay 0 so that every sensor has 5 slots. Invalid = nodata.

    Returns
        X     (n, T, 9) float16  5 satellite features + 4 weather features (0 where invalid)
        V     (n, T)    bool     True where the observation is valid
        doy   (T,)      float32  day of year of every scene
        sid   (T,)      int8     sensor id of every scene (0 liss3, 1 awifs, 2 eos04)
        names (T,)      list     "sensor:date:scene" label of every scene
    """
    import rasterio
    rows = load_scene_table()
    n, T = len(sel), len(rows)
    X = np.zeros((n, T, N_FEAT), np.float16)
    V = np.zeros((n, T), bool)
    doy = np.zeros(T, np.float32)
    sid = np.zeros(T, np.int8)
    names = []
    g = get_grid()
    wt = _weather_tables(rows)                                    # weather per scene and cell (None = no weather files)
    if wt is None:
        print("!! no weather files (weather_IMD_cell_map_*.tif / daily_values_*.npz) in %s: weather features are 0" % OUT)
    else:
        with rasterio.open(wt[0]) as cm:                          # cell id of every pixel: band 1 rain cell, band 2 temp cell
            rid = np.clip(cm.read(1, window=window).ravel()[sel], 0, None)
            tid = np.clip(cm.read(2, window=window).ravel()[sel], 0, None)

    # Cloud masks: one band per scene, made by cloudmask_scenes.py. If the file does not belong to this stack
    # (different number of scenes or different grid) we stop: using a wrong mask would silently corrupt training.
    mk = None
    if os.path.exists(MASKS_TIF):
        mk = rasterio.open(MASKS_TIF)
        if mk.count != len(rows) or (mk.height, mk.width) != (g["H"], g["W"]):
            mk.close()
            raise SystemExit("%s does not match the scene table. Re-run cloudmask_scenes.py." % MASKS_TIF)
    else:
        print("!! no cloud mask file (%s): cloudy pixels are NOT removed. Run cloudmask_scenes.py." % MASKS_TIF)

    with rasterio.open(MODEL_TIF) as ds:
        if (ds.height, ds.width) != (g["H"], g["W"]):
            raise SystemExit("%s is on a different grid. Re-run preprocess_stack_scenes.py." % MODEL_TIF)
        scaled_sar = ds.tags().get("sar_encoding", "") == "scaled_0_10000"   # filled stack: SAR already 0..10000
        for t, r in enumerate(rows):
            s, d = r["sensor"], r["date"]
            f, l = int(r["first_band"]), int(r["last_band"])            # bands of this scene inside the stack
            a = ds.read(list(range(f, l + 1)), window=window).reshape(l - f + 1, -1)[:, sel]   # (bands, n)
            if SENSORS[s]["kind"] == "optical":
                ok = (a > 0).all(axis=0)                                  # DN 0 = nodata in any band -> invalid
                if mk is not None:                                        # 1 = clear in the scene's own mask band
                    ok &= mk.read(t + 1, window=window).ravel()[sel] == 1
                feats = optical_features(a.T.astype(np.float32) / 10000.0)   # DN -> reflectance (x 0.0001)
            else:
                ok = (a != SAR_NODATA).all(axis=0)
                if mk is not None:                                        # mask 1 = real SAR data (filled pixels are 0)
                    ok &= mk.read(t + 1, window=window).ravel()[sel] == 1
                feats = np.zeros((n, 5), np.float32)
                if scaled_sar:                                            # 0..10000 -> dB (inverse of the fill pass)
                    feats[:, :2] = a.T.astype(np.float32) / 10000.0 * (SAR_DB_MAX - SAR_DB_MIN) + SAR_DB_MIN
                else:
                    feats[:, :2] = a.T.astype(np.float32) / 100.0           # stored dB*100 -> dB
                feats[:, 2] = 1.0 if r["orbit"] == "A" else 0.0             # ascending-pass flag
            if wt is not None:                                              # weather columns 5-8 of this scene's token
                w = np.stack([wt[1][t][rid], wt[2][t][rid], wt[3][t][tid], wt[4][t][tid]], axis=1)
                feats = np.concatenate([feats, w.astype(np.float32)], axis=1)
            else:
                feats = np.concatenate([feats, np.zeros((n, 4), np.float32)], axis=1)
            feats[~ok] = 0                                                  # invalid observations carry zeros
            X[:, t, :] = feats
            V[:, t] = ok
            doy[t] = doy_of(d)
            sid[t] = SENS_ID[s]
            names.append("%s:%s:%s" % (s, d, r["scene"]))
    if mk is not None:
        mk.close()
    return X, V, doy, sid, names


# ----------------------------------------------------------------------------------------------
# 5. EMBEDDING STORAGE (float32 on a 0-1 scale)
# An embedding tif holds float32 numbers between 0 and 1 (128 bands), nodata = -1.0.
# How a model output becomes such a number: z = (unit vector) * sqrt(128) has about unit variance per dimension;
# u = (clip(z, -4, 4) + 4) / 8 maps it to 0..1. The constants are FIXED, so every season, tile and model uses the
# same scale, and z is recovered exactly (inside the clip range) with z = u * 8 - 4.
# Nodata is -1 because every real value is >= 0, so "valid" simply means "value >= 0".
# ----------------------------------------------------------------------------------------------
EMB_NODATA = -1.0   # stored value meaning "no embedding here" (outside the district or no valid observation)
Z_LIM = 4.0         # z values are clipped to [-4, 4] before they are mapped to 0..1


def z_to_u(z):
    """z (any shape, ~unit variance) -> float32 in 0..1 (what is written into the embedding tif)."""
    return ((np.clip(z, -Z_LIM, Z_LIM) + Z_LIM) / (2 * Z_LIM)).astype(np.float32)


def u_to_z(u):
    """float 0..1 -> float32 on the unit-variance scale the models work in (0 is the centre)."""
    return (np.asarray(u, np.float32) * (2 * Z_LIM) - Z_LIM).astype(np.float32)


def emb_valid(a, axis=0):
    """True where an embedding exists: any band >= 0 along `axis` (nodata is -1 in every band)."""
    return (np.asarray(a) >= 0).any(axis=axis)