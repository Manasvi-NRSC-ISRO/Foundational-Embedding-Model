# season_pipeline.ps1
# Runs the WHOLE pipeline for ONE new season, one step after the other, and stops at the first step that fails (so that a
# broken step never feeds garbage into the next one). It only chains the commands that were run by hand for the other seasons
# (Apr-Jun, Jul-Aug, Nov-Dec); nothing in it is new science.
#
#   step 1  preprocess_stack_scenes.py   raw zips -> ONE stack tif of the season (satellite_stack_<tag>.tif) + scene table
#   step 2  cloudmask_scenes.py          cloud mask of every optical scene (satellite_cloudmask_<tag>.tif)
#   step 3  fill_rescale_stack.py        fills no-data inside the district, SAR rescaled to 0..10000 (..._filled_<tag>.tif)
#   step 4  weather\weather_prep.py      IMD rain / Tmax / Tmin of the season -> weather_IMD_*_<tag>.npz / .tif
#   step 5  merge_stack_layers.py        satellite stack + weather in ONE tif (satellite_weather_stack_<tag>.tif)
#   step 6  build_samples.py             100,000 random district pixels with all their observations (samples.npz)
#   step 7  barlow_twins.py              self-supervised training, 20 epochs (model.pt)
#   step 8  embed_to_tif.py              the model applied to every pixel -> embedding_barlow_<tag>.tif (128 bands)
#   step 9  validate_embeddings.py       the 8 standard checks
#   step 10 validate_embeddings.py       the 2 extra checks (--robust, --probe); a failure here does NOT stop the script
#
# HOW TO START (project folder, virtual environment ACTIVE: the prompt must start with "(venv)"):
#       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#       .\season_pipeline.ps1 -Name 2025_Sep_Oct -Data "Z:\Manasvi\bhoonidhi_SepOct" -Start 2025-09-01 -End 2025-10-31
#       .\season_pipeline.ps1 -Name 2025_Jan_Mar -Data "Z:\Manasvi\bhoonidhi_JanMar" -Start 2025-01-01 -End 2025-03-31
#
#   -Name   name of the season folder that is created under outputs\ (outputs\<Name>)
#   -Data   folder with the downloaded zips of this season (the one made by bhoonidhi_download.py --out ...)
#   -Start / -End   first and last day of the season, YYYY-MM-DD. They select the scenes of the stack (--periods) AND the
#                   days of the weather file, so use whole months as for the other seasons.
#   -Grd    folder with the IMD .grd files (default: GRD_DATA inside the project). The weather step needs the daily files of
#           the season; if they are missing, step 4 stops with an error and you have to add them first.
#   -FromStep   restart at this step number (1..10) after a failure, so that finished steps are not repeated.
#               Example: .\season_pipeline.ps1 -Name 2025_Sep_Oct -Data ... -Start ... -End ... -FromStep 4
#
# IMPORTANT
#   * Do NOT run any other python job at the same time (RAM): a second heavy job caused a MemoryError before. QGIS closed.
#   * Do not open, rename or move files of the season folder while the script runs.
#   * The steps 1-5 read the zips: they must be COMPLETE (the download has finished) before you start.
#   * Steps 7 and 8 take about 3 hours together on the CPU (as for Jul-Aug); the whole run is several hours.
#   * Nothing of another season is touched: every file is written into outputs\<Name>.

param(
    [Parameter(Mandatory = $true)] [string]$Name,
    [Parameter(Mandatory = $true)] [string]$Data,
    [Parameter(Mandatory = $true)] [string]$Start,
    [Parameter(Mandatory = $true)] [string]$End,
    [string]$Grd = "D:\Manasvi\Embedding-Foundational-Model\GRD_DATA",
    [int]$FromStep = 1
)

# ---- 0) settings ------------------------------------------------------------------------------------------------------
# common.py reads these two environment variables: EMB_OUT = season output folder, EMB_DATA = folder with the zips.
$env:EMB_OUT  = Join-Path "outputs" $Name
$env:EMB_DATA = $Data
$period = "${Start}:${End}"                      # the START:END text that preprocess_stack_scenes.py and weather_prep.py take

# Stop at once if the virtual environment is not active (the packages are not found), instead of failing in every step.
python -c "import numpy, torch, rasterio" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "STOP: venv not active (numpy/torch/rasterio not found). Activate it first." -ForegroundColor Red; exit 1 }

# Stop at once if the zip folder or the weather folder does not exist.
if (-not (Test-Path $Data)) { Write-Host "STOP: zip folder $Data not found" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Grd))  { Write-Host "STOP: weather folder $Grd not found" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force $env:EMB_OUT | Out-Null

Write-Host "Season $Name   zips: $Data   days: $Start .. $End   output: $env:EMB_OUT   from step $FromStep" -ForegroundColor Yellow

# Helper: run one step (a script block with ONE python command) and stop the whole script if it fails. Steps below the
# requested -FromStep are skipped, so a failed run can be continued where it stopped.
function Step([int]$n, [string]$title, [scriptblock]$cmd) {
    if ($n -lt $FromStep) { Write-Host "skip step $n ($title)"; return }
    Write-Host ""
    Write-Host "=== step $n : $title  ($(Get-Date -Format 'HH:mm'))" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STOP: step $n failed -> $title.  After fixing it restart with  -FromStep $n" -ForegroundColor Red
        exit 1
    }
}

# ---- 1) stack: every zip of the season -> ONE tif with one group of bands per scene ---------------------------------------
Step 1 "preprocess_stack_scenes (stack of the season)" { python preprocess_stack_scenes.py --periods $period }

# ---- 2) cloud mask of every optical scene (brightness rule, see the header of cloudmask_scenes.py) ------------------------
Step 2 "cloudmask_scenes" { python cloudmask_scenes.py }

# ---- 3) fill no-data inside the district and rescale SAR to 0..10000 (the raw stack is never changed) -----------------------
Step 3 "fill_rescale_stack" { python fill_rescale_stack.py }

# ---- 4) IMD weather of the season: daily values + the cell map on the 30 m grid -----------------------------------------------
Step 4 "weather_prep" { python weather\weather_prep.py --root $Grd --periods $period }

# ---- 5) satellite stack + weather in ONE tif (default: the filled stack and the weather tif of the season folder) ---------
Step 5 "merge_stack_layers" { python merge_stack_layers.py }

# ---- 6) training samples: 100,000 random pixels inside the district, each with all its observations ----------------------
Step 6 "build_samples (100000 pixels)" { python build_samples.py --n 100000 }

# ---- 7) Barlow Twins training, 20 epochs (the same as the other seasons) -------------------------------------------------------
Step 7 "barlow_twins (20 epochs)" { python barlow_twins.py --epochs 20 }

# ---- 8) apply the trained model to every pixel of the district -> embedding_barlow_<tag>.tif ------------------------------
Step 8 "embed_to_tif barlow" { python embed_to_tif.py --kind barlow --block 8 }

# ---- 9) the 8 standard checks of the embedding -------------------------------------------------------------------------------------
Step 9 "validate_embeddings barlow" { python validate_embeddings.py --kind barlow }

# ---- 10) the 2 extra checks: robustness to missing scenes / sensors and the multi-task probe -------------------------------
# These need torch and the raw scenes of a small sample of pixels. A failure here is only reported: the embedding itself is
# finished and checked by step 9.
if (10 -ge $FromStep) {
    Write-Host ""
    Write-Host "=== step 10 : validate_embeddings --robust --probe  ($(Get-Date -Format 'HH:mm'))" -ForegroundColor Cyan
    python validate_embeddings.py --kind barlow --robust --probe
    if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: the extra checks failed (the embedding is fine). Run them again by hand." -ForegroundColor Yellow }
}

# back to the Apr-Jun folder and the default zips so that later commands in this window work as before
$env:EMB_OUT  = "outputs\2025_Apr_Jun"
Remove-Item Env:\EMB_DATA -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "ALL DONE for $Name  ($(Get-Date -Format 'HH:mm'))" -ForegroundColor Green
