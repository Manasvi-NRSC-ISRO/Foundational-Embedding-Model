# jute_maps.ps1
# Makes the pictures for looking at (to show the supervisor): for each of the three seasons, the jute map predicted from the
# Barlow Twins embedding (similarity method) and a figure that puts it next to the reference jute map:
#       green = jute in both, red = false alarm (predicted jute, map says not), blue = missed jute.
#
# Start it from the project folder with the virtual environment ACTIVE (the prompt must start with "(venv)"):
#       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#       .\jute_maps.ps1
#
# Takes about 30-45 minutes. Do not run another python job at the same time (RAM) and keep QGIS closed.
#
# What it writes into every season folder (outputs\2025_Apr_Jun, outputs\2025_Jul_Aug, outputs\2025_Nov_Dec):
#       jute_simmap_barlow.tif   float32 map of the similarity score of the whole district (for QGIS)
#       jute_simpred_barlow.tif  uint8 map, 1 = jute, 0 = not jute, 255 = nodata (for QGIS)
#       jute_compare_sim_barlow.png   the figure: reference map | predicted map | where they differ
#       jute_pixel_counts_sim_barlow.txt / .csv   numbers of pixels: train / test blocks, true jute, false alarms, misses
# Old maps of the same name (made on 11 Nov with the all-pixel labels) are MOVED to a sub-folder old_maps, not deleted.
# The fingerprints are learned on the CLEAN labels (jute share >= 90 % or <= 10 %); the figure compares with the FULL reference map,
# so it shows the honest picture of the whole district, mixed cells included.

# ---- 0) settings ------------------------------------------------------------------------------------------------------
$CLEAN = "outputs\inspect_check\jute_labels_30m_pure.tif"   # labels for learning the fingerprints (made by jute_inspect.py --pure 0.9)
$FULL  = "outputs\jute_labels_30m.tif"                      # the full reference map at 30 m, used only for the picture

# season folder = embedding file of that season
$seasons = @(
    @{ dir = "outputs\2025_Apr_Jun"; emb = "embedding_barlow_2025-04-14_to_2025-06-30.tif" },
    @{ dir = "outputs\2025_Jul_Aug"; emb = "embedding_barlow_2025-07-02_to_2025-08-31.tif" },
    @{ dir = "outputs\2025_Nov_Dec"; emb = "embedding_barlow_2025-11-02_to_2025-12-31.tif" }
)

# Stop at once if the virtual environment is not active.
python -c "import numpy, rasterio, matplotlib" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "STOP: venv not active. Activate it first." -ForegroundColor Red; exit 1 }

# ---- 1) one season after the other ----------------------------------------------------------------------------------
foreach ($s in $seasons) {
    $dir = $s.dir
    $embfile = Join-Path $dir $s.emb
    Write-Host ""
    Write-Host "=== $dir  ($(Get-Date -Format 'HH:mm'))" -ForegroundColor Cyan
    if (-not (Test-Path $embfile)) { Write-Host "STOP: $embfile not found" -ForegroundColor Red; exit 1 }

    # all jute scripts write into the folder named by EMB_OUT
    $env:EMB_OUT = $dir

    # move the old maps of this season (if any) into old_maps\ so nothing is overwritten or lost
    $old = Get-ChildItem $dir -File | Where-Object { $_.Name -match "^jute_(simmap|simpred|compare_sim)_barlow\.(tif|png)$" }
    if ($old) {
        New-Item -ItemType Directory -Force (Join-Path $dir "old_maps") | Out-Null
        $old | ForEach-Object { Move-Item $_.FullName (Join-Path $dir ("old_maps\" + $_.Name)) -Force }
        Write-Host "  old maps moved to $dir\old_maps"
    }

    # similarity method with the whole-district maps switched on (no --no_maps), one split (seed 0), one embedding (Barlow Twins)
    python jute\jute_similarity.py --seeds 0 --labels $CLEAN --emb "barlow=$embfile"
    if ($LASTEXITCODE -ne 0) { Write-Host "STOP: jute_similarity.py failed in $dir" -ForegroundColor Red; exit 1 }

    # the figure: reference map | predicted map | differences. --split_labels tells the script which labels the fingerprints were
    # learned with, so that it can also COUNT the pixels (whole district / train blocks / test blocks: true jute, false alarms,
    # misses) and write jute_pixel_counts_sim_barlow.txt / .csv next to the figure (seed 0 = the split of the maps).
    python jute\jute_map_compare.py --labels $FULL --split_labels $CLEAN --seed 0
    if ($LASTEXITCODE -ne 0) { Write-Host "STOP: jute_map_compare.py failed in $dir" -ForegroundColor Red; exit 1 }
}

# back to the Apr-Jun folder so that later commands in this window work as before
$env:EMB_OUT = "outputs\2025_Apr_Jun"
Write-Host ""
Write-Host "ALL DONE  ($(Get-Date -Format 'HH:mm'))" -ForegroundColor Green