# overnight_pure.ps1
# Runs, one after the other, every jute test on the "pure" labels (30 m pixels whose jute share is clear: >= 90 % or <= 10 %).
# Start it from the project folder with the virtual environment active:
#       .\overnight_pure.ps1
# Total time: about 2.5 hours. Do not start any other python job while it runs (RAM crash).
# Every step copies its result file to a name of its own, because the scripts always write to the same file name.

# ---- settings -------------------------------------------------------------------------------------------------------
# Season folder of the first run (Apr-Jun) and the label file without mixed pixels (made by jute_inspect.py --pure 0.9).
$AJ = "outputs\2025_Apr_Jun"
$ND = "outputs\2025_Nov_Dec"
$L  = "outputs\inspect_check\jute_labels_30m_pure.tif"

# every result goes to the Apr-Jun folder first
$env:EMB_OUT = $AJ

# ---- 1) similarity method, Apr-Jun and Nov-Dec embeddings together (256 values per pixel), clean pixels only ----------
Write-Host "STEP 1/5: similarity, both seasons"
python jute\jute_similarity.py --seeds 0 1 2 --no_maps --labels $L --both $AJ $ND
Copy-Item "$AJ\jute_similarity_results.txt" "$AJ\jute_similarity_results_pure_both.txt"

# ---- 2) fine-tuning of the pre-trained Barlow Twins encoder (3 splits), clean pixels only ---------------------------------
Write-Host "STEP 2/5: fine-tuning"
python jute\jute_finetune.py --seeds 0 1 2 --labels $L
Copy-Item "$AJ\jute_finetune_results.txt" "$AJ\jute_finetune_results_pure.txt"

# ---- 3) classifier on the embeddings of both seasons, clean pixels only ----------------------------------------------------
Write-Host "STEP 3/5: classifier, both seasons"
python jute\jute_classify.py --jute_values 1 --seeds 0 1 2 --no_raw --labels $L --both $AJ $ND
Copy-Item "$AJ\jute_results.txt" "$AJ\jute_results_pure_both.txt"

# ---- 4) classifier with the raw-data baseline (21 hand-made features) next to the embeddings, clean pixels only -------------
Write-Host "STEP 4/5: classifier with raw baseline"
python jute\jute_classify.py --jute_values 1 --seeds 0 1 2 --labels $L
Copy-Item "$AJ\jute_results.txt" "$AJ\jute_results_pure_with_raw.txt"

# ---- 5) control: similarity with the Nov-Dec embeddings alone (jute is not in the field), clean pixels only ----------------
Write-Host "STEP 5/5: Nov-Dec control"
$env:EMB_OUT = $ND
python jute\jute_similarity.py --seeds 0 1 2 --no_maps --labels $L
Copy-Item "$ND\jute_similarity_results.txt" "$ND\jute_similarity_results_pure.txt"

# back to the Apr-Jun folder so that later commands in this window work as before
$env:EMB_OUT = $AJ
Write-Host "ALL DONE"
