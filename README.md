# Embedding pipeline (LISS-III + AWiFS + EOS-04 SAR) - Murshidabad, 30 m

Put these files in the repo folder (next to `bhoonidhi_year/` and `boundaries/murshidabad.geojson`).
`pip install -r requirements.txt`. Full documentation: the Claude Docs document "Embedding Foundation Model for Indian Satellites".

## Main route (reads the zips directly; every scene is one observation)
| Step | Command | Output |
|---|---|---|
| 0 Download | `python bhoonidhi_v9.py --start 2025-04-01 --end 2025-06-30 --sensors LISS3 AWIFS --workers 1` (`--retry-404` later for missing scenes) | `bhoonidhi_year/` |
| 1 Single stack of all scenes (AWiFS, LISS-III, EOS-04), clipped to the district | `python stack_scenes.py` | `outputs/stack_all_scenes.tif`, `_bands.csv`, `scene_coverage.csv` |
| 2 Cloud mask per scene | `python cloudmask_scenes.py` | `outputs/stack_all_masks.tif`, `stack_all_masks_stats.csv` |
| 3 Training pixels | `python build_samples.py --n 100000` | `outputs/samples.npz` |
| 4a Barlow Twins | `python train.py --epochs 20` then `python embed_to_tif.py` | `model.pt`, `embedding_30m.tif`, `embedding_pca.*` |
| 4b Denoising autoencoder | `python train_dae.py --epochs 20` then `python embed_to_tif.py --model outputs\model_dae.pt --out outputs\embedding_dae_30m.tif` | `model_dae.pt`, `embedding_dae_30m.tif` |
| 4c Convolutional autoencoder (on the DAE embedding) | `python conv_ae.py train --epochs 30` then `python conv_ae.py apply` | `model_conv.pt`, `embedding_conv_30m.tif` |

All of it: `python run_all.py --dae --conv` (omit flags for Barlow Twins only).

## Checking the results
- `python mask_check.py` (cloudiest scenes: image beside mask), `python mask_diag.py --band N`
- `python similarity_auto.py [--emb outputs\embedding_dae_30m.tif --tag dae]` (6 similarity maps), `python similarity.py --lon .. --lat ..`
- `python eval_dae.py` (reconstruction and gap-filling scores on unseen pixels)
- `python pixel_info.py --x .. --y ..` (one pixel's time series; needs the per-date stacks from `preprocess.py`)
- `python export_embeddings.py`, `python export_csv.py` (embedding as numpy / csv)

## Legacy route (per-date stacks; only for pixel_info.py and old masks)
`python preprocess.py liss3 awifs eos04` then `python cloudmask.py liss3 awifs`; `python run_all.py --legacy` does this first.

## Notes
- `stack_scenes.py` takes every zip in `bhoonidhi_year`; do not delete zips you still want in the stack.
- Close QGIS before running anything that writes to `outputs/` (it locks open files).
- Whenever the stack changes (new scenes, new mask) the models must be retrained: the number of observations is fixed inside `model.pt`.
- Embedding tif: int8, 128 bands, nodata -128. emb = q/32; unit vector = emb/sqrt(128).
- SAR: gamma0 dB = 10 log10(DN^2) - 68.255; scenes averaged in linear power, 18 m -> 30 m by area mean.
- Cloud mask values: 0 no data, 1 clear, 2 cloud, 255 outside the district.
