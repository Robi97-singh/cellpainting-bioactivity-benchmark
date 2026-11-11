# Comparison with the original repository

This repository extends the code of Fredin Haslum et al., Nature Communications 15, 3470 (2024)
(https://github.com/cfredinh/bioactive, Apache-2.0). The comparison below was made file by file against a
fresh clone of that repository (content hashes plus diffs). Credit for the original pipeline belongs to the
original authors; see NOTICE.

## 1. Files kept exactly as in the original

`LICENSE`, `MLP_predictor.py`, `params/params.json`, `docker_image/Dockerfile`, `defaults/__init__.py`,
`utils/__init__.py`, `utils/launch.py`, `utils/pretrained_loader.py`, `utils/system_def.py`,
`data_prep/README.md`, `data_prep/step_by_step.md`, `data_prep/config.json`,
`data_prep/download_and_process.py`, `data_prep/prep_activity_data.py`, `data_prep/data_splitting.py`,
`data_prep/remove_missing_files.py`.

## 2. Original files that were modified

| File | Change |
|---|---|
| `defaults/bases.py` | `torch.load(..., weights_only=False)` for current PyTorch. Checkpoints also store and restore `epoch`, `best_val_target` and `best_val_loss`, so training can be resumed. |
| `defaults/wrappers.py` | Chooses the new model classes from `backbone_type` (DINOv2, DINOv3, CLIP/BiomedCLIP, ConvNeXt, Cell-DINO, LoRA ViT) and registers the `BioActBrightfield` dataset. |
| `defaults/models.py` | Adds the classes `DINOv2Classifier`, `DINOv3Classifier`, `CLIPVisionClassifier`, `ConvNeXtClassifier`, `CellDINOClassifier` and `LoRAViTClassifier` (input-channel adaptation, optional embedding output) and handling for backbones with a `classifier` attribute and for EfficientNet. |
| `defaults/datasets.py` | Adds `BioActBrightfield` (one-channel brightfield input). Imports and function signatures were reorganised; the sparse-matrix loaders import `scipy.sparse` explicitly. |
| `defaults/trainer.py` | Adds logging (hyperparameters, per-epoch summary) and writes a metrics JSON file. |
| `classification.py` | Adds command-line options `--model` and `--model_name`, the input resolution switch (224 / 448 px, `_apply_resolution`) and fold selection (`_fold_splits`). |
| `utils/_utils.py` | Removes the `verbose` argument of the scheduler constructor (newer PyTorch). The focal-loss tensors use the GPU when available and the CPU otherwise. |
| `utils/dist_utills.py` | Returns early when CUDA is not available, so the code also runs on CPU-only machines. |
| `utils/helpfuns.py` | `torch.load(..., weights_only=False)`. |
| `utils/metrics.py` | Writes `test_labels.csv` and `test_preds.csv` to the folder given by `BIOACT_RESULTS_DIR`. |
| `README.md` | Rewritten for this project. |

The loss, optimiser and evaluation logic of the original training code are not otherwise rewritten. The large
files (`models.py`, `datasets.py`, `trainer.py`) were compared by their definitions and imports, not
line by line.

## 3. Dataset construction: ours against the original

| Our script | Original | Differences |
|---|---|---|
| `data_prep/prep_paper_faithful.py` | `prep_activity_data.py` | ChEMBL 37 is used (the original expects ChEMBL 33). Only the Potency rows of the overlapping compounds are read from the database, with the same filter as the original. Unused variables are removed and progress prints are added. Label rules and thresholds are unchanged: active +1, inactive -1, at least 100 compounds per assay, then at least 50 actives and 50 inactives per assay, first over all compounds and again within source_11. |
| `data_prep/split_paper_faithful.py` | `data_splitting.py` | Random seeds are fixed (42), while the original does not seed, so the original folds cannot be recreated exactly. Compounds whose InChI cannot be parsed are skipped. Clusters left over after the sixth fold go into the last fold. The unused Murcko-scaffold step is removed. Fingerprints (Morgan, radius 2, 1024 bits), Butina cutoff (0.7) and six compound-level folds are the same. |
| `data_prep/remove_missing_paper.py` | `remove_missing_files.py` | Rows without an image path are dropped first, then rows whose file does not exist. The result is the same. Per-fold counts are printed. |

All three scripts read their paths from `BIOACTIVE_DATA_DIR`.

**Assays.** Using JUMP-CP source_11 and ChEMBL 37 gives 29 assays, the same count as in the paper. The
ChEMBL 33 version was not re-run, and the 29 assay IDs were not compared with the paper's list (the
training CSV lists them as its numeric column names, so anyone with the paper's list can check).

## 4. Additions

| Folder | Content |
|---|---|
| root | `extract_embeddings.py`, `train_head.py`, `dump_predictions.py` (frozen embeddings and trained heads) |
| `benchmark/` | aggregation of the cross-validation results, statistics (Friedman, Wilcoxon), figures |
| `brightfield/` | brightfield-only and fluorescence-adaptation experiments |
| `explainability/` | Grad-CAM, ViT-CX, occlusion, per-channel ablation, centre-box control, foreground/background masking test |
| `cluster/` | job generators and example job files (Kubernetes) |
| `params/` | parameter files for every model, resolution and fold |
| `results_showcase/` | result tables and selected figures |
| `verify/` | `verify_claims.py`, which recomputes the numbers quoted in the README |
| `docs/` | methodology notes and this comparison |
