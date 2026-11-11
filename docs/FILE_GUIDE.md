# File guide


Script locations: `classification.py`, `extract_embeddings.py`, `train_head.py` and `dump_predictions.py` are in the repository root; every other script is in `analysis/`, `brightfield/`, `explainability/` or `cluster/` (see Repository layout).

| File | Role | Regime |
|---|---|---|
| `data_prep/prep_paper_faithful.py` | Builds labeled compound dataset (JUMP metadata + ChEMBL join) | Fluorescence data prep |
| `data_prep/split_paper_faithful.py` | Butina clustering → compound-level 6-fold `split_number` | Fluorescence data prep |
| `brightfield/01_build_brightfield_manifest.py` | Joins `training_paper.csv` against per-plate S3 load-data files to locate brightfield TIFFs | Brightfield data prep |
| `brightfield/02_download_and_process_brightfield.py` | Downloads TIFFs from the public JUMP-CP S3 bucket, percentile-normalizes, writes 1-channel PNGs | Brightfield data prep |
| `brightfield/04_compute_brightfield_stats.py` | Computes real per-channel mean/std from downloaded images | Brightfield data prep |
| `classification.py` | Main CLI: trains (or `--test` evaluates) one model/resolution/fold. Used directly for fine-tuned CNNs; its arg-parsing is reused by `extract_embeddings.py` | Both (entry point) |
| `extract_embeddings.py` | Runs a frozen backbone once per CV split, caches `{embeddings, labels, uids}` to disk — mirrors `classification.py`'s CLI exactly | Frozen probe |
| `train_head.py` | Trains a linear head on cached embeddings; evaluates on TEST split with compound-level aggregation, writes `per_assay_auc.csv` | Frozen probe |
| `aggregate_cv.py` | Aggregates the 6 CV folds into mean ± std test ROC-AUC per model/resolution | Both (convergence point) |
| `cv_rescore.py` | Re-scores saved per-image predictions (`oof_predictions.csv`) at compound level into `cv_summary.csv` and `cv_per_assay.csv` per arm (used for ResNet-18 and EfficientNet-B3) | Fine-tuned CNN (re-scoring) |
| `plot_cv_extra.py` | Per-fold and per-assay plots from `cv_summary.csv` and `cv_per_assay.csv` | Fine-tuned CNN (viz) |
| `make_benchmark_figures.py` | Cross-arm publication figures + `benchmark_summary.csv` (model comparison, resolution effect, Friedman/Nemenyi) | Both (final viz) |
| `make_plots_v3.py` | Per-run diagnostic plots for one `model_name` | Both (final viz) |
| `run_gradcam_v2.py` | Grad-CAM visualization for a fine-tuned ResNet-50 | Fine-tuned CNN (inference/viz) |
| `run_vitcx.py` | Explanations (ViT-CX, occlusion, ViT Grad-CAM, grad x act) and deletion/insertion metrics for frozen heads | Frozen probe (explainability) |
| `run_gradcam_r65.py` | `run_gradcam_v2.py` plus `#k` assay indices, used for the 65 random targets | Fine-tuned CNN (explainability) |
| `sample_targets.py` | Draws random (compound, assay) targets from out-of-fold predictions | Explainability |
| `agg_r65.py`, `r65_enrich.py`, `r65_resnet.py`, `contact_sheet.py` | Aggregation, enrichment and labelling helpers for the 65-target replication | Explainability |
| `defaults/models.py` | Model definitions, including channel-adaptation logic for brightfield (1-channel) arms | Shared |
| `defaults/datasets.py` | Dataset classes, including `BioActBrightfield` | Shared |
