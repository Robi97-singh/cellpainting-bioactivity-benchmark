# Evaluation notes and history

Why the frozen-backbone results were re-scored at compound level, how the recipe was fixed and re-run, and the original single-fold design that the current benchmark replaces.

## Evaluation protocol fix — frozen-backbone models (2026-10-03)

### What was wrong

All six-fold CV results for frozen-backbone models (DINOv2 small/base/large,
DINOv3-Base, Cell-DINO, CLIP ViT-L/14, BiomedCLIP — anything trained via
`extract_embeddings.py` + `train_head.py`) were computed on **raw per-image
predictions, with no aggregation across a compound's multiple site images**.

The end-to-end CNN pipeline (`classification.py` → `defaults/trainer.py` →
`utils/metrics.py`'s `evaluate_predictions()`) has always aggregated
predictions to compound level before scoring:

```python
preds[assays] = sigmoid(preds[assays].values)
preds_combined = preds.groupby("compound").mean()
```

`train_head.py` — which evaluates a linear head on cached, frozen-backbone
embeddings — never had an equivalent step. It scored every image
independently:

```python
test_logits = head(Xte.to(device)).cpu().numpy()
test_mean, _ = mean_roc_auc(Yte.numpy(), test_logits, do_sigmoid=True)
```

### Why that matters

Every assay label is a compound-level property (ChEMBL potency annotation,
not an image-level one) — all site images of the same well carry an
identical label. Scoring each image as an independent test case is
pseudo-replication: correlated, repeated observations of one ground truth
were fed into the AUC calculation as if they were independent samples, and
skipping the averaging step left in per-image noise (site-to-site cell
confluence, debris, illumination variation) that the multi-site imaging
protocol is specifically designed to be averaged away. Single-image scoring
systematically under-estimates a model's true compound-level discriminative
power.

### Impact

Found while building a matched fluorescence baseline for a brightfield
imaging ablation and noticing the evaluation code paths diverged. A
retrospective audit of all frozen-backbone results showed every one
under-reporting AUC by ~0.04–0.05:

| Model | Per-image (uncorrected) | Compound-level (corrected) | Δ |
|---|---|---|---|
| DINOv2-Large r448 | 0.5868 | **0.6412** | +0.0544 |
| DINOv3-Base r448 | 0.5863 | 0.6398 | +0.0535 |
| DINOv2-Base r448 | 0.5842 | 0.6371 | +0.0529 |
| Cell-DINO r224 | 0.5892 | 0.6333 | +0.0441 |
| Cell-DINO r448 | 0.5899 | 0.6333 | +0.0434 |
| DINOv2-Large r224 | 0.5783 | 0.6299 | +0.0516 |
| DINOv3-Base r224 | 0.5761 | 0.6281 | +0.0520 |
| DINOv2-Base r224 | 0.5749 | 0.6272 | +0.0523 |
| DINOv2-Small r224 | 0.5761 | 0.6259 | +0.0498 |
| CLIP ViT-L/14 r224 | 0.5785 | 0.6257 | +0.0472 |
| BiomedCLIP r224 | 0.5754 | 0.6214 | +0.0460 |

Mean over 6 folds, all 29 assays, all runs of the first author. (DINOv3 r224/r448
and CLIP r224 results from Bhartendu Parihar's runs could not be corrected
retroactively — raw per-image predictions were never saved for those runs,
only the aggregate `per_assay_auc.csv`.)

**Under the verified recipe the best frozen probe is DINOv2-Large r448 (0.6412), but the frozen
probes span only 0.621-0.641, within the fold-to-fold standard deviation; the 0.6448 reported for
DINOv3-Base r448 on 2026-10-03 reproduces to within 0.005 (0.6398).** End-to-end CNN results (ResNet-50, ResNet-18,
EfficientNet-B3) and the LoRA parameter-efficient arm were unaffected — both
go through `trainer.py`/`metrics.py` and already aggregated by compound
correctly.

### Recipe and re-run verification (2026-10-06)

**Frozen-probe recipe (identical for every frozen backbone):** frozen backbone, cached CLS
embeddings, `Linear(D, 29)` head, masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr 0.02,
train-only feature standardization, early stopping on compound-level validation ROC-AUC (patience 6).
The learning rate was chosen in a fold-0 sweep on Cell-DINO (during the original per-image
evaluation phase) and then fixed for all frozen backbones. ResNet-50 and LoRA use their own recipes, so the end-to-end-vs-frozen comparison may include a tuning advantage for the end-to-end trained
arms, and Cell-DINO may hold a small one among the frozen probes.

**Re-run.** The results first reported after this fix (2026-10-03 batch) could not be reproduced
from the code as committed at that time: the `train_head.py` default was `--lr 0.005` (it is now 0.02), whereas the original runs were
launched with `--lr 0.02` (recorded in their `*_head_metrics.json`). Three of them (DINOv2,
DINOv2-Large and Cell-DINO at 448) had in fact been run at lr 0.005 and stopped at the 100-epoch cap
without converging, which produced the earlier finding that 448 "hurts" those backbones. All 11 frozen
configurations (66 runs) were therefore re-run on the committed code with an explicit `--lr 0.02`, and
the replaced results are backed up. Relative to the original batch, the 224 results moved by -0.001
to -0.007 and DINOv3-Base r448 by -0.005, which is attributable to early stopping and checkpoint
selection now using compound-level validation (it stops earlier); the 448 runs that had used the
wrong lr rose by 0.010-0.020. Independent checks: per-assay AUC recomputed from the saved
compound-level `test_preds.csv`/`test_labels.csv` (labels 0 = untested excluded) equals the reported
values to ~1e-16 on three spot-checked runs, and no compound appears in more than one split.
Per-model `results/<group>/<model>/README.md` narratives and
`results/benchmark_figures/README.md` were rewritten from these numbers.

### Fix applied

- `train_head.py` now builds a `(plate, well) → compound` lookup from
  `training_paper.csv`, maps each cached embedding's `uid`
  (`PLATE/WELL_SITE.png`) back to its compound, averages sigmoid
  probabilities per compound, and computes AUC on the aggregated values —
  matching `evaluate_predictions()`'s logic exactly. The old per-image mean
  is still recorded (as `test_mean_roc_auc_per_image_UNCORRECTED` in
  `head_metrics.json`) for audit purposes, but is no longer the reported
  metric.
- All historical results for the 8 affected models (48 fold-runs) were
  corrected in place by re-aggregating their already-saved per-image
  `test_preds.csv`/`test_labels.csv` — no retraining was needed, since the
  frozen backbone and trained head weights were untouched by this fix. The
  original per-image files are preserved alongside the corrected ones as
  `test_preds_per_image_UNCORRECTED.csv`, `test_labels_per_image_UNCORRECTED.csv`,
  and `per_assay_auc_per_image_UNCORRECTED.csv` in each fold's `plots/`
  directory.

### Resolved (2026-10-03)

- **The top "Results" table has been regenerated** from the corrected 6-fold
  CV numbers (see above), replacing the old single-fixed-fold comparison.
- **All benchmark figures/tables were regenerated** (`make_benchmark_figures.py`,
  `make_adaptation_figure.py`, `make_pr_calibration_pareto.py`) from the
  corrected `per_assay_auc.csv` files -- see `results/benchmark_figures/`.
  (Wilcoxon was dropped from this pass; Friedman + Nemenyi remain.) All 11
  affected `results/<group>/<model>/README.md` narratives were fully
  rewritten, not just number-swapped, since several statistical conclusions
  changed or reversed under correction (see each folder's README for specifics;
  the resolution-effect reversal and the Cell-DINO/ResNet50 significance
  collapse are the two largest).
- The collaborator's `dinov3_r224`, `dinov3_r448` and `clip_r224` runs were re-run with the patched `train_head.py` against his cached embeddings. That re-run used the default lr 0.005 instead of the locked 0.02, so it is not comparable and is not included here; the results table uses this project's own re-runs at lr 0.02. His DINOv3 r448 embeddings are identical copies of this account's own, so they are one extraction, not two independent ones.
- Additionally found and fixed during this pass: `train_head.py`'s
  **validation-time** scoring (used for early stopping / checkpoint
  selection) had the same per-image aggregation bug as the test-time scoring
  above. Patched to score validation at compound level too. Spot-checked
  before trusting the already-reported numbers: the old per-image validation
  metric and the corrected compound-level one agreed on epoch-to-epoch
  ordering 97-99% of the time, and picked checkpoints within 0.0001 ROC-AUC
  of each other -- so no retraining was needed and no already-reported
  number changes, but future runs are now scored correctly end to end.
- `clip_vitl14_adamw_224` (a different training recipe -- AdamW, full-image,
  not the locked SGD linear-probe recipe shared by every frozen arm) was
  removed from `results/` as an invalid peer comparison, rather than
  corrected -- it was never comparing like with like.


## Experimental design (original single-fold comparison, superseded above)

*The design below describes the project's original ResNet-50 vs. DINOv2-Base/CLIP/BiomedCLIP
comparison: a single fixed fold, with the three ViTs fully trained end-to-end (AdamW). This
predates, and uses a different recipe and evaluation protocol from, the frozen-linear-probe
6-fold CV benchmark reported in "Results" above (locked SGD recipe, `extract_embeddings.py` +
`train_head.py`, compound-level aggregation). It is kept here as a record of the project's first
pass; the current main benchmark is the one above.*

- **Single fixed fold** (folds 0–3 train / 4 val / 5 test) — identical across all models, so model-to-model comparison is internally valid. Comparison to the paper's 6-fold CV mean is indicative.
- **Per-architecture hyperparameters** — ResNet-50: SGD, lr 1e-3, batch 64. All three ViTs: AdamW, lr 1e-4, batch 16 (matched). ReduceLROnPlateau scheduling for all.
- **Resolution** — ResNet-50 and DINOv2 at 448; CLIP-family at native 224 (positional embeddings do not transfer to 448). Field of view held identical across all models.
- **Single seed** — run-to-run variance ≈ ±0.02 ROC-AUC; only differences above this scale are interpreted. Model ordering between the 448 px end-to-end CNNs (0.669 to 0.679) and the frozen probes (0.621 to 0.641) is robust to it; the ordering among the three end-to-end CNNs (differences of 0.003 to 0.010 at 448 px) is not.

Full reasoning and limitations are in `docs/Project_Bioactive_Methodology.md`.
