# Results in detail

Details behind the summary in the README: statistics, the collaborator note, the brightfield study and the explainability study.

## Benchmark statistics


Mean ROC-AUC over 29 assays, 6-fold CV, compound-level aggregation (matching the reference
paper's protocol). **Public JUMP-CP benchmark (Haslum et al. 2024): 0.660 ± 0.094 (6-fold CV)** --
kept here as the external reference point; it is a different number from any row in this table,
all of which are this project's own reproduction/extension results. Resolution has a small positive effect on the DINO-family frozen probes (224 -> 448: DINOv2-Base
+0.010, DINOv2-Large +0.011, DINOv3-Base +0.012, Cell-DINO 0.000; all within the 0.01-0.02
fold-to-fold standard deviation) and a similar-sized one on fine-tuned ResNet-50 (+0.015). See
`results/benchmark_figures/README.md` for the full resolution comparison.

**Key finding:** under compound-level aggregation, one locked linear-probe recipe for every frozen
backbone, and 6-fold CV with Friedman + Nemenyi testing, fine-tuned CNNs are the strongest arms.
ResNet-50 at 448 px has the highest mean (0.679) and ranks first in all 6 folds among the original
arms (ResNet-50 and the frozen probes; mean 0.621-0.641 for the frozen probes). ResNet-18 (0.676) and
EfficientNet-B3 (0.669) at 448 px, added later (see the paragraph on EfficientNet-B3 and ResNet-18
below), are within one fold SD of ResNet-50; with them included the best of the 16 arms varies by fold
and the three CNNs are not significantly different from one another. In the 9-arm 224 comparison (six
frozen probes + ResNet-50, ResNet-18 and EfficientNet-B3 at 224 px; Friedman over the 29 assays, chi2
= 67.2, p = 2e-11) all three CNNs are significantly better than DINOv2-Base (p <= 0.002), DINOv3-Base
(p <= 0.001), CLIP ViT-L/14 (p <= 0.001) and BiomedCLIP (p < 0.001), marginally better than
DINOv2-Large (p = 0.019-0.048), and not significantly different from Cell-DINO (p = 0.40-0.60). No
frozen pair separates (largest gap: Cell-DINO vs BiomedCLIP, p = 0.15), so neither pretraining domain
nor model scale gives a reliable advantage. A broader 16-arm test over the six folds
(`cv_friedman.py`: frozen probes at both resolutions + the three CNNs; chi2 = 71.8, p = 2e-9) is more
conservative (critical difference 9.4 ranks): the three 448 px CNNs (mean ranks 2.2-2.5) separate from
DINOv2 r224, DINOv3-Base r224, DINOv2-Small, CLIP and BiomedCLIP, but not from the best frozen 448 px
probes or Cell-DINO. LoRA-adapted DINOv2-Small (0.6373) is +0.011 above its frozen backbone and level
with the best frozen 448 probes; it is not part of the significance tests. Caveat: the frozen recipe
(lr 0.02) was selected in a fold-0 sweep on Cell-DINO and applied unchanged to all frozen backbones,
whereas ResNet-50 and LoRA use individually tuned recipes. See `results/benchmark_figures/README.md`
and `docs/Project_Bioactive_Methodology.md` (updated 2026-10-21 with the corrected
benchmark, the brightfield vs fluorescence comparison and the explainability results).

## Contributions

EfficientNet-B3 and ResNet-18: the configurations (`params/params_efficientnet_b3*.json`, `params/params_resnet18*.json` and the EfficientNet-B3 model registration in `defaults/models.py`) are by Bhartendu Parihar, who also trained both models at 448 px and at 224 px (resize 540 then crop 224, a field of view of 0.415). Their results in the table above are re-computed from his saved checkpoints at compound level with `cv_rescore.py` (jobs: `jobs/robin-bioact-effnet-eval.yaml`, `jobs/robin-bioact-r18-eval.yaml`) and are checked by `verify/verify_claims.py` like the other models. Details and caveats are in `results/efficientnet_b3/README.md` and `results/resnet18/README.md`: the results come from re-scoring his saved checkpoints.

## Brightfield vs fluorescence (ResNet-50, 448 px, same 102 plates)

To test whether the fluorescence channels carry information that brightfield lacks, a brightfield ResNet-50 and a five-channel fluorescence ResNet-50 were trained at 448 px on the same 102 plates, one job per fold, over all six folds (same split and compound-level aggregation as above).

| Input | Mean test ROC-AUC (6-fold CV) |
|---|---|
| Fluorescence (5 channels) | 0.710 ± 0.048 |
| Brightfield (1 channel) | 0.694 |

Brightfield minus fluorescence is -0.016 AUC (paired t-test p = 0.19; brightfield better on 2 of 6 folds; rough 95% range -0.042 to +0.011). There is no significant difference, but fluorescence is ahead on four of six folds (folds 2 to 5, by 0.013 to 0.056), so a small fluorescence advantage is not ruled out. These numbers use the 102-plate subset and are not comparable with the full-dataset ResNet-50 row above. Aggregation: `python3 analysis/aggregate_cv.py <model> 448 --outdir results/cv/<name>`; comparison: `compare_resnet.py`.

## Explainability of the frozen models (2026-10)

Explanations were produced for the frozen DINOv3-Base (448 px) and Cell-DINO (224 px) linear heads (standardized CLS embedding + linear head, same heads as the benchmark) and compared with ResNet-50 Grad-CAM as the reference. Methods: ViT-CX (primary, Xie et al. 2023), occlusion, ViT Grad-CAM and gradient-times-activation. The explained score is the assay logit. Faithfulness is deletion and insertion AUC, each standardized (z) against 10 smooth random maps and judged against two image-independent control maps (centre blob and border). Cell focus is the enrichment of the map on the brightest 3% of pixels (1.0 = chance). Intervals are 95% bootstrap intervals over targets.

Main results on 65 randomly sampled (compound, assay) targets (13 per fold, folds 0 to 4, 30 positives and 35 negatives, 27 assays; 57 normal, 6 sparse and 2 noise images, labelled by eye):

* Occlusion is the only method that clearly beats the control maps on both deletion and insertion (DINOv3 -2.63 / +2.57, Cell-DINO -5.03 / +2.21), but its maps are at chance on cell enrichment.
* ViT-CX is not clearly better than a centre blob on random targets (DINOv3 deletion z -0.95 vs -0.40 for the control; insertion z -0.84). Enrichment is 1.30 on DINOv3 and 1.01 on Cell-DINO.
* Gradient-times-activation has the highest enrichment (3.27 DINOv3, 2.29 Cell-DINO), but this is partly mechanical because the map is weighted by activation.
* ResNet-50 Grad-CAM is at chance on enrichment (1.07, 95% CI 0.90 to 1.28). Its deletion and insertion scores were not evaluated.
* An earlier 16-target pilot gave much stronger numbers (e.g. DINOv3 ViT-CX deletion z -7.5), but those targets were the ResNet's most confident cases and half were noise images, so the pilot overstated the methods.

Scripts (usage is in each script's docstring or argument parser): `run_vitcx.py` (explanations and faithfulness metrics; `METHOD=vitcx|occlusion|gradcam|centre`), `sample_targets.py` (random target sampler), `run_gradcam_r65.py` (ResNet Grad-CAM with `#k` assay indices), `make_panel.py` and `strat_panel.py` (pilot figures and intervals), `report_fig.py` (three-example report figure), `agg_r65.py` (65-target faithfulness aggregation), `contact_sheet.py` (numbered contact sheets for labelling image types), `r65_enrich.py` and `r65_resnet.py` (cell enrichment). Caveats: intervals cover target sampling only; the Cell-DINO wrapper resizes images to 128 px internally; the DINOv3 embeddings were extracted under another cluster account (training count 111,968 vs 111,984).
