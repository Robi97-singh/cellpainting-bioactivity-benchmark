# Foreground / background masking test (ResNet-50, 448 px, 65 random targets)

Question: does the model's score depend on cell pixels more than on other image regions?

Method (run_fgbg_mask_r65.py): a cell mask is built from the image (each stain scaled to its 1-99.9
percentile, maximum over the five stains, Gaussian smoothing sigma 2 px, Otsu threshold). The model is
run on the image with (a) cell pixels, (b) background pixels, or (c) a randomly moved copy of the cell
mask (same area, wrong place) replaced by each stain's median. Score = assay logit; change = logit
after minus logit before.

Result: removing cell pixels changes the score by about 0.5 logit units, no more than removing the same
area in the wrong place (larger for the cell mask in 49% of targets, p = 0.56). Background removal
changes it slightly more, but replaces twice the area. The score is not more sensitive to cell-like
pixels than to other regions of similar size.

Control check: the moved copy of the mask lies on average 29% on real mask pixels (range 0-70%; mask
area 34%), so it is mostly background.

Limits: the mask is a brightness proxy covering about a third of the image, not a cell segmentation;
median replacement can itself change the input; one random shift per target; 65 targets, ResNet only.

Files: fgbg_mask.csv (per target and variant), fgbg_summary.csv / fgbg_summary.md (table and tests).
