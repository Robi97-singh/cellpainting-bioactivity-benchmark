# ViT explainability methods: summary (65 random targets)

Figure: vit_summary.png. Table: vit_summary.md / vit_summary.csv. Data: ../random_65_targets/agg_r65.csv
and bootstrap_ci.csv. Script: vit_summary.py (repository root).

Four methods give a heatmap of which image regions drive the prediction of the frozen DINOv3-Base and
Cell-DINO: ViT-CX, occlusion, ViT Grad-CAM and gradient x activation. Three scores, each averaged over
the 65 targets with a 95% bootstrap interval:
- Deletion z: hide the marked regions first; more negative = the score drops faster than for a random map.
- Insertion z: reveal only the marked regions first; more positive = the score recovers faster.
- Foreground enrichment: weight on the brightest 3% of pixels (a proxy for cells) relative to the image
  average; 1.0 = chance.
The diamonds are a centre-box control: a fixed box in the image centre, which knows nothing about the image.

What it shows:
- Occlusion has the best deletion and insertion scores (deletion z -5.0 Cell-DINO, -2.6 DINOv3), but this
  is partly built in: occlusion makes its map by hiding regions, which is also how deletion is scored.
- On deletion, apart from occlusion, the intervals of all methods overlap the centre-box control's
  (about -0.9 to -1.5 for the methods, -0.4 to -0.5 for the control), so none is clearly separated from a
  map that knows nothing about the image. DINOv3 ViT-CX is below the random baseline on insertion (-0.84).
- On foreground enrichment only gradient x activation clearly prefers bright pixels (3.3 DINOv3, 2.3
  Cell-DINO), and this is partly built in because its map is weighted by activation. ViT-CX (1.30 and
  1.01), Cell-DINO occlusion (1.06) and the ResNet-50 Grad-CAM (1.07) are close to chance.
- Enrichment was not computed for ViT Grad-CAM or for DINOv3 occlusion (shown as "not computed").
- A first pilot on 16 targets made the methods look much stronger; with 65 random targets most of that
  disappeared, so the pilot should not be used to rank the methods.
