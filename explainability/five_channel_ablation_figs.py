#!/usr/bin/env python3
"""
five_channel_ablation_figs.py - one figure per random target, labelled to match the code.

Row 1: the five stains in grey (DNA, AGP, ER, Mito, RNA) and the ONE Grad-CAM heatmap
       (ResNet-50, last conv block, computed from all 5 channels).
Row 2: the heatmap recomputed with one stain removed (replaced by its median), the score
       change for that assay, and the overlay on the stain whose removal changed the score most.
Maps are scaled 0-1 individually. Needs ablation.csv and the saved .npy maps of
run_gradcam_r65_ablate.py plus the image .npz files used by the earlier figure script.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_panel import fit

CH = ["DNA", "AGP", "ER", "Mito", "RNA"]


def scale(c):
    lo, hi = np.percentile(c, [1, 99.5])
    return np.clip((c - lo) / (hi - lo + 1e-8), 0, 1)


def norm01(m):
    m = m.astype(np.float32)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="xai_scratch_r65 folder")
ap.add_argument("--abl", required=True, help="folder with ablation.csv and the ablation maps")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0, help="0 = all targets")
ap.add_argument("--res", type=int, default=224, help="model input size shown in the title")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ref = sorted(glob.glob(os.path.join(a.root, "vitcx_dinov3*r448_r448")))[0]
D = pd.read_csv(os.path.join(a.abl, "ablation.csv"))
D["assay"] = D["assay"].astype(str)
keys = D[D.variant == "all"][["fold", "compound", "assay", "reason", "label", "logit"]].drop_duplicates()

done = skipped = 0
for _, k in keys.iterrows():
    if a.n and done >= a.n:
        break
    fold, comp, assay, reason = int(k.fold), k.compound, k.assay, k.reason
    npz = os.path.join(ref, "f%d_%s_%s.npz" % (fold, comp, assay))
    adir = os.path.join(a.abl, assay.replace("/", "_"))
    if not os.path.isfile(npz):
        skipped += 1
        continue
    x = np.load(npz)["x"].astype(np.float32)
    assert x.ndim == 3 and x.shape[0] == 5, x.shape
    sub = D[(D.compound == comp) & (D.assay == assay) & (D.reason == reason)].set_index("variant")
    try:
        maps = {v: norm01(fit(np.load(os.path.join(adir, "%s_%s_%s.npy" % (comp, reason, v))), x.shape[1:]))
                for v in ["all"] + ["no_" + c for c in CH]}
    except FileNotFoundError:
        skipped += 1
        continue
    d = {c: float(sub.loc["no_" + c, "d_logit"]) for c in CH}
    best = max(CH, key=lambda c: abs(d[c]))
    grey = [scale(x[c]) for c in range(5)]

    fig, ax = plt.subplots(2, 6, figsize=(18, 7.6))
    for j, c in enumerate(CH):
        ax[0, j].imshow(grey[j], cmap="gray"); ax[0, j].set_title(c, fontsize=11)
    ax[0, 5].imshow(maps["all"], cmap="jet", vmin=0, vmax=1)
    ax[0, 5].set_title("Grad-CAM heatmap\n(all 5 channels)", fontsize=11)
    for j, c in enumerate(CH):
        ax[1, j].imshow(maps["no_" + c], cmap="jet", vmin=0, vmax=1)
        ax[1, j].set_title("without %s\nscore change %+.2f" % (c, d[c]), fontsize=10)
    bi = CH.index(best)
    ax[1, 5].imshow(grey[bi], cmap="gray"); ax[1, 5].imshow(maps["all"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[1, 5].set_title("overlay on %s\n(removing it changed the score most)" % best, fontsize=10)
    for q in ax.ravel():
        q.axis("off")
    fig.suptitle("ResNet-50 (%d px) | fold %d | compound %s | assay %s | label %+d (%s) | logit %.2f | "
                 "Grad-CAM from last conv block; maps scaled 0-1 separately; score change = logit change"
                 % (a.res, fold, comp, assay, int(k.label), reason.replace("random_", ""), k.logit), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.5)
    fig.savefig(os.path.join(a.out, "f%d_%s_%s.png" % (fold, comp, assay)), dpi=100)
    plt.close(fig)
    done += 1
print(done, "figures written to", a.out, "| skipped", skipped)
