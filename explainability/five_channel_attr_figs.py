#!/usr/bin/env python3
"""
five_channel_attr_figs.py - one figure per random target, labelled to match the code.

Row 1: the five stains in grey (DNA, AGP, ER, Mito, RNA) + ONE Grad-CAM heatmap
       (last conv block; all 5 stains together, because the first conv mixes them).
Row 2: per-stain gradient x input (|attribution| of each input channel, smoothed for
       display, each scaled 0-1 on its own) + bar chart of each stain's share of the total.
Row 3: Grad-CAM recomputed with one stain replaced by its median, with the score (logit)
       change; last panel = overlay on the stain whose removal changed the score most.
Inputs: ablation.csv + maps from run_gradcam_r65_ablate.py, gradxinput maps from
run_gradin_r65.py, and the image .npz files in <root>/vitcx_dinov3*r448_r448.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from make_panel import fit

CH = ["DNA", "AGP", "ER", "Mito", "RNA"]


def scale(c):
    lo, hi = np.percentile(c, [1, 99.5])
    return np.clip((c - lo) / (hi - lo + 1e-8), 0, 1)


def norm01(m):
    m = m.astype(np.float32)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def attr_map(a, shape):
    """|gradient x input| of one stain, smoothed for display, scaled 0-1 on its own."""
    a = gaussian_filter(np.abs(a.astype(np.float32)), sigma=max(1.0, a.shape[0] / 112.0))
    a = np.clip(a / (np.percentile(a, 99.5) + 1e-12), 0, 1)
    return a if a.shape == tuple(shape) else fit(a, shape)


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="xai_scratch_r65 folder")
ap.add_argument("--abl", required=True, help="folder with ablation.csv and the ablation maps")
ap.add_argument("--gradin", required=True, help="folder with run_gradin_r65.py maps")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=224, help="model input size shown in the title")
ap.add_argument("--n", type=int, default=0, help="0 = all targets")
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
    sub_a = assay.replace("/", "_")
    npz = os.path.join(ref, "f%d_%s_%s.npz" % (fold, comp, assay))
    gp = os.path.join(a.gradin, sub_a, "%s_%s_gradxinput.npy" % (comp, reason))
    if not (os.path.isfile(npz) and os.path.isfile(gp)):
        skipped += 1
        continue
    x = np.load(npz)["x"].astype(np.float32)
    assert x.ndim == 3 and x.shape[0] == 5, x.shape
    sub = D[(D.compound == comp) & (D.assay == assay) & (D.reason == reason)].set_index("variant")
    try:
        maps = {v: norm01(fit(np.load(os.path.join(a.abl, sub_a, "%s_%s_%s.npy" % (comp, reason, v))), x.shape[1:]))
                for v in ["all"] + ["no_" + c for c in CH]}
    except FileNotFoundError:
        skipped += 1
        continue
    d = {c: float(sub.loc["no_" + c, "d_logit"]) for c in CH}
    best = max(CH, key=lambda c: abs(d[c]))
    grey = [scale(x[c]) for c in range(5)]
    ga = np.load(gp)
    gmaps = [attr_map(ga[c], x.shape[1:]) for c in range(5)]
    mass = np.abs(ga.astype(np.float32)).reshape(5, -1).sum(1)
    share = 100.0 * mass / max(mass.sum(), 1e-12)

    fig, ax = plt.subplots(3, 6, figsize=(18, 11.2))
    # row 1
    for j, c in enumerate(CH):
        ax[0, j].imshow(grey[j], cmap="gray"); ax[0, j].set_title(c, fontsize=11)
    ax[0, 5].imshow(maps["all"], cmap="jet", vmin=0, vmax=1)
    ax[0, 5].set_title("one Grad-CAM map\n(from all 5 stains together)", fontsize=11)
    # row 2
    for j, c in enumerate(CH):
        ax[1, j].imshow(gmaps[j], cmap="magma", vmin=0, vmax=1)
        ax[1, j].set_title("%s: gradient x input\n(%.0f%% of total)" % (c, share[j]), fontsize=10)
    bar = ax[1, 5]
    bar.barh(CH[::-1], share[::-1], color="#555555")
    bar.axvline(20, color="#bbbbbb", lw=1, ls="--")
    bar.set_xlim(0, 60)
    bar.set_title("share of total |gradient x input|\n(dashed line = equal share, 20%)", fontsize=10)
    bar.tick_params(labelsize=9)
    # row 3
    for j, c in enumerate(CH):
        ax[2, j].imshow(maps["no_" + c], cmap="jet", vmin=0, vmax=1)
        ax[2, j].set_title("Grad-CAM, %s removed\nscore change %+.2f" % (c, d[c]), fontsize=10)
    bi = CH.index(best)
    ax[2, 5].imshow(grey[bi], cmap="gray")
    ax[2, 5].imshow(maps["all"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[2, 5].set_title("Grad-CAM over %s\n(its removal changed the score most)" % best, fontsize=10)

    for q in ax.ravel():
        if q is not bar:
            q.axis("off")
    fig.suptitle("ResNet-50 (%d px) | fold %d | compound %s | assay %s | label %+d (%s) | logit %.2f\n"
                 "row 1: Grad-CAM, last conv block | row 2: per-stain gradient x input, each scaled 0-1 on its own | "
                 "row 3: Grad-CAM with one stain replaced by its median; score change = logit change"
                 % (a.res, fold, comp, assay, int(k.label), reason.replace("random_", ""), k.logit), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    fig.savefig(os.path.join(a.out, "f%d_%s_%s.png" % (fold, comp, assay)), dpi=100)
    plt.close(fig)
    done += 1
print(done, "figures written to", a.out, "| skipped", skipped)
