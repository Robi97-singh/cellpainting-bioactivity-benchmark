#!/usr/bin/env python3
"""
vit_summary.py - one figure and one table for the ViT explainability methods (65 random targets).

Reads agg_r65.csv (deletion / insertion z with 95% bootstrap CI) and bootstrap_ci.csv (foreground
enrichment with 95% bootstrap CI) from --dir and writes vit_summary.png, vit_summary.csv and
vit_summary.md to --out.
  deletion z    more negative = removing the marked regions lowers the score faster than random maps
  insertion z   more positive = revealing the marked regions restores the score faster than random maps
  enrichment    mean map value on the brightest 3% of pixels / mean over the image (1.0 = chance)
Centre-box control = a fixed box in the image centre used as a "map" (no information about the image).
ViT-CX uses the 'paper' variant. Missing values (no enrichment computed) are left blank.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="folder with agg_r65.csv and bootstrap_ci.csv")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

agg = pd.read_csv(os.path.join(a.dir, "agg_r65.csv"))
agg = agg[agg["group"] == "all"].set_index(["run", "variant"])
boot = pd.read_csv(os.path.join(a.dir, "bootstrap_ci.csv"))
boot = boot[boot["metric"] == "enrichment"].set_index("method")

MODELS = [("DINOv3", "dinov3", "#4c6e91"), ("Cell-DINO", "celldino", "#c27c3a")]
rows = []
for model, key, col in MODELS:
    spec = [("ViT-CX", (key, "paper"), "ViT-CX"),
            ("occlusion", ("occlusion-" + key, "occlusion"), "occlusion"),
            ("ViT Grad-CAM", ("gradcam-" + key, "vitgradcam"), "ViT Grad-CAM"),
            ("gradient x activation", ("gradcam-" + key, "gradxact"), "grad x act"),
            ("centre-box control", ("centre-" + key, "centre"), None)]
    for name, idx, bname in spec:
        r = agg.loc[idx]
        e = boot.loc[model + " " + bname] if bname and (model + " " + bname) in boot.index else None
        rows.append(dict(model=model, method=name, color="#8a8a8a" if bname is None else col,
                         z_del=r.z_del, del_lo=r.del_lo, del_hi=r.del_hi,
                         z_ins=r.z_ins, ins_lo=r.ins_lo, ins_hi=r.ins_hi,
                         enr=np.nan if e is None else e["mean"],
                         enr_lo=np.nan if e is None else e["mean_lo"],
                         enr_hi=np.nan if e is None else e["mean_hi"]))
if "ResNet Grad-CAM" in boot.index:
    e = boot.loc["ResNet Grad-CAM"]
    rows.append(dict(model="ResNet-50", method="Grad-CAM", color="#555555", z_del=np.nan, del_lo=np.nan,
                     del_hi=np.nan, z_ins=np.nan, ins_lo=np.nan, ins_hi=np.nan,
                     enr=e["mean"], enr_lo=e["mean_lo"], enr_hi=e["mean_hi"]))
T = pd.DataFrame(rows)
T["label"] = T["model"] + ": " + T["method"]

out = T[["model", "method", "z_del", "del_lo", "del_hi", "z_ins", "ins_lo", "ins_hi", "enr", "enr_lo", "enr_hi"]].round(2)
out.to_csv(os.path.join(a.out, "vit_summary.csv"), index=False)


def cell(v, lo, hi):
    return "" if pd.isna(v) else "%.2f [%.2f, %.2f]" % (v, lo, hi)


with open(os.path.join(a.out, "vit_summary.md"), "w") as f:
    f.write("| Model | Method | Deletion z (lower = better) | Insertion z (higher = better) | Foreground enrichment (1 = chance) |\n|---|---|---|---|---|\n")
    for _, r in T.iterrows():
        f.write("| %s | %s | %s | %s | %s |\n" % (r.model, r.method, cell(r.z_del, r.del_lo, r.del_hi),
                                                  cell(r.z_ins, r.ins_lo, r.ins_hi), cell(r.enr, r.enr_lo, r.enr_hi)))

ypos, y = [], 0.0
prev = None
for _, r in T.iterrows():
    if prev is not None and r.model != prev:
        y += 0.6
    ypos.append(y)
    y += 1.0
    prev = r.model
T["y"] = ypos

fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.4), sharey=True)
panels = [("z_del", "del_lo", "del_hi", "Deletion z\nmore negative = better", 0.0),
          ("z_ins", "ins_lo", "ins_hi", "Insertion z\nmore positive = better", 0.0),
          ("enr", "enr_lo", "enr_hi", "Foreground enrichment\nhigher = more weight on bright (cell-like) pixels", 1.0)]
for k, (v, lo, hi, title, ref) in enumerate(panels):
    q = ax[k]
    q.axvline(ref, color="#aa3333", ls="--", lw=1)
    for _, r in T.iterrows():
        if pd.isna(r[v]):
            if k == 2 and r.method != "centre-box control":
                q.text(ref, r.y, " not computed", va="center", fontsize=8, color="#888888")
            continue
        ctrl = r.method == "centre-box control"
        q.errorbar(r[v], r.y, xerr=[[r[v] - r[lo]], [r[hi] - r[v]]], fmt="o" if not ctrl else "D",
                   color=r.color, ecolor=r.color, capsize=3, ms=6, mfc=r.color if not ctrl else "white", lw=1.4)
        q.text(r[hi], r.y, "  %.2f" % r[v], va="center", fontsize=8, color="#333333",
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    q.set_title(title, fontsize=10)
    q.grid(axis="x", color="#e6e6e6", lw=0.8)
    q.set_axisbelow(True)
    for s in ("top", "right"):
        q.spines[s].set_visible(False)
ax[0].set_yticks(T["y"])
ax[0].set_yticklabels(T["label"], fontsize=9)
ax[0].invert_yaxis()
fig.suptitle("ViT attribution methods on the 65 random targets: mean with 95% bootstrap interval. "
             "Dashed line = random-map baseline (z) or chance (enrichment). Diamonds = centre-box control.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(a.out, "vit_summary.png"), dpi=150)
print(open(os.path.join(a.out, "vit_summary.md")).read())
print("written to", a.out)
