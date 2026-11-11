#!/usr/bin/env python3
"""
make_control_plot.py - table and plots for the cross-modality control (point 6).

Conditions (mean test ROC-AUC over six Butina folds, 29 assays):
  fluorescence-trained ResNet-50 on fluorescence images   (training-code scoring)
  brightfield-trained ResNet-50 on brightfield images     (training-code scoring)
  fluorescence-trained model on brightfield copied x5, brightfield normalisation   (cv_rescore.py)
  same, fluorescence normalisation                                                (cv_rescore.py)
Writes control_table.md, control_table.csv, control_auc.png and, if per-assay files
exist for both, control_per_assay.png.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--cv", required=True, help="results/cv folder")
ap.add_argument("--rescore", required=True, help="rescore_bf5 folder")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def folds(path):
    d = pd.read_csv(path)
    d = d[d["fold"].astype(str).str.fullmatch(r"\d+")]
    return d["mean_test_roc_auc"].astype(float).to_numpy()


conds = [
    ("Fluorescence-trained,\nfluorescence images", os.path.join(a.cv, "f102_resnet_r448", "cv_summary.csv"), "training code", "#4c6e91"),
    ("Brightfield-trained,\nbrightfield images", os.path.join(a.cv, "brightfield_resnet_r448", "cv_summary.csv"), "training code", "#8a8a8a"),
    ("Fluorescence-trained on\nbrightfield (fluor. norm)", os.path.join(a.rescore, "fluor_on_bf_fluornorm", "cv_summary.csv"), "cv_rescore.py", "#c27c3a"),
    ("Fluorescence-trained on\nbrightfield (bf norm)", os.path.join(a.rescore, "fluor_on_bf_bfnorm", "cv_summary.csv"), "cv_rescore.py", "#d9a86c"),
]
rows, vals = [], []
for name, p, how, col in conds:
    v = folds(p)
    vals.append(v)
    rows.append(dict(condition=name.replace("\n", " "), mean_auc=round(v.mean(), 4), sd_over_folds=round(v.std(ddof=1), 4),
                     min_fold=round(v.min(), 3), max_fold=round(v.max(), 3), scoring=how))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(a.out, "control_table.csv"), index=False)
with open(os.path.join(a.out, "control_table.md"), "w") as f:
    f.write("| Condition | Mean AUC | SD over folds | Fold range | Scoring |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write("| %s | %.3f | %.3f | %.2f-%.2f | %s |\n" % (r["condition"], r["mean_auc"], r["sd_over_folds"],
                                                              r["min_fold"], r["max_fold"], r["scoring"]))
print(T.to_string(index=False))

fig, ax = plt.subplots(figsize=(8.2, 4.6))
for i, ((name, _, how, col), v) in enumerate(zip(conds, vals)):
    ax.bar(i, v.mean(), color=col, width=0.62)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color="black", s=14, zorder=3)
    ax.text(i, v.mean() + 0.015, "%.3f" % v.mean(), ha="center", fontsize=9)
ax.axhline(0.5, color="#aa3333", ls="--", lw=1)
ax.text(len(conds) - 0.45, 0.505, "chance", color="#aa3333", fontsize=8, ha="right")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[0] for c in conds], fontsize=8)
ax.set_ylim(0.4, 0.8)
ax.set_ylabel("mean test ROC-AUC over 29 assays")
ax.set_title("Same 102 plates, same six folds. Dots = single folds.\n"
             "Grey/blue: training-code scoring; orange: cv_rescore.py (different scoring method)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "control_auc.png"), dpi=150)
plt.close(fig)

try:
    ctrl = pd.read_csv(os.path.join(a.rescore, "fluor_on_bf_fluornorm", "cv_per_assay.csv"))
    ref = pd.read_csv(os.path.join(a.cv, "brightfield_resnet_r448", "cv_per_assay.csv"))
    m = ctrl[["assay", "mean"]].merge(ref[["assay", "mean"]], on="assay", suffixes=("_ctrl", "_bf"))
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(m["mean_bf"], m["mean_ctrl"], s=22, color="#c27c3a")
    ax.plot([0.3, 1], [0.3, 1], color="#999999", lw=1)
    ax.axhline(0.5, color="#aa3333", ls="--", lw=0.8)
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
    ax.set_xlabel("brightfield-trained, per-assay AUC")
    ax.set_ylabel("fluorescence-trained on brightfield")
    ax.set_title("Per assay (%d assays); diagonal = equal" % len(m), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "control_per_assay.png"), dpi=150)
    plt.close(fig)
    print("per-assay plot written (%d assays)" % len(m))
except Exception as e:
    print("per-assay plot skipped:", repr(e))
print("written to", a.out)
