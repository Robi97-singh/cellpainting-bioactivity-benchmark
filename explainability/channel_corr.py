#!/usr/bin/env python3
"""
channel_corr.py - how similar are the five stains in the 65 random targets?

For every target image it computes the Pearson correlation between each pair of
channels (over all pixels), then summarises over targets. Correlation is unaffected
by per-channel scaling, so the image normalisation does not matter.
Optional --gradin: the same for the |gradient x input| maps of the five stains.
Prints a mean 5x5 matrix and, per pair, min / median / max over targets.
"""
import argparse, glob, os
import numpy as np
import pandas as pd

CH = ["DNA", "AGP", "ER", "Mito", "RNA"]
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="xai_scratch_r65 folder")
ap.add_argument("--abl", required=True, help="folder with ablation.csv (defines the 65 targets)")
ap.add_argument("--gradin", default=None, help="gradin folder (optional)")
a = ap.parse_args()

ref = sorted(glob.glob(os.path.join(a.root, "vitcx_dinov3*r448_r448")))[0]
D = pd.read_csv(os.path.join(a.abl, "ablation.csv"))
D["assay"] = D["assay"].astype(str)
keys = D[D.variant == "all"][["fold", "compound", "assay", "reason"]].drop_duplicates()


def corr_matrix(arr):
    f = arr.reshape(5, -1).astype(np.float64)
    if (f.std(1) == 0).any():
        return None
    return np.corrcoef(f)


def summarise(title, mats):
    m = np.array(mats)
    print("\n%s  (%d targets)" % (title, len(m)))
    print("mean correlation matrix")
    print(pd.DataFrame(m.mean(0), index=CH, columns=CH).round(2).to_string())
    print("per pair over targets: min / median / max")
    for i, j in PAIRS:
        v = m[:, i, j]
        print("  %-4s-%-4s  %.2f / %.2f / %.2f" % (CH[i], CH[j], v.min(), np.median(v), v.max()))
    cells = [m[:, i, j] for i, j in PAIRS if i > 0]
    allc = np.concatenate(cells)
    print("pairs among AGP/ER/Mito/RNA: mean %.2f, share of target-pairs above 0.9: %.0f%%"
          % (allc.mean(), 100 * (allc > 0.9).mean()))


img, att, skipped = [], [], 0
for _, k in keys.iterrows():
    npz = os.path.join(ref, "f%d_%s_%s.npz" % (int(k.fold), k.compound, k.assay))
    if not os.path.isfile(npz):
        skipped += 1
        continue
    x = np.load(npz)["x"].astype(np.float32)
    c = corr_matrix(x)
    if c is None:
        skipped += 1
        continue
    img.append(c)
    if a.gradin:
        gp = os.path.join(a.gradin, k.assay.replace("/", "_"), "%s_%s_gradxinput.npy" % (k.compound, k.reason))
        if os.path.isfile(gp):
            ca = corr_matrix(np.abs(np.load(gp).astype(np.float32)))
            if ca is not None:
                att.append(ca)

summarise("IMAGE channels", img)
if att:
    summarise("|gradient x input| maps", att)
print("\nskipped:", skipped)
