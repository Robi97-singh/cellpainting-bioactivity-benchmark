#!/usr/bin/env python3
"""
run_fgbg_mask_r65.py - foreground / background masking test on the 65 random targets.

Same arguments and target list as run_gradcam_r65.py. For every (compound, assay) target the
cell mask is built from the image (per-channel 1-99.9 percentile scaling, maximum over the five
stains, Gaussian smoothing sigma 2 px, Otsu threshold) and the model is run on:
  all              the normal image
  bg_removed       background pixels (outside the mask) replaced, per channel, by that channel's median
  fg_removed       cell pixels (inside the mask) replaced by the channel median
  shifted_removed  control: a copy of the cell mask moved by a random offset (same area, wrong place),
                   replaced by the channel median
Records the assay logit and its change from 'all', plus the area replaced. If the model relies on
cells, fg_removed should lower the score more than bg_removed and more than the shifted control.
Writes <out>/fgbg_mask.csv and prints a summary. Channel order: DNA, AGP, ER, Mito, RNA.
"""
import os, sys, json, functools, zlib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_gradcam_r65 as g

print = functools.partial(print, flush=True)


def otsu(v, bins=256):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.cumsum((h * c)[::-1]) / np.maximum(np.cumsum(h[::-1]), 1))[::-1]
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[1:]) ** 2
    return c[np.argmax(between)]


def cell_mask(x):
    """x: numpy [5,H,W] -> bool mask of cells."""
    s = []
    for c in range(x.shape[0]):
        lo, hi = np.percentile(x[c], [1, 99.9])
        s.append(np.clip((x[c] - lo) / (hi - lo + 1e-9), 0, 1))
    s = gaussian_filter(np.max(s, 0), 2.0)
    return s > otsu(s.ravel())


def variants(x, seed):
    """x: tensor [1,5,H,W] -> list of (name, tensor, area_replaced)."""
    mask = cell_mask(x[0].float().cpu().numpy())
    H, W = mask.shape
    rng = np.random.default_rng(seed)
    dy, dx = int(rng.integers(H // 4, 3 * H // 4)), int(rng.integers(W // 4, 3 * W // 4))
    shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    fill = x.flatten(2).median(2).values[:, :, None, None]   # per-channel median
    t = lambda m: torch.from_numpy(m).to(x.device)[None, None]
    out = [("all", x, 0.0),
           ("bg_removed", torch.where(t(mask), x, fill), float(1 - mask.mean())),
           ("fg_removed", torch.where(t(mask), fill, x), float(mask.mean())),
           ("shifted_removed", torch.where(t(shifted), fill, x), float(shifted.mean()))]
    return out


def run_one(trainer, x, lbl, cmpd, assay_name, assay_idx, reason, fold, rows):
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    orig = model.forward
    model.forward = lambda z: orig(z, return_embedding=False)
    label = lbl[0, assay_idx].item() if lbl.dim() > 1 else lbl.item()
    seed = zlib.crc32((str(cmpd) + '|' + str(assay_name)).encode())
    try:
        model.eval()
        base = None
        for name, xv, area in variants(x, seed):
            with torch.no_grad():
                logit = model(xv)[0, assay_idx].item()
            if name == "all":
                base = logit
            rows.append(dict(fold=fold, compound=cmpd, assay=assay_name, assay_idx=assay_idx,
                             reason=reason, label=label, variant=name, area_replaced=area,
                             logit=logit, d_logit=logit - base))
    finally:
        model.forward = orig


def main():
    args = g.parse_args()
    targets_df = pd.read_csv(args.targets)
    with open(args.params) as f:
        base_params = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "fgbg_mask.csv")

    folds = sorted(targets_df["fold"].unique())
    trainer_tmp = g.load_fold_model(folds[0], args, base_params)
    int_to_labels = trainer_tmp.testloader.dataset.int_to_labels
    label_to_int = {v: k for k, v in int_to_labels.items()}

    rows, failed = [], 0
    for fold in folds:
        ft = targets_df[targets_df["fold"] == fold]
        lookup = {}
        for _, row in ft.iterrows():
            cmpd = row["compound_id"]
            assay_name = row["assay"]
            reason = row["selection_reason"].replace("_high_conf", "")
            key = str(assay_name)
            if key.startswith("#"):
                assay_name = int_to_labels[int(key[1:])]
                key = str(assay_name)
            if key not in label_to_int:
                failed += 1
                continue
            lookup.setdefault(cmpd, []).append((assay_name, label_to_int[key], reason))
        if not lookup:
            continue
        print("Fold %s: %d compounds" % (fold, len(lookup)))
        trainer = trainer_tmp if fold == folds[0] else g.load_fold_model(fold, args, base_params)
        found = set()
        for batch_idx, (images, labels, ids, compounds) in enumerate(trainer.testloader):
            if batch_idx % 200 == 0:
                print("  batch %d, found %d/%d" % (batch_idx, len(found), len(lookup)))
            for i, cmpd in enumerate(compounds):
                if cmpd in lookup and cmpd not in found:
                    found.add(cmpd)
                    x = images[i:i + 1].to(trainer.device_id)
                    lb = labels[i:i + 1]
                    for assay_name, assay_idx, reason in lookup[cmpd]:
                        print("  mask test %s x %s (%s)" % (cmpd, assay_name, reason))
                        run_one(trainer, x, lb, cmpd, assay_name, assay_idx, reason, fold, rows)
            if len(found) == len(lookup):
                break
        failed += sum(len(v) for k, v in lookup.items() if k not in found)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("\nDone. %d rows, %d failed targets. Metrics: %s" % (len(df), failed, csv_path))
    if len(df):
        d = df[df.variant != "all"]
        s = d.groupby(["reason", "variant"]).agg(
            mean_area=("area_replaced", "mean"), mean_d_logit=("d_logit", "mean"),
            median_d_logit=("d_logit", "median"), n=("d_logit", "size")).round(3)
        print(s.to_string())


if __name__ == "__main__":
    main()
