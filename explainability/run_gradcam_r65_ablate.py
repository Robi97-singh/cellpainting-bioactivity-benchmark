#!/usr/bin/env python3
"""
run_gradcam_r65_ablate.py - per-channel ablation of ResNet Grad-CAM.

Same arguments and same target list as run_gradcam_r65.py. For every
(compound, assay) target it runs Grad-CAM on:
  all      the normal 5-channel image
  no_DNA, no_AGP, no_ER, no_Mito, no_RNA
           the same image with ONE channel replaced by its own median value
           (a flat "empty" channel; no assumption about the normalisation)
and records, per variant:
  logit / prob      the model score for that assay
  d_logit           logit minus the 'all' logit (large negative = model needs that stain)
  map_corr          Pearson correlation of the heatmap with the 'all' heatmap
  top3_overlap      fraction of the 'all' top-3% pixels that are still in the top 3%
Heatmaps are saved as .npy, metrics in <out>/ablation.csv.
Channel order (from the authors' data prep): DNA, AGP, ER, Mito, RNA.
"""
import os, sys, json, functools
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_gradcam_r65 as g
from pytorch_grad_cam import GradCAM

print = functools.partial(print, flush=True)
CH = ["DNA", "AGP", "ER", "Mito", "RNA"]


def variants(x):
    out = [("all", x)]
    for c, name in enumerate(CH):
        y = x.clone()
        y[:, c] = y[:, c].median()
        out.append(("no_" + name, y))
    return out


def top_idx(m, frac=0.03):
    k = max(1, int(frac * m.size))
    return set(np.argsort(m.ravel())[-k:].tolist())


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_one(trainer, x, lbl, cmpd, assay_name, assay_idx, reason, fold,
            out_dir, rows, model_type):
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    layer = g.get_target_layer(model, model_type)
    orig = model.forward
    model.forward = lambda z: orig(z, return_embedding=False)
    label = lbl[0, assay_idx].item() if lbl.dim() > 1 else lbl.item()
    adir = os.path.join(out_dir, str(assay_name).replace("/", "_"))
    os.makedirs(adir, exist_ok=True)
    try:
        cam = GradCAM(model=model, target_layers=[layer])
        base_map, base_top, base_logit = None, None, None
        for name, xv in variants(x):
            with torch.no_grad():
                logit = model(xv)[0, assay_idx].item()
            m = cam(input_tensor=xv, targets=[g.AssayTarget(assay_idx)])[0]
            np.save(os.path.join(adir, "%s_%s_%s.npy" % (cmpd, reason, name)), m)
            if name == "all":
                base_map, base_top, base_logit = m, top_idx(m), logit
            rows.append(dict(
                fold=fold, compound=cmpd, assay=assay_name, assay_idx=assay_idx,
                reason=reason, label=label, variant=name,
                logit=logit, prob=float(torch.sigmoid(torch.tensor(logit))),
                d_logit=logit - base_logit,
                map_corr=corr(m, base_map),
                top3_overlap=len(top_idx(m) & base_top) / max(1, len(base_top)),
            ))
    finally:
        model.forward = orig


def main():
    args = g.parse_args()
    targets_df = pd.read_csv(args.targets)
    with open(args.params) as f:
        base_params = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "ablation.csv")

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
                        print("  ablate %s x %s (%s)" % (cmpd, assay_name, reason))
                        run_one(trainer, x, lb, cmpd, assay_name, assay_idx,
                                reason, fold, args.out, rows, args.model)
            if len(found) == len(lookup):
                break
        failed += sum(len(v) for k, v in lookup.items() if k not in found)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("\nDone. %d rows, %d failed targets. Metrics: %s" % (len(df), failed, csv_path))
    if len(df):
        s = df[df.variant != "all"].groupby(["reason", "variant"])[
            ["d_logit", "map_corr", "top3_overlap"]].mean().round(3)
        print(s.to_string())


if __name__ == "__main__":
    main()
