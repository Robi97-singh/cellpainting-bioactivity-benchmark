#!/usr/bin/env python3
"""
run_gradin_r65.py - per-stain gradient x input maps for the random-65 targets.

Same arguments and target list as run_gradcam_r65.py. For each (compound, assay):
  logit = model(x)[0, assay_idx];  logit.backward()
  attr  = x.grad * x            (shape 5 x H x W, one map per input channel, signed)
Saved per target: <out>/<assay>/<compound>_<reason>_gradxinput.npy  (float16, 5xHxW)
Metrics in <out>/gradxinput.csv: share of total |attr| carried by each stain.
Channel order (authors' data prep): DNA, AGP, ER, Mito, RNA.
Unlike Grad-CAM (one map after the first conv has mixed the channels), this gives
a separate map for every input channel. Attribution is a local gradient: it says
where the score is sensitive, not that the pixels are causally responsible.
"""
import os, sys, json, functools
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_gradcam_r65 as g

print = functools.partial(print, flush=True)
CH = ["DNA", "AGP", "ER", "Mito", "RNA"]


def run_one(trainer, x, lbl, cmpd, assay_name, assay_idx, reason, fold, out_dir, rows):
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    orig = model.forward
    model.forward = lambda z: orig(z, return_embedding=False)
    try:
        model.eval()
        xr = x.clone().detach().requires_grad_(True)
        logit = model(xr)[0, assay_idx]
        model.zero_grad(set_to_none=True)
        logit.backward()
        attr = (xr.grad * xr).detach()[0].float().cpu().numpy()  # 5 x H x W
    finally:
        model.forward = orig
    adir = os.path.join(out_dir, str(assay_name).replace("/", "_"))
    os.makedirs(adir, exist_ok=True)
    np.save(os.path.join(adir, "%s_%s_gradxinput.npy" % (cmpd, reason)), attr.astype(np.float16))
    mass = np.abs(attr).reshape(5, -1).sum(1)
    share = mass / max(mass.sum(), 1e-12)
    row = dict(fold=fold, compound=cmpd, assay=assay_name, assay_idx=assay_idx,
               reason=reason, label=lbl[0, assay_idx].item() if lbl.dim() > 1 else lbl.item(),
               logit=float(logit.item()))
    for c, name in enumerate(CH):
        row["share_" + name] = float(share[c])
    rows.append(row)


def main():
    args = g.parse_args()
    targets_df = pd.read_csv(args.targets)
    with open(args.params) as f:
        base_params = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "gradxinput.csv")

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
                        print("  gradxinput %s x %s (%s)" % (cmpd, assay_name, reason))
                        run_one(trainer, x, lb, cmpd, assay_name, assay_idx, reason, fold, args.out, rows)
            if len(found) == len(lookup):
                break
        failed += sum(len(v) for k, v in lookup.items() if k not in found)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("\nDone. %d rows, %d failed targets. Metrics: %s" % (len(df), failed, csv_path))
    if len(df):
        print(df[["share_" + c for c in CH]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
