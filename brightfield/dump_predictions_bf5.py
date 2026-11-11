#!/usr/bin/env python3
"""
dump_predictions_bf5.py - cross-modality control (inference only).

Runs the FLUORESCENCE-trained ResNet-50 (fluor102 checkpoints, 5 input channels)
on BRIGHTFIELD images of the same 102 plates / same folds. The single brightfield
channel is copied into all 5 input channels. No training happens.

--norm bf     normalise with the brightfield mean/std (0.4972 / 0.1916)
--norm fluor  normalise with the fluorescence per-channel mean/std
Writes <save_dir>/oof_predictions.csv in the same format as dump_predictions.py.
It never touches the default results folder.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import defaults.wrappers as W
from defaults.datasets import BioAct, BioActBrightfield
import dump_predictions as dp

NORM = {"mode": "bf"}


class BioActBF5(BioActBrightfield):
    """Brightfield image copied into 5 identical channels (5-channel model input)."""
    img_channels = 5

    def init_stats(self):
        if NORM["mode"] == "fluor":
            BioAct.init_stats(self)
        else:
            self.mean = (0.4972,) * 5
            self.std = (0.1916,) * 5

    def get_image_data(self, path):
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[:, :, 0]
        return np.repeat(img[:, :, None], 5, axis=2)


_orig_mapper = W.DefaultWrapper.dataset_mapper
W.DefaultWrapper.dataset_mapper = property(
    lambda self: {**_orig_mapper.fget(self), "BioActBF5": BioActBF5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", choices=["bf", "fluor"], default="bf")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--params", default="params/params_fluor102_resnet_cluster.json")
    ap.add_argument("--model_name", default="bioact_fluor102_resnet")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(6)))
    a = ap.parse_args()
    NORM["mode"] = a.norm

    with open(a.params) as f:
        base = json.load(f)
    base["dataset_params"]["dataset"] = "BioActBF5"
    base["dataset_params"]["data_location"] = "data/brightfield_images"

    args = argparse.Namespace(model="resnet", res=a.res, model_name=a.model_name,
                              batch_size=a.batch_size, fold=None)
    dfs = []
    for fold in a.folds:
        print("\n=== fold %d (norm=%s) ===" % (fold, a.norm))
        ck = os.path.join(os.environ.get("BIOACT_SAVE_DIR", "src/models"),
                          "checkpoints", "%s_r%d_fold%d" % (a.model_name, a.res, fold))
        if not os.path.exists(ck):
            print("[ERROR] checkpoint missing: " + ck)
            sys.exit(1)
        df = dp.run_fold(fold, args, base)
        if df is not None:
            dfs.append(df)
    if not dfs:
        print("[ERROR] no predictions produced")
        sys.exit(1)
    pooled = pd.concat(dfs, ignore_index=True)
    os.makedirs(a.save_dir, exist_ok=True)
    out = os.path.join(a.save_dir, "oof_predictions.csv")
    pooled.to_csv(out, index=False)
    print("Saved %d rows, %d compounds, folds %s -> %s" % (
        len(pooled), pooled["compound_id"].nunique(),
        sorted(pooled["fold"].unique().tolist()), out))


if __name__ == "__main__":
    main()
