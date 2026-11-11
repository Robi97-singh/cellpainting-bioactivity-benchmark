# How to rerun an experiment

One command submits one experiment as a job on the GPU cluster:

    cluster/run_experiment.sh <model> <resolution> [fold|all]

    ./cluster/run_experiment.sh resnet 448 all       # ResNet-50, 448 px, all six folds
    ./cluster/run_experiment.sh celldino 224 3       # frozen Cell-DINO, 224 px, fold 3
    DRY=1 ./cluster/run_experiment.sh resnet 224 0   # print the job without submitting

Follow a run with `kubectl logs -f job/<name> -n <namespace>`; the script prints the name.

## Models

| Model name | Kind | What runs |
|---|---|---|
| resnet, resnet18, effnet, lora_vit_s | End To End  | `classification.py`, then `analysis/make_plots_v3.py` |
| dino_small, dino, dino_large, dinov3, celldino, clip, biomedclip | frozen | `extract_embeddings.py`, then `train_head.py` (learning rate 0.02) |

Resolution is 224 or 448. CLIP and BiomedCLIP are limited to 224 by their positional embeddings.
Folds are 0 to 5, or `all` for a sequential loop over the six folds in one job.
`MODE=finetune` or `MODE=frozen` forces the other path for a model.
`PARAMS=<file>` and `BATCH=<n>` override the parameter file and batch size.

## One-time setup

1. **Cluster.** A Kubernetes cluster with GPU nodes, and the container image
   `abrainone/ai-linux:cu12.6.3-latest`. Dockerfile: `docker_image/Dockerfile`.
2. **Python environment.** Create a virtualenv at `/shared/ssd/home/<user>/venv` (ours used
   Python 3.12) and install `requirements.txt` plus `easydict`.
3. **Data.** The repository expects `data/images/` and `data/training_paper.csv`
   (brightfield: `data/brightfield_images/`, `data/brightfield_training_paper.csv`).
   Build them with the scripts in `data_prep/` and `brightfield/`, then link them:

       ln -s /shared/hdd/data/bioactive data

4. **Cell-DINO (only for celldino).** Clone `https://github.com/facebookresearch/dinov2`
   into `dinov2_repo/` (we used commit 7764ea0). Put the Cell-DINO checkpoint at
   `celldino_weights/cell_dino_vits8_pretrain_cp.pth`. The weights are not in this
   repository; request them from the Cell-DINO authors.
5. **Hugging Face token (only for gated models).** Create a Kubernetes secret named
   `hf-token` with key `token`. The job reads it if it exists.
6. **Permissions.** The job runs as your user id with group 1376. If your cluster uses
   other ids, set `RUN_UID` and `RUN_GID`. The repository, venv and data must be readable
   by that user and group, and the repository must live under `/shared/ssd/home/<user>/`.

## Where results go

- End To End runs write checkpoints and metrics to `src/models/`.
- Frozen runs write embeddings to `<save_dir>/embeddings/` and the per-assay ROC-AUC to `<save_dir>/results/<fold>/plots/per_assay_auc.csv`, with `<save_dir>` = `/shared/ssd/logs/<user>/rerun/<model>_r<res>/`.

Compare against the tables in `results/`. Runs are not bit-for-bit reproducible, so
expect small differences; the fold-to-fold spread is on the order of 0.01 to 0.03
ROC-AUC. Then run `python3 verify/verify_claims.py` to check the published numbers.

## What is tested

The submit script was tested end to end up to the start of training for `resnet 224 0`
and the start of embedding extraction for `celldino 224 0`. The other models use the
same code paths but were not smoke-tested through this script. Brightfield and
explainability runs have their own job files in `cluster/jobs/`.
