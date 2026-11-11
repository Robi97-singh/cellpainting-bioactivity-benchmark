#!/usr/bin/env bash
# Submit one replication run as a Kubernetes job on the cluster.
#
#   cluster/run_experiment.sh <model> <res> [fold|all]
#
#   ./cluster/run_experiment.sh resnet 448 all        # ResNet-50, 448 px, all six folds
#   ./cluster/run_experiment.sh celldino 224 3        # frozen Cell-DINO, 224 px, fold 3
#   DRY=1 ./cluster/run_experiment.sh resnet 224 0    # print the job, do not submit
#
# Fine-tuned models  (resnet resnet18 effnet lora_vit_s):  classification.py
# Frozen models      (dino dino_small dino_large dinov3 celldino clip biomedclip):
#                    extract_embeddings.py, then train_head.py
# Force the other path with MODE=finetune or MODE=frozen.
#
# Settings you may need to override (environment variables):
#   NS        cluster namespace = user name (default: current kubectl namespace)
#   RUN_UID   numeric user id the pod runs as (default: your own id); RUN_GID group (default 1376, the shared group that owns the venv)
#   VENV      virtualenv to activate (default /shared/ssd/home/$NS/venv)
#   PARAMS    params file, replaces the automatic choice
#   BATCH     batch size for fine-tuned runs
set -euo pipefail

usage() { sed -n '2,15p' "$0"; exit 1; }
MODEL="${1:-}"; RES="${2:-}"; FOLD="${3:-all}"
[ -n "$MODEL" ] && [ -n "$RES" ] || usage
case "$RES" in 224|448) ;; *) echo "res must be 224 or 448" >&2; exit 1;; esac
case "$FOLD" in all|0|1|2|3|4|5) ;; *) echo "fold must be 0-5 or all" >&2; exit 1;; esac
FOLDS="$FOLD"; [ "$FOLD" = all ] && FOLDS="0 1 2 3 4 5"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true)}"
[ -n "$NS" ] || { echo "Set NS=<your cluster user name>" >&2; exit 1; }
HOME_DIR="/shared/ssd/home/$NS"; LOGS="/shared/ssd/logs/$NS"; DATA="/shared/hdd/data/bioactive"
VENV="${VENV:-$HOME_DIR/venv}"; RUN_UID="${RUN_UID:-$(id -u)}"; RUN_GID="${RUN_GID:-1376}"
IMAGE="abrainone/ai-linux:cu12.6.3-latest"
case "$REPO" in "$HOME_DIR"/*) ;; *) echo "Repository must be under $HOME_DIR (it is mounted into the pod); it is in $REPO" >&2; exit 1;; esac

if [ -z "${MODE:-}" ]; then
  case "$MODEL" in
    resnet|resnet18|effnet|lora_vit_s) MODE=finetune;;
    dino|dino_small|dino_large|dinov3|celldino|clip|biomedclip) MODE=frozen;;
    *) echo "Unknown model '$MODEL'. Set MODE=finetune or MODE=frozen to force one." >&2; exit 1;;
  esac
fi

NAME="rerun-$(echo "$MODEL" | tr '_' '-')-r$RES-f$FOLD-$(date +%H%M%S)"
RUN="bioact_${MODEL}_r${RES}"

if [ "$MODE" = finetune ]; then
  case "$MODEL" in
    lora_vit_s) P="params/params_lora_vit_small_cluster.json"; B=32;;
    effnet)     if [ "$RES" = 224 ]; then P='params/params_efficientnet_b3_r224_fold$fold.json'; else P='params/params_efficientnet_b3_fold$fold.json'; fi; B=64;;
    resnet18)   if [ "$RES" = 224 ]; then P='params/params_resnet18_r224_fold$fold.json'; else P='params/params_resnet18_fold$fold.json'; fi; B=64;;
    dino) P="params/params_cluster_base.json"; B=32;;
    clip|biomedclip) P="params/params_cluster_base.json"; B=16;;
    *) P="params/params_cluster_base.json"; B=64;;
  esac
  P="${PARAMS:-$P}"; B="${BATCH:-$B}"
  read -r -d '' CMD <<BODY || true
for fold in $FOLDS; do
  python classification.py --params_path $P --model $MODEL --res $RES --fold \$fold --batch_size $B --model_name $RUN || { echo "FAILED fold \$fold"; exit 1; }
  python analysis/make_plots_v3.py ${RUN}_fold\$fold || true
done
BODY
else
  case "$MODEL" in
    celldino)   PX="params/params_celldino_cluster.json";;
    dino)       PX="params/params_dinov2_base_cluster.json";;
    dino_large) PX="params/params_dinov2_large_cluster.json";;
    biomedclip) PX="params/params_biomedclip_cluster.json";;
    *)          PX="params/params_cluster_base.json";;
  esac
  PX="${PARAMS:-$PX}"; SD="$LOGS/rerun/${MODEL}_r${RES}"
  read -r -d '' CMD <<BODY || true
for fold in $FOLDS; do
  python extract_embeddings.py --params_path $PX --model $MODEL --res $RES --fold \$fold --save_dir $SD || { echo "FAILED extract fold \$fold"; exit 1; }
  python train_head.py --model $MODEL --res $RES --fold \$fold --lr 0.02 --save_dir $SD || { echo "FAILED head fold \$fold"; exit 1; }
done
BODY
fi
CMD_INDENTED="$(printf '%s\n' "$CMD" | sed 's/^/          /')"

YAML="apiVersion: batch/v1
kind: Job
metadata:
  name: $NAME
  namespace: $NS
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      securityContext:
        runAsUser: $RUN_UID
        runAsGroup: $RUN_GID
        fsGroup: $RUN_GID
      containers:
      - name: train
        image: $IMAGE
        workingDir: $REPO
        command: [\"bash\", \"-lc\"]
        args:
        - |
          source $VENV/bin/activate
$CMD_INDENTED
        env:
        - { name: WANDB_MODE, value: \"disabled\" }
        - { name: BIOACT_SAVE_DIR, value: \"$LOGS\" }
        - { name: HF_HOME, value: \"$HOME_DIR/hf_cache\" }
        - { name: PYTHONUNBUFFERED, value: \"1\" }
        - { name: PYTHONPATH, value: \".\" }
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef: { name: hf-token, key: token, optional: true }
        resources:
          requests: { nvidia.com/gpu: 1, cpu: 8, memory: 32Gi }
          limits:   { nvidia.com/gpu: 1, cpu: 16, memory: 40Gi }
        volumeMounts:
        - { mountPath: $HOME_DIR, name: home }
        - { mountPath: $DATA, name: data, readOnly: true }
        - { mountPath: $LOGS, name: logs }
        - { mountPath: /dev/shm, name: dshm }
      volumes:
      - name: home
        hostPath: { path: $HOME_DIR, type: Directory }
      - name: data
        hostPath: { path: $DATA, type: Directory }
      - name: logs
        hostPath: { path: $LOGS, type: Directory }
      - name: dshm
        emptyDir: { medium: Memory, sizeLimit: 16Gi }"

if [ "${DRY:-0}" = 1 ]; then printf '%s\n' "$YAML"; exit 0; fi
printf '%s\n' "$YAML" | kubectl apply -f -
echo "Submitted $NAME.  Follow with: kubectl logs -f job/$NAME -n $NS"
