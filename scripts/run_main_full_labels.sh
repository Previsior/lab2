#!/usr/bin/env bash
set -euo pipefail

DATASET="cifar10"
BACKBONE="resnet18"
GPU="0"

usage() {
    cat <<USAGE
Usage: ${0##*/} [--dataset NAME] [--backbone NAME] [--gpu ID]
  --dataset   Dataset name (default: ${DATASET})
  --backbone  Backbone architecture (default: ${BACKBONE})
  --gpu       GPU id to use (default: ${GPU})
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export CUDA_VISIBLE_DEVICES="${GPU}"

python supervised_train.py \
    --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 \
    --label-fraction 1.0 --experiment-name "${DATASET}_${BACKBONE}_supervised_full"

SIMCLR_EXP="${DATASET}_${BACKBONE}_simclr_full"
python simclr_pretrain.py \
    --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 \
    --augment full --loss nt_xent --temperature 0.5 \
    --experiment-name "${SIMCLR_EXP}"

SIMCLR_ENCODER="checkpoints/${SIMCLR_EXP}/simclr_encoder.pt"
SIMCLR_PROJ="checkpoints/${SIMCLR_EXP}/simclr_projection.pt"

LINEAR_EXP="${DATASET}_${BACKBONE}_simclr_linear_full"
python linear_eval.py \
    --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
    --encoder-type resnet --encoder-checkpoint "${SIMCLR_ENCODER}" --projection-checkpoint "${SIMCLR_PROJ}" \
    --proj-mode head_on_hfeat \
    --label-fraction 1.0 --batch-size 256 --epochs 50 --lr-head 1e-3 \
    --experiment-name "${LINEAR_EXP}"

FT_EXP="${DATASET}_${BACKBONE}_simclr_finetune_full"
python finetune.py \
    --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
    --encoder-type resnet --encoder-checkpoint "${SIMCLR_ENCODER}" --projection-checkpoint "${SIMCLR_PROJ}" \
    --proj-mode head_on_hfeat \
    --label-fraction 1.0 --batch-size 256 --epochs 50 \
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 \
    --experiment-name "${FT_EXP}"

AE_EXP="${DATASET}_${BACKBONE}_ae_full"
python ae_pretrain.py \
    --dataset "${DATASET}" --data-dir ./data \
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 \
    --latent-dim 128 --experiment-name "${AE_EXP}"

AE_ENCODER="checkpoints/${AE_EXP}/ae_encoder.pt"

AE_FT_EXP="${DATASET}_${BACKBONE}_ae_finetune_full"
python finetune.py \
    --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
    --encoder-type ae --ae-latent-dim 128 --encoder-checkpoint "${AE_ENCODER}" \
    --label-fraction 1.0 --batch-size 256 --epochs 50 \
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 \
    --experiment-name "${AE_FT_EXP}"
