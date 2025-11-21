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

AUGMENTS=("only_crop" "crop_color" "crop_color_gray" "full")

for AUG in "${AUGMENTS[@]}"; do
    echo "Running SimCLR with augmentation ${AUG}"
    EXP_NAME="${DATASET}_${BACKBONE}_abl_aug_${AUG}"

    python simclr_pretrain.py \
        --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
        --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 \
        --augment "${AUG}" --loss nt_xent --temperature 0.5 \
        --experiment-name "${EXP_NAME}"

    ENCODER_CKPT="checkpoints/${EXP_NAME}/simclr_encoder.pt"
    PROJ_CKPT="checkpoints/${EXP_NAME}/simclr_projection.pt"

    LINEAR_EXP="${EXP_NAME}_linear"
    python linear_eval.py \
        --dataset "${DATASET}" --data-dir ./data --backbone "${BACKBONE}" \
        --encoder-type resnet --encoder-checkpoint "${ENCODER_CKPT}" --projection-checkpoint "${PROJ_CKPT}" \
        --proj-mode head_on_hfeat \
        --label-fraction 1.0 --batch-size 256 --epochs 50 --lr-head 1e-3 \
        --experiment-name "${LINEAR_EXP}"
done
