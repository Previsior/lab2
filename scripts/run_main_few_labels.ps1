param(
    [string]$dataset = "cifar10", 
    [string]$backbone = "resnet18" 
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

# Supervised baseline with 15% labels
python supervised_train.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --label-fraction 0.15 --experiment-name "${dataset}_${backbone}_supervised_few"

# SimCLR pretraining still uses all unlabeled training data
$simclrExp = "${dataset}_${backbone}_simclr_few"
python simclr_pretrain.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --augment full --loss nt_xent --temperature 0.5 `
    --experiment-name $simclrExp

$simclrEncoder = Join-Path "checkpoints" "$simclrExp/simclr_encoder.pt"
$simclrProj = Join-Path "checkpoints" "$simclrExp/simclr_projection.pt"

# Linear evaluation with 15% labels
$linearExp = "${dataset}_${backbone}_simclr_linear_few"
python linear_eval.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type resnet --encoder-checkpoint $simclrEncoder --projection-checkpoint $simclrProj `
    --proj-mode head_on_hfeat `
    --label-fraction 0.15 --batch-size 256 --epochs 50 --lr-head 1e-3 `
    --experiment-name $linearExp

# Fine-tuning with 15% labels
$ftExp = "${dataset}_${backbone}_simclr_finetune_few"
python finetune.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type resnet --encoder-checkpoint $simclrEncoder --projection-checkpoint $simclrProj `
    --proj-mode head_on_hfeat `
    --label-fraction 0.15 --batch-size 256 --epochs 50 `
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 `
    --experiment-name $ftExp

# Autoencoder pretraining baseline
$aeExp = "${dataset}_${backbone}_ae_few"
python ae_pretrain.py `
    --dataset $dataset --data-dir ./data `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --latent-dim 128 --experiment-name $aeExp

$aeEncoder = Join-Path "checkpoints" "$aeExp/ae_encoder.pt"

# Fine-tune AE encoder with 15% labels
$aeFtExp = "${dataset}_${backbone}_ae_finetune_few"
python finetune.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type ae --ae-latent-dim 128 --encoder-checkpoint $aeEncoder `
    --label-fraction 0.15 --batch-size 256 --epochs 50 `
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 `
    --experiment-name $aeFtExp

