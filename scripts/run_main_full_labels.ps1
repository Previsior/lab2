param(
    [string]$dataset = "cifar10", 
    [string]$backbone = "resnet18" 
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

# Supervised ResNet-18 baseline with full labels
python supervised_train.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --label-fraction 1.0 --experiment-name "${dataset}_${backbone}_supervised_full"

# SimCLR pretraining (FullAug + NT-Xent)
$simclrExp = "${dataset}_${backbone}_simclr_full"
python simclr_pretrain.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --augment full --loss nt_xent --temperature 0.5 `
    --experiment-name $simclrExp

$simclrEncoder = Join-Path "checkpoints" "$simclrExp/simclr_encoder.pt"
$simclrProj = Join-Path "checkpoints" "$simclrExp/simclr_projection.pt"

# Linear evaluation on frozen encoder (full labels)
$linearExp = "${dataset}_${backbone}_simclr_linear_full"
python linear_eval.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type resnet --encoder-checkpoint $simclrEncoder --projection-checkpoint $simclrProj `
    --proj-mode head_on_hfeat `
    --label-fraction 1.0 --batch-size 256 --epochs 50 --lr-head 1e-3 `
    --experiment-name $linearExp

# Full-network fine-tuning from SimCLR encoder
$ftExp = "${dataset}_${backbone}_simclr_finetune_full"
python finetune.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type resnet --encoder-checkpoint $simclrEncoder --projection-checkpoint $simclrProj `
    --proj-mode head_on_hfeat `
    --label-fraction 1.0 --batch-size 256 --epochs 50 `
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 `
    --experiment-name $ftExp

# Autoencoder baseline pretraining
$aeExp = "${dataset}_${backbone}_ae_full"
python ae_pretrain.py `
    --dataset $dataset --data-dir ./data `
    --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
    --latent-dim 128 --experiment-name $aeExp

$aeEncoder = Join-Path "checkpoints" "$aeExp/ae_encoder.pt"

# Fine-tune starting from AE encoder
$aeFtExp = "${dataset}_${backbone}_ae_finetune_full"
python finetune.py `
    --dataset $dataset --data-dir ./data --backbone $backbone `
    --encoder-type ae --ae-latent-dim 128 --encoder-checkpoint $aeEncoder `
    --label-fraction 1.0 --batch-size 256 --epochs 50 `
    --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 1e-4 `
    --experiment-name $aeFtExp

