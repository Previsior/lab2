param(
    [string]$dataset = "cifar10", 
    [string]$backbone = "resnet18" 
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

$losses = @("nt_xent", "cosine_margin", "pos_only")

foreach ($lossName in $losses) {
    Write-Host "Running SimCLR loss ablation: $lossName"
    $expName = "${dataset}_${backbone}_abl_loss_$lossName"

    python simclr_pretrain.py `
        --dataset $dataset --data-dir ./data --backbone $backbone `
        --batch-size 256 --epochs 100 --lr 3e-4 --weight-decay 1e-4 `
        --augment full --loss $lossName --temperature 0.5 `
        --experiment-name $expName

    $encoderCkpt = Join-Path "checkpoints" "$expName/simclr_encoder.pt"
    $projCkpt = Join-Path "checkpoints" "$expName/simclr_projection.pt"

    $linearExp = "${expName}_linear"
    python linear_eval.py `
        --dataset $dataset --data-dir ./data --backbone $backbone `
        --encoder-type resnet --encoder-checkpoint $encoderCkpt --projection-checkpoint $projCkpt `
        --proj-mode head_on_hfeat `
        --label-fraction 1.0 --batch-size 256 --epochs 50 --lr-head 1e-3 `
        --experiment-name $linearExp
}

