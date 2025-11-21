import argparse
from pathlib import Path

import torch
from torch import nn

from data import get_simclr_dataloader
from losses import CosineMarginLoss, NTXentLoss, PositiveOnlySimilarityLoss
from models import ProjectionHead, ResNetEncoder, SimCLRModel
from utils import CSVLogger, create_experiment_dirs, save_metrics, set_seed
from utils.lr_schedulers import build_warmup_cosine_scheduler
from utils.train_helpers import train_simclr_epoch


def build_loss(name: str, temperature: float) -> nn.Module:
    name = name.lower()
    if name == "nt_xent":
        return NTXentLoss(temperature=temperature)
    if name == "cosine_margin":
        return CosineMarginLoss()
    if name == "pos_only":
        return PositiveOnlySimilarityLoss()
    raise ValueError(f"Unknown loss: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR pretraining")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10, stl10)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where to store/download data")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for pretraining")
    parser.add_argument("--epochs", type=int, default=200, help="Number of pretraining epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", type=str, default="simclr_cifar10_resnet18", help="Experiment name")
    parser.add_argument(
        "--augment",
        type=str,
        default="full",
        choices=["only_crop", "crop_color", "crop_color_gray", "full"],
        help="Augmentation variant",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="nt_xent",
        choices=["nt_xent", "cosine_margin", "pos_only"],
        help="Contrastive loss type",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for NT-Xent")
    parser.add_argument(
        "--proj-mode",
        type=str,
        default="head_on_hfeat",
        choices=["no_head", "head_on_hfeat", "head_on_zfeat"],
        help="Where to apply the projection head",
    )
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection output dimension")
    parser.add_argument("--proj-hidden-dim", type=int, default=512, help="Projection hidden dimension")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs before cosine decay")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_simclr_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.num_workers,
    )

    encoder = ResNetEncoder(args.backbone, dataset=args.dataset)
    projection_head = ProjectionHead(encoder.feature_dim, hidden_dim=args.proj_hidden_dim, out_dim=args.proj_dim)
    model = SimCLRModel(encoder, projection_head, proj_mode=args.proj_mode).to(device)

    criterion = build_loss(args.loss, args.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    results_dir, ckpt_dir = create_experiment_dirs(args.experiment_name)
    log_path = results_dir / "pretrain_log.csv"
    logger = CSVLogger(log_path, fieldnames=["epoch", "loss"])

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_simclr_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        logger.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), ckpt_dir / "simclr_encoder.pt")
    torch.save(projection_head.state_dict(), ckpt_dir / "simclr_projection.pt")

    save_metrics(results_dir / "pretrain_metrics.json", {"final_loss": avg_loss})
    print(f"Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
