import argparse
from pathlib import Path

import torch

from data import get_classification_dataloaders
from data.augmentations import get_classification_transform
from models import AutoEncoder
from utils import CSVLogger, build_warmup_cosine_scheduler, create_experiment_dirs, save_metrics, set_seed
from utils.train_helpers import train_autoencoder_epoch


def get_input_size(dataset: str) -> int:
    return 32 if dataset.lower() == "cifar10" else 96


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder pretraining")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--latent-dim", type=int, default=128, help="AE latent dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", type=str, default="ae_cifar10", help="Experiment name")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs before cosine decay")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir, ckpt_dir = create_experiment_dirs(args.experiment_name)
    logger = CSVLogger(results_dir / "ae_pretrain_log.csv", fieldnames=["epoch", "loss"])

    image_size = get_input_size(args.dataset)
    transform = get_classification_transform(train=False, image_size=image_size)
    train_loader, _ = get_classification_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=1.0,
        seed=args.seed,
        train_transform=transform,
        test_transform=transform,
        drop_last=False,
    )

    autoencoder = AutoEncoder(input_size=image_size, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_autoencoder_epoch(autoencoder, train_loader, optimizer, device, scheduler)
        logger.log({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f}")

    torch.save(autoencoder.encoder.state_dict(), ckpt_dir / "ae_encoder.pt")
    torch.save(autoencoder.state_dict(), ckpt_dir / "ae_full.pt")
    save_metrics(results_dir / "ae_metrics.json", {"final_loss": epoch_loss})
    print(f"AE checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
