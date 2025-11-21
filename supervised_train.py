import argparse

import torch
from torch import nn

from data import get_classification_dataloaders
from models import LinearClassifier, ResNetEncoder
from utils import CSVLogger, build_warmup_cosine_scheduler, create_experiment_dirs, save_metrics, set_seed
from utils.train_helpers import build_feature_extractor, evaluate_classifier, train_classifier_epoch


def get_num_classes(dataset: str) -> int:
    if dataset.lower() in {"cifar10", "stl10"}:
        return 10
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised training from scratch")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--label-fraction", type=float, default=1.0, help="Fraction of labels to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", type=str, default="supervised_baseline", help="Experiment name")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs for LR scheduling")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir, ckpt_dir = create_experiment_dirs(args.experiment_name)
    logger = CSVLogger(results_dir / "supervised_log.csv", fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

    train_loader, test_loader = get_classification_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=args.label_fraction,
        seed=args.seed,
        drop_last=False,
    )

    encoder = ResNetEncoder(args.backbone, dataset=args.dataset).to(device)
    classifier = LinearClassifier(encoder.feature_dim, num_classes=get_num_classes(args.dataset)).to(device)

    optimizer = torch.optim.AdamW(
        [{"params": encoder.parameters()}, {"params": classifier.parameters()}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    feature_extractor = build_feature_extractor(encoder=encoder, projection_head=None, use_projection=False, device=device)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        train_loss, train_acc = train_classifier_epoch(
            feature_extractor,
            classifier,
            train_loader,
            criterion,
            optimizer,
            device,
            freeze_encoder=False,
            scheduler=scheduler,
        )

        encoder.eval()
        test_loss, test_acc = evaluate_classifier(feature_extractor, classifier, test_loader, device)

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.3f}"
        )

    torch.save(encoder.state_dict(), ckpt_dir / "supervised_encoder.pt")
    torch.save(classifier.state_dict(), ckpt_dir / "supervised_classifier.pt")
    save_metrics(results_dir / "supervised_metrics.json", {"test_acc": test_acc, "test_loss": test_loss})
    print(f"Supervised training complete. Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
