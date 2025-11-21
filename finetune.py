import argparse

import torch
from torch import nn

from data import get_classification_dataloaders
from models import AEEncoder, LinearClassifier, ProjectionHead, ResNetEncoder
from utils import CSVLogger, build_warmup_cosine_scheduler, create_experiment_dirs, save_metrics, set_seed
from utils.train_helpers import build_feature_extractor, evaluate_classifier, train_classifier_epoch


def get_num_classes(dataset: str) -> int:
    if dataset.lower() in {"cifar10", "stl10"}:
        return 10
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_input_size(dataset: str) -> int:
    return 32 if dataset.lower() == "cifar10" else 96


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained encoder with a linear head")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--encoder-type", type=str, default="resnet", choices=["resnet", "ae"], help="Encoder type")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone (when encoder-type=resnet)")
    parser.add_argument("--ae-latent-dim", type=int, default=128, help="Latent dimension for AE encoder")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Learning rate for encoder (and projection)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", type=str, default="finetune", help="Experiment name")
    parser.add_argument("--encoder-checkpoint", type=str, required=True, help="Path to pretrained encoder weights")
    parser.add_argument("--projection-checkpoint", type=str, default="", help="Optional projection head checkpoint")
    parser.add_argument(
        "--proj-mode",
        type=str,
        default="head_on_hfeat",
        choices=["no_head", "head_on_hfeat", "head_on_zfeat"],
        help="Projection usage from pretraining",
    )
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection output dimension")
    parser.add_argument("--proj-hidden-dim", type=int, default=512, help="Projection hidden dimension")
    parser.add_argument("--label-fraction", type=float, default=1.0, help="Fraction of labels to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs for LR scheduling")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir, ckpt_dir = create_experiment_dirs(args.experiment_name)
    log_path = results_dir / "finetune_log.csv"
    logger = CSVLogger(log_path, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

    train_loader, test_loader = get_classification_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=args.label_fraction,
        seed=args.seed,
        drop_last=False,
    )

    if args.encoder_type == "resnet":
        encoder = ResNetEncoder(args.backbone, dataset=args.dataset)
    else:
        encoder = AEEncoder(input_size=get_input_size(args.dataset), latent_dim=args.ae_latent_dim)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
    encoder.to(device)

    use_projection = args.encoder_type == "resnet" and args.proj_mode == "head_on_zfeat"
    projection_head = None
    if use_projection:
        projection_head = ProjectionHead(encoder.feature_dim, hidden_dim=args.proj_hidden_dim, out_dim=args.proj_dim)
        if args.projection_checkpoint:
            projection_head.load_state_dict(torch.load(args.projection_checkpoint, map_location=device))
        projection_head.to(device)

    feat_dim = projection_head.out_dim if use_projection and projection_head is not None else encoder.feature_dim
    classifier = LinearClassifier(feat_dim, num_classes=get_num_classes(args.dataset)).to(device)

    param_groups = [{"params": encoder.parameters(), "lr": args.lr_backbone}]
    if use_projection and projection_head is not None:
        param_groups.append({"params": projection_head.parameters(), "lr": args.lr_backbone})
    param_groups.append({"params": classifier.parameters(), "lr": args.lr_head})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()

    feature_extractor = build_feature_extractor(
        encoder=encoder,
        projection_head=projection_head,
        use_projection=use_projection,
        device=device,
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        if projection_head is not None:
            projection_head.train()

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
        if projection_head is not None:
            projection_head.eval()
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

    torch.save(encoder.state_dict(), ckpt_dir / "finetuned_encoder.pt")
    if projection_head is not None:
        torch.save(projection_head.state_dict(), ckpt_dir / "finetuned_projection.pt")
    torch.save(classifier.state_dict(), ckpt_dir / "finetuned_classifier.pt")
    save_metrics(results_dir / "finetune_metrics.json", {"test_acc": test_acc, "test_loss": test_loss})
    print(f"Finished fine-tuning. Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
