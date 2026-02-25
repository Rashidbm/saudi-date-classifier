"""
Shared training loop with 2-phase transfer learning strategy.

Usage:
    python -m src.train --model efficientnet
    python -m src.train --model vit
    python -m src.train --model resnet

Phase 1: Frozen backbone, train head only (higher LR)
Phase 2: Unfreeze all layers, fine-tune (lower LR)
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.utils import load_config, seed_everything, get_device, setup_logging


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model_name: str, config: dict | None = None) -> str:
    """
    Full training pipeline for a given model.

    Args:
        model_name: One of 'efficientnet', 'vit', 'resnet'
        config: Optional config dict (loads default if None)

    Returns:
        Path to the best checkpoint
    """
    if config is None:
        config = load_config()

    seed_everything(config["training"]["seed"])
    device = get_device()

    # Output directory
    output_dir = Path("outputs") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir))

    logger.info(f"Training {model_name.upper()} on {device}")
    logger.info(f"Config: {config['training']}")

    # Data
    train_loader, val_loader, _, class_names = create_dataloaders(config)
    num_classes = len(class_names)

    # Model
    build_fn = MODEL_BUILDERS[model_name]
    model = build_fn(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss function (with class weights if needed)
    criterion = nn.CrossEntropyLoss()

    # ===== PHASE 1: Frozen backbone =====
    logger.info("\n========== PHASE 1: Training Head Only ==========")
    model.freeze_backbone()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr_phase1"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["lr_phase1"],
        epochs=config["training"]["epochs_phase1"],
        steps_per_epoch=len(train_loader),
    )

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "best_model.pth"
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(1, config["training"]["epochs_phase1"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            f"[Phase1] Epoch {epoch:>2}/{config['training']['epochs_phase1']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "phase": 1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "config": config,
            }, best_checkpoint_path)
            logger.info(f"  ✅ New best! Val Acc: {val_acc:.4f} → saved checkpoint")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # ===== PHASE 2: Full fine-tuning =====
    logger.info("\n========== PHASE 2: Full Fine-Tuning ==========")
    model.unfreeze_backbone()

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr_phase2"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["lr_phase2"],
        epochs=config["training"]["epochs_phase2"],
        steps_per_epoch=len(train_loader),
    )

    patience_counter = 0

    for epoch in range(1, config["training"]["epochs_phase2"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            f"[Phase2] Epoch {epoch:>2}/{config['training']['epochs_phase2']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "phase": 2,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "config": config,
            }, best_checkpoint_path)
            logger.info(f"  ✅ New best! Val Acc: {val_acc:.4f} → saved checkpoint")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    logger.info(f"\n🏁 Training complete! Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Checkpoint saved: {best_checkpoint_path}")

    return str(best_checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train a date fruit classifier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["efficientnet", "vit", "resnet"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = train_model(args.model, config)
    print(f"\nDone! Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
