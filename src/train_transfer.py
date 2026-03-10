"""
Transfer Learning Training Script for ViT
Two-phase training: Phase 1 (frozen backbone), Phase 2 (fine-tuning)
Optimized for RTX 4050 (6GB VRAM)

Author: Ahmad
Branch: Ahmad-VIT
Purpose: Training script for Vision Transformer on Saudi date classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.models.vit_pretrained import PretrainedViTClassifier
from src.dataset import DateFruitDataset, get_train_transforms, get_val_transforms
from src.utils import load_config


# Configuration - Optimized for RTX 4050 (6GB VRAM)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 0  # Set to 0 to avoid Windows multiprocessing issues
CHECKPOINT_DIR = "checkpoints"

# Phase 1: Train classifier head only (frozen backbone)
PHASE1_EPOCHS = 10
PHASE1_LR = 0.001

# Phase 2: Fine-tune all parameters (unfrozen backbone)
PHASE2_EPOCHS = 30
PHASE2_LR = 0.0001

WEIGHT_DECAY = 0.0001
GRADIENT_ACCUMULATION = 2

# Model configuration
MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_CLASSES = 9

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Metrics tracking for plotting
metrics = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rate': [],
    'phase': [],  # Track which phase we're in
}


def load_data():
    """Load training and validation datasets from CSV files."""
    config = load_config("configs/default.yaml")
    
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    
    train_dataset = DateFruitDataset(
        csv_path="data/train.csv",
        transform=train_transforms
    )
    
    val_dataset = DateFruitDataset(
        csv_path="data/val.csv",
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Statistics
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Optimizer step every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item() * accumulation_steps:.4f} | "
                  f"Acc: {100 * correct / total:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, phase, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"  [OK] Checkpoint saved: {filepath}")


def plot_metrics(metrics, save_dir="checkpoints"):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transfer Learning - ViT Baseline (Two-Phase Training)', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, metrics['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs, metrics['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    axes[1, 0].plot(epochs, metrics['learning_rate'], color='green', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy Focus
    axes[1, 1].fill_between(epochs, metrics['val_acc'], alpha=0.3, color='blue')
    axes[1, 1].plot(epochs, metrics['val_acc'], label='Validation Accuracy', color='blue', linewidth=2.5, marker='s', markersize=5)
    max_acc_idx = np.argmax(metrics['val_acc'])
    axes[1, 1].scatter(epochs[max_acc_idx], metrics['val_acc'][max_acc_idx], color='red', s=100, zorder=5, label=f'Best: {metrics["val_acc"][max_acc_idx]:.2f}%')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 1].set_title('Best Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Metrics plot saved: {plot_path}")
    plt.close()


def main():
    """Main training function with two-phase approach."""
    
    print(f"\n{'='*70}")
    print(f"Transfer Learning - ViT Baseline for Saudi Date Classifier")
    print(f"GPU: RTX 4050 (6GB VRAM) | Two-Phase Training")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} (Gradient Accumulation: {GRADIENT_ACCUMULATION})")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"\nPhase 1 (Frozen Backbone): {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}")
    print(f"Phase 2 (Fine-tuning):     {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}\n")
    
    # Load data
    print("Loading datasets...")
    try:
        train_loader, val_loader = load_data()
        print(f"[OK] Training samples: {len(train_loader.dataset)}")
        print(f"[OK] Validation samples: {len(val_loader.dataset)}")
        print(f"[OK] Training batches: {len(train_loader)}")
        print(f"[OK] Validation batches: {len(val_loader)}\n")
    except FileNotFoundError as e:
        print(f"\n[ERR] Error: {e}")
        print("Please make sure data/train.csv and data/val.csv exist\n")
        return
    
    # Initialize model
    print("Initializing pretrained ViT model...")
    model = PretrainedViTClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
    )
    model = model.to(DEVICE)
    
    total_params, trainable_params = model.get_trainable_params()
    print(f"[OK] Total parameters: {total_params:,}")
    print(f"[OK] Trainable parameters: {trainable_params:,}\n")
    
    criterion = nn.CrossEntropyLoss()
    
    # ============================================================
    # PHASE 1: Train classifier head only (frozen backbone)
    # ============================================================
    print("="*70)
    print(f"PHASE 1: Training Classifier Head (Frozen Backbone)")
    print("="*70)
    
    model.freeze_backbone()
    total_params, trainable_params = model.get_trainable_params()
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, PHASE1_EPOCHS + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, GRADIENT_ACCUMULATION)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch [{epoch}/{PHASE1_EPOCHS}] ({elapsed_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        metrics['phase'].append(1)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "phase1_best.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, 1, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Load best model from phase 1
    best_phase1_path = os.path.join(CHECKPOINT_DIR, "phase1_best.pth")
    if os.path.exists(best_phase1_path):
        checkpoint = torch.load(best_phase1_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n[OK] Loaded best Phase 1 model")
    
    # ============================================================
    # PHASE 2: Fine-tune all parameters (unfrozen backbone)
    # ============================================================
    print("\n" + "="*70)
    print(f"PHASE 2: Fine-tuning All Parameters (Unfrozen Backbone)")
    print("="*70)
    
    model.unfreeze_backbone()
    total_params, trainable_params = model.get_trainable_params()
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)
    
    best_val_acc_phase2 = best_val_acc
    patience_counter = 0
    
    for epoch in range(1, PHASE2_EPOCHS + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, GRADIENT_ACCUMULATION)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch [{epoch}/{PHASE2_EPOCHS}] ({elapsed_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        metrics['phase'].append(2)
        
        # Save best model
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, 2, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Final summary
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc_phase2:.2f}%")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
    print("="*70 + "\n")
    
    # Plot metrics
    plot_metrics(metrics, CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
