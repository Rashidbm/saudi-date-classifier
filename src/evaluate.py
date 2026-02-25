"""
Evaluation pipeline: metrics, confusion matrix, per-class breakdown, comparison plots.

Usage:
    python -m src.evaluate --model efficientnet
    python -m src.evaluate --model vit
    python -m src.evaluate --model resnet
    python -m src.evaluate --compare   # Compare all models
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.utils import load_config, get_device, seed_everything


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on entire dataloader.

    Returns:
        all_labels, all_preds, all_probs (numpy arrays)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Percentages
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} (% per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_model_comparison(results: dict, save_path: str) -> None:
    """
    Plot bar chart comparing accuracy and F1 across models.

    Args:
        results: {model_name: {"accuracy": float, "f1": float, ...}}
    """
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    f1_scores = [results[m]["f1"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score", color="#4CAF50")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison: Saudi Date Variety Classification")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved: {save_path}")


def evaluate_model(model_name: str, config: dict | None = None) -> dict:
    """
    Evaluate a trained model on the test set.

    Returns dict with accuracy, f1, precision, recall, and per-class report.
    """
    if config is None:
        config = load_config()

    seed_everything(config["training"]["seed"])
    device = get_device()

    # Load checkpoint
    checkpoint_path = Path("outputs") / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"Train the model first: python -m src.train --model {model_name}")
        return {}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    # Build model and load weights
    build_fn = MODEL_BUILDERS[model_name]
    model = build_fn(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"\nEvaluating {model_name.upper()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Best val acc: {checkpoint['val_acc']:.4f} (phase {checkpoint['phase']}, epoch {checkpoint['epoch']})")

    # Data
    _, _, test_loader, _ = create_dataloaders(config)

    # Predictions
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    output_dir = Path("outputs") / model_name
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"{model_name.upper()} Confusion Matrix",
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def compare_models(config: dict | None = None) -> None:
    """Evaluate and compare all trained models."""
    if config is None:
        config = load_config()

    results = {}
    for model_name in MODEL_BUILDERS:
        checkpoint_path = Path("outputs") / model_name / "best_model.pth"
        if checkpoint_path.exists():
            results[model_name] = evaluate_model(model_name, config)
        else:
            print(f"Skipping {model_name} (no checkpoint found)")

    if len(results) > 1:
        output_dir = Path("outputs") / "comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_model_comparison(results, str(output_dir / "model_comparison.png"))

        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 55)
        for name, r in results.items():
            print(
                f"  {name:<13} {r['accuracy']*100:>9.1f}% {r['f1']*100:>9.1f}% "
                f"{r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}%"
            )

        best_model = max(results, key=lambda k: results[k]["accuracy"])
        print(f"\n🏆 Best model: {best_model.upper()} ({results[best_model]['accuracy']*100:.1f}% accuracy)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate date fruit classifier")
    parser.add_argument("--model", type=str, choices=["efficientnet", "vit", "resnet"])
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.compare:
        compare_models(config)
    elif args.model:
        evaluate_model(args.model, config)
    else:
        print("Specify --model <name> or --compare")


if __name__ == "__main__":
    main()
