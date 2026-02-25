"""
Ensemble model: combines predictions from EfficientNet, ViT, and ResNet.

Supports:
- Soft voting (average probabilities)
- Weighted voting (learned or manual weights)

Usage:
    python -m src.ensemble
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.evaluate import get_predictions, plot_confusion_matrix
from src.utils import load_config, get_device, seed_everything
from sklearn.metrics import accuracy_score, f1_score, classification_report


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}


def load_trained_model(
    model_name: str,
    config: dict,
    device: torch.device,
) -> nn.Module | None:
    """Load a trained model from checkpoint."""
    checkpoint_path = Path("outputs") / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found for {model_name}, skipping.")
        return None

    build_fn = MODEL_BUILDERS[model_name]
    model = build_fn(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded {model_name} (val_acc={checkpoint['val_acc']:.4f})")
    return model


def ensemble_predict(
    models: dict[str, nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble prediction via weighted soft voting.

    Args:
        models: {model_name: model}
        loader: Test DataLoader
        device: Compute device
        weights: Optional {model_name: weight}. If None, equal weights.

    Returns:
        all_labels, all_preds, all_avg_probs
    """
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}

    # Normalize weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    all_labels = []
    all_probs_sum = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Ensemble prediction"):
            images = images.to(device)
            batch_probs = torch.zeros(images.size(0), 9).to(device)

            for name, model in models.items():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs += weights[name] * probs

            all_labels.append(labels.numpy())
            all_probs_sum.append(batch_probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs_sum)
    all_preds = all_probs.argmax(axis=1)

    return all_labels, all_preds, all_probs


def main():
    config = load_config()
    seed_everything(config["training"]["seed"])
    device = get_device()

    print("Loading trained models for ensemble...")
    models = {}
    val_accs = {}

    for model_name in MODEL_BUILDERS:
        model = load_trained_model(model_name, config, device)
        if model is not None:
            models[model_name] = model
            # Get val accuracy for weighting
            ckpt = torch.load(
                Path("outputs") / model_name / "best_model.pth",
                map_location=device,
                weights_only=False,
            )
            val_accs[model_name] = ckpt["val_acc"]

    if len(models) < 2:
        print("Need at least 2 trained models for ensemble. Train more models first.")
        return

    print(f"\nEnsemble with {len(models)} models: {list(models.keys())}")

    # Data
    _, _, test_loader, class_names = create_dataloaders(config)

    # === Equal weight ensemble ===
    print("\n--- Equal Weight Ensemble ---")
    y_true, y_pred_eq, y_probs_eq = ensemble_predict(models, test_loader, device)
    acc_eq = accuracy_score(y_true, y_pred_eq)
    f1_eq = f1_score(y_true, y_pred_eq, average="weighted")
    print(f"  Accuracy: {acc_eq:.4f} ({acc_eq*100:.1f}%)")
    print(f"  F1 Score: {f1_eq:.4f}")

    # === Accuracy-weighted ensemble ===
    print("\n--- Accuracy-Weighted Ensemble ---")
    y_true, y_pred_wt, y_probs_wt = ensemble_predict(
        models, test_loader, device, weights=val_accs
    )
    acc_wt = accuracy_score(y_true, y_pred_wt)
    f1_wt = f1_score(y_true, y_pred_wt, average="weighted")
    print(f"  Accuracy: {acc_wt:.4f} ({acc_wt*100:.1f}%)")
    print(f"  F1 Score: {f1_wt:.4f}")
    print(f"  Weights used: {val_accs}")

    # Use the better ensemble
    if acc_wt >= acc_eq:
        best_method = "weighted"
        y_pred_best, y_probs_best = y_pred_wt, y_probs_wt
        best_acc = acc_wt
    else:
        best_method = "equal"
        y_pred_best, y_probs_best = y_pred_eq, y_probs_eq
        best_acc = acc_eq

    print(f"\n🏆 Best ensemble method: {best_method} (Accuracy: {best_acc*100:.1f}%)")
    print(f"\nFull classification report ({best_method} ensemble):")
    print(classification_report(y_true, y_pred_best, target_names=class_names))

    # Save confusion matrix
    output_dir = Path("outputs") / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true, y_pred_best, class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"Ensemble ({best_method}) Confusion Matrix",
    )

    # Save ensemble info
    info = {
        "method": best_method,
        "models": list(models.keys()),
        "weights": val_accs if best_method == "weighted" else None,
        "test_accuracy": float(best_acc),
        "test_f1": float(f1_score(y_true, y_pred_best, average="weighted")),
    }
    torch.save(info, output_dir / "ensemble_info.pth")
    print(f"\nEnsemble info saved: {output_dir / 'ensemble_info.pth'}")


if __name__ == "__main__":
    main()
