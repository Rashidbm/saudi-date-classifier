"""
Ensemble: Combines ResNet50, EfficientNet-B0, and ViT using soft voting.

Downloads checkpoints from HuggingFace, runs all three on the test set,
averages softmax probabilities, and reports the ensemble accuracy.

Usage:
    python -m src.ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.dataset import DateFruitDataset, get_val_transforms
from src.utils import load_config, get_device, seed_everything


HF_REPO_ID = "Rashidbm/saudi-date-classifier"
CHECKPOINTS = {
    "resnet": "arabic_dates_resnet50_best_V2.pth",
    "efficientnet": "efficientnet_best.pth",
    "vit": "vit_best_model.pth",
}

CLASS_NAMES = [
    "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali",
    "Rutab", "Shaishe", "Sokari", "Sugaey",
]


def build_resnet50(num_classes=9, dropout=0.3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def build_efficientnet(num_classes=9, dropout=0.3):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


class PretrainedViTClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.backbone(x).logits


def load_checkpoint(model, path, device):
    """Load checkpoint, handle both 'model_state_dict' and raw state dicts."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def load_all_models(device):
    """Download checkpoints from HuggingFace and load all three models."""
    print("Downloading checkpoints from HuggingFace...")
    paths = {
        name: hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
        for name, fname in CHECKPOINTS.items()
    }

    print("Loading models...")
    models_dict = {}

    models_dict["resnet"] = load_checkpoint(
        build_resnet50(num_classes=9), paths["resnet"], device
    )
    models_dict["efficientnet"] = load_checkpoint(
        build_efficientnet(num_classes=9), paths["efficientnet"], device
    )
    models_dict["vit"] = load_checkpoint(
        PretrainedViTClassifier(num_classes=9), paths["vit"], device
    )

    print("All models loaded.")
    return models_dict


@torch.no_grad()
def evaluate_single(model, loader, device, name):
    """Evaluate a single model, return (accuracy, all_probs, all_labels)."""
    all_probs = []
    all_labels = []

    for images, labels, _ in tqdm(loader, desc=f"Evaluating {name}"):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = all_probs.argmax(dim=1)
    acc = accuracy_score(all_labels.numpy(), preds.numpy()) * 100
    return acc, all_probs, all_labels


def main():
    config = load_config()
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Load all models
    models_dict = load_all_models(device)

    # Load test set
    transform = get_val_transforms(config)
    test_dataset = DateFruitDataset("data/test.csv", transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=0
    )
    print(f"\nTest set: {len(test_dataset)} images")

    # Evaluate each model
    results = {}
    for name, model in models_dict.items():
        acc, probs, labels = evaluate_single(model, test_loader, device, name)
        results[name] = {"accuracy": acc, "probs": probs, "labels": labels}

    # Ensemble (soft voting - average of softmax probabilities)
    ensemble_probs = sum(r["probs"] for r in results.values()) / len(results)
    ensemble_preds = ensemble_probs.argmax(dim=1).numpy()
    true_labels = results["vit"]["labels"].numpy()
    ensemble_acc = accuracy_score(true_labels, ensemble_preds) * 100

    print(f"\n{'='*50}")
    print(f"INDIVIDUAL vs ENSEMBLE")
    print(f"{'='*50}")
    for name, r in results.items():
        print(f"  {name.upper():<15} {r['accuracy']:>6.2f}%")
    print(f"  {'ENSEMBLE':<15} {ensemble_acc:>6.2f}%")

    print(f"\nEnsemble Classification Report:")
    print(classification_report(true_labels, ensemble_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, ensemble_preds))


if __name__ == "__main__":
    main()
