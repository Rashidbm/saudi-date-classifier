"""
Explainability: Grad-CAM heatmaps and t-SNE feature visualization.

Usage:
    python -m src.explainability --gradcam
    python -m src.explainability --tsne
    python -m src.explainability --all
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from tqdm import tqdm


def reshape_transform(tensor, height=14, width=14):
    """Reshape ViT output from (B, 197, 768) to (B, 768, 14, 14) for Grad-CAM.
    ViT outputs a sequence of patch tokens. We drop the CLS token and reshape
    the remaining 196 tokens into a 14x14 spatial grid."""
    # Remove CLS token (first token)
    result = tensor[:, 1:, :]
    # Reshape to spatial grid: (B, 196, 768) -> (B, 14, 14, 768) -> (B, 768, 14, 14)
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

from src.dataset import DateFruitDataset, get_val_transforms
from src.models.vit_pretrained import PretrainedViTClassifier
from src.utils import load_config, get_device, seed_everything

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali",
    "Rutab", "Shaishe", "Sokari", "Sugaey",
]


def load_vit_model(checkpoint_path: str, device: torch.device) -> PretrainedViTClassifier:
    """Load trained ViT from checkpoint."""
    model = PretrainedViTClassifier(num_classes=9)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to 0-1 RGB numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam(
    model: PretrainedViTClassifier,
    dataset: DateFruitDataset,
    device: torch.device,
    samples_per_class: int = 2,
    save_path: str = "results/gradcam_grid.png",
) -> None:
    """Generate Grad-CAM heatmap grid for each variety."""
    # ViT target layer: last layernorm before attention
    target_layer = model.backbone.vit.encoder.layer[-1].layernorm_before
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(
        num_classes, samples_per_class * 2,
        figsize=(4 * samples_per_class * 2, 3 * num_classes),
    )

    # Group images by class
    class_indices = {v: [] for v in CLASS_NAMES}
    for idx in range(len(dataset)):
        _, label, variety = dataset[idx]
        if len(class_indices[variety]) < samples_per_class:
            class_indices[variety].append(idx)

    for row, variety in enumerate(CLASS_NAMES):
        indices = class_indices.get(variety, [])
        for col, idx in enumerate(indices):
            image_tensor, label, _ = dataset[idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]

            rgb_img = denormalize(image_tensor)

            # Original image
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(rgb_img)
            ax_orig.set_title(variety, fontsize=10, fontweight="bold")
            ax_orig.axis("off")

            # Grad-CAM overlay
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            ax_cam = axes[row, col * 2 + 1]
            ax_cam.imshow(cam_image)
            ax_cam.set_title("Grad-CAM", fontsize=10)
            ax_cam.axis("off")

    plt.suptitle("Grad-CAM: What the ViT Model Focuses On", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_tsne(
    model: PretrainedViTClassifier,
    dataset: DateFruitDataset,
    device: torch.device,
    save_path: str = "results/tsne.png",
) -> None:
    """Generate t-SNE plot of learned feature embeddings."""
    model.eval()
    all_features = []
    all_labels = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            # Get CLS token features from ViT
            outputs = model.backbone.vit(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    print(f"Running t-SNE on {features.shape[0]} samples, {features.shape[1]} dimensions...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))

    for i, variety in enumerate(CLASS_NAMES):
        mask = labels == i
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[i]],
            label=variety,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.legend(fontsize=10, loc="best")
    ax.set_title("t-SNE: How ViT Clusters Saudi Date Varieties", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vit_best_model.pth")
    args = parser.parse_args()

    if args.all:
        args.gradcam = True
        args.tsne = True

    if not args.gradcam and not args.tsne:
        print("Specify --gradcam, --tsne, or --all")
        return

    config = load_config()
    seed_everything(42)
    device = get_device()

    print(f"Device: {device}")
    print(f"Loading model from {args.checkpoint}...")
    model = load_vit_model(args.checkpoint, device)

    transform = get_val_transforms(config)
    dataset = DateFruitDataset("data/test.csv", transform=transform)
    print(f"Test set: {len(dataset)} images")

    if args.gradcam:
        print("\nGenerating Grad-CAM...")
        generate_gradcam(model, dataset, device)

    if args.tsne:
        print("\nGenerating t-SNE...")
        generate_tsne(model, dataset, device)


if __name__ == "__main__":
    main()
