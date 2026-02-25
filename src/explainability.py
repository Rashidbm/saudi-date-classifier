"""
Explainability: Grad-CAM heatmaps and t-SNE/UMAP feature space visualization.

Usage:
    python -m src.explainability --model efficientnet --gradcam
    python -m src.explainability --model efficientnet --tsne
    python -m src.explainability --model efficientnet --all

Member 5's responsibility.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.dataset import DateFruitDataset, get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.utils import load_config, get_device, seed_everything


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}


def get_target_layer(model: nn.Module, model_name: str):
    """Get the appropriate target layer for Grad-CAM based on model architecture."""
    if model_name == "efficientnet":
        # Last convolutional block in EfficientNet
        return [model.backbone.conv_head]
    elif model_name == "resnet":
        # Last residual block in ResNet
        return [model.backbone.layer4[-1]]
    elif model_name == "vit":
        # Last attention block in ViT (Grad-CAM works differently for ViT)
        return [model.backbone.encoder.layer[-1].layernorm_before]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable RGB numpy array (0-1 range)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam_grid(
    model: nn.Module,
    model_name: str,
    dataset: DateFruitDataset,
    device: torch.device,
    num_per_class: int = 2,
    save_path: str = "outputs/gradcam.png",
) -> None:
    """
    Generate a grid of Grad-CAM heatmaps showing what the model focuses on
    for each date variety.
    """
    model.eval()
    target_layers = get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers)

    class_names = dataset.class_names
    num_classes = len(class_names)

    fig, axes = plt.subplots(
        num_classes, num_per_class * 2,  # Original + CAM side by side
        figsize=(4 * num_per_class * 2, 3 * num_classes),
    )

    # Group images by class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label, variety = dataset[idx]
        if variety not in class_indices:
            class_indices[variety] = []
        if len(class_indices[variety]) < num_per_class:
            class_indices[variety].append(idx)

    for row, variety in enumerate(class_names):
        indices = class_indices.get(variety, [])
        for col, idx in enumerate(indices):
            image_tensor, label, _ = dataset[idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]

            # Denormalize for display
            rgb_img = denormalize(image_tensor)

            # Show original
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(rgb_img)
            ax_orig.set_title(f"{variety}", fontsize=10, fontweight="bold")
            ax_orig.axis("off")

            # Show Grad-CAM overlay
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            ax_cam = axes[row, col * 2 + 1]
            ax_cam.imshow(cam_image)
            ax_cam.set_title("Grad-CAM", fontsize=10)
            ax_cam.axis("off")

    plt.suptitle(
        f"Grad-CAM: What {model_name.upper()} Looks At",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM grid saved: {save_path}")


def generate_tsne(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    save_path: str = "outputs/tsne.png",
) -> None:
    """
    Generate t-SNE visualization of learned feature embeddings.
    Shows how the model clusters different date varieties in feature space.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_varieties = []

    with torch.no_grad():
        for images, labels, varieties in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            features = model.get_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_varieties.extend(varieties)

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    print(f"Running t-SNE on {features.shape[0]} samples, {features.shape[1]} dims...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, variety in enumerate(class_names):
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

    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.set_title("t-SNE Feature Space: Saudi Date Varieties", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate explainability visualizations")
    parser.add_argument("--model", type=str, required=True, choices=["efficientnet", "vit", "resnet"])
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    if args.all:
        args.gradcam = True
        args.tsne = True

    if not args.gradcam and not args.tsne:
        print("Specify --gradcam, --tsne, or --all")
        return

    config = load_config(args.config)
    seed_everything(config["training"]["seed"])
    device = get_device()

    # Load model
    checkpoint_path = Path("outputs") / args.model / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        return

    build_fn = MODEL_BUILDERS[args.model]
    model = build_fn(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    class_names = checkpoint["class_names"]

    output_dir = Path("outputs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gradcam:
        print(f"\nGenerating Grad-CAM for {args.model.upper()}...")
        val_transform = get_val_transforms(config)
        dataset = DateFruitDataset("data/test.csv", transform=val_transform)
        generate_gradcam_grid(
            model, args.model, dataset, device,
            save_path=str(output_dir / "gradcam_grid.png"),
        )

    if args.tsne:
        print(f"\nGenerating t-SNE for {args.model.upper()}...")
        from src.dataset import create_dataloaders
        _, _, test_loader, _ = create_dataloaders(config)
        generate_tsne(
            model, test_loader, class_names, device,
            save_path=str(output_dir / "tsne.png"),
        )


if __name__ == "__main__":
    main()
