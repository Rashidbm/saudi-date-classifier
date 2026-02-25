"""
PyTorch Dataset and DataLoader factory for Saudi date fruit images.

Handles:
- Loading images from CSV manifests (train.csv, val.csv, test.csv)
- Albumentations augmentation pipelines (train vs val/test)
- DataLoader creation with proper config
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.utils import load_config

# ImageNet normalization stats (used with pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(config: dict) -> A.Compose:
    """Build training augmentation pipeline."""
    aug = config["augmentation"]
    size = config["data"]["image_size"]

    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=aug["horizontal_flip"]),
        A.VerticalFlip(p=aug["vertical_flip"]),
        A.Rotate(limit=aug["rotation_limit"], p=0.5),
        A.ColorJitter(
            brightness=aug["color_jitter_brightness"],
            contrast=aug["color_jitter_contrast"],
            saturation=aug["color_jitter_saturation"],
            hue=aug["color_jitter_hue"],
            p=0.5,
        ),
        A.GaussNoise(var_limit=aug["gaussian_noise_var_limit"], p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(config: dict) -> A.Compose:
    """Build validation/test transform pipeline (no augmentation)."""
    size = config["data"]["image_size"]

    return A.Compose([
        A.Resize(size + 32, size + 32),  # Resize slightly larger
        A.CenterCrop(size, size),         # Then center crop
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DateFruitDataset(Dataset):
    """
    PyTorch Dataset for Saudi date fruit images.

    Args:
        csv_path: Path to the CSV manifest (train.csv, val.csv, or test.csv)
        transform: Albumentations transform pipeline
    """

    def __init__(self, csv_path: str, transform: A.Compose | None = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Verify at least some images exist
        sample_path = Path(self.df.iloc[0]["image_path"])
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Image not found: {sample_path}\n"
                "Make sure the dataset is extracted to data/raw/"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor of shape (3, H, W) normalized
            label: Integer class index
            variety: String variety name
        """
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label_idx"])
        variety = row["variety"]

        # Load image with OpenCV (Albumentations uses numpy/cv2)
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Fallback: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label, variety

    @property
    def class_names(self) -> list[str]:
        """Return sorted list of variety names."""
        return sorted(self.df["variety"].unique().tolist())

    @property
    def num_classes(self) -> int:
        """Return number of unique classes."""
        return self.df["label_idx"].nunique()

    @property
    def class_counts(self) -> dict[str, int]:
        """Return dict of {variety: count}."""
        return dict(self.df["variety"].value_counts().sort_index())


def create_dataloaders(
    config: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Create train, val, and test DataLoaders from CSV manifests.

    Args:
        config: Configuration dict. If None, loads from default.yaml.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if config is None:
        config = load_config()

    # Build transform pipelines
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Create datasets
    train_dataset = DateFruitDataset("data/train.csv", transform=train_transform)
    val_dataset = DateFruitDataset("data/val.csv", transform=val_transform)
    test_dataset = DateFruitDataset("data/test.csv", transform=val_transform)

    # Create DataLoaders
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.class_names
    print(f"DataLoaders ready: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_loader, val_loader, test_loader, class_names
