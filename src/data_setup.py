"""
Data Setup: Scan raw dataset folders, build manifest CSVs, stratified train/val/test split.

Usage:
    python -m src.data_setup

Expects dataset extracted to data/raw/ with structure:
    data/raw/
    ├── Ajwa/
    │   ├── img001.jpg
    │   └── ...
    ├── Galaxy/
    ├── Medjool/
    └── ...
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

from src.utils import load_config, seed_everything


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def scan_dataset(raw_dir: str) -> pd.DataFrame:
    """
    Scan the raw dataset directory and build a manifest DataFrame.

    Returns DataFrame with columns: image_path, variety, label_idx
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"ERROR: Raw data directory not found: {raw_path.resolve()}")
        print("\nPlease download the dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment")
        print(f"\nExtract the ZIP so variety folders are directly inside: {raw_path.resolve()}/")
        sys.exit(1)

    # Discover variety folders
    variety_dirs = sorted([
        d for d in raw_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if len(variety_dirs) == 0:
        print(f"ERROR: No variety folders found in {raw_path.resolve()}")
        print("Expected folders like: Ajwa/, Galaxy/, Medjool/, etc.")
        sys.exit(1)

    # Build class-to-index mapping
    class_names = [d.name for d in variety_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} varieties: {class_names}")

    # Collect all image paths
    records = []
    skipped = 0

    for variety_dir in variety_dirs:
        variety = variety_dir.name
        # Walk recursively to handle nested structures
        for img_path in sorted(variety_dir.rglob("*")):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Verify image is readable
            try:
                with Image.open(img_path) as img:
                    img.verify()
                records.append({
                    "image_path": str(img_path),
                    "variety": variety,
                    "label_idx": class_to_idx[variety],
                })
            except Exception:
                skipped += 1

    if skipped > 0:
        print(f"Warning: Skipped {skipped} corrupted/unreadable images")

    df = pd.DataFrame(records)
    print(f"Total valid images: {len(df)}")

    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test with stratification by variety.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["variety"],
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["variety"],
        random_state=seed,
    )

    return train_df, val_df, test_df


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print detailed split statistics."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(train_df):>5} images ({len(train_df)/len(train_df)+len(val_df)+len(test_df):.0%}... )")
    print(f"  Val:   {len(val_df):>5} images")
    print(f"  Test:  {len(test_df):>5} images")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  Total: {total:>5} images")

    print(f"\n{'Variety':<15} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 45)

    all_varieties = sorted(train_df["variety"].unique())
    for variety in all_varieties:
        n_train = len(train_df[train_df["variety"] == variety])
        n_val = len(val_df[val_df["variety"] == variety])
        n_test = len(test_df[test_df["variety"] == variety])
        n_total = n_train + n_val + n_test
        print(f"  {variety:<13} {n_train:>6} {n_val:>6} {n_test:>6} {n_total:>6}")

    print("=" * 60)


def main():
    """Main data setup pipeline."""
    config = load_config()
    seed_everything(config["data"]["seed"])

    # Step 1: Scan dataset
    print("Step 1: Scanning dataset...")
    df = scan_dataset(config["data"]["raw_dir"])

    # Step 2: Stratified split
    print("\nStep 2: Splitting dataset...")
    splits = config["data"]["splits"]
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=splits[0],
        val_ratio=splits[1],
        test_ratio=splits[2],
        seed=config["data"]["seed"],
    )

    # Step 3: Save CSVs
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print(f"\nSaved: data/train.csv ({len(train_df)} rows)")
    print(f"Saved: data/val.csv ({len(val_df)} rows)")
    print(f"Saved: data/test.csv ({len(test_df)} rows)")

    # Step 4: Print summary
    print_split_summary(train_df, val_df, test_df)

    # Save class mapping for reference
    class_names = sorted(df["variety"].unique())
    class_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"\nClass mapping: {class_map}")

    print("\n✅ Data setup complete! Ready for training.")


if __name__ == "__main__":
    main()
