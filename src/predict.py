"""
Single-image inference: load a trained model and predict date variety with confidence %.

Usage:
    python -m src.predict --image path/to/date.jpg --model efficientnet
    python -m src.predict --image path/to/date.jpg --model ensemble
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.utils import load_config, get_device, HERITAGE_INFO


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}


def load_model(model_name: str, config: dict, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Load a trained model from checkpoint."""
    checkpoint_path = Path("outputs") / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {checkpoint_path}")

    build_fn = MODEL_BUILDERS[model_name]
    model = build_fn(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]
    return model, class_names


def predict_image(
    image_path: str,
    model: nn.Module,
    class_names: list[str],
    config: dict,
    device: torch.device,
) -> dict:
    """
    Predict date variety from an image.

    Returns:
        {
            "variety": str,            # Predicted variety name
            "confidence": float,       # Top confidence (0-100%)
            "all_confidences": dict,   # {variety: confidence%} for all classes
            "heritage": dict,          # Heritage info for predicted variety
        }
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply val transforms
    transform = get_val_transforms(config)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Parse results
    confidences = {
        class_names[i]: round(probs[i].item() * 100, 2)
        for i in range(len(class_names))
    }

    # Sort by confidence descending
    confidences = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True))

    top_variety = list(confidences.keys())[0]
    top_confidence = list(confidences.values())[0]

    heritage = HERITAGE_INFO.get(top_variety, {})

    return {
        "variety": top_variety,
        "confidence": top_confidence,
        "all_confidences": confidences,
        "heritage": heritage,
    }


def ensemble_predict_image(
    image_path: str,
    config: dict,
    device: torch.device,
) -> dict:
    """Predict using ensemble of all available models."""
    # Load all available models
    models = {}
    class_names = None

    for model_name in MODEL_BUILDERS:
        try:
            model, names = load_model(model_name, config, device)
            models[model_name] = model
            class_names = names
        except FileNotFoundError:
            continue

    if not models:
        raise RuntimeError("No trained models found. Train at least one model first.")

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_val_transforms(config)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Average predictions
    avg_probs = torch.zeros(len(class_names)).to(device)
    with torch.no_grad():
        for model in models.values():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            avg_probs += probs

    avg_probs /= len(models)

    # Parse results
    confidences = {
        class_names[i]: round(avg_probs[i].item() * 100, 2)
        for i in range(len(class_names))
    }
    confidences = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True))

    top_variety = list(confidences.keys())[0]
    heritage = HERITAGE_INFO.get(top_variety, {})

    return {
        "variety": top_variety,
        "confidence": list(confidences.values())[0],
        "all_confidences": confidences,
        "heritage": heritage,
        "models_used": list(models.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict date fruit variety")
    parser.add_argument("--image", type=str, required=True, help="Path to date fruit image")
    parser.add_argument(
        "--model", type=str, default="ensemble",
        choices=["efficientnet", "vit", "resnet", "ensemble"],
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    if args.model == "ensemble":
        result = ensemble_predict_image(args.image, config, device)
        print(f"Models used: {result.get('models_used', [])}")
    else:
        model, class_names = load_model(args.model, config, device)
        result = predict_image(args.image, model, class_names, config, device)

    # Display results
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULT")
    print(f"{'='*50}")
    print(f"  Variety: {result['variety']}")
    print(f"  Confidence: {result['confidence']}%")

    if result["heritage"]:
        h = result["heritage"]
        print(f"\n  Arabic: {h.get('arabic', 'N/A')}")
        print(f"  Region: {h.get('region', 'N/A')}")
        print(f"  Flavor: {h.get('flavor', 'N/A')}")
        print(f"  Significance: {h.get('significance', 'N/A')}")

    print(f"\n  All Confidences:")
    for variety, conf in result["all_confidences"].items():
        bar = "█" * int(conf / 2)
        print(f"    {variety:<12} {conf:>6.2f}% {bar}")


if __name__ == "__main__":
    main()
