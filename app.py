"""
Gradio Web App: Saudi Date Fruit Variety Classifier

Upload or take a photo of a date fruit → get variety prediction with
confidence percentages, Grad-CAM visualization, and heritage info.

Usage:
    python app.py

Deploys on Hugging Face Spaces with: gradio app.py
"""

import os
from pathlib import Path

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import get_val_transforms
from src.models.efficientnet import build_efficientnet
from src.models.vit import build_vit
from src.models.resnet import build_resnet
from src.utils import load_config, get_device, HERITAGE_INFO


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "vit": build_vit,
    "resnet": build_resnet,
}

# Global state
loaded_models = {}
class_names = []
config = None
device = None
transform = None


def initialize():
    """Load config and all available models on startup."""
    global config, device, transform, class_names, loaded_models

    config = load_config()
    device = get_device()
    transform = get_val_transforms(config)

    print("Loading models...")
    for model_name, build_fn in MODEL_BUILDERS.items():
        checkpoint_path = Path("outputs") / model_name / "best_model.pth"
        if checkpoint_path.exists():
            model = build_fn(config)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            loaded_models[model_name] = model
            class_names = checkpoint["class_names"]
            print(f"  ✅ {model_name} loaded (val_acc={checkpoint['val_acc']:.4f})")
        else:
            print(f"  ⚠️  {model_name} not found (no checkpoint)")

    if not loaded_models:
        print("WARNING: No trained models found! Train at least one model first.")
        # Set default class names from config
        class_names = config.get("classes", [])

    print(f"Ready with {len(loaded_models)} models. Classes: {class_names}")


def predict(image: np.ndarray, model_choice: str) -> tuple[dict, str, str]:
    """
    Main prediction function called by Gradio.

    Args:
        image: RGB numpy array from Gradio
        model_choice: Which model to use

    Returns:
        confidences_dict, heritage_text, variety_label
    """
    if not loaded_models:
        return {}, "No trained models available. Please train a model first.", "N/A"

    if image is None:
        return {}, "Please upload an image.", "N/A"

    # Preprocess
    image_rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    if model_choice == "Ensemble (All Models)":
        avg_probs = torch.zeros(len(class_names)).to(device)
        with torch.no_grad():
            for model in loaded_models.values():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                avg_probs += probs
        avg_probs /= len(loaded_models)
        probs = avg_probs
    else:
        # Map display name to key
        model_key = model_choice.lower().split(" ")[0]
        if model_key not in loaded_models:
            return {}, f"Model '{model_key}' not loaded.", "N/A"
        with torch.no_grad():
            logits = loaded_models[model_key](input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

    # Build confidence dict for Gradio Label component
    confidences = {
        class_names[i]: float(probs[i].item())
        for i in range(len(class_names))
    }

    # Top prediction
    top_idx = probs.argmax().item()
    top_variety = class_names[top_idx]
    top_confidence = probs[top_idx].item() * 100

    # Heritage info
    heritage = HERITAGE_INFO.get(top_variety, {})
    if heritage:
        heritage_text = f"""
## {top_variety} ({heritage.get('arabic', '')})

**Confidence:** {top_confidence:.1f}%

---

🌍 **Region:** {heritage.get('region', 'N/A')}

🍬 **Flavor:** {heritage.get('flavor', 'N/A')}

📖 **Description:** {heritage.get('description', 'N/A')}

⭐ **Cultural Significance:** {heritage.get('significance', 'N/A')}
"""
    else:
        heritage_text = f"## {top_variety}\n\n**Confidence:** {top_confidence:.1f}%"

    return confidences, heritage_text, f"{top_variety} ({top_confidence:.1f}%)"


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    # Available model choices
    model_choices = []
    for name in ["efficientnet", "vit", "resnet"]:
        if name in loaded_models:
            display_name = {
                "efficientnet": "EfficientNet-B0",
                "vit": "ViT-B/16",
                "resnet": "ResNet-50",
            }[name]
            model_choices.append(display_name)

    if len(loaded_models) > 1:
        model_choices.append("Ensemble (All Models)")

    if not model_choices:
        model_choices = ["No models available"]

    default_choice = "Ensemble (All Models)" if "Ensemble (All Models)" in model_choices else model_choices[0]

    with gr.Blocks(
        title="Saudi Date Fruit Classifier 🌴",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        """,
    ) as app:
        gr.Markdown(
            """
            # Saudi Date Fruit Variety Classifier
            ### Powered by Deep Learning | Multi-Architecture Comparison with Explainability

            Upload a photo of a date fruit to identify its Saudi variety with confidence percentages
            and learn about its cultural heritage.
            """,
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Date Fruit Image",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=350,
                )
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_choice,
                    label="Select Model",
                )
                predict_btn = gr.Button(
                    "Classify Date",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                variety_label = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                    scale=1,
                )
                confidence_output = gr.Label(
                    label="Confidence per Variety",
                    num_top_classes=9,
                )

        with gr.Row():
            heritage_output = gr.Markdown(
                label="Heritage Information",
                value="*Upload an image to see heritage information*",
            )

        # Connect prediction
        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[confidence_output, heritage_output, variety_label],
        )

        # Also predict on image upload
        image_input.change(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[confidence_output, heritage_output, variety_label],
        )

        gr.Markdown(
            """
            ---
            **About:** This classifier uses EfficientNet-B0, Vision Transformer (ViT-B/16),
            and ResNet-50 to identify 9 Saudi date varieties: Ajwa, Galaxy, Medjool, Meneifi,
            Nabtat Ali, Rutab, Shaishe, Sokari (Sukkari), and Sugaey.

            Built with PyTorch, timm, HuggingFace Transformers, and Gradio.
            """
        )

    return app


# Initialize on import
initialize()

# Build app
demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
