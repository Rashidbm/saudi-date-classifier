"""
Saudi Date Fruit Variety Classifier - Gradio Web App

Upload a date fruit image, get the variety prediction from the ensemble model
(ResNet + EfficientNet + ViT) with confidence percentages and heritage info.

Run locally: python app.py
Deploy on Hugging Face Spaces by pushing this file + requirements.
"""

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import get_val_transforms
from src.ensemble import (
    CLASS_NAMES,
    build_resnet50,
    build_efficientnet,
    PretrainedViTClassifier,
    load_checkpoint,
    CHECKPOINTS,
    HF_REPO_ID,
)
from src.utils import load_config, get_device, HERITAGE_INFO
from huggingface_hub import hf_hub_download


print("Loading models...")
config = load_config()
device = get_device()
transform = get_val_transforms(config)

# Download and load all three models
paths = {
    name: hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
    for name, fname in CHECKPOINTS.items()
}

models_dict = {
    "resnet": load_checkpoint(build_resnet50(num_classes=9), paths["resnet"], device),
    "efficientnet": load_checkpoint(build_efficientnet(num_classes=9), paths["efficientnet"], device),
    "vit": load_checkpoint(PretrainedViTClassifier(num_classes=9), paths["vit"], device),
}
print(f"All models loaded on {device}")


def predict(image: np.ndarray, model_choice: str):
    """Predict date variety and return confidences + heritage info."""
    if image is None:
        return {}, "Please upload an image.", "N/A"

    # Convert to RGB if needed and preprocess
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        if model_choice == "Ensemble (All 3 Models)":
            probs_sum = torch.zeros(9).to(device)
            for m in models_dict.values():
                probs_sum += F.softmax(m(input_tensor), dim=1)[0]
            probs = probs_sum / len(models_dict)
        else:
            key = model_choice.lower().split(" ")[0]
            probs = F.softmax(models_dict[key](input_tensor), dim=1)[0]

    # Build confidence dict
    confidences = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(9)}

    # Top prediction
    top_idx = probs.argmax().item()
    top_variety = CLASS_NAMES[top_idx]
    top_conf = probs[top_idx].item() * 100

    # Heritage info
    h = HERITAGE_INFO.get(top_variety, {})
    heritage_text = f"""## {top_variety} ({h.get('arabic', '')})

**Confidence:** {top_conf:.1f}%

**Region:** {h.get('region', 'N/A')}

**Description:** {h.get('description', 'N/A')}

**Flavor:** {h.get('flavor', 'N/A')}

**Cultural Significance:** {h.get('significance', 'N/A')}
"""

    return confidences, heritage_text, f"{top_variety} ({top_conf:.1f}%)"


def build_app():
    model_choices = [
        "Ensemble (All 3 Models)",
        "ViT (Vision Transformer)",
        "EfficientNet (EfficientNet-B0)",
        "ResNet (ResNet-50)",
    ]

    with gr.Blocks(title="Saudi Date Fruit Classifier", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # Saudi Date Fruit Variety Classifier
            Upload a photo of a date fruit to identify its Saudi variety.
            The ensemble combines three deep learning models for 90%+ accuracy.
            """
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
                    value="Ensemble (All 3 Models)",
                    label="Model",
                )
                predict_btn = gr.Button("Classify", variant="primary", size="lg")

            with gr.Column(scale=1):
                variety_label = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Label(
                    label="Confidence per Variety",
                    num_top_classes=9,
                )

        with gr.Row():
            heritage_output = gr.Markdown(
                value="Upload an image to see heritage information."
            )

        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[confidence_output, heritage_output, variety_label],
        )

        image_input.change(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[confidence_output, heritage_output, variety_label],
        )

        gr.Markdown(
            """
            ---
            Classifies 9 Saudi date varieties: Ajwa, Galaxy, Medjool, Meneifi,
            Nabtat Ali, Rutab, Shaishe, Sokari (Sukkari), and Sugaey.
            """
        )

    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
