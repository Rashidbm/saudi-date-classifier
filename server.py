"""
FastAPI backend for the Saudi Date Classifier.

Serves a custom static site (static/index.html) and a /api/predict endpoint
that runs the ResNet/EfficientNet/ViT ensemble on uploaded images and returns
predictions, confidence breakdown, heritage info, and a Grad-CAM overlay.

Run: python server.py
"""

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import get_val_transforms
from src.ensemble import (
    CHECKPOINTS,
    CLASS_NAMES,
    HF_REPO_ID,
    PretrainedViTClassifier,
    build_efficientnet,
    build_resnet50,
    load_checkpoint,
)
from src.explainability import reshape_transform
from src.utils import HERITAGE_INFO, get_device, load_config

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
RESULTS_DIR = ROOT / "results"

print("Loading models...")
config = load_config()
device = get_device()
transform = get_val_transforms(config)

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

vit_target_layer = models_dict["vit"].backbone.vit.encoder.layer[-1].layernorm_before
gradcam = GradCAM(
    model=models_dict["vit"],
    target_layers=[vit_target_layer],
    reshape_transform=reshape_transform,
)

app = FastAPI(title="Saudi Date Classifier")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/tsne.png")
def tsne_image():
    path = RESULTS_DIR / "tsne.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="t-SNE image not generated yet")
    return FileResponse(path)


def _encode_png(rgb: np.ndarray) -> str:
    pil = Image.fromarray(rgb.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form("ensemble")):
    data = await file.read()
    try:
        pil = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    image = np.array(pil)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        if model == "ensemble":
            probs_sum = torch.zeros(9).to(device)
            for m in models_dict.values():
                probs_sum += F.softmax(m(input_tensor), dim=1)[0]
            probs = probs_sum / len(models_dict)
        elif model in models_dict:
            probs = F.softmax(models_dict[model](input_tensor), dim=1)[0]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    confidences = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(9)}
    top_idx = int(probs.argmax().item())
    top_variety = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx].item())

    gradcam_b64 = None
    try:
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=None)[0]
        resized_rgb = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(resized_rgb, grayscale_cam, use_rgb=True)
        gradcam_b64 = _encode_png(cam_image)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")

    return JSONResponse(
        {
            "variety": top_variety,
            "confidence": top_conf,
            "confidences": confidences,
            "heritage": HERITAGE_INFO.get(top_variety, {}),
            "gradcam": gradcam_b64,
            "model": model,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7864)
