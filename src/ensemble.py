
!pip install -q huggingface_hub transformers scikit-learn
import os

print(os.listdir("/kaggle/input"))

DATASET_DIR = "/kaggle/input/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment"  # change if needed
print(os.listdir(DATASET_DIR)[:20])

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification
from huggingface_hub import hf_hub_download

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

HF_REPO_ID = "Rashidbm/saudi-date-classifier"

RESNET_FILE = "arabic_dates_resnet50_best_V2.pth"
EFFICIENTNET_FILE = "efficientnet_best.pth"
VIT_FILE = "vit_best_model.pth"

EXPECTED_CLASSES = [
    "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali",
    "Rutab", "Shaishe", "Sokari", "Sugaey"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transforms)

class_names = full_dataset.classes
print("Classes found:", class_names)

if class_names != EXPECTED_CLASSES:
    raise ValueError(f"Class order mismatch.\nExpected: {EXPECTED_CLASSES}\nFound: {class_names}")

n = len(full_dataset)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(n, generator=generator).tolist()

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

test_dataset = Subset(full_dataset, test_idx)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Total images:", n)
print("Test images:", len(test_dataset))

# ResNet50
def build_resnet50(num_classes=9, dropout=0.3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


# EfficientNet-B0
def build_efficientnet(num_classes=9, pretrained=True, dropout=0.3):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


# ViT wrapper
class PretrainedViTClassifier(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", num_classes=9, dropout=0.1):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits

resnet_path = hf_hub_download(repo_id=HF_REPO_ID, filename=RESNET_FILE)
efficientnet_path = hf_hub_download(repo_id=HF_REPO_ID, filename=EFFICIENTNET_FILE)
vit_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VIT_FILE)

print("Downloaded:")
print(resnet_path)
print(efficientnet_path)
print(vit_path)

def load_checkpoint_flex(model, path):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


resnet_model = build_resnet50(num_classes=len(class_names), dropout=0.3)
resnet_model = load_checkpoint_flex(resnet_model, resnet_path)

efficientnet_model = build_efficientnet(num_classes=len(class_names), pretrained=True, dropout=0.3)
efficientnet_model = load_checkpoint_flex(efficientnet_model, efficientnet_path)

vit_model = PretrainedViTClassifier(
    model_name="google/vit-base-patch16-224-in21k",
    num_classes=len(class_names),
    dropout=0.1,
)
vit_model = load_checkpoint_flex(vit_model, vit_path)

print("All models loaded successfully.")

@torch.no_grad()
def evaluate_ensemble(resnet_model, efficientnet_model, vit_model, loader, class_names):
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits_resnet = resnet_model(images)
        logits_eff = efficientnet_model(images)
        logits_vit = vit_model(images)

        probs_resnet = F.softmax(logits_resnet, dim=1)
        probs_eff = F.softmax(logits_eff, dim=1)
        probs_vit = F.softmax(logits_vit, dim=1)

        ensemble_probs = (probs_resnet + probs_eff + probs_vit) / 3.0
        preds = ensemble_probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nEnsemble Test Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


evaluate_ensemble(resnet_model, efficientnet_model, vit_model, test_loader, class_names)
