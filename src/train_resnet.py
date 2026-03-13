import random
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pickle

from src.models.resnet import build_resnet50


DATA_DIR = "data/raw"   # change this if needed
MODEL_SAVE_PATH = "arabic_dates_resnet50_best.pth"
PICKLE_SAVE_PATH = "arabic_dates_classnames.pkl"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SPLITS = (0.70, 0.15, 0.15)   # train, val, test
SEED = 42

EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

DROPOUT = 0.3

EXPECTED_CLASSES = [
    "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali",
    "Rutab", "Shaishe", "Sokari", "Sugaey"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class AddGaussianNoise:
    def __init__(self, var_min=10.0, var_max=50.0, p=0.3):
        self.p = p
        self.sigma_min = (var_min ** 0.5) / 255.0
        self.sigma_max = (var_max ** 0.5) / 255.0

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 1.0)


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    AddGaussianNoise(var_min=10.0, var_max=50.0, p=0.3),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
eval_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

if class_names != EXPECTED_CLASSES:
    raise ValueError(f"Class order mismatch.\nExpected: {EXPECTED_CLASSES}\nFound: {class_names}")

n = len(train_dataset)
n_train = int(SPLITS[0] * n)
n_val = int(SPLITS[1] * n)
n_test = n - n_train - n_val

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(n, generator=generator).tolist()

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_data = Subset(train_dataset, train_idx)
val_data = Subset(eval_dataset, val_idx)
test_data = Subset(eval_dataset, test_idx)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


model = build_resnet50(num_classes=num_classes, dropout=DROPOUT)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for inputs, labels in tqdm(loader, leave=False, desc=("Training" if train else "Validating")):
            inputs, labels = inputs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


best_val_loss = float("inf")


def save_best():
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": class_names
    }, MODEL_SAVE_PATH)


# Phase 1: head only
for name, param in model.named_parameters():
    param.requires_grad = name.startswith("fc.")

optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_PHASE1,
    weight_decay=WEIGHT_DECAY
)

print("\n🚀 Phase 1 (Head only)")
patience_left = EARLY_STOPPING_PATIENCE

for epoch in range(EPOCHS_PHASE1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)

    print(f"Phase1 Ep {epoch+1:02d}/{EPOCHS_PHASE1} | "
          f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
          f"Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        save_best()
        patience_left = EARLY_STOPPING_PATIENCE
        print(f"💾 Saved best model (Val Loss: {best_val_loss:.4f})")
    else:
        patience_left -= 1
        if patience_left == 0:
            print("🛑 Early stopping triggered in Phase 1")
            break


ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])


# Phase 2: fine-tune all
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR_PHASE2,
    weight_decay=WEIGHT_DECAY
)

print("\n🔥 Phase 2 (Fine-tune all)")
patience_left = EARLY_STOPPING_PATIENCE

for epoch in range(EPOCHS_PHASE2):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)

    print(f"Phase2 Ep {epoch+1:02d}/{EPOCHS_PHASE2} | "
          f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
          f"Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        save_best()
        patience_left = EARLY_STOPPING_PATIENCE
        print(f"💾 Saved best model (Val Loss: {best_val_loss:.4f})")
    else:
        patience_left -= 1
        if patience_left == 0:
            print("🛑 Early stopping triggered in Phase 2")
            break


print(f"\n✅ Training done. Best Val Loss: {best_val_loss:.4f}")
print(f"✅ Best model saved to {MODEL_SAVE_PATH}")

ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"🧪 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

with open(PICKLE_SAVE_PATH, "wb") as f:
    pickle.dump(class_names, f)
print(f"📦 Saved class names to {PICKLE_SAVE_PATH}")