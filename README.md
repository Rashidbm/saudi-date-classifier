# Saudi Date Fruit Variety Classifier

Multi-architecture deep learning classifier for 9 Saudi date varieties with explainability and heritage context.

**Varieties:** Ajwa | Galaxy | Medjool | Meneifi | Nabtat Ali | Rutab | Shaishe | Sokari (Sukkari) | Sugaey

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Rashidbm/saudi-date-classifier.git
cd saudi-date-classifier
```

### 2. Install dependencies
```bash
pip install -e .
```

### 3. Download the dataset
1. Go to: https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP so the variety folders are inside `data/raw/`:
```
data/raw/
├── Ajwa/
├── Galaxy/
├── Medjool/
├── Meneifi/
├── Nabtat Ali/
├── Rutab/
├── Shaishe/
├── Sokari/
└── Sugaey/
```

### 4. Prepare the data splits
```bash
python -m src.data_setup
```
This creates `data/train.csv`, `data/val.csv`, and `data/test.csv` with stratified 70/15/15 splits.

### 5. Train a model
```bash
python -m src.train --model efficientnet
python -m src.train --model vit
python -m src.train --model resnet
```

### 6. Evaluate
```bash
python -m src.evaluate --model efficientnet
python -m src.evaluate --compare    # Compare all trained models
```

### 7. Run the ensemble
```bash
python -m src.ensemble
```

### 8. Explainability
```bash
python -m src.explainability --model efficientnet --all
```

### 9. Launch the web app
```bash
python app.py
```

---

## Team Assignments

| Member | Responsibility | Files |
|--------|---------------|-------|
| **Member 1 (Lead)** | Data pipeline + coordination | `src/data_setup.py`, `src/dataset.py`, `src/utils.py`, `configs/`, `notebooks/eda.ipynb` |
| **Member 2** | EfficientNet-B0 model | `src/models/efficientnet.py`, train & evaluate with `--model efficientnet` |
| **Member 3** | Vision Transformer (ViT) | `src/models/vit.py`, train & evaluate with `--model vit` |
| **Member 4** | ResNet-50 + Ensemble | `src/models/resnet.py`, `src/ensemble.py`, model comparison analysis |
| **Member 5** | Explainability + Gradio App | `src/explainability.py`, `src/predict.py`, `app.py` |

> **Note:** Starter template code is provided for all files. Each member should review, modify, and improve their assigned files.

---

## Project Structure

```
saudi-date-classifier/
├── pyproject.toml              # Dependencies
├── configs/
│   └── default.yaml            # Shared hyperparameters
├── data/
│   └── raw/                    # Dataset (download from Kaggle)
├── src/
│   ├── data_setup.py           # Folder scanner, CSV builder, splitter
│   ├── dataset.py              # PyTorch Dataset + augmentations
│   ├── models/
│   │   ├── efficientnet.py     # [Member 2]
│   │   ├── vit.py              # [Member 3]
│   │   └── resnet.py           # [Member 4]
│   ├── train.py                # Shared training loop
│   ├── ensemble.py             # [Member 4] Multi-model ensemble
│   ├── evaluate.py             # Metrics + comparison
│   ├── explainability.py       # [Member 5] Grad-CAM + t-SNE
│   ├── predict.py              # [Member 5] Single-image inference
│   └── utils.py                # Seed, device, heritage knowledge
├── notebooks/
│   └── eda.ipynb               # Data exploration
├── app.py                      # [Member 5] Gradio web app
└── outputs/                    # Checkpoints + plots (gitignored)
```

---

## Architecture

- **EfficientNet-B0** (timm) - Best accuracy/efficiency tradeoff
- **ViT-B/16** (HuggingFace) - Transformer-based approach
- **ResNet-50** (timm) - Strong CNN baseline
- **Ensemble** - Weighted soft voting across all models

Training uses 2-phase transfer learning:
1. **Phase 1:** Frozen backbone, train classification head only (lr=1e-3)
2. **Phase 2:** Unfreeze all layers, full fine-tuning (lr=1e-5)

---

## Dataset

**Date Fruit Image Dataset in Controlled Environment**
- 1,658 images, 9 Saudi varieties
- Canon DSLR, white background, ring light
- CC BY 4.0 license
- [Kaggle link](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment)
