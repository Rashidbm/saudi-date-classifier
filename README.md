# Saudi Date Fruit Variety Classifier

Classifies 9 Saudi date varieties using EfficientNet, ViT, and ResNet with an ensemble and Grad-CAM explainability.

## Setup

```bash
git clone https://github.com/Rashidbm/saudi-date-classifier.git
cd saudi-date-classifier
pip install -e .
```

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment

Extract so the folders sit inside `data/raw/`:

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

Then run:

```bash
python -m src.data_setup
```

This splits the data into train/val/test CSVs (70/15/15).

## Training

```bash
python -m src.train --model efficientnet
python -m src.train --model vit
python -m src.train --model resnet
```

## Evaluation

```bash
python -m src.evaluate --model efficientnet
python -m src.evaluate --compare
```

## Ensemble

```bash
python -m src.ensemble
```

## Web App

```bash
python app.py
```

## Project Structure

```
src/
├── data_setup.py       # Scans dataset folders, builds CSV splits
├── dataset.py          # PyTorch Dataset, augmentations, DataLoaders
├── utils.py            # Seeding, device selection, config loading, heritage info
├── models/
│   ├── efficientnet.py
│   ├── vit.py
│   └── resnet.py
├── train.py
├── ensemble.py
├── evaluate.py
├── explainability.py
└── predict.py
configs/
└── default.yaml
app.py
```

## Dataset Info

- 1,658 images, 9 Saudi varieties
- Controlled environment (white background, ring light)
- CC BY 4.0
