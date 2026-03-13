import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes, dropout=0.3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model