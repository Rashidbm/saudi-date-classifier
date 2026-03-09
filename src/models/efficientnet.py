import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet(num_classes=9, pretrained=True, dropout=0.3):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def freeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = True
