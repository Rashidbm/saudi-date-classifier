"""
ResNet-50 model for Saudi date fruit classification.
Uses timm library with ImageNet pretrained weights.

Member 4's responsibility.
"""

import torch
import torch.nn as nn
import timm


class DateResNet(nn.Module):
    """
    ResNet-50 with custom classification head for date variety classification.

    Architecture:
        - Backbone: ResNet-50 (pretrained on ImageNet)
        - Custom head: Dropout -> Linear(2048, num_classes)
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-50 without default classifier
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,  # Remove default head
        )
        self.feature_dim = self.backbone.num_features  # 2048 for ResNet-50

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)  # (B, 2048)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (for t-SNE/UMAP and ensemble)."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone for Phase 1 training (head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("ResNet backbone FROZEN")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("ResNet backbone UNFROZEN")


def build_resnet(config: dict) -> DateResNet:
    """Factory function to build ResNet from config."""
    model_cfg = config["models"]["resnet"]
    num_classes = len(config["classes"])
    return DateResNet(
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"],
    )
