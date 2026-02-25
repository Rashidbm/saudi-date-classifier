"""
EfficientNet-B0 model for Saudi date fruit classification.
Uses timm library with ImageNet pretrained weights.

Member 2's responsibility.
"""

import torch
import torch.nn as nn
import timm


class DateEfficientNet(nn.Module):
    """
    EfficientNet-B0 with custom classification head for date variety classification.

    Architecture:
        - Backbone: EfficientNet-B0 (pretrained on ImageNet)
        - Custom head: Dropout -> Linear(1280, num_classes)
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0 without default classifier
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Remove default head, returns features
        )
        self.feature_dim = self.backbone.num_features  # 1280 for B0

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
        features = self.backbone(x)  # (B, 1280)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (for t-SNE/UMAP and ensemble)."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone for Phase 1 training (head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("EfficientNet backbone FROZEN")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("EfficientNet backbone UNFROZEN")


def build_efficientnet(config: dict) -> DateEfficientNet:
    """Factory function to build EfficientNet from config."""
    model_cfg = config["models"]["efficientnet"]
    num_classes = len(config["classes"])
    return DateEfficientNet(
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"],
    )
