"""
Vision Transformer (ViT-B/16) model for Saudi date fruit classification.
Uses HuggingFace transformers with ImageNet-21k pretrained weights.

Member 3's responsibility.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class DateViT(nn.Module):
    """
    Vision Transformer (ViT-B/16) with custom classification head.

    Architecture:
        - Backbone: ViT-B/16 (pretrained on ImageNet-21k)
        - Custom head: Dropout -> Linear(768, num_classes)
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()

        model_name = "google/vit-base-patch16-224-in21k"

        if pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.backbone = ViTModel(config)

        self.feature_dim = self.backbone.config.hidden_size  # 768 for ViT-B

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
        # ViT expects pixel_values, returns last_hidden_state and pooler_output
        outputs = self.backbone(pixel_values=x)
        # Use [CLS] token output for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        logits = self.head(cls_output)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (for t-SNE/UMAP and ensemble)."""
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

    def freeze_backbone(self) -> None:
        """Freeze backbone for Phase 1 training (head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("ViT backbone FROZEN")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("ViT backbone UNFROZEN")


def build_vit(config: dict) -> DateViT:
    """Factory function to build ViT from config."""
    model_cfg = config["models"]["vit"]
    num_classes = len(config["classes"])
    return DateViT(
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"],
    )
