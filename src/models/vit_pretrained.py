"""
Pretrained ViT Model for Transfer Learning
Loads pretrained ViT from HuggingFace and adapts it for Saudi date classification.

Author: Ahmad
Branch: Ahmad-VIT
Purpose: Vision Transformer model for Saudi date classification ensemble
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor


class PretrainedViTClassifier(nn.Module):
    """
    Pretrained ViT model from HuggingFace with custom classifier head.
    Supports freezing/unfreezing backbone for transfer learning.
    """
    
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        num_classes=9,
        dropout=0.1,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Load pretrained ViT
        self.backbone = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
        self.num_classes = num_classes
        self.backbone_frozen = False
        
    def freeze_backbone(self):
        """Freeze all backbone parameters except classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        self.backbone_frozen = True
        print("[OK] Backbone frozen - only classifier head trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        print("[OK] Backbone unfrozen - all parameters trainable")
    
    def get_trainable_params(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        return total, trainable
    
    def forward(self, x):
        """Forward pass."""
        outputs = self.backbone(x)
        return outputs.logits


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing Pretrained ViT")
    print("=" * 60)
    
    model = PretrainedViTClassifier(
        model_name="google/vit-base-patch16-224-in21k",
        num_classes=9
    )
    model = model.to(device)
    
    total, trainable = model.get_trainable_params()
    print(f"Total parameters: {total:,}")
    print(f"Trainable: {trainable:,}\n")
    
    # Phase 1: Freeze backbone
    model.freeze_backbone()
    total, trainable = model.get_trainable_params()
    print(f"After freezing backbone:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}\n")
    
    # Phase 2: Unfreeze backbone
    model.unfreeze_backbone()
    total, trainable = model.get_trainable_params()
    print(f"After unfreezing backbone:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}\n")
    
    # Test forward pass
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("=" * 60)
