import torch
import torch.nn as nn
import timm


class ViTFineTuned(nn.Module):
    """
    Pretrained Vision Transformer fine-tuned for binary classification.

    Strategy:
        Phase 1 — freeze backbone, train head only (fast convergence)
        Phase 2 — unfreeze all, train end-to-end with low LR

    Args:
        backbone (str):   timm model name, e.g. "vit_base_patch16_224"
        pretrained (bool): load ImageNet weights
    """

    def __init__(self, backbone="vit_base_patch16_224", pretrained=True):
        super().__init__()

        self.vit = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,      # remove original head → outputs [B, hidden_dim]
        )
        hidden_dim = self.vit.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),   # binary output
        )

        # start with backbone frozen
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        print("[ViT] Backbone frozen — training head only")

    def unfreeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = True
        print("[ViT] Backbone unfrozen — full fine-tuning")

    def forward(self, x):
        features = self.vit(x)     # [B, hidden_dim]
        return self.head(features)  # [B, 1] — raw logits


def build_vit(cfg):
    """
    Instantiate ViTFineTuned from config.

    Args:
        cfg (dict): config from vit_config.py

    Returns:
        ViTFineTuned
    """
    return ViTFineTuned(
        backbone=cfg.get("vit_backbone", "vit_base_patch16_224"),
        pretrained=cfg.get("pretrained", True),
    )


if __name__ == "__main__":
    model = ViTFineTuned()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")