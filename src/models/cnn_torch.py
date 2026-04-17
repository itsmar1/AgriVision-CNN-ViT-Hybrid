import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNTorch(nn.Module):
    """
    Custom CNN for binary satellite image classification.

    Architecture:
        5 × ConvBlock (32 → 64 → 128 → 256 → 512 channels)
        Global Average Pooling
        Dropout → Linear → Sigmoid

    Args:
        dropout (float): dropout probability before the classifier head
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),    # 224 → 112
            ConvBlock(32,  64),    # 112 →  56
            ConvBlock(64,  128),   #  56 →  28
            ConvBlock(128, 256),   #  28 →  14
            ConvBlock(256, 512),   #  14 →   7
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → [B, 512, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x   # raw logits — use BCEWithLogitsLoss during training


def build_cnn_torch(cfg):
    """
    Instantiate CNNTorch from a config dict.

    Args:
        cfg (dict): config from cnn_config.py

    Returns:
        CNNTorch
    """
    return CNNTorch(dropout=cfg.get("dropout", 0.3))


if __name__ == "__main__":
    model = CNNTorch()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # expect [2, 1]
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")