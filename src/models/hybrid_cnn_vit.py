import torch
import torch.nn as nn
import torchvision.models as tvm


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT for binary satellite image classification.

    Architecture:
        1. CNN encoder  — ResNet-18 backbone (without final pool + FC)
                          outputs spatial feature map [B, 512, 7, 7]
        2. Patch tokens — flatten spatial dims → 49 tokens of size 512
                          linear projection to vit_hidden_dim
                          + learnable positional embeddings
                          + prepended [CLS] token
        3. ViT encoder  — stack of Transformer encoder blocks
        4. Head         — [CLS] token → LayerNorm → Linear → sigmoid

    The CNN captures local textures and edges; the Transformer reasons
    over their global spatial relationships.

    Args:
        cnn_pretrained (bool):  load ImageNet weights for ResNet-18
        vit_hidden_dim (int):   token projection dimension
        vit_num_heads  (int):   attention heads per transformer block
        vit_num_layers (int):   number of transformer encoder blocks
        vit_mlp_dim    (int):   feedforward hidden dim inside transformer
        vit_dropout    (float): dropout inside transformer blocks
    """

    def __init__(
        self,
        cnn_pretrained=True,
        vit_hidden_dim=768,
        vit_num_heads=8,
        vit_num_layers=4,
        vit_mlp_dim=1024,
        vit_dropout=0.1,
    ):
        super().__init__()

        # ── CNN encoder ──────────────────────────────────────────────────────
        resnet = tvm.resnet18(weights="IMAGENET1K_V1" if cnn_pretrained else None)
        # keep everything up to (but not including) avgpool and fc
        self.cnn_encoder = nn.Sequential(*list(resnet.children())[:-2])
        # output: [B, 512, 7, 7]  for a 224×224 input

        cnn_out_ch = 512
        num_patches = 7 * 7   # 49 spatial tokens

        # ── Patch projection ─────────────────────────────────────────────────
        self.patch_proj = nn.Linear(cnn_out_ch, vit_hidden_dim)
        self.pos_embed  = nn.Parameter(
            torch.randn(1, num_patches + 1, vit_hidden_dim) * 0.02
        )
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, vit_hidden_dim))

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_hidden_dim,
            nhead=vit_num_heads,
            dim_feedforward=vit_mlp_dim,
            dropout=vit_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=vit_num_layers
        )

        # ── Classification head ───────────────────────────────────────────────
        self.norm = nn.LayerNorm(vit_hidden_dim)
        self.head = nn.Linear(vit_hidden_dim, 1)

    def forward(self, x):
        B = x.size(0)

        # 1. CNN feature map
        feat = self.cnn_encoder(x)           # [B, 512, 7, 7]

        # 2. Flatten spatial dims → token sequence
        feat = feat.flatten(2)               # [B, 512, 49]
        feat = feat.permute(0, 2, 1)         # [B, 49, 512]
        tokens = self.patch_proj(feat)        # [B, 49, vit_hidden_dim]

        # 3. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 50, D]

        # 4. Add positional embeddings
        tokens = tokens + self.pos_embed          # [B, 50, D]

        # 5. Transformer encoder
        tokens = self.transformer(tokens)          # [B, 50, D]

        # 6. Classify from [CLS] token
        cls_out = self.norm(tokens[:, 0])          # [B, D]
        return self.head(cls_out)                  # [B, 1] — raw logits


def build_hybrid(cfg):
    """
    Instantiate HybridCNNViT from config.

    Args:
        cfg (dict): config from hybrid_config.py

    Returns:
        HybridCNNViT
    """
    return HybridCNNViT(
        cnn_pretrained=cfg.get("cnn_pretrained", True),
        vit_hidden_dim=cfg.get("vit_hidden_dim", 768),
        vit_num_heads=cfg.get("vit_num_heads", 8),
        vit_num_layers=cfg.get("vit_num_layers", 4),
        vit_mlp_dim=cfg.get("vit_mlp_dim", 1024),
        vit_dropout=cfg.get("vit_dropout", 0.1),
    )


if __name__ == "__main__":
    model = HybridCNNViT()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")