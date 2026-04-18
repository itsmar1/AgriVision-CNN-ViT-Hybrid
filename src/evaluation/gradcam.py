import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN and Hybrid models.

    Hooks onto the target convolutional layer, computes the gradient of
    the predicted class score w.r.t. the feature map activations, and
    produces a spatial heatmap highlighting discriminative regions.

    Args:
        model       (nn.Module): trained CNN or HybridCNNViT
        target_layer (nn.Module): the conv layer to hook
                                  e.g. model.features[-1]  (CNNTorch)
                                       model.cnn_encoder[-1] (Hybrid)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # register hooks
        self._fwd = target_layer.register_forward_hook(self._save_activations)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            input_tensor (Tensor): [1, 3, H, W]  (single image, on device)

        Returns:
            np.ndarray: heatmap [H, W] in range [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)          # [1, 1]
        self.model.zero_grad()
        output.squeeze().backward()

        # global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


# ── Attention Rollout (ViT) ───────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.

    Recursively multiplies attention matrices across all transformer
    blocks to trace the information flow to the [CLS] token.

    Supports timm ViT models (ViTFineTuned) and the HybridCNNViT's
    internal transformer encoder.

    Args:
        model      (nn.Module): ViTFineTuned or HybridCNNViT
        head_fusion (str):      how to fuse multi-head attention
                                "mean" | "min" | "max"
    """

    def __init__(self, model, head_fusion="mean"):
        self.model = model
        self.head_fusion = head_fusion
        self.attention_maps = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def _hook(module, input, output):
            # output is (attn_output, attn_weights) when need_weights=True
            # for nn.MultiheadAttention; timm uses its own attn module
            pass

        # for timm ViT — hook each block's attn.proj
        if hasattr(self.model, "vit"):
            for block in self.model.vit.blocks:
                h = block.attn.register_forward_hook(
                    lambda m, i, o: self.attention_maps.append(
                        o.detach() if not isinstance(o, tuple) else o[1].detach()
                    )
                )
                self._hooks.append(h)

    def generate(self, input_tensor, grid_size=14):
        """
        Compute attention rollout heatmap.

        Args:
            input_tensor (Tensor): [1, 3, H, W]
            grid_size    (int):    number of patches per side

        Returns:
            np.ndarray: heatmap [H, W] normalised to [0, 1]
        """
        self.attention_maps = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        rollout = None
        for attn in self.attention_maps:
            if attn.dim() == 4:             # [B, heads, tokens, tokens]
                if self.head_fusion == "mean":
                    attn = attn.mean(dim=1)
                elif self.head_fusion == "max":
                    attn = attn.max(dim=1).values
                else:
                    attn = attn.min(dim=1).values

                # add residual identity
                I = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
                attn = (attn + I) / 2
                attn = attn / attn.sum(dim=-1, keepdim=True)

                rollout = attn if rollout is None else torch.bmm(attn, rollout)

        if rollout is None:
            return np.zeros((grid_size, grid_size))

        # [CLS] row attention over patch tokens
        cls_attn = rollout[0, 0, 1:].reshape(grid_size, grid_size).cpu().numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        return cls_attn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ── Overlay utilities ─────────────────────────────────────────────────────────

def overlay_heatmap(image_tensor, heatmap, alpha=0.5):
    """
    Overlay a heatmap on an image tensor.

    Args:
        image_tensor (Tensor):   [3, H, W] normalised image
        heatmap      (np.ndarray): [H, W] values in [0, 1]
        alpha        (float):    heatmap opacity

    Returns:
        np.ndarray: [H, W, 3] blended image in [0, 1]
    """
    # denormalise (rough — assumes ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    H, W = img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H))
    ) / 255.0
    colormap = cm.get_cmap("jet")(heatmap_resized)[..., :3]
    blended = (1 - alpha) * img + alpha * colormap
    return np.clip(blended, 0, 1)


def save_gradcam_grid(model, target_layer, images, labels, probs,
                      figures_dir, run_name, device, n=8):
    """
    Generate and save a grid of Grad-CAM overlays.

    Args:
        model        (nn.Module):  trained CNN or Hybrid
        target_layer (nn.Module):  convolutional layer to hook
        images       (Tensor):     [N, 3, H, W]
        labels       (list[int]):  true labels
        probs        (list[float]):predicted probabilities
        figures_dir  (str):        output directory
        run_name     (str):        used in filename
        device       (str):        "cuda" or "cpu"
        n            (int):        number of samples
    """
    gradcam = GradCAM(model, target_layer)
    n = min(n, len(images))
    CLASS_NAMES = ["non_agri", "agri"]

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(n):
        inp = images[i].unsqueeze(0).to(device).requires_grad_(True)
        heatmap = gradcam.generate(inp)
        overlay = overlay_heatmap(images[i], heatmap)
        pred = int(probs[i] >= 0.5)
        color = "green" if pred == labels[i] else "red"
        axes[i].imshow(overlay)
        axes[i].set_title(
            f"True: {CLASS_NAMES[labels[i]]}\n"
            f"Pred: {CLASS_NAMES[pred]} ({probs[i]:.2f})",
            fontsize=8, color=color,
        )
        axes[i].axis("off")

    gradcam.remove_hooks()
    plt.suptitle(f"Grad-CAM — {run_name}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{run_name}_gradcam.png")
    plt.savefig(out)
    plt.close()
    print(f"[GradCAM] Saved → {out}")