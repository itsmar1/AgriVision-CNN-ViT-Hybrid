import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss — reduces the relative loss for well-classified
    examples and focuses training on hard, misclassified ones.

    Particularly useful when class_0_non_agri >> class_1_agri.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): weight for the positive class (default 0.25)
        gamma (float): focusing parameter (default 2.0)
                       gamma=0 reduces to standard BCE
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits  (Tensor): raw model outputs [B, 1]
            targets (Tensor): binary labels     [B, 1]  float
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with label smoothing.

    Smoothes hard 0/1 targets to (eps/2, 1-eps/2) to prevent
    overconfident predictions.

    Args:
        smoothing (float): smoothing factor, typically 0.05–0.15
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def build_criterion(cfg):
    """
    Select loss function from config.

    Config key: "loss"  →  "bce" | "focal" | "label_smoothing"
    Default: "bce"

    Args:
        cfg (dict): any model config dict

    Returns:
        nn.Module
    """
    loss_type = cfg.get("loss", "bce")

    if loss_type == "focal":
        return FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
        )
    elif loss_type == "label_smoothing":
        return LabelSmoothingBCE(smoothing=cfg.get("label_smoothing", 0.1))
    else:
        return nn.BCEWithLogitsLoss()