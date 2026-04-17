import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_scheduler(optimizer, cfg, num_epochs):
    """
    Build a learning rate scheduler.

    Strategy: linear warmup for `warmup_epochs`, then cosine annealing
    down to lr * 0.01 for the remaining epochs.

    Args:
        optimizer  (torch.optim.Optimizer): optimizer to schedule
        cfg        (dict):                  config dict
        num_epochs (int):                   total training epochs

    Returns:
        torch.optim.lr_scheduler  (SequentialLR or CosineAnnealingLR)
    """
    warmup_epochs = cfg.get("warmup_epochs", 0)

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=cfg.get("learning_rate", 1e-3) * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=cfg.get("learning_rate", 1e-3) * 0.01,
        )

    return scheduler