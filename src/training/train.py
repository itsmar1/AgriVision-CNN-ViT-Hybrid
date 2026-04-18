import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.scheduler import build_scheduler
from src.utils.logger import CSVLogger
from src.utils.checkpoint import save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run one evaluation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, cfg, device=None):
    """
    Full training loop with:
      - cosine LR scheduler + optional linear warmup
      - early stopping on validation loss
      - per-epoch CSV logging
      - best model checkpointing
      - live train/val loss and accuracy curves saved to figures/

    Args:
        model        (nn.Module):   model to train
        train_loader (DataLoader):  training set
        val_loader   (DataLoader):  validation set
        cfg          (dict):        config dict
        device       (str|None):    "cuda", "cpu", or auto-detect

    Returns:
        dict: history with keys train_loss, val_loss, train_acc, val_acc
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    model = model.to(device)

    epochs = cfg["debug_epochs"] if cfg.get("debug") else cfg["epochs"]
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, cfg, num_epochs=epochs)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["figures_dir"], exist_ok=True)

    run_name = cfg.get("run_name", "run")
    logger = CSVLogger(os.path.join(cfg["log_dir"], f"{run_name}.csv"))

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 7)

    # ── ViT two-phase fine-tuning ────────────────────────────────────────────
    freeze_epochs = cfg.get("freeze_backbone_epochs", 0)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # phase transition: unfreeze backbone after freeze_epochs
        if freeze_epochs and epoch == freeze_epochs + 1:
            if hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()
                # rebuild optimizer with lower backbone LR
                optimizer = torch.optim.AdamW([
                    {"params": model.vit.parameters(),
                     "lr": cfg.get("learning_rate_backbone", 1e-5)},
                    {"params": model.head.parameters(),
                     "lr": cfg.get("learning_rate_head", 1e-3)},
                ], weight_decay=cfg.get("weight_decay", 1e-4))
                scheduler = build_scheduler(optimizer, cfg,
                                            num_epochs=epochs - epoch + 1)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # log to CSV
        logger.log({"epoch": epoch,
                    "train_loss": train_loss, "val_loss": val_loss,
                    "train_acc": train_acc,   "val_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0]})

        # accumulate history for plotting
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            cfg["checkpoint_dir"], run_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # save training curves
    _plot_history(history, cfg["figures_dir"], run_name)
    return history


def _plot_history(history, figures_dir, run_name):
    """Save a two-panel loss / accuracy curve figure."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle(run_name, fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f"{run_name}_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Train] Curves saved → {out_path}")