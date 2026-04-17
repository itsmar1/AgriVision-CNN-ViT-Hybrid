import os
import torch


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, run_name):
    """
    Save model and optimizer state to disk.

    Saves two files:
      - {run_name}_best.pt  : always overwritten with the current best
      - {run_name}_e{epoch}.pt : epoch-specific snapshot (optional archive)

    Args:
        model          (nn.Module):          model to save
        optimizer      (torch.optim):        optimizer state
        epoch          (int):                current epoch number
        val_loss       (float):              validation loss at this epoch
        checkpoint_dir (str):                directory to save into
        run_name       (str):                prefix for filenames
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    torch.save(state, best_path)
    print(f"[Checkpoint] Saved best → {best_path}  (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(model, checkpoint_path, optimizer=None, device=None):
    """
    Load model weights (and optionally optimizer state) from a checkpoint.

    Args:
        model           (nn.Module):         model to load weights into
        checkpoint_path (str):               path to .pt checkpoint file
        optimizer       (torch.optim|None):  if provided, also restore optimizer
        device          (str|None):          target device, auto-detects if None

    Returns:
        tuple: (model, optimizer_or_None, epoch, val_loss)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

    epoch    = state.get("epoch", 0)
    val_loss = state.get("val_loss", float("inf"))
    print(f"[Checkpoint] Loaded ← {checkpoint_path}  (epoch {epoch}, val_loss {val_loss:.4f})")

    return model, optimizer, epoch, val_loss