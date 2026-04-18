import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ── Consistent plot style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

CLASS_NAMES = ["non_agri", "agri"]


def plot_confusion_matrix(cm, figures_dir, run_name):
    """
    Save a seaborn confusion matrix heatmap.

    Args:
        cm          (np.ndarray):  2×2 confusion matrix
        figures_dir (str):         output directory
        run_name    (str):         used in filename and title
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{run_name}_confusion.png")
    plt.savefig(out)
    plt.close()
    print(f"[Viz] Confusion matrix saved → {out}")


def plot_roc_curve(probs, labels, figures_dir, run_name):
    """
    Save a ROC curve with AUC annotation.

    Args:
        probs       (np.ndarray): predicted probabilities [N]
        labels      (np.ndarray): ground-truth labels     [N]
        figures_dir (str):        output directory
        run_name    (str):        used in filename and title
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {run_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{run_name}_roc.png")
    plt.savefig(out)
    plt.close()
    print(f"[Viz] ROC curve saved → {out}")
    return fpr, tpr, roc_auc


def plot_training_curves(history, figures_dir, run_name):
    """
    Save a two-panel loss / accuracy curve figure.

    Args:
        history     (dict):  keys train_loss, val_loss, train_acc, val_acc
        figures_dir (str):   output directory
        run_name    (str):   used in filename and title
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    ax1.plot(epochs, history["val_loss"],   label="val",   marker="o", ms=3)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train", marker="o", ms=3)
    ax2.plot(epochs, history["val_acc"],   label="val",   marker="o", ms=3)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle(run_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{run_name}_curves.png")
    plt.savefig(out)
    plt.close()
    print(f"[Viz] Training curves saved → {out}")


def plot_roc_overlay(roc_data, figures_dir):
    """
    Overlay ROC curves for all models on one figure for final comparison.

    Args:
        roc_data    (list[dict]):  each dict has keys:
                                   name, fpr, tpr, auc
        figures_dir (str):         output directory
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for entry in roc_data:
        ax.plot(entry["fpr"], entry["tpr"], lw=2,
                label=f"{entry['name']} (AUC={entry['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = os.path.join(figures_dir, "roc_overlay.png")
    plt.savefig(out)
    plt.close()
    print(f"[Viz] ROC overlay saved → {out}")


def plot_sample_predictions(images, labels, probs, figures_dir, run_name, n=8):
    """
    Save a grid of sample predictions with true/predicted labels.

    Args:
        images      (Tensor): batch of images [N, 3, H, W]  (unnormalized)
        labels      (list):   true labels
        probs       (list):   predicted probabilities
        figures_dir (str):    output directory
        run_name    (str):    used in filename
        n           (int):    number of samples to display
    """
    import torchvision.transforms.functional as TF

    n = min(n, len(images))
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    axes = axes.flatten()

    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        pred = int(probs[i] >= 0.5)
        color = "green" if pred == labels[i] else "red"
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {CLASS_NAMES[labels[i]]}\n"
            f"Pred: {CLASS_NAMES[pred]} ({probs[i]:.2f})",
            fontsize=8, color=color,
        )
        axes[i].axis("off")

    plt.suptitle(f"Sample Predictions — {run_name}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{run_name}_samples.png")
    plt.savefig(out)
    plt.close()
    print(f"[Viz] Sample predictions saved → {out}")