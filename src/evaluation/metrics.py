import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run inference over a DataLoader and collect all predictions and labels.

    Args:
        model  (nn.Module):   trained model
        loader (DataLoader):  val or test loader
        device (str):         "cuda" or "cpu"

    Returns:
        tuple:
            probs  (np.ndarray): sigmoid probabilities [N]
            preds  (np.ndarray): binary predictions    [N]
            labels (np.ndarray): ground-truth labels   [N]
    """
    model.eval()
    model.to(device)
    all_probs, all_labels = [], []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)          # [B]
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(targets.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return probs, preds, labels


def compute_metrics(probs, preds, labels):
    """
    Compute a full suite of binary classification metrics.

    Args:
        probs  (np.ndarray): predicted probabilities [N]
        preds  (np.ndarray): binary predictions      [N]
        labels (np.ndarray): ground-truth labels     [N]

    Returns:
        dict with keys:
            accuracy, precision, recall, f1, roc_auc,
            confusion_matrix, classification_report
    """
    return {
        "accuracy":               accuracy_score(labels, preds),
        "precision":              precision_score(labels, preds, zero_division=0),
        "recall":                 recall_score(labels, preds,    zero_division=0),
        "f1":                     f1_score(labels, preds,        zero_division=0),
        "roc_auc":                roc_auc_score(labels, probs),
        "confusion_matrix":       confusion_matrix(labels, preds),
        "classification_report":  classification_report(
                                      labels, preds,
                                      target_names=["non_agri", "agri"]),
    }


def print_metrics(metrics, run_name=""):
    """Pretty-print a metrics dict."""
    header = f"── Metrics: {run_name} " if run_name else "── Metrics "
    print(f"\n{header}{'─' * (50 - len(header))}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['classification_report']}")