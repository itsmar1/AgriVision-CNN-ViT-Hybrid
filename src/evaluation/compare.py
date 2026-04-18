import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt


def benchmark_inference(model, input_size=(1, 3, 224, 224), device="cpu", n_runs=50):
    """
    Measure average inference time per image in milliseconds.

    Args:
        model      (nn.Module): trained model
        input_size (tuple):     input tensor shape
        device     (str):       "cuda" or "cpu"
        n_runs     (int):       number of forward passes to average

    Returns:
        float: mean inference time in ms
    """
    model.eval().to(device)
    dummy = torch.randn(*input_size).to(device)

    # warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_comparison_table(results):
    """
    Build a comparison DataFrame from a list of model result dicts.

    Args:
        results (list[dict]): each dict must have keys:
            name, accuracy, precision, recall, f1, roc_auc,
            params_total, inference_ms

    Returns:
        pd.DataFrame
    """
    rows = []
    for r in results:
        rows.append({
            "Model":           r["name"],
            "Accuracy":        f"{r['accuracy']:.4f}",
            "Precision":       f"{r['precision']:.4f}",
            "Recall":          f"{r['recall']:.4f}",
            "F1 Score":        f"{r['f1']:.4f}",
            "ROC-AUC":         f"{r['roc_auc']:.4f}",
            "Params (M)":      f"{r['params_total'] / 1e6:.1f}",
            "Inference (ms)":  f"{r['inference_ms']:.1f}",
        })
    return pd.DataFrame(rows).set_index("Model")


def save_comparison_table(df, figures_dir):
    """
    Save the comparison DataFrame as a styled PNG table.

    Args:
        df          (pd.DataFrame): output of build_comparison_table
        figures_dir (str):          output directory
    """
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.7 + 1.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # style header row
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternate row shading
    for i in range(1, len(df) + 1):
        color = "#EAF2FB" if i % 2 == 0 else "white"
        for j in range(-1, len(df.columns)):
            tbl[i, j].set_facecolor(color)

    plt.title("Model Comparison — Geo Classify", fontsize=13,
              fontweight="bold", pad=10)
    plt.tight_layout()
    out = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Compare] Table saved → {out}")

    # also save as CSV for reference
    csv_out = os.path.join(figures_dir, "model_comparison.csv")
    df.to_csv(csv_out)
    print(f"[Compare] CSV saved   → {csv_out}")