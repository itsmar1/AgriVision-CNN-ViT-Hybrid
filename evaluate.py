"""
evaluate.py — CLI entry point for evaluating trained models.

Usage:
    python evaluate.py --model cnn_torch
    python evaluate.py --model vit
    python evaluate.py --model hybrid
    python evaluate.py --model all     # evaluate + compare all models
"""

import argparse
import os
import torch

from config.cnn_config import CNN_CONFIG
from config.vit_config import VIT_CONFIG
from config.hybrid_config import HYBRID_CONFIG
from src.models.cnn_torch import build_cnn_torch
from src.models.vit import build_vit
from src.models.hybrid_cnn_vit import build_hybrid
from data.loaders.split import split_dataset
from data.loaders.augment import get_torch_transforms
from data.loaders.dataset import get_dataloaders
from src.evaluation.metrics import collect_predictions, compute_metrics, print_metrics
from src.evaluation.viz import (
    plot_confusion_matrix, plot_roc_curve, plot_roc_overlay
)
from src.evaluation.compare import (
    build_comparison_table, save_comparison_table,
    benchmark_inference, count_parameters,
)
from src.utils.checkpoint import load_checkpoint


MODEL_REGISTRY = {
    "cnn_torch": ("cnn_torch",  CNN_CONFIG),
    "vit":       ("vit",        VIT_CONFIG),
    "hybrid":    ("hybrid",     HYBRID_CONFIG),
}


def load_model(model_name, cfg, device):
    builders = {
        "cnn_torch": build_cnn_torch,
        "vit":       build_vit,
        "hybrid":    build_hybrid,
    }
    model = builders[model_name](cfg)
    ckpt_path = os.path.join(
        cfg["checkpoint_dir"], f"{cfg['run_name']}_best.pt"
    )
    model, _, _, _ = load_checkpoint(model, ckpt_path, device=device)
    return model.to(device)


def evaluate_model(model_name, cfg, device):
    """Run full evaluation for one model. Returns metrics dict + roc data."""
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = split_dataset(cfg["data_dir"])

    eval_tf = get_torch_transforms(cfg, mode="eval")
    _, _, test_loader = get_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        train_transform=eval_tf,
        eval_transform=eval_tf,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = load_model(model_name, cfg, device)
    probs, preds, labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(probs, preds, labels)
    print_metrics(metrics, run_name=cfg["run_name"])

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"],
                          cfg["figures_dir"], cfg["run_name"])
    fpr, tpr, roc_auc = plot_roc_curve(probs, labels,
                                        cfg["figures_dir"], cfg["run_name"])

    total_params, _ = count_parameters(model)
    inf_ms = benchmark_inference(model, device=device)

    return {
        **metrics,
        "name":          cfg["run_name"],
        "params_total":  total_params,
        "inference_ms":  inf_ms,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate geo_classify models")
    parser.add_argument(
        "--model",
        choices=["cnn_torch", "vit", "hybrid", "all"],
        required=True,
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    all_results = []
    roc_data    = []

    for name in targets:
        model_key, cfg = MODEL_REGISTRY[name]
        print(f"\n{'='*50}\nEvaluating: {name.upper()}\n{'='*50}")
        result = evaluate_model(model_key, cfg, device)
        all_results.append(result)
        roc_data.append({
            "name": result["name"],
            "fpr":  result["fpr"],
            "tpr":  result["tpr"],
            "auc":  result["auc"],
        })

    if len(all_results) > 1:
        figures_dir = CNN_CONFIG["figures_dir"]
        plot_roc_overlay(roc_data, figures_dir)
        df = build_comparison_table(all_results)
        save_comparison_table(df, figures_dir)
        print("\n" + df.to_string())


if __name__ == "__main__":
    main()