"""
train.py — CLI entry point for training any model in the geo_classify pipeline.

Usage:
    python train.py --model cnn_torch
    python train.py --model cnn_keras
    python train.py --model vit
    python train.py --model hybrid
    python train.py --model cnn_torch --debug     # fast local smoke-test
"""

import argparse
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
from src.training.train import train


def get_config(model_name):
    return {
        "cnn_torch": CNN_CONFIG,
        "cnn_keras": CNN_CONFIG,
        "vit":       VIT_CONFIG,
        "hybrid":    HYBRID_CONFIG,
    }[model_name]


def run_pytorch(model_name, cfg):
    """Train a PyTorch model (CNN, ViT, or Hybrid)."""

    # ── data ────────────────────────────────────────────────────────────────
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = split_dataset(
        cfg["data_dir"],
        debug=cfg.get("debug", False),
        debug_samples=cfg.get("debug_samples", 100),
    )

    train_tf = get_torch_transforms(cfg, mode="train")
    eval_tf  = get_torch_transforms(cfg, mode="eval")

    train_loader, val_loader, _ = get_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # ── model ────────────────────────────────────────────────────────────────
    builders = {
        "cnn_torch": build_cnn_torch,
        "vit":       build_vit,
        "hybrid":    build_hybrid,
    }
    model = builders[model_name](cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train(model, train_loader, val_loader, cfg, device=device)
    return history


def run_keras(cfg):
    """Train the Keras CNN."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from src.models.cnn_keras import build_cnn_keras
    from src.training.callbacks import build_keras_callbacks
    from data.loaders.augment import get_keras_transforms

    H, W = cfg["input_size"]
    train_kwargs, eval_kwargs = get_keras_transforms(cfg)

    train_gen_factory = ImageDataGenerator(**{
        k: v for k, v in train_kwargs.items() if v is not None
    })
    eval_gen_factory = ImageDataGenerator(**eval_kwargs)

    train_gen = train_gen_factory.flow_from_directory(
        cfg["data_dir"], target_size=(H, W),
        batch_size=cfg["batch_size"],
        class_mode="binary", subset="training", seed=42,
    )
    val_gen = train_gen_factory.flow_from_directory(
        cfg["data_dir"], target_size=(H, W),
        batch_size=cfg["batch_size"],
        class_mode="binary", subset="validation", seed=42,
    )

    model = build_cnn_keras(cfg)
    callbacks = build_keras_callbacks(cfg)

    epochs = cfg["debug_epochs"] if cfg.get("debug") else cfg["epochs"]
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks,
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train geo_classify models")
    parser.add_argument(
        "--model",
        choices=["cnn_torch", "cnn_keras", "vit", "hybrid"],
        required=True,
        help="Which model to train",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode (small subset, 2 epochs)",
    )
    args = parser.parse_args()

    cfg = get_config(args.model)
    if args.debug:
        cfg = {**cfg, "debug": True}

    print(f"\n{'='*55}")
    print(f"  Training: {args.model.upper()}  |  debug={cfg.get('debug', False)}")
    print(f"{'='*55}\n")

    if args.model == "cnn_keras":
        run_keras(cfg)
    else:
        run_pytorch(args.model, cfg)


if __name__ == "__main__":
    main()