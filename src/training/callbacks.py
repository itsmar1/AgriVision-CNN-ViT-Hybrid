import os
from tensorflow import keras


def build_keras_callbacks(cfg):
    """
    Build a standard set of Keras callbacks for training.

    Includes:
      - ModelCheckpoint : saves best weights by val_loss
      - EarlyStopping   : stops when val_loss stops improving
      - ReduceLROnPlateau: halves LR after 3 stagnant epochs
      - CSVLogger       : writes per-epoch metrics to a CSV file

    Args:
        cfg (dict): config from cnn_config.py

    Returns:
        list[keras.callbacks.Callback]
    """
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    run_name = cfg.get("run_name", "keras_run")
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{run_name}_best.keras")
    log_path  = os.path.join(cfg["log_dir"],         f"{run_name}.csv")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.get("early_stopping_patience", 7),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path, append=False),
    ]

    return callbacks