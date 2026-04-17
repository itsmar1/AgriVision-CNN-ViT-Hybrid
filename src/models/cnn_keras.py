import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_keras(cfg):
    """
    Build a Keras CNN for binary satellite image classification.

    Architecture mirrors CNNTorch:
        5 × (Conv2D → BatchNorm → ReLU → MaxPool)
        GlobalAveragePooling2D
        Dropout → Dense(1, sigmoid)

    Args:
        cfg (dict): config from cnn_config.py

    Returns:
        keras.Model (compiled)
    """
    H, W = cfg["input_size"]
    dropout = cfg.get("dropout", 0.3)
    lr = cfg.get("learning_rate", 1e-3)

    inputs = keras.Input(shape=(H, W, 3), name="input")

    x = inputs
    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2, 2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="CNN_Keras")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )

    return model


if __name__ == "__main__":
    from config.cnn_config import CONFIG
    model = build_cnn_keras(CONFIG)
    model.summary()