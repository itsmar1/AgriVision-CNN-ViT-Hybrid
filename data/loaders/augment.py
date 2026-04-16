import torchvision.transforms as T


# ImageNet mean and std — standard for models pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_torch_transforms(cfg, mode="train"):
    """
    Build torchvision transform pipelines.

    Train pipeline:  augmentation → resize → tensor → normalise
    Eval pipeline:   resize → tensor → normalise  (no augmentation)

    Args:
        cfg  (dict):  config dict (from cnn_config / vit_config / hybrid_config)
        mode (str):   "train" or "eval"

    Returns:
        torchvision.transforms.Compose
    """
    h, w = cfg["input_size"]

    if mode == "train":
        transforms = [
            T.Resize((h, w)),
            T.RandomHorizontalFlip(p=0.5 if cfg.get("aug_hflip") else 0.0),
            T.RandomVerticalFlip(p=0.5  if cfg.get("aug_vflip")  else 0.0),
            T.RandomRotation(degrees=cfg.get("aug_rotation", 0)),
            T.ColorJitter(
                brightness=cfg.get("aug_color_jitter", 0),
                contrast=cfg.get("aug_color_jitter", 0),
                saturation=cfg.get("aug_color_jitter", 0),
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    else:
        transforms = [
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    return T.Compose(transforms)


def get_keras_transforms(cfg):
    """
    Return Keras/TensorFlow ImageDataGenerator kwargs for train and eval.

    Used in Part 2 (Keras CNN baseline) only.

    Args:
        cfg (dict): config dict from cnn_config

    Returns:
        tuple: (train_datagen_kwargs, eval_datagen_kwargs)
    """
    train_kwargs = {
        "rescale": 1.0 / 255.0,
        "horizontal_flip": cfg.get("aug_hflip", False),
        "vertical_flip":   cfg.get("aug_vflip",  False),
        "rotation_range":  cfg.get("aug_rotation", 0),
        "brightness_range": [
            1.0 - cfg.get("aug_color_jitter", 0),
            1.0 + cfg.get("aug_color_jitter", 0),
        ] if cfg.get("aug_color_jitter") else None,
        "fill_mode": "nearest",
        "validation_split": 0.15,
    }

    eval_kwargs = {
        "rescale": 1.0 / 255.0,
    }

    return train_kwargs, eval_kwargs