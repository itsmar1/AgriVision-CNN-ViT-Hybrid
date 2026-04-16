CONFIG = {
    "model": "cnn",
    "framework": "pytorch",       # "pytorch" or "keras"

    # data
    "data_dir": "data/images_dataSAT",
    "input_size": (224, 224),
    "num_classes": 1,             # binary — sigmoid output
    "num_workers": 2,

    # training
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "early_stopping_patience": 7,

    # scheduler — cosine annealing
    "scheduler": "cosine",
    "warmup_epochs": 3,

    # augmentation
    "aug_hflip": True,
    "aug_vflip": True,
    "aug_rotation": 15,           # degrees
    "aug_color_jitter": 0.2,

    # output
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs",
    "figures_dir": "outputs/figures",
    "run_name": "cnn_baseline",

    # debug mode — small subset for local testing
    "debug": False,
    "debug_samples": 100,
    "debug_epochs": 2,
}