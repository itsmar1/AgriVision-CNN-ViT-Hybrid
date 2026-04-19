VIT_CONFIG = {
    "model": "vit",
    "framework": "pytorch",

    # data
    "data_dir": "data/images_dataSAT",
    "input_size": (224, 224),
    "num_classes": 1,
    "num_workers": 2,

    # pretrained backbone
    "vit_backbone": "vit_base_patch16_224",   # timm model name
    "pretrained": True,
    "freeze_backbone_epochs": 5,              # freeze then unfreeze

    # training
    "epochs": 20,
    "batch_size": 16,                         # smaller — ViT is memory hungry
    "learning_rate": 1e-3,
    "learning_rate_head": 1e-3,               # LR for classifier head
    "learning_rate_backbone": 1e-5,           # LR for backbone after unfreeze
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,

    # scheduler
    "scheduler": "cosine",
    "warmup_epochs": 2,

    # augmentation
    "aug_hflip": True,
    "aug_vflip": True,
    "aug_rotation": 15,
    "aug_color_jitter": 0.2,

    # output
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs",
    "figures_dir": "outputs/figures",
    "run_name": "vit_finetune",

    "debug": False,
    "debug_samples": 100,
    "debug_epochs": 2,
}