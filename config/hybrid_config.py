HYBRID_CONFIG = {
    "model": "hybrid_cnn_vit",
    "framework": "pytorch",

    # data
    "data_dir": "data/images_dataSAT",
    "input_size": (224, 224),
    "num_classes": 1,
    "num_workers": 2,

    # CNN encoder
    "cnn_backbone": "resnet18",               # feature extractor
    "cnn_pretrained": True,
    "cnn_out_channels": 512,                  # feature map channels

    # ViT encoder
    "vit_hidden_dim": 768,                    # projection dim for tokens
    "vit_num_heads": 8,
    "vit_num_layers": 4,                      # transformer encoder blocks
    "vit_mlp_dim": 1024,
    "vit_dropout": 0.1,

    # training
    "epochs": 25,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 7,

    # scheduler
    "scheduler": "cosine",
    "warmup_epochs": 3,

    # augmentation
    "aug_hflip": True,
    "aug_vflip": True,
    "aug_rotation": 15,
    "aug_color_jitter": 0.2,

    # output
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs",
    "figures_dir": "outputs/figures",
    "run_name": "hybrid_cnn_vit",

    "debug": False,
    "debug_samples": 100,
    "debug_epochs": 2,
}