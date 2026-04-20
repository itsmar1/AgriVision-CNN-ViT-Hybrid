# 🛰️ AgriVision — Hybrid CNN-ViT for Agricultural Land Detection

Binary classification of satellite imagery using CNNs, Vision Transformers,
and a hybrid CNN-ViT architecture. Built on EuroSAT (Sentinel-2) imagery.

---

## Results

| Model | Accuracy | F1 | ROC-AUC | Params |
|---|---|---|---|---|
| CNN (PyTorch) | 0.9618 | 0.9617 | 0.9947 | 1.6M |
| CNN (Keras) | 0.9651 | 0.9649 | 0.9950 | 1.6M |
| ViT fine-tuned | 0.9921 | 0.9922 | 0.9998 | 85.8M |
| **Hybrid CNN-ViT** | 0.9855 | 0.9855 | 0.9992 | 27.4M |


---

## Project Structure

```
geo_classify/
├── data/               # loaders, augmentation, splits, raw images
├── src/
│   ├── models/         # CNN (PyTorch + Keras), ViT, Hybrid
│   ├── training/       # train loop, scheduler, callbacks, losses
│   ├── evaluation/     # metrics, plots, Grad-CAM, comparison
│   └── utils/          # CSV logger, checkpointing
├── notebooks/          # Final Report: Models comparison
├── config/             # cnn_config.py, vit_config.py, hybrid_config.py
├── outputs/            # checkpoints, logs, figures  [git-ignored]
├── train.py            # CLI training entry point
├── evaluate.py         # CLI evaluation + comparison
└── app.py              # Gradio deployment UI
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
Download EuroSAT from Kaggle:
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Organise into:
```
data/images_dataSAT/
├── class_0_non_agri/   ← Forest, River, SeaLake, Industrial, Residential, ...
└── class_1_agri/       ← AnnualCrop, PermanentCrop
```

### 3. Debug run (local CPU — smoke test only)
```bash
python train.py --model cnn_torch --debug
```

### 4. Full training (Kaggle GPU)
```bash
python train.py --model cnn_torch
python train.py --model cnn_keras
python train.py --model vit
python train.py --model hybrid
```

### 5. Evaluate all models + generate comparison
```bash
python evaluate.py --model all
```

### 6. Launch Gradio demo
```bash
python app.py
```

---

## Dataset
EuroSAT (Sentinel-2 RGB) via Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Binary class structure:
- class_1_agri: AnnualCrop (3,000) + PermanentCrop (2,500) = 5,500 images
- class_0_non_agri: Forest + River + SeaLake + Industrial + Residential + HerbaceousVegetation = 5,500 images

---

## Architecture — Hybrid CNN-ViT

```
Input [B, 3, 224, 224]
    │
    ▼
CNN Encoder (ResNet-18, pretrained)
    │  [B, 512, 7, 7]  — local features: edges, textures, patterns
    ▼
Flatten + Linear Projection
    │  [B, 49, 768]    — 49 spatial tokens
    ▼
Prepend [CLS] + Positional Embeddings
    │  [B, 50, 768]
    ▼
Transformer Encoder (4 × Multi-Head Self-Attention blocks)
    │                  — global spatial reasoning
    ▼
[CLS] token → LayerNorm → Linear(1)
    │
    ▼
Sigmoid → Agricultural (1) / Non-Agricultural (0)
```


---

## Stack

- **PyTorch** 2.0+ · **timm** — model training and ViT backbone
- **Keras / TensorFlow** — CNN baseline comparison
- **scikit-learn** — metrics and data splitting
- **Gradio** — deployment UI
- **matplotlib / seaborn** — visualisation

---

## Contributing

Contributions are welcome! Feel free to:

*   Open issues for bugs or suggestions
*   Submit pull requests with improvements

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.