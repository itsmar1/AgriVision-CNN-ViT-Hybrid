"""
app.py — Gradio web UI for the AgriVision inference pipeline.

Allows a user to upload a satellite image tile and receive:
  - Predicted class (Agricultural / Non-Agricultural)
  - Confidence score
  - Grad-CAM heatmap overlay

Deploy to Hugging Face Spaces:
    1. Push this file + requirements.txt + outputs/checkpoints/ to a Space repo
    2. Set Space SDK to "Gradio"

Local run:
    python app.py
"""

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import torchvision.transforms as T

from config.hybrid_config import HYBRID_CONFIG as CFG
from src.models.hybrid_cnn_vit import build_hybrid
from src.utils.checkpoint import load_checkpoint
from src.evaluation.gradcam import GradCAM, overlay_heatmap

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "🌿 Non-Agricultural Land", 1: "🌾 Agricultural Land"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH    = os.path.join(CFG["checkpoint_dir"], f"{CFG['run_name']}_best.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Load model ───────────────────────────────────────────────────────────────
print(f"[App] Loading model from {CKPT_PATH} on {DEVICE}...")
model = build_hybrid(CFG)
model, _, _, _ = load_checkpoint(model, CKPT_PATH, device=DEVICE)
model.eval().to(DEVICE)
print("[App] Model ready.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(pil_image):
    """
    Run inference on a PIL image and return prediction + Grad-CAM overlay.

    Args:
        pil_image (PIL.Image): uploaded satellite tile

    Returns:
        tuple:
            label_str  (str):         predicted class label
            confidence (float):       confidence score in [0, 1]
            overlay    (np.ndarray):  Grad-CAM overlay [H, W, 3] uint8
    """
    if pil_image is None:
        return "No image provided.", 0.0, None

    # preprocess
    tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # inference
    with torch.enable_grad():
        logit = model(tensor)
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= 0.5)

    label_str  = CLASS_NAMES[pred]
    confidence = prob if pred == 1 else 1 - prob

    # Grad-CAM on the last CNN layer
    target_layer = model.cnn_encoder[-1][-1].conv2   # last ResNet BasicBlock conv
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(tensor)
    gradcam.remove_hooks()

    overlay = overlay_heatmap(tensor.squeeze(0).detach(), heatmap)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return label_str, round(confidence, 4), overlay_uint8


# ── Gradio interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="AgriVision — Agricultural Land Detection") as demo:

    gr.Markdown(
        """
        # 🛰️ AgriVision — Agricultural Land Detection
        Upload a satellite image tile. The model will classify it as
        **Agricultural** or **Non-Agricultural** land and highlight the
        discriminative regions using **Grad-CAM**.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Satellite Tile")
            run_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_out      = gr.Textbox(label="Predicted Class")
            confidence_out = gr.Number(label="Confidence Score")
            gradcam_out    = gr.Image(label="Grad-CAM Heatmap")

    gr.Examples(
        examples=[],   # add sample image paths here after training
        inputs=image_input,
    )

    run_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_out, confidence_out, gradcam_out],
    )

if __name__ == "__main__":
    demo.launch(share=False)