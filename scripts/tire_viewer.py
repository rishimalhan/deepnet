#!/usr/bin/env python3
# tire_viewer.py
# Minimal web UI to sample a random tire image and show the model's prediction.
#
# Usage:
#   python tire_viewer.py --port 8111 \
#     --data-root ../data/tire_dataset \
#     --ckpt checkpoints_tires/resnet18_tires_best.pth
#
# Notes:
# - Uses MPS on Apple Silicon if available, else CPU. No CUDA assumptions.
# - Expects the same folder layout you trained with:
#     tire_dataset/
#       flat.class/      *.jpg
#       full.class/      *.jpg
#       no_tire.class/   *.jpg

import os
import io
import argparse
import random
import base64
from pathlib import Path

from PIL import Image
from flask import Flask, render_template_string, redirect, url_for

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


# -----------------------------
# Config & Device
# -----------------------------
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -----------------------------
# Model / Data loader helpers
# -----------------------------
def build_model(num_classes: int, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def try_load_weights(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both raw state_dict and full checkpoint dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and all(
        k.startswith(("layer", "conv1", "bn1", "fc")) for k in ckpt.keys()
    ):
        model.load_state_dict(ckpt)
    else:
        # Fallback: if someone saved with different key name
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state)
    model.eval()


def build_val_transform(img_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def list_all_images(dataset_root: Path):
    # Use ImageFolder to recover class mapping & enumerate all samples
    ds = datasets.ImageFolder(root=str(dataset_root))
    # samples is list of (path, class_index)
    return ds, ds.samples, ds.class_to_idx


def load_image_bytes_base64(img_path: Path, max_edge=600):
    # Load and downscale a display copy (no normalization), return base64 for inline HTML.
    with Image.open(img_path).convert("RGB") as im:
        w, h = im.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def predict_one(model, device, img_path: Path, tfm, idx_to_class):
    with Image.open(img_path).convert("RGB") as im:
        x = tfm(im).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.tensor(probs).argmax().item())
        pred_label = idx_to_class[pred_idx]
    # Bundle per-class probs with labels in original order
    class_probs = [(idx_to_class[i], float(p)) for i, p in enumerate(probs)]
    return pred_label, class_probs


# -----------------------------
# Flask app
# -----------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Tire Classifier Viewer</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }
    .container { max-width: 900px; margin: 0 auto; }
    .header { display:flex; align-items:center; justify-content:space-between; }
    h1 { font-size: 24px; margin: 0 0 8px; }
    .meta { color: #555; font-size: 14px; }
    .btn { display:inline-block; padding:10px 16px; border:1px solid #333; border-radius:10px; text-decoration:none; color:#111; }
    .btn:hover { background:#f3f3f3; }
    .card { margin-top: 24px; padding: 16px; border:1px solid #ddd; border-radius:12px; }
    .row { display:flex; gap: 20px; align-items:flex-start; }
    img { max-width:100%; height:auto; border-radius: 10px; border:1px solid #eee; }
    table { border-collapse: collapse; min-width: 260px; }
    th, td { padding: 8px 10px; border-bottom:1px solid #eee; text-align:left; }
    .pred { font-size: 18px; font-weight: 600; margin: 8px 0 12px; }
    .small { font-size:12px; color:#666; }
    .footer { margin-top:32px; color:#777; font-size:12px; }
    code { background:#f6f6f6; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>🚗 Tire Classifier — Random Sample</h1>
      <div class="meta">
        Device: <b>{{ device_name }}</b> • Classes: {{ class_names|join(", ") }}
      </div>
    </div>
    <div>
      <a class="btn" href="{{ url_for('random_sample') }}">▶︎ {{ 'Start' if not has_sample else 'Next random' }}</a>
    </div>
  </div>

  {% if has_sample %}
  <div class="card">
    <div class="row">
      <div style="flex: 1;">
        <img src="{{ image_b64 }}" alt="random tire image"/>
        <div class="small">{{ img_relpath }}</div>
      </div>
      <div style="flex: 1; min-width:280px;">
        <div class="pred">Prediction: {{ pred_label }}</div>
        <table>
          <thead><tr><th>Class</th><th>Probability</th></tr></thead>
          <tbody>
            {% for name, p in class_probs %}
            <tr><td>{{ name }}</td><td>{{ "%0.2f%%" % (p*100.0) }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  {% else %}
  <div class="card">
    <p>Click <b>Start</b> to fetch a random image from <code>{{ data_root }}</code> and run inference.</p>
  </div>
  {% endif %}

  <div class="footer">
    Checkpoint: <code>{{ ckpt_path }}</code> • Img tfm: Resize({{ img_size }}), CenterCrop({{ crop_size }})
  </div>
</div>
</body>
</html>
"""


def create_app(args):
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Resolve paths
    data_root = Path(args.data_root).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()

    # Build dataset listing + class mapping from folder names
    ds, samples, class_to_idx = list_all_images(data_root)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Build model & load weights
    device = get_device()
    model = build_model(num_classes=len(class_to_idx), device=device)
    try_load_weights(model, ckpt_path, device)

    # Same validation transform as training script
    tfm = build_val_transform(img_size=args.img_size, crop_size=args.crop_size)

    # Keep state in closures
    state = {
        "has_sample": False,
        "last_img_path": None,
        "last_pred_label": None,
        "last_class_probs": None,
        "last_image_b64": None,
    }

    @app.route("/")
    def home():
        return render_template_string(
            HTML,
            device_name=str(device),
            class_names=class_names,
            has_sample=state["has_sample"],
            image_b64=state["last_image_b64"],
            pred_label=state["last_pred_label"],
            class_probs=state["last_class_probs"],
            img_relpath=(
                str(Path(state["last_img_path"]).relative_to(data_root))
                if state["last_img_path"]
                else ""
            ),
            data_root=str(data_root),
            ckpt_path=str(ckpt_path),
            img_size=args.img_size,
            crop_size=args.crop_size,
        )

    @app.route("/random")
    def random_sample():
        # choose a random (path, class_idx)
        if not samples:
            # no images found
            state.update(
                {
                    "has_sample": False,
                    "last_img_path": None,
                    "last_pred_label": None,
                    "last_class_probs": None,
                    "last_image_b64": None,
                }
            )
            return redirect(url_for("home"))

        path, _ = random.choice(samples)
        img_path = Path(path)

        # Predict
        pred_label, class_probs = predict_one(
            model, device, img_path, tfm, idx_to_class
        )

        # Build display image as base64 (downscaled)
        image_b64 = load_image_bytes_base64(img_path)

        # Update state
        state.update(
            {
                "has_sample": True,
                "last_img_path": str(img_path),
                "last_pred_label": pred_label,
                "last_class_probs": class_probs,
                "last_image_b64": image_b64,
            }
        )
        return redirect(url_for("home"))

    # helpful alias
    app.add_url_rule("/start", view_func=random_sample)

    return app


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Web viewer for tire classifier")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8111)
    p.add_argument("--data-root", type=str, default="../data/tire_dataset")
    p.add_argument(
        "--ckpt",
        type=str,
        default="../model/checkpoints_tires/resnet18_tires_best.pth",
        help="Path to best checkpoint. Supports state_dict or full dict with 'model_state'.",
    )
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--crop_size", type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    # threaded=True is fine for this small viewer; model inference is guarded by GIL + torch ops.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
