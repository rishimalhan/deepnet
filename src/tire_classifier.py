#!/usr/bin/env python3
"""
Simple 3-class tire classifier (Full / Flat / No-tire)
Folder layout:
tire_dataset/
  flat.class/      *.jpg
  full.class/      *.jpg
  no_tire.class/   *.jpg
"""

import os
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, val_ratio=0.2, seed=42):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    v = int(n * val_ratio)
    return idx[v:], idx[:v]  # train_idx, val_idx


def top1_acc(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--crop_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--eval_only", action="store_true", help="Skip training, just evaluate"
    )
    ap.add_argument(
        "--predict", type=str, default="", help="Run single-image prediction path"
    )
    args = ap.parse_args()

    set_seed(args.seed)

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"Device: {device}")

    # Transforms (standard ImageNet normalization)
    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(args.crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset (ImageFolder maps subfolder name -> class id)
    root = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "data",
        "tire_dataset",
    )
    ds_full_train = datasets.ImageFolder(root=root, transform=train_tfms)
    ds_full_val = datasets.ImageFolder(
        root=root, transform=val_tfms
    )  # same root; we’ll split by indices

    class_to_idx = ds_full_train.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Classes:")
    for k, v in class_to_idx.items():
        print(f"  {v}: {k}")

    # Split indices once, wrap with Subset for each transform variant
    train_idx, val_idx = split_indices(len(ds_full_train), args.val_ratio, args.seed)
    ds_train = Subset(ds_full_train, train_idx)
    ds_val = Subset(ds_full_val, val_idx)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    # Model: ResNet-18 (simple + fast)
    num_classes = len(class_to_idx)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Eval-only (e.g., after restoring a checkpoint externally)
    if args.eval_only:
        val_loss, val_acc = evaluate(model, dl_val, criterion, device)
        print(f"[Eval-only] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if args.predict:
            predict_image(
                model, args.predict, device, idx_to_class, args.img_size, args.crop_size
            )
        return

    # Train
    out_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "model",
        "checkpoints_tires",
    )
    os.makedirs(out_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, dl_val, criterion, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Save checkpoint and best
        ckpt_path = out_dir / f"resnet18_tires_e{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
            },
            ckpt_path,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "resnet18_tires_best.pth")

    print(f"Best val_acc: {best_acc:.4f}")

    # Optional single-image prediction
    if args.predict:
        predict_image(
            model, args.predict, device, idx_to_class, args.img_size, args.crop_size
        )


# -----------------------------
# Train / Eval / Predict
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += imgs.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += imgs.size(0)
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def predict_image(model, img_path, device, idx_to_class, img_size=256, crop_size=224):
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from PIL import Image

    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.tensor(probs).argmax().item())

    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"\nPrediction for {Path(img_path).name}: {classes[pred_idx]}")
    print("Class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {classes[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
