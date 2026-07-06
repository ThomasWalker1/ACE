"""
train_split_proxy.py — Finetune ResNet-50 with DCT proxy regularization on intermediate features.

Adds `λ * bpp_proxy(normalize(features))` to the cross-entropy loss during finetuning.
Features are tanh-normalized per-channel before applying the proxy, bringing them into
the proxy's operating range [-1.5, 1.5].

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python train/train_split_proxy.py \
        --train_dir data/imagenet/train \
        --val_dir   data/imagenet/val \
        --split_point layer2 --lam 0.1 --epochs 10 --lr 1e-4 \
        --output_dir checkpoints/split_proxy
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ── Feature hook ──────────────────────────────────────────────────────────────


class SplitResNet50(nn.Module):
    """ResNet-50 with a feature extraction hook at a configurable split point.
    Forward returns (logits, features_at_split)."""

    def __init__(self, split_point: str):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
        self.split_point = split_point

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        if self.split_point == "layer1":
            return x
        x = self.layer2(x)
        if self.split_point == "layer2":
            return x
        return self.layer3(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.split_point == "layer1":
            x = self.layer2(x)
        if self.split_point in ("layer1", "layer2"):
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features


def normalize_features(feat: torch.Tensor) -> torch.Tensor:
    """Normalize features to proxy operating range.
    Mean-center per channel (ResNet ReLU features are non-negative; centering is
    needed so AC coefficients in the proxy's DCT are symmetric around 0).
    Then tanh-scale to ≈ (-0.75, 0.75) ⊂ [-1.5, 1.5]."""
    B, C, H, W = feat.shape
    flat = feat.reshape(B, C, -1)
    mean = flat.mean(dim=-1).reshape(B, C, 1, 1)
    std = flat.std(dim=-1).reshape(B, C, 1, 1).clamp(min=1e-6)
    return torch.tanh((feat - mean) / (2.0 * std)) * 0.75


# ── Training ──────────────────────────────────────────────────────────────────


def proxy_on_features(feat: torch.Tensor, n_channels: int = 64, n_bins: int = 32) -> torch.Tensor:
    """Apply proxy to a random subset of channels to keep memory tractable.
    Full (B, 512, 28, 28) × n_bins=64 intermediate tensor is ~16 GB; subsampling to
    n_channels=64 with n_bins=32 gives ~1 GB before backward."""
    if feat.shape[1] > n_channels:
        idx = torch.randperm(feat.shape[1], device=feat.device)[:n_channels]
        feat = feat[:, idx]
    return bpp_proxy(normalize_features(feat), n_bins=n_bins).mean()


def set_bn_eval(model):
    """Freeze all BatchNorm layers: keeps running stats and affine params fixed.
    Prevents BN drift from corrupting pretrained accuracy during finetuning."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)


def train_one_epoch(model, loader, optimizer, lam, device, epoch, freeze_bn: bool):
    model.train()
    if freeze_bn:
        set_bn_eval(model)
    ce_total = proxy_total = n = 0
    t0 = time.time()

    for x, y in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, feat = model(x)
        ce_loss = F.cross_entropy(logits, y)
        proxy_loss = proxy_on_features(feat)
        loss = ce_loss + lam * proxy_loss

        loss.backward()
        optimizer.step()

        ce_total += ce_loss.item() * y.size(0)
        proxy_total += proxy_loss.item() * y.size(0)
        n += y.size(0)

    elapsed = time.time() - t0
    return {"ce": ce_total / n, "proxy": proxy_total / n, "time": elapsed}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--split_point", default="layer2", choices=["layer1", "layer2", "layer3"])
    parser.add_argument("--lam", type=float, default=0.1, help="Proxy regularization weight")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument(
        "--freeze_bn",
        action="store_true",
        default=True,
        help="Freeze BatchNorm running stats and affine params (prevents BN drift)",
    )
    parser.add_argument("--no_freeze_bn", dest="freeze_bn", action="store_false")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0-1). Use 0.1 for quick runs.",
    )
    parser.add_argument("--output_dir", default="checkpoints/split_proxy")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_name = f"{args.split_point}_lam{args.lam}_v2"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Multi-GPU
    model = SplitResNet50(args.split_point).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )

    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    if args.train_subset < 1.0:
        n = int(len(train_dataset) * args.train_subset)
        idx = torch.randperm(len(train_dataset))[:n].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, idx)
        print(
            f"Using {n} / {len(datasets.ImageFolder(args.train_dir))} training images ({args.train_subset * 100:.0f}%)"
        )
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # SGD + momentum preserves pretrained accuracy better than Adam for ImageNet finetuning
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    log = []

    freeze_str = "BN frozen" if args.freeze_bn else "BN unfrozen"
    print(
        f"Starting finetuning: split={args.split_point}, λ={args.lam}, epochs={args.epochs}, "
        f"lr={args.lr}, SGD+momentum, {freeze_str}"
    )
    print(f"Saving to {out_dir}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, args.lam, device, epoch, freeze_bn=args.freeze_bn
        )
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:2d} | CE={train_stats['ce']:.4f} | "
            f"proxy={train_stats['proxy']:.4f} | val_acc={val_acc:.2f}% | "
            f"t={train_stats['time']:.0f}s"
        )

        entry = {"epoch": epoch, "val_acc": val_acc, **train_stats}
        log.append(entry)

        if val_acc > best_acc:
            best_acc = val_acc
            m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": m.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                out_dir / "best.pth",
            )
            print(f"  -> saved best ({val_acc:.2f}%)")

        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

    # Save final
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": args.epochs,
            "state_dict": m.state_dict(),
            "val_acc": val_acc,
            "args": vars(args),
        },
        out_dir / "final.pth",
    )
    print(f"\nDone. Best val_acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
