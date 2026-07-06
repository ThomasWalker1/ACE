"""
train_augmix_cifar.py — Train ResNet-18 on CIFAR-10 with AugMix and/or DCT proxy.

Conditions:
  baseline    : RandomCrop + RandomHorizontalFlip
  proxy_aug   : baseline + DCT proxy gradient steps (prob=0.5)
  augmix      : AugMix with JSD consistency loss
  augmix_proxy: AugMix + DCT proxy gradient steps applied to the original view

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/train_augmix_cifar.py \
        --condition augmix --seed 0 \
        --output_dir checkpoints/aceaug
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ace import bpp_proxy
from augmix import augmentations as _augmix_ops


# ── Normalisation ─────────────────────────────────────────────────────────────

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)[None, :, None, None]) / STD.to(x.device)[None, :, None, None]


# ── AugMix (unnormalized output) ──────────────────────────────────────────────


def _apply_op(img_hwc: np.ndarray, op, severity: int) -> np.ndarray:
    """Apply one official AugMix op. img_hwc: (H,W,C) float32 [0,1]."""
    pil = Image.fromarray(np.clip(img_hwc * 255, 0, 255).astype(np.uint8))
    pil = op(pil, severity)
    return np.asarray(pil).astype(np.float32) / 255.0


def augmix(
    img_hwc: np.ndarray, severity: int = 3, width: int = 3, alpha: float = 1.0
) -> np.ndarray:
    """AugMix using official ops; returns unnormalized (H,W,C) float in [0,1]."""
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(img_hwc)
    for w in ws:
        aug = img_hwc.copy()
        for _ in range(np.random.randint(1, 4)):
            op = np.random.choice(_augmix_ops.augmentations)
            aug = _apply_op(aug, op, severity)
        mix += w * aug
    return np.clip((1 - m) * img_hwc + m * mix, 0, 1)


# ── JSD consistency loss ──────────────────────────────────────────────────────


def jsd_loss(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    p_mean = (p1 + p2 + p3) / 3.0
    h_mean = -(p_mean * (p_mean + 1e-8).log()).sum(1).mean()
    h_avg = (
        -(p1 * (p1 + 1e-8).log()).sum(1).mean()
        - (p2 * (p2 + 1e-8).log()).sum(1).mean()
        - (p3 * (p3 + 1e-8).log()).sum(1).mean()
    ) / 3.0
    return h_mean - h_avg


# ── Datasets ──────────────────────────────────────────────────────────────────


def _load_batch(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b"labels"], dtype=np.int64)
    return imgs, labels


def _crop_flip_hwc(img_hwc: np.ndarray) -> np.ndarray:
    """Random crop (pad=4) + random horizontal flip on (H,W,C) float array."""
    pil = Image.fromarray(np.clip(img_hwc * 255, 0, 255).astype(np.uint8))
    pil = ImageOps.expand(pil, border=4, fill=0)
    x = np.random.randint(0, 9)
    y = np.random.randint(0, 9)
    pil = pil.crop((x, y, x + 32, y + 32))
    if np.random.rand() > 0.5:
        pil = ImageOps.mirror(pil)
    return np.asarray(pil).astype(np.float32) / 255.0


class CIFAR10Plain(Dataset):
    """Standard CIFAR-10: returns (img_tensor, label). Crop+flip on train."""

    def __init__(self, data_dir: str, split: str):
        data_dir = Path(data_dir)
        batches = (
            [data_dir / f"data_batch_{i}" for i in range(1, 6)]
            if split == "train"
            else [data_dir / "test_batch"]
        )
        imgs, labels = [], []
        for b in batches:
            im, lb = _load_batch(b)
            imgs.append(im)
            labels.append(lb)
        self.imgs = np.concatenate(imgs)  # (N, 3, 32, 32) float
        self.labels = np.concatenate(labels)
        self.train = split == "train"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].transpose(1, 2, 0)  # (32,32,3) float
        if self.train:
            img = _crop_flip_hwc(img)
        return torch.from_numpy(img.transpose(2, 0, 1)), int(self.labels[idx])


class CIFAR10AugMix(Dataset):
    """CIFAR-10 AugMix: returns (x_orig, x_aug1, x_aug2, label), all unnormalized."""

    def __init__(self, data_dir: str, severity: int = 3, width: int = 3):
        data_dir = Path(data_dir)
        imgs, labels = [], []
        for i in range(1, 6):
            im, lb = _load_batch(data_dir / f"data_batch_{i}")
            imgs.append(im)
            labels.append(lb)
        self.imgs = np.concatenate(imgs)
        self.labels = np.concatenate(labels)
        self.severity = severity
        self.width = width

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = _crop_flip_hwc(self.imgs[idx].transpose(1, 2, 0))  # (H,W,C) float
        aug1 = augmix(img, self.severity, self.width)
        aug2 = augmix(img, self.severity, self.width)

        def to_t(a):
            return torch.from_numpy(a.transpose(2, 0, 1))

        return to_t(img), to_t(aug1), to_t(aug2), int(self.labels[idx])


# ── Proxy augmentation ────────────────────────────────────────────────────────


def proxy_augment(x: torch.Tensor, T: int, lr: float, lam: float) -> torch.Tensor:
    x_aug = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x_aug], lr=lr)
    for _ in range(T):
        opt.zero_grad()
        xc = x_aug.clamp(0.0, 1.0)
        loss = F.mse_loss(xc, x) + lam * bpp_proxy(xc).mean()
        loss.backward()
        opt.step()
    return x_aug.detach().clamp(0.0, 1.0)


# ── Model ─────────────────────────────────────────────────────────────────────


def build_resnet18(num_classes: int = 10) -> nn.Module:
    net = models.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    device = torch.device(args.device)
    run_name = f"seed{args.seed}" if args.seed is not None else "default"
    out = Path(args.output_dir) / args.condition / run_name
    out.mkdir(parents=True, exist_ok=True)

    use_augmix = args.condition in ("augmix", "augmix_proxy")
    use_proxy = args.condition in ("proxy_aug", "augmix_proxy")

    if use_augmix:
        train_ds = CIFAR10AugMix(args.data_dir, args.aug_severity, args.aug_width)
    else:
        train_ds = CIFAR10Plain(args.data_dir, "train")

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_ds = CIFAR10Plain(args.data_dir, "val")
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = build_resnet18().to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    with open(out / "log.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_acc\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            if use_augmix:
                x_orig, x_aug1, x_aug2, y = [b.to(device) for b in batch]

                if use_proxy:
                    mask = torch.rand(x_orig.shape[0]) < args.aug_prob
                    if mask.any():
                        with torch.enable_grad():
                            x_p = proxy_augment(x_orig[mask], args.T, args.proxy_lr, args.lam)
                        x_orig = x_orig.clone()
                        x_orig[mask] = x_p

                logits_orig = model(normalise(x_orig))
                p_orig = F.softmax(logits_orig, dim=1)
                p_aug1 = F.softmax(model(normalise(x_aug1)), dim=1)
                p_aug2 = F.softmax(model(normalise(x_aug2)), dim=1)

                ce = F.cross_entropy(logits_orig, y)
                jsd = jsd_loss(p_orig, p_aug1, p_aug2)
                loss = ce + args.jsd_weight * jsd
                batch_n = x_orig.shape[0]
                total_correct += (logits_orig.detach().argmax(1) == y).sum().item()
            else:
                x, y = [b.to(device) for b in batch]

                if use_proxy:
                    mask = torch.rand(x.shape[0]) < args.aug_prob
                    if mask.any():
                        with torch.enable_grad():
                            x_p = proxy_augment(x[mask], args.T, args.proxy_lr, args.lam)
                        x = x.clone()
                        x[mask] = x_p

                logits = model(normalise(x))
                loss = F.cross_entropy(logits, y)
                batch_n = x.shape[0]
                total_correct += (logits.argmax(1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_n
            total_n += batch_n

        sched.step()
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n * 100

        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xv, yv in val_ld:
                xv, yv = xv.to(device), yv.to(device)
                correct += (model(normalise(xv)).argmax(1) == yv).sum().item()
                n += yv.shape[0]
        val_acc = correct / n * 100

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"train {train_acc:.2f}% | val {val_acc:.2f}%"
        )
        with open(out / "log.csv", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{val_acc:.2f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_acc": val_acc},
                out / "best.pt",
            )

    print(f"\nBest val acc: {best_acc:.2f}%  →  {out / 'best.pt'}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints/aceaug")
    p.add_argument(
        "--condition", default="augmix", choices=["baseline", "proxy_aug", "augmix", "augmix_proxy"]
    )
    # AugMix parameters
    p.add_argument("--jsd_weight", type=float, default=12.0)
    p.add_argument("--aug_severity", type=int, default=3)
    p.add_argument("--aug_width", type=int, default=3)
    # Proxy parameters
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--proxy_lr", type=float, default=0.01)
    p.add_argument("--aug_prob", type=float, default=0.5)
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}")

    train(args)


if __name__ == "__main__":
    main()
