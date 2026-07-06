"""
eval_adv_purification.py — Adversarial purification via DCT proxy gradient descent.

Uses a pretrained ResNet-50. For each batch:
  1. Generate adversarial examples (FGSM or PGD)
  2. Apply defenses: none / JPEG / DCT-proxy purification
  3. Measure top-1 accuracy recovery

Theoretical connection: adversarial perturbations concentrate in high-frequency
DCT bands (Yin et al. 2019). Minimising bpp_proxy suppresses AC coefficient
entropy, preferentially removing adversarial noise while preserving
low-frequency semantic content.

Usage:
    CUDA_VISIBLE_DEVICES=5 python eval/eval_adv_purification.py \
        --data_dir data/imagenet/val \
        --n_images 1000 --attack pgd --eps 4
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

# ── ImageNet normalisation ────────────────────────────────────────────────────

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def normalise(x):  # [0,1] → ImageNet-normalised
    return (x - MEAN.to(x.device)[None, :, None, None]) / STD.to(x.device)[None, :, None, None]


def denormalise(x):  # ImageNet-normalised → [0,1]
    return x * STD.to(x.device)[None, :, None, None] + MEAN.to(x.device)[None, :, None, None]


# ── Dataset ───────────────────────────────────────────────────────────────────


class ImageNetSubset(Dataset):
    def __init__(self, root, n_images, transform=None):
        root = Path(root)
        all_classes = sorted(root.iterdir())
        self.samples = []
        for ci, cls_dir in enumerate(all_classes):
            for img_path in sorted(cls_dir.glob("*.JPEG")):
                self.samples.append((img_path, ci))
        # evenly spaced subsample
        step = max(1, len(self.samples) // n_images)
        self.samples = self.samples[::step][:n_images]
        self.transform = transform or T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Attacks ───────────────────────────────────────────────────────────────────


def fgsm(model, x, y, eps):
    """Fast Gradient Sign Method. x in [0,1], eps in [0,1]."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(normalise(x_adv)), y)
    loss.backward()
    return (x + eps * x_adv.grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, steps=10, alpha=None):
    """PGD attack (Madry et al.). x in [0,1], eps in [0,1]."""
    if alpha is None:
        alpha = eps / 4
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(normalise(x_adv)), y)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


# ── Defenses ──────────────────────────────────────────────────────────────────


def jpeg_defense(x, quality=75):
    """JPEG compress/decompress each image in the batch."""
    out = []
    for img in x:
        pil = T.ToPILImage()(img.cpu())
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(T.ToTensor()(Image.open(buf)))
    return torch.stack(out).to(x.device)


def dct_purify(x, n_steps, lr, lam, bpp_bins=64):
    """DCT proxy gradient descent: minimise AC entropy + L2 fidelity."""
    if n_steps == 0:
        return x
    x_pur = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x_pur], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xc = x_pur.clamp(0.0, 1.0)
        rate = bpp_proxy(xc, n_bins=bpp_bins).mean()
        fidelity = F.mse_loss(xc, x)
        loss = fidelity + lam * rate
        loss.backward()
        opt.step()
    return x_pur.detach().clamp(0.0, 1.0)


def gaussian_noise_defense(x, sigma):
    """Add Gaussian noise then clip to [0,1]."""
    return (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)


def gaussian_blur_defense(x, sigma):
    """Gaussian blur each image; kernel size chosen to cover ±3σ."""
    if sigma <= 0:
        return x
    ks = int(6 * sigma + 1) | 1  # odd kernel size covering 3σ each side
    ks = max(ks, 3)
    return TF.gaussian_blur(x, kernel_size=ks, sigma=sigma)


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def accuracy(model, x, y):
    pred = model(normalise(x)).argmax(dim=1)
    return (pred == y).float().mean().item()


def evaluate_with_clean(model, loader, attack_fn, defense_fns, device):
    """
    defense_fns: dict[name -> callable(x_adv) -> x_defended]
      Special keys handled automatically: "clean" uses x, "adversarial" uses x_adv.
    Returns dict: {name: top-1 accuracy (%)}.
    """
    totals = {name: 0.0 for name in ["clean", "adversarial"] + list(defense_fns)}
    n_batches = 0

    for x, y in tqdm(loader, desc="Batches", leave=False):
        x, y = x.to(device), y.to(device)

        with torch.enable_grad():
            x_adv = attack_fn(model, x, y)

        with torch.no_grad():
            totals["clean"] += accuracy(model, x, y)
            totals["adversarial"] += accuracy(model, x_adv, y)

        for name, fn in defense_fns.items():
            needs_grad = name == "dct_purify"
            ctx = torch.enable_grad() if needs_grad else torch.no_grad()
            with ctx:
                x_def = fn(x_adv)
            with torch.no_grad():
                totals[name] += accuracy(model, x_def, y)
        n_batches += 1

    return {k: round(v / n_batches * 100, 2) for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/adv_purification")
    p.add_argument("--n_images", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    p.add_argument("--eps", type=float, default=4.0, help="Perturbation budget in [0,255].")
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--n_pur_steps", type=int, default=10)
    p.add_argument("--pur_lr", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--jpeg_quality", type=int, default=75)
    p.add_argument(
        "--noise_sigma", type=float, default=0.05, help="Gaussian noise σ (in [0,1] pixel units)."
    )
    p.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur σ (pixels).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    eps = args.eps / 255.0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Attack: {args.attack.upper()}, ε={args.eps}/255")
    print(f"DCT purify: T={args.n_pur_steps}, lr={args.pur_lr}, λ={args.lam}")
    print(f"Noise σ={args.noise_sigma:.3f}, Blur σ={args.blur_sigma:.1f}")
    print(f"Images: {args.n_images}, batch: {args.batch_size}")

    model = models.resnet50(weights="IMAGENET1K_V1").to(args.device).eval()

    dataset = ImageNetSubset(args.data_dir, args.n_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if args.attack == "fgsm":

        def attack_fn(m, x, y):
            return fgsm(m, x, y, eps)
    else:

        def attack_fn(m, x, y):
            return pgd(m, x, y, eps, steps=args.pgd_steps, alpha=eps / 4)

    defense_fns = {
        "jpeg": lambda x_adv: jpeg_defense(x_adv, args.jpeg_quality),
        "gaussian_noise": lambda x_adv: gaussian_noise_defense(x_adv, args.noise_sigma),
        "gaussian_blur": lambda x_adv: gaussian_blur_defense(x_adv, args.blur_sigma),
        "dct_purify": lambda x_adv: dct_purify(x_adv, args.n_pur_steps, args.pur_lr, args.lam),
    }

    results = evaluate_with_clean(model, loader, attack_fn, defense_fns, args.device)

    print(f"\n--- Top-1 Accuracy ({args.attack.upper()}, ε={args.eps}/255) ---")
    print(f"{'Defense':<20}  {'Accuracy':>10}")
    print("-" * 35)
    for name, acc in results.items():
        print(f"{name:<20}  {acc:>9.2f}%")

    adv_acc = results["adversarial"]
    clean_acc = results["clean"]
    recovery = {}
    for name, acc in results.items():
        if name in ("clean", "adversarial"):
            continue
        rec = (acc - adv_acc) / max(clean_acc - adv_acc, 1e-6) * 100
        recovery[name] = round(rec, 1)
        print(f"  {name}: {rec:.1f}% of accuracy gap recovered")

    summary = {
        "attack": args.attack,
        "eps": args.eps,
        "n_images": args.n_images,
        "n_pur_steps": args.n_pur_steps,
        "lam": args.lam,
        "noise_sigma": args.noise_sigma,
        "blur_sigma": args.blur_sigma,
        "accuracy": results,
        "recovery_pct": recovery,
    }
    out_path = out / f"results_{args.attack}_eps{int(args.eps)}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
