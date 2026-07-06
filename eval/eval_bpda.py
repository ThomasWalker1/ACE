"""
eval_bpda.py — Adaptive attack (BPDA) against DCT purification.

An adaptive adversary who knows the purification defence crafts adversarial
examples that are designed to survive DCT proxy smoothing.

BPDA (Athalye et al. 2018): substitute the identity function for the
(non-differentiable) purification step during backward pass, so gradients
flow directly through the classifier without the purification barrier.
In our case, the purification IS differentiable (it's Adam on the proxy),
so we backpropagate through the full purification loop ("white-box BPDA").

Two settings compared:
  - Standard PGD (no knowledge of defence) → then purified
  - BPDA-PGD (crafted through purification) → then purified

Expected result: BPDA partially defeats the defence. Reporting this honestly
is necessary for any robustness claim to survive peer review.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval/eval_bpda.py \
        --data_dir data/imagenet/val --n_images 500 --eps 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def normalise(x):
    return (x - MEAN.to(x.device)[None, :, None, None]) / STD.to(x.device)[None, :, None, None]


class ImageNetSubset(Dataset):
    def __init__(self, root, n_images, transform=None):
        root = Path(root)
        all_classes = sorted(root.iterdir())
        self.samples = []
        for ci, cls_dir in enumerate(all_classes):
            for img_path in sorted(cls_dir.glob("*.JPEG")):
                self.samples.append((img_path, ci))
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


@torch.no_grad()
def accuracy(model, x, y):
    return (model(normalise(x)).argmax(1) == y).float().mean().item()


def dct_purify(x, n_steps, lr, lam):
    """Standard DCT purification (no gradient through purification)."""
    if n_steps == 0:
        return x
    x_pur = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x_pur], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xc = x_pur.clamp(0.0, 1.0)
        (F.mse_loss(xc, x) + lam * bpp_proxy(xc).mean()).backward()
        opt.step()
    return x_pur.detach().clamp(0.0, 1.0)


def dct_purify_differentiable(x_adv, n_steps, lr, lam):
    """Purification with BPDA straight-through estimator.

    Forward: run real Adam purification loop.
    Backward: straight-through — gradients see identity (x_adv).
    This lets the adversarial PGD loop backpropagate through the defence.
    """
    x_adv_det = x_adv.detach()
    x_pur = x_adv_det.clone().requires_grad_(True)
    opt = torch.optim.Adam([x_pur], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xc = x_pur.clamp(0.0, 1.0)
        (F.mse_loss(xc, x_adv_det) + lam * bpp_proxy(xc).mean()).backward()
        opt.step()
    purified = x_pur.detach().clamp(0.0, 1.0)
    # Straight-through: in forward pass use purified; in backward pass use x_adv
    return purified - x_adv.detach() + x_adv


def pgd_standard(model, x, y, eps, steps, alpha, n_pur_steps, pur_lr, lam):
    """Standard PGD — no knowledge of defence."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(normalise(x_adv)), y)
        loss.backward()
        x_adv = (x_adv + alpha * x_adv.grad.sign()).detach()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv


def pgd_bpda(model, x, y, eps, steps, alpha, n_pur_steps, pur_lr, lam):
    """BPDA-PGD: backpropagate through purification via straight-through."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        # BPDA: purify in forward, identity in backward
        x_pur = dct_purify_differentiable(x_adv, n_pur_steps, pur_lr, lam)
        loss = F.cross_entropy(model(normalise(x_pur.clamp(0, 1))), y)
        loss.backward()
        x_adv = (x_adv + alpha * x_adv.grad.sign()).detach()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/adv_purification")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Smaller batch due to double backward through purification",
    )
    p.add_argument("--eps", type=float, default=4.0, help="ε in [0,255]")
    p.add_argument(
        "--pgd_steps", type=int, default=20, help="PGD steps (more for BPDA to be effective)"
    )
    p.add_argument(
        "--n_pur_steps", type=int, default=10, help="Purification steps (use optimal T=10)"
    )
    p.add_argument("--pur_lr", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    eps = args.eps / 255.0
    alpha = eps / 4.0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"BPDA adaptive attack: ε={args.eps}/255, PGD steps={args.pgd_steps}")
    print(f"Purification: T={args.n_pur_steps}, λ={args.lam}")

    model = models.resnet50(weights="IMAGENET1K_V1").to(args.device).eval()

    dataset = ImageNetSubset(args.data_dir, args.n_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    totals = {
        "clean": 0.0,
        "pgd_unpurified": 0.0,
        "pgd_purified": 0.0,
        "bpda_unpurified": 0.0,
        "bpda_purified": 0.0,
    }
    n_batches = 0

    for x, y in tqdm(loader, desc="Batches"):
        x, y = x.to(args.device), y.to(args.device)

        with torch.enable_grad():
            x_pgd = pgd_standard(
                model, x, y, eps, args.pgd_steps, alpha, args.n_pur_steps, args.pur_lr, args.lam
            )
            x_bpda = pgd_bpda(
                model, x, y, eps, args.pgd_steps, alpha, args.n_pur_steps, args.pur_lr, args.lam
            )

            x_pgd_pur = dct_purify(x_pgd, args.n_pur_steps, args.pur_lr, args.lam)
            x_bpda_pur = dct_purify(x_bpda, args.n_pur_steps, args.pur_lr, args.lam)

        totals["clean"] += accuracy(model, x, y)
        totals["pgd_unpurified"] += accuracy(model, x_pgd, y)
        totals["pgd_purified"] += accuracy(model, x_pgd_pur, y)
        totals["bpda_unpurified"] += accuracy(model, x_bpda, y)
        totals["bpda_purified"] += accuracy(model, x_bpda_pur, y)
        n_batches += 1

    results = {k: round(v / n_batches * 100, 2) for k, v in totals.items()}
    clean = results["clean"]

    def rec(acc):
        adv = min(results["pgd_unpurified"], results["bpda_unpurified"])
        return round((acc - adv) / max(clean - adv, 1e-6) * 100, 1)

    print(f"\n--- BPDA Adaptive Attack (ε={args.eps}/255) ---")
    print(f"{'Condition':<25}  {'Accuracy':>10}  {'Recovery':>10}")
    print("-" * 50)
    print(f"{'Clean':<25}  {clean:>9.2f}%")
    for k in ["pgd_unpurified", "pgd_purified", "bpda_unpurified", "bpda_purified"]:
        print(f"{k:<25}  {results[k]:>9.2f}%  {rec(results[k]):>8.1f}%")

    summary = {
        "eps": args.eps,
        "pgd_steps": args.pgd_steps,
        "n_pur_steps": args.n_pur_steps,
        "lam": args.lam,
        "n_images": args.n_images,
        "accuracy": results,
    }
    out_path = out / f"bpda_eps{int(args.eps)}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
