"""
eval_freq_analysis.py — Per-band DCT entropy analysis.

Decomposes bpp_proxy into frequency sub-bands (DCT zigzag order) and plots
per-band entropy for clean, adversarial, and DCT-purified images.

This validates Proposition 4: FGSM/PGD concentrate energy in high-frequency
DCT bands, and DCT purification selectively suppresses them.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval/eval_freq_analysis.py \
        --data_dir data/imagenet/val \
        --n_images 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import dct_matrix as _dct_matrix

# ── Zigzag scan order for 8×8 blocks ─────────────────────────────────────────


def _zigzag_bands(block_size=8):
    """Return list of (i,j) pairs in zigzag order, excluding DC (0,0)."""
    b = block_size
    order = []
    for s in range(2 * b - 1):
        if s % 2 == 0:
            for i in range(min(s, b - 1), max(0, s - b + 1) - 1, -1):
                j = s - i
                if 0 <= i < b and 0 <= j < b:
                    order.append((i, j))
        else:
            for i in range(max(0, s - b + 1), min(s, b - 1) + 1):
                j = s - i
                if 0 <= i < b and 0 <= j < b:
                    order.append((i, j))
    # remove DC
    return [(i, j) for (i, j) in order if not (i == 0 and j == 0)]


_ZIGZAG_COORDS = _zigzag_bands(8)  # 63 AC positions in zigzag order


def per_band_entropy(img: Tensor, block_size: int = 8, n_bins: int = 64) -> Tensor:
    """Compute soft-histogram entropy for each of the 63 AC DCT positions.

    Returns:
        entropy: (63,) tensor of entropy values (bits) in zigzag order
    """
    DCT_RANGE = 1.5
    sigma_bin = (2.0 * DCT_RANGE) / n_bins * 1.5

    B, C, H, W = img.shape
    bs = block_size

    pad_h = (bs - H % bs) % bs
    pad_w = (bs - W % bs) % bs
    if pad_h > 0 or pad_w > 0:
        img = nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = img.shape[2], img.shape[3]
    nH, nW = Hp // bs, Wp // bs

    blocks = img.reshape(B, C, nH, bs, nW, bs).permute(0, 1, 2, 4, 3, 5)
    blocks = blocks - blocks.mean(dim=(-2, -1), keepdim=True)

    D = _dct_matrix(bs, img.device)
    flat = blocks.reshape(-1, bs, bs)
    coeffs = D @ flat @ D.t()  # (N_blocks*B*C, bs, bs)

    centers = torch.linspace(-DCT_RANGE, DCT_RANGE, n_bins, device=img.device)
    entropies = []

    for i, j in _ZIGZAG_COORDS:
        ac = coeffs[:, i, j]  # (N_blocks*B*C,)
        d_exp = ac.unsqueeze(-1)
        c_exp = centers.view(1, n_bins)
        weights = torch.exp(-0.5 * ((d_exp - c_exp) / sigma_bin) ** 2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        hist = weights.mean(dim=0).clamp(min=1e-9)
        ent = -(hist * hist.log()).sum() / math.log(2)
        entropies.append(ent.item())

    return entropies


# ── ImageNet subset ───────────────────────────────────────────────────────────


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


MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def normalise(x):
    return (x - MEAN.to(x.device)[None, :, None, None]) / STD.to(x.device)[None, :, None, None]


def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(normalise(x_adv)), y)
    loss.backward()
    return (x + eps * x_adv.grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, steps=10, alpha=None):
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


def dct_purify(x, n_steps=20, lr=0.01, lam=0.05):
    if n_steps == 0:
        return x
    x_pur = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x_pur], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xc = x_pur.clamp(0.0, 1.0)
        from ace import bpp_proxy

        rate = bpp_proxy(xc).mean()
        fidelity = F.mse_loss(xc, x)
        loss = fidelity + lam * rate
        loss.backward()
        opt.step()
    return x_pur.detach().clamp(0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/frequency_analysis")
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eps", type=float, default=4.0, help="ε in [0,255]")
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--n_pur_steps", type=int, default=20)
    p.add_argument("--pur_lr", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    eps = args.eps / 255.0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = models.resnet50(weights="IMAGENET1K_V1").to(args.device).eval()

    dataset = ImageNetSubset(args.data_dir, args.n_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    n_bands = len(_ZIGZAG_COORDS)
    accum = {
        "clean": [0.0] * n_bands,
        "fgsm": [0.0] * n_bands,
        "pgd": [0.0] * n_bands,
        "purified_fgsm": [0.0] * n_bands,
        "purified_pgd": [0.0] * n_bands,
    }
    n_batches = 0

    for x, y in tqdm(loader, desc="Batches"):
        x, y = x.to(args.device), y.to(args.device)

        with torch.enable_grad():
            x_fgsm = fgsm(model, x, y, eps)
            x_pgd = pgd(model, x, y, eps, steps=args.pgd_steps)

        with torch.enable_grad():
            x_pur_fgsm = dct_purify(x_fgsm, args.n_pur_steps, args.pur_lr, args.lam)
            x_pur_pgd = dct_purify(x_pgd, args.n_pur_steps, args.pur_lr, args.lam)

        with torch.no_grad():
            for key, img in [
                ("clean", x),
                ("fgsm", x_fgsm),
                ("pgd", x_pgd),
                ("purified_fgsm", x_pur_fgsm),
                ("purified_pgd", x_pur_pgd),
            ]:
                ent = per_band_entropy(img)
                for b in range(n_bands):
                    accum[key][b] += ent[b]

        n_batches += 1

    # Average
    for key in accum:
        accum[key] = [v / n_batches for v in accum[key]]

    # Band grouping: low (bands 0-20), mid (21-41), high (42-62)
    def band_mean(lst, lo, hi):
        return sum(lst[lo : hi + 1]) / (hi - lo + 1)

    summary = {}
    for key in accum:
        ents = accum[key]
        summary[key] = {
            "per_band": [round(v, 6) for v in ents],
            "low_freq_mean": round(band_mean(ents, 0, 20), 6),
            "mid_freq_mean": round(band_mean(ents, 21, 41), 6),
            "high_freq_mean": round(band_mean(ents, 42, 62), 6),
            "total_mean": round(sum(ents) / len(ents), 6),
        }

    print("\n--- Per-band entropy summary ---")
    print(f"{'Condition':<20}  {'Low':>8}  {'Mid':>8}  {'High':>8}  {'Total':>8}")
    print("-" * 58)
    for key in ["clean", "fgsm", "pgd", "purified_fgsm", "purified_pgd"]:
        s = summary[key]
        print(
            f"{key:<20}  {s['low_freq_mean']:>8.4f}  {s['mid_freq_mean']:>8.4f}  "
            f"{s['high_freq_mean']:>8.4f}  {s['total_mean']:>8.4f}"
        )

    result = {
        "n_images": args.n_images,
        "eps": args.eps,
        "n_pur_steps": args.n_pur_steps,
        "lam": args.lam,
        "band_coords": _ZIGZAG_COORDS,
        "freq_analysis": summary,
    }
    out_path = out / "freq_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
