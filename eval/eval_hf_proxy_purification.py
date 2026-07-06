"""
eval_hf_proxy_purification.py — Selective high-frequency proxy for adversarial purification.

Motivation: the full proxy reduces AC entropy uniformly across all frequency bands
(Table 1 in the paper), which causes over-smoothing of semantic content at T>10.
Restricting entropy to high-frequency AC coefficients only should let the optimizer
target adversarial content more selectively while leaving low-frequency semantic
structure intact.

Frequency bands (by DCT coefficient frequency sum i+j):
  Full proxy  : all 63 AC coefficients (i+j >= 1)
  Mid+High    : 42 coefficients (i+j >= 5)
  High only   : 21 coefficients (i+j >= 9)

We compare the three variants on PGD-10 eps=4/255 at T=10 steps (where full proxy peaks).
Then sweep T for the best variant to find if it can exceed 47.9% recovery.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval/eval_hf_proxy_purification.py \
        --data_dir data/imagenet/val --n_images 500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as TF
from torch import Tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Selective proxy
# ---------------------------------------------------------------------------


def _dct_matrix(n: int, device: torch.device) -> Tensor:
    k = torch.arange(n, dtype=torch.float32, device=device)
    m = torch.arange(n, dtype=torch.float32, device=device)
    D = torch.cos(math.pi / n * k.unsqueeze(1) * (m.unsqueeze(0) + 0.5))
    D[0] *= math.sqrt(1.0 / n)
    D[1:] *= math.sqrt(2.0 / n)
    return D


def _hf_mask(block_size: int, min_freq_sum: int, device: torch.device) -> Tensor:
    """Bool mask over (block_size, block_size): True for AC positions with i+j >= min_freq_sum."""
    i = torch.arange(block_size, device=device).unsqueeze(1)
    j = torch.arange(block_size, device=device).unsqueeze(0)
    freq_sum = i + j  # (bs, bs)
    mask = (freq_sum >= min_freq_sum) & ~(freq_sum == 0)  # exclude DC (0,0)
    return mask


def bpp_proxy_selective(
    img: Tensor,
    block_size: int = 8,
    n_bins: int = 64,
    sigma_bin: float = None,
    min_freq_sum: int = 1,  # 1=all AC, 5=mid+high, 9=high only
) -> Tensor:
    """BPP proxy restricted to AC coefficients with i+j >= min_freq_sum."""
    DCT_RANGE = 1.5
    if sigma_bin is None:
        sigma_bin = (2.0 * DCT_RANGE) / n_bins * 1.5

    B, C, H, W = img.shape
    bs = block_size

    pad_h = (bs - H % bs) % bs
    pad_w = (bs - W % bs) % bs
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = img.shape[2], img.shape[3]
    nH, nW = Hp // bs, Wp // bs

    blocks = img.reshape(B, C, nH, bs, nW, bs).permute(0, 1, 2, 4, 3, 5)
    blocks = blocks - blocks.mean(dim=(-2, -1), keepdim=True)

    D = _dct_matrix(bs, img.device)
    flat = blocks.reshape(-1, bs, bs)
    coeffs = D @ flat @ D.t()

    mask = _hf_mask(bs, min_freq_sum, img.device)
    ac = coeffs[:, mask].reshape(B, C, -1)  # (B, C, n_selected)

    centers = torch.linspace(-DCT_RANGE, DCT_RANGE, n_bins, device=img.device)
    d_exp = ac.unsqueeze(-1)
    c_exp = centers.view(1, 1, 1, n_bins)
    weights = torch.exp(-0.5 * ((d_exp - c_exp) / sigma_bin) ** 2)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

    hist = weights.mean(dim=2).clamp(min=1e-9)
    entropy_bits = -(hist * hist.log()).sum(dim=-1) / math.log(2)  # (B, C)
    return entropy_bits.mean(dim=1)  # (B,)


# ---------------------------------------------------------------------------
# Purification and attack
# ---------------------------------------------------------------------------


def dct_purify(x_adv_01: Tensor, lam: float, n_steps: int, lr: float, min_freq_sum: int) -> Tensor:
    """Purify in [0,1] space — proxy requires [0,1] inputs."""
    x = x_adv_01.clone().detach()
    opt = torch.optim.Adam([x.requires_grad_(True)], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xc = x.clamp(0, 1)
        rate = bpp_proxy_selective(xc, min_freq_sum=min_freq_sum).mean()
        fid = 0.5 * F.mse_loss(xc, x_adv_01)
        (fid + lam * rate).backward()
        opt.step()
    return x.detach().clamp(0, 1)


def pgd_attack(
    model, x_01: Tensor, y: Tensor, eps: float, n_steps: int, step_size: float, normalize_fn
) -> Tensor:
    """PGD in [0,1] space; normalizes internally for model inference."""
    x_adv = x_01.clone().detach() + torch.zeros_like(x_01).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(normalize_fn(x_adv)), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_01 + eps), x_01 - eps).clamp(0, 1)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/adv_purification")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--eps", type=float, default=4.0, help="PGD epsilon in /255 units")
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    eps = args.eps / 255.0
    step_size = 2.5 * eps / args.pgd_steps

    # Model
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    normalize_fn = TF.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    preprocess = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.ToTensor()])

    from PIL import Image

    data_dir = Path(args.data_dir)
    classes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    samples = []
    for label_idx, cls_dir in enumerate(classes):
        for img_path in sorted(cls_dir.glob("*.JPEG"))[:2]:
            samples.append((img_path, label_idx))
        if len(samples) >= args.n_images:
            break
    samples = samples[: args.n_images]
    print(f"Loaded {len(samples)} images, eps={args.eps}/255, PGD-{args.pgd_steps}")

    variants = [
        ("full", 1, [5, 10, 20]),
        ("mid_high", 5, [5, 10, 20]),
        ("high", 9, [5, 10, 20]),
    ]

    results = {"eps": args.eps, "pgd_steps": args.pgd_steps, "n_images": len(samples)}
    clean_correct = 0

    # First pass: evaluate clean and generate adversarial examples in [0,1] space
    print("Generating adversarial examples...")
    adv_list = []  # [0,1] adversarial images
    label_list = []

    for img_path, label in tqdm(samples):
        img_01 = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        lbl = torch.tensor([label], device=device)

        with torch.no_grad():
            pred = model(normalize_fn(img_01)).argmax(dim=1)
        if pred.item() == label:
            clean_correct += 1

        # Attack in [0,1]; model sees normalized image internally
        x_adv = pgd_attack(model, img_01, lbl, eps, args.pgd_steps, step_size, normalize_fn)
        adv_list.append(x_adv.cpu())
        label_list.append(lbl.cpu())

    clean_acc = 100.0 * clean_correct / len(samples)
    adv_correct = 0
    for x_adv, lbl in zip(adv_list, label_list):
        with torch.no_grad():
            pred = model(normalize_fn(x_adv.to(device))).argmax(dim=1)
        if pred.item() == lbl.item():
            adv_correct += 1
    adv_acc = 100.0 * adv_correct / len(samples)

    print(f"Clean: {clean_acc:.2f}%  |  PGD adversarial: {adv_acc:.2f}%")
    results["clean_acc"] = clean_acc
    results["adv_acc"] = adv_acc
    results["variants"] = {}

    # Second pass: purify in [0,1]; normalize only for classification
    for name, min_freq_sum, t_list in variants:
        n_selected = int(_hf_mask(8, min_freq_sum, torch.device("cpu")).sum().item())
        print(f"\n--- {name} proxy (min_freq_sum={min_freq_sum}, {n_selected}/63 AC coefs) ---")
        results["variants"][name] = {
            "min_freq_sum": min_freq_sum,
            "n_ac_coefficients": n_selected,
            "T_results": {},
        }

        for T in t_list:
            correct = 0
            for x_adv, lbl in tqdm(zip(adv_list, label_list), total=len(adv_list), desc=f"T={T}"):
                x_adv_d = x_adv.to(device)
                x_pur = dct_purify(x_adv_d, args.lam, T, args.lr, min_freq_sum)
                with torch.no_grad():
                    pred = model(normalize_fn(x_pur)).argmax(dim=1)
                if pred.item() == lbl.item():
                    correct += 1
            acc = 100.0 * correct / len(samples)
            gap = clean_acc - adv_acc
            recovery = (acc - adv_acc) / gap * 100.0 if gap > 0 else 0.0
            print(f"  T={T:>2}: acc={acc:.2f}%  gap_recovery={recovery:.1f}%")
            results["variants"][name]["T_results"][str(T)] = {
                "accuracy": acc,
                "gap_recovery": recovery,
            }

    # Print summary table
    print("\n=== Summary (PGD-10, ε=4/255) ===")
    print(f"{'Variant':>10}  {'#AC':>4}  ", end="")
    for _, _, t_list in variants:
        for T in t_list:
            print(f"  T={T:<2}", end="")
    print()
    for name, min_freq_sum, t_list in variants:
        n_ac = results["variants"][name]["n_ac_coefficients"]
        print(f"{name:>10}  {n_ac:>4}  ", end="")
        for T in t_list:
            rec = results["variants"][name]["T_results"][str(T)]["gap_recovery"]
            print(f"  {rec:>5.1f}%", end="")
        print()

    out_path = Path(args.output_dir) / "hf_proxy_purification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
