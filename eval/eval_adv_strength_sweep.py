"""
eval_adv_strength_sweep.py — Sweep purification strength for all methods.

Produces JSON data for the gap-recovery-vs-strength figure.

  DCT proxy:       sweep T ∈ {0, 2, 5, 10, 15, 20, 30}
  Gaussian blur:   sweep σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
  Gaussian noise:  sweep σ ∈ {0.01, 0.02, 0.04, 0.06, 0.08, 0.10}
  JPEG:            sweep q ∈ {90, 80, 70, 60, 50, 40}

All evaluated on the same attack/dataset so results are directly comparable.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/eval_adv_strength_sweep.py \\
        --data_dir data/imagenet/val \\
        --attack pgd --eps 4 --n_images 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eval_adv_purification import (
    ImageNetSubset,
    fgsm,
    pgd,
    jpeg_defense,
    gaussian_noise_defense,
    gaussian_blur_defense,
    dct_purify,
    accuracy,
)

# ── Sweep configs ─────────────────────────────────────────────────────────────

DCT_T_VALUES = [0, 2, 5, 10, 15, 20, 30]
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NOISE_SIGMAS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
JPEG_QUALITIES = [100, 90, 80, 70, 60, 50, 40]

# ── Evaluation ────────────────────────────────────────────────────────────────


def sweep_accuracy(model, loader, attack_fn, defense_fn, device):
    """
    Measure clean, adversarial, defended (on adv), and clean-defended accuracy.

    clean_defended: defense applied to clean images — measures cost to clean accuracy.
    This is the critical measurement for Pareto comparison: a defense that recovers
    adversarial accuracy at the cost of badly hurting clean accuracy is not useful.
    """
    clean_total = adv_total = def_total = clean_def_total = 0.0
    n = 0
    needs_grad = getattr(defense_fn, "__name__", "") == "dct_purify"
    ctx = torch.enable_grad if needs_grad else torch.no_grad
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            clean_total += accuracy(model, x, y)
            adv_total += accuracy(model, x_adv, y)
        with ctx():
            x_def = defense_fn(x_adv)
            x_def_clean = defense_fn(x)
        with torch.no_grad():
            def_total += accuracy(model, x_def, y)
            clean_def_total += accuracy(model, x_def_clean, y)
        n += 1
    c = round(clean_total / n * 100, 2)
    a = round(adv_total / n * 100, 2)
    d = round(def_total / n * 100, 2)
    cd = round(clean_def_total / n * 100, 2)
    rec = round((d - a) / max(c - a, 1e-6) * 100, 1)
    clean_drop = round(c - cd, 2)
    return {
        "clean": c,
        "adversarial": a,
        "defended": d,
        "clean_defended": cd,
        "recovery_pct": rec,
        "clean_drop_pp": clean_drop,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/adv_strength_sweep")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    p.add_argument("--eps", type=float, default=4.0)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--pur_lr", type=float, default=0.01)
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

    if args.attack == "fgsm":

        def attack_fn(m, x, y):
            return fgsm(m, x, y, eps)
    else:

        def attack_fn(m, x, y):
            return pgd(m, x, y, eps, steps=args.pgd_steps, alpha=eps / 4)

    results = {}

    # ── DCT proxy sweep ───────────────────────────────────────────────────────
    print("Sweeping DCT proxy T …")
    dct_results = []
    for T in DCT_T_VALUES:
        print(f"  T={T}")

        def fn(x_adv, T=T):
            return dct_purify(x_adv, T, args.pur_lr, args.lam)

        fn.__name__ = "dct_purify"
        r = sweep_accuracy(model, loader, attack_fn, fn, args.device)
        dct_results.append({"T": T, **r})
    results["dct_proxy"] = dct_results

    # ── Gaussian blur sweep ───────────────────────────────────────────────────
    print("Sweeping Gaussian blur σ …")
    blur_results = []
    for sigma in BLUR_SIGMAS:
        print(f"  σ={sigma}")

        def fn(x_adv, s=sigma):
            return gaussian_blur_defense(x_adv, s)

        r = sweep_accuracy(model, loader, attack_fn, fn, args.device)
        blur_results.append({"sigma": sigma, **r})
    results["gaussian_blur"] = blur_results

    # ── Gaussian noise sweep ──────────────────────────────────────────────────
    print("Sweeping Gaussian noise σ …")
    noise_results = []
    for sigma in NOISE_SIGMAS:
        print(f"  σ={sigma}")

        def fn(x_adv, s=sigma):
            return gaussian_noise_defense(x_adv, s)

        r = sweep_accuracy(model, loader, attack_fn, fn, args.device)
        noise_results.append({"sigma": sigma, **r})
    results["gaussian_noise"] = noise_results

    # ── JPEG quality sweep ────────────────────────────────────────────────────
    print("Sweeping JPEG quality …")
    jpeg_results = []
    for q in JPEG_QUALITIES:
        print(f"  q={q}")

        def fn(x_adv, quality=q):
            return jpeg_defense(x_adv, quality)

        r = sweep_accuracy(model, loader, attack_fn, fn, args.device)
        jpeg_results.append({"quality": q, **r})
    results["jpeg"] = jpeg_results

    out_path = out / f"strength_sweep_{args.attack}_eps{int(args.eps)}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"attack": args.attack, "eps": args.eps, "n_images": args.n_images, "results": results},
            f,
            indent=2,
        )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
