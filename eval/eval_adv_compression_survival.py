"""
eval_adv_compression_survival.py — Experiment 2: Compression-Surviving Adversarial Perturbations.

Craft PGD adversarial examples with and without a proxy constraint, then measure
how well each attack survives JPEG/HEVC compression. The proxy constraint keeps
the perturbation's AC entropy low, making it closer to the codec's null space.

Hypothesis: proxy-constrained PGD produces perturbations that survive compression
better than standard PGD (same ε budget).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/eval_adv_compression_survival.py \
        --data_dir data/imagenet/val \
        --n_images 500 --eps 4 8 --device cuda

Results are saved to results/adversarial_compression/
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

FFMPEG = os.environ.get("FFMPEG_PATH", shutil.which("ffmpeg") or "ffmpeg")


def normalise(x: Tensor) -> Tensor:
    return (x - MEAN.to(x.device)[None, :, None, None]) / STD.to(x.device)[None, :, None, None]


# ── Dataset ───────────────────────────────────────────────────────────────────


class ImageNetSubset(Dataset):
    def __init__(self, root: str, n_images: int):
        root = Path(root)
        self.samples = []
        classes = sorted(path for path in root.iterdir() if path.is_dir())
        class_to_label = {path.name: index for index, path in enumerate(classes)}
        for cls_dir in classes:
            for img_path in sorted(cls_dir.glob("*.JPEG")):
                self.samples.append((img_path, class_to_label[cls_dir.name]))
        step = max(1, len(self.samples) // n_images)
        self.samples = self.samples[::step][:n_images]

        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Attacks ───────────────────────────────────────────────────────────────────


def pgd_standard(
    model: torch.nn.Module, x: Tensor, y: Tensor, eps: float, steps: int = 10, alpha: float = None
) -> Tensor:
    """Standard PGD-10 (Madry et al.). x in [0,1], eps in pixel units [0,1]."""
    if alpha is None:
        alpha = eps / 4
    delta = torch.empty_like(x).uniform_(-eps, eps).clamp(-eps, eps)
    x_adv = (x + delta).clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(normalise(x_adv)), y)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


def pgd_proxy_constrained(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    eps: float,
    steps: int = 10,
    alpha: float = None,
    mu: float = 0.5,
) -> Tensor:
    """
    Proxy-constrained PGD. At each step, the PGD gradient is augmented with
    -mu * grad(bpp_proxy(delta)) to keep the perturbation's AC entropy low.

    The proxy is evaluated on the perturbation delta (shifted to [0,1]) rather
    than the full adversarial image, so it targets the perturbation's own
    frequency content, not the clean image's.
    """
    if alpha is None:
        alpha = eps / 4
    delta = torch.empty_like(x).uniform_(-eps, eps).clamp(-eps, eps)
    x_adv = (x + delta).clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        ce_loss = F.cross_entropy(model(normalise(x_adv)), y)

        # proxy on the perturbation (shifted to [0,1] range)
        delta_viz = ((x_adv - x) / (2 * eps) + 0.5).clamp(0.0, 1.0)
        proxy_loss = bpp_proxy(delta_viz).mean()

        # maximize CE (attack) while minimizing proxy (keep perturbation compressible)
        loss = ce_loss - mu * proxy_loss
        loss.backward()

        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


# ── Compression ───────────────────────────────────────────────────────────────


def jpeg_compress_batch(x: Tensor, quality: int) -> Tensor:
    """Round-trip JPEG compression on a batch. x in [0,1]."""
    out = []
    for img in x:
        pil = T.ToPILImage()(img.cpu())
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(T.ToTensor()(Image.open(buf)))
    return torch.stack(out).to(x.device)


def hevc_compress_batch(x: Tensor, crf: int) -> Tensor:
    """Round-trip HEVC compression on a batch via ffmpeg."""
    out = []
    for img in x:
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]
        with (
            tempfile.NamedTemporaryFile(suffix=".png") as fin,
            tempfile.NamedTemporaryFile(suffix=".hevc") as fhevc,
            tempfile.NamedTemporaryFile(suffix=".png") as fout,
        ):
            Image.fromarray(img_np).save(fin.name)
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    fin.name,
                    "-c:v",
                    "libx265",
                    "-crf",
                    str(crf),
                    "-preset",
                    "fast",
                    "-x265-params",
                    "log-level=none",
                    fhevc.name,
                ],
                check=True,
            )
            subprocess.run(
                [FFMPEG, "-y", "-loglevel", "error", "-i", fhevc.name, fout.name], check=True
            )
            rec = np.array(Image.open(fout.name).convert("RGB"))
        out.append(torch.from_numpy(rec).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out).to(x.device)


# ── Metrics ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def accuracy(model: torch.nn.Module, x: Tensor, y: Tensor) -> float:
    return (model(normalise(x)).argmax(dim=1) == y).float().mean().item()


def perturbation_bpp(x_clean: Tensor, x_adv: Tensor, eps: float) -> float:
    """Mean proxy BPP of the perturbation (shifted to [0,1])."""
    delta_viz = ((x_adv - x_clean) / (2 * eps) + 0.5).clamp(0.0, 1.0)
    with torch.no_grad():
        return bpp_proxy(delta_viz).mean().item()


# ── Main evaluation loop ──────────────────────────────────────────────────────


def run_experiment(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loading dataset from {args.data_dir} ({args.n_images} images)...")
    dataset = ImageNetSubset(args.data_dir, args.n_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    jpeg_qualities = args.jpeg_qualities
    hevc_crfs = args.hevc_crfs
    eps_values = [e / 255.0 for e in args.eps]
    mu_values = args.mu

    all_results = {}

    for eps_raw, eps in zip(args.eps, eps_values):
        print(f"\n{'=' * 60}")
        print(f"ε = {eps_raw}/255")
        print(f"{'=' * 60}")

        eps_results = {"clean_acc": 0.0, "attacks": {}}
        attack_configs = [
            ("pgd_standard", lambda m, x, y, e=eps: pgd_standard(m, x, y, e, steps=args.pgd_steps)),
        ]
        for mu in mu_values:
            attack_configs.append(
                (
                    f"pgd_proxy_mu{mu}",
                    lambda m, x, y, e=eps, mu_=mu: pgd_proxy_constrained(
                        m, x, y, e, steps=args.pgd_steps, mu=mu_
                    ),
                )
            )

        # Initialize accumulators
        clean_acc_total = 0.0
        attack_accs = {name: {"pre_compression": 0.0} for name, _ in attack_configs}
        for name, _ in attack_configs:
            for q in jpeg_qualities:
                attack_accs[name][f"post_jpeg_q{q}"] = 0.0
            for crf in hevc_crfs:
                attack_accs[name][f"post_hevc_crf{crf}"] = 0.0
            attack_accs[name]["perturbation_bpp"] = 0.0
        n_batches = 0

        for x, y in tqdm(loader, desc=f"ε={eps_raw}/255"):
            x, y = x.to(device), y.to(device)
            n_batches += 1

            with torch.no_grad():
                clean_acc_total += accuracy(model, x, y)

            for name, attack_fn in attack_configs:
                with torch.enable_grad():
                    x_adv = attack_fn(model, x, y)

                with torch.no_grad():
                    attack_accs[name]["pre_compression"] += accuracy(model, x_adv, y)
                    attack_accs[name]["perturbation_bpp"] += perturbation_bpp(x, x_adv, eps)

                    for q in jpeg_qualities:
                        x_jpeg = jpeg_compress_batch(x_adv, q)
                        attack_accs[name][f"post_jpeg_q{q}"] += accuracy(model, x_jpeg, y)

                    for crf in hevc_crfs:
                        x_hevc = hevc_compress_batch(x_adv, crf)
                        attack_accs[name][f"post_hevc_crf{crf}"] += accuracy(model, x_hevc, y)

        # Normalize
        eps_results["clean_acc"] = clean_acc_total / n_batches * 100
        for name, _ in attack_configs:
            eps_results["attacks"][name] = {
                k: v / n_batches * 100 if k != "perturbation_bpp" else v / n_batches
                for k, v in attack_accs[name].items()
            }

        all_results[f"eps_{eps_raw}"] = eps_results

        # Print summary
        print(f"\nClean accuracy: {eps_results['clean_acc']:.1f}%")
        print(
            f"\n{'Attack':<30} {'Pre-comp':>10} {'JPEG-50':>10} {'JPEG-75':>10} "
            f"{'HEVC-32':>10} {'HEVC-37':>10} {'δ-BPP':>8}"
        )
        print("-" * 90)
        for name, stats in eps_results["attacks"].items():
            pre = stats["pre_compression"]
            bpp = stats["perturbation_bpp"]
            parts = [f"{name:<30}", f"{pre:>9.1f}%"]
            for q in jpeg_qualities:
                parts.append(f"{stats.get(f'post_jpeg_q{q}', float('nan')):>9.1f}%")
            for crf in hevc_crfs:
                parts.append(f"{stats.get(f'post_hevc_crf{crf}', float('nan')):>9.1f}%")
            parts.append(f"{bpp:>7.3f}")
            print(" ".join(parts))

    # Save results
    out_path = output_dir / f"results_n{args.n_images}_steps{args.pgd_steps}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compression-surviving adversarial perturbations")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results/adversarial_compression")
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--eps",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Perturbation budgets in pixel units (will be divided by 255)",
    )
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument(
        "--mu",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 3.0],
        help="Proxy constraint weights to sweep",
    )
    parser.add_argument("--jpeg_qualities", type=int, nargs="+", default=[50, 75])
    parser.add_argument("--hevc_crfs", type=int, nargs="+", default=[32, 37])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
