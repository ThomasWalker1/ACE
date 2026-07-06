"""
eval_cifar_c.py — Evaluate a CIFAR-10 checkpoint on CIFAR-10-C corruption benchmark.

Reports top-1 accuracy per corruption × severity, mean accuracy per corruption,
mCE (mean Corruption Error, relative to AlexNet baseline), and clean accuracy.

AlexNet CIFAR-10-C baseline error rates (Hendrycks & Dietterich 2019, Table 2):
  Used only for normalising CE → mCE.  If you don't need mCE, set --no_mce.

Usage:
    python eval/eval_cifar_c.py \
        --checkpoint_baseline  <path>/baseline/best.pt \
        --checkpoint_proxy_aug <path>/proxy_aug/best.pt \
        --cifar_c_dir data/CIFAR-10-C \
        --cifar_dir   data/cifar-10-batches-py \
        --output      results/aceaug/cifar_c_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 15 standard corruptions for mCE ──────────────────────────────────────────

STANDARD_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",  # noise
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",  # blur
    "snow",
    "frost",
    "fog",
    "brightness",  # weather
    "contrast",
    "elastic_transform",
    "pixelate",  # digital
    "jpeg_compression",
]

# AlexNet CIFAR-10-C error rates (%) per corruption, averaged over severities 1–5
# Source: Hendrycks & Dietterich 2019, Table 2
ALEXNET_ERROR = {
    "gaussian_noise": 89.0,
    "shot_noise": 82.7,
    "impulse_noise": 82.8,
    "defocus_blur": 75.5,
    "glass_blur": 90.5,
    "motion_blur": 76.6,
    "zoom_blur": 78.9,
    "snow": 77.1,
    "frost": 74.5,
    "fog": 75.7,
    "brightness": 56.6,
    "contrast": 83.6,
    "elastic_transform": 61.0,
    "pixelate": 56.0,
    "jpeg_compression": 60.0,
}

# ── Model ─────────────────────────────────────────────────────────────────────

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])


def normalise(x, device):
    return (x - MEAN.to(device)[None, :, None, None]) / STD.to(device)[None, :, None, None]


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    net = models.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_resnet18_cifar().to(device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, "
        f"val_acc {ckpt.get('val_acc', '?'):.2f}%)"
    )
    return model


# ── Evaluation helpers ────────────────────────────────────────────────────────


def evaluate_loader(model, loader, device) -> float:
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(normalise(x, device)).argmax(1) == y).sum().item()
            n += y.shape[0]
    return correct / n * 100


def eval_clean(model, cifar_dir: str, device) -> float:
    with open(Path(cifar_dir) / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = torch.from_numpy(d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
    labels = torch.from_numpy(np.array(d[b"labels"], dtype=np.int64))
    ds = TensorDataset(imgs, labels)
    loader = DataLoader(ds, batch_size=256, num_workers=4)
    return evaluate_loader(model, loader, device)


def eval_corruption(model, cifar_c_dir: str, corruption: str, device) -> dict[int, float]:
    """Returns {severity: accuracy} for severities 1–5."""
    imgs = np.load(Path(cifar_c_dir) / f"{corruption}.npy")  # (50000, 32, 32, 3)
    labels = np.load(Path(cifar_c_dir) / "labels.npy")  # (50000,)

    accs = {}
    for sev in range(1, 6):
        sl = slice((sev - 1) * 10000, sev * 10000)
        x = torch.from_numpy(imgs[sl].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        y = torch.from_numpy(labels[sl].astype(np.int64))
        loader = DataLoader(TensorDataset(x, y), batch_size=256, num_workers=4)
        accs[sev] = evaluate_loader(model, loader, device)
    return accs


# ── Main ──────────────────────────────────────────────────────────────────────


def run_eval(
    model, name: str, cifar_dir: str, cifar_c_dir: str, device, corruptions: list[str]
) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  Evaluating: {name}")
    print(f"{'─' * 60}")

    clean = eval_clean(model, cifar_dir, device)
    print(f"  Clean accuracy: {clean:.2f}%")

    results = {"name": name, "clean_acc": round(clean, 2), "corruptions": {}}
    ces, mces = [], []

    for corr in tqdm(corruptions, desc="Corruptions"):
        sev_accs = eval_corruption(model, cifar_c_dir, corr, device)
        mean_acc = sum(sev_accs.values()) / len(sev_accs)
        mean_err = 100.0 - mean_acc

        results["corruptions"][corr] = {
            "per_severity": {str(k): round(v, 2) for k, v in sev_accs.items()},
            "mean_acc": round(mean_acc, 2),
            "mean_err": round(mean_err, 2),
        }
        ces.append(mean_err)

        if corr in ALEXNET_ERROR:
            mces.append(mean_err / ALEXNET_ERROR[corr] * 100)

    results["mean_corruption_acc"] = round(
        sum(v["mean_acc"] for v in results["corruptions"].values()) / len(corruptions), 2
    )
    results["mCE"] = round(sum(mces) / len(mces), 1) if mces else None

    print(f"  Mean corruption acc (15 std): {results['mean_corruption_acc']:.2f}%")
    if results["mCE"] is not None:
        print(f"  mCE (vs AlexNet):             {results['mCE']:.1f}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_baseline", required=True)
    p.add_argument("--checkpoint_proxy_aug", required=True)
    p.add_argument("--cifar_c_dir", required=True)
    p.add_argument("--cifar_dir", required=True)
    p.add_argument("--output", default="results/aceaug/cifar_c_results.json")
    p.add_argument(
        "--extra_corruptions",
        nargs="*",
        default=[],
        help="Additional corruption names beyond the standard 15",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    corruptions = STANDARD_CORRUPTIONS + args.extra_corruptions

    baseline = load_model(args.checkpoint_baseline, device)
    proxy_aug = load_model(args.checkpoint_proxy_aug, device)

    res_base = run_eval(baseline, "baseline", args.cifar_dir, args.cifar_c_dir, device, corruptions)
    res_proxy = run_eval(
        proxy_aug, "proxy_aug", args.cifar_dir, args.cifar_c_dir, device, corruptions
    )

    # Side-by-side summary
    print(f"\n{'─' * 60}")
    print("  Summary (mean accuracy over 15 standard corruptions)")
    print(f"{'─' * 60}")
    print(f"  {'Corruption':<25}  {'Baseline':>10}  {'Proxy aug':>10}  {'Δ':>8}")
    print(f"  {'─' * 25}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
    for corr in corruptions:
        if corr not in res_base["corruptions"]:
            continue
        b = res_base["corruptions"][corr]["mean_acc"]
        q = res_proxy["corruptions"][corr]["mean_acc"]
        print(f"  {corr:<25}  {b:>9.2f}%  {q:>9.2f}%  {q - b:>+7.2f}pp")
    print(
        f"\n  {'Clean acc':<25}  {res_base['clean_acc']:>9.2f}%  "
        f"{res_proxy['clean_acc']:>9.2f}%  "
        f"{res_proxy['clean_acc'] - res_base['clean_acc']:>+7.2f}pp"
    )
    print(
        f"  {'Mean corr. acc':<25}  {res_base['mean_corruption_acc']:>9.2f}%  "
        f"{res_proxy['mean_corruption_acc']:>9.2f}%  "
        f"{res_proxy['mean_corruption_acc'] - res_base['mean_corruption_acc']:>+7.2f}pp"
    )
    if res_base["mCE"] and res_proxy["mCE"]:
        print(
            f"  {'mCE':<25}  {res_base['mCE']:>9.1f}   "
            f"{res_proxy['mCE']:>9.1f}   "
            f"{res_proxy['mCE'] - res_base['mCE']:>+7.1f}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"baseline": res_base, "proxy_aug": res_proxy}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
