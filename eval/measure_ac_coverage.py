"""
measure_ac_coverage.py — Report what fraction of natural-image AC DCT-II
coefficients fall outside the proxy's bin range [-1.5, 1.5].

Addresses reviewer question: "What fraction of natural image AC coefficients
in the Kodak and ImageNet validation sets fall outside [−1.5, 1.5]? What
happens to out-of-range values?"

Usage:
    python eval/measure_ac_coverage.py \
        --kodak_dir  data/kodak \
        --imagenet_dir data/imagenet/val \
        --n_imagenet 500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# ── DCT helpers ────────────────────────────────────────────────────────────────


def _dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix of size n×n."""
    k = torch.arange(n, dtype=torch.float32, device=device)
    m = torch.arange(n, dtype=torch.float32, device=device)
    D = torch.cos(math.pi / n * k.unsqueeze(1) * (m.unsqueeze(0) + 0.5))
    D[0] *= math.sqrt(1.0 / n)
    D[1:] *= math.sqrt(2.0 / n)
    return D


def compute_ac_coefficients(img_tensor: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    img_tensor: (C, H, W) float in [0, 1].
    Returns all AC DCT-II coefficients as a 1-D tensor.
    """
    C, H, W = img_tensor.shape
    bs = block_size

    # Pad to block multiples
    pad_h = (bs - H % bs) % bs
    pad_w = (bs - W % bs) % bs
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    Hp, Wp = img_tensor.shape[1], img_tensor.shape[2]
    nH, nW = Hp // bs, Wp // bs

    # Extract blocks
    blocks = img_tensor.reshape(C, nH, bs, nW, bs).permute(0, 1, 3, 2, 4)
    # (C, nH, nW, bs, bs)
    blocks = blocks - blocks.mean(dim=(-2, -1), keepdim=True)

    D = _dct_matrix(bs, img_tensor.device)
    flat = blocks.reshape(-1, bs, bs)  # (N, bs, bs)
    coeffs = D @ flat @ D.t()  # (N, bs, bs)

    ac_mask = torch.ones(bs, bs, dtype=torch.bool)
    ac_mask[0, 0] = False
    return coeffs[:, ac_mask].reshape(-1)  # (N * 63,)


def load_image(path: Path, max_side: int = 768) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return T.ToTensor()(img)  # (3, H, W) in [0, 1]


def coverage_stats(
    ac: torch.Tensor, lo: float = -1.5, hi: float = 1.5, quantile_sample: int = 2_000_000
) -> dict:
    total = ac.numel()
    outside = ((ac < lo) | (ac > hi)).sum().item()
    # torch.quantile fails on very large tensors; subsample for percentiles
    ac_q = ac if total <= quantile_sample else ac[torch.randperm(total)[:quantile_sample]]
    ac_q_sorted, _ = ac_q.sort()

    def pct(p):
        idx = int(p * len(ac_q_sorted))
        return round(ac_q_sorted[idx].item(), 4)

    return {
        "total_coefficients": total,
        "outside_range": outside,
        "fraction_outside": round(outside / total, 6),
        "pct_outside": round(outside / total * 100, 4),
        "p1": pct(0.01),
        "p5": pct(0.05),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": round(ac.min().item(), 4),
        "max": round(ac.max().item(), 4),
        "std": round(ac.std().item(), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def collect_ac(image_paths: list[Path], desc: str) -> torch.Tensor:
    all_ac = []
    for p in tqdm(image_paths, desc=desc):
        try:
            img = load_image(p)
            ac = compute_ac_coefficients(img)
            all_ac.append(ac)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return torch.cat(all_ac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak_dir", default="data/kodak")
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument(
        "--n_imagenet",
        type=int,
        default=500,
        help="Number of ImageNet images to sample (evenly spaced)",
    )
    ap.add_argument("--output", default="results/ac_coverage.json")
    args = ap.parse_args()

    results = {}

    # ── Kodak ──────────────────────────────────────────────────────────────────
    kodak_dir = Path(args.kodak_dir)
    kodak_paths = (
        sorted(kodak_dir.glob("*.png"))
        + sorted(kodak_dir.glob("*.PNG"))
        + sorted(kodak_dir.glob("*.jpg"))
        + sorted(kodak_dir.glob("*.JPEG"))
    )
    if not kodak_paths:
        print(f"WARNING: no images found in {kodak_dir}")
    else:
        print(f"Kodak: {len(kodak_paths)} images")
        ac_kodak = collect_ac(kodak_paths, "Kodak")
        results["kodak"] = coverage_stats(ac_kodak)
        results["kodak"]["n_images"] = len(kodak_paths)
        print(f"  Kodak: {results['kodak']['pct_outside']:.3f}% outside [-1.5, 1.5]")

    # ── ImageNet ───────────────────────────────────────────────────────────────
    inet_dir = Path(args.imagenet_dir)
    all_inet = []
    for cls_dir in sorted(inet_dir.iterdir()):
        for p in sorted(cls_dir.glob("*.JPEG")):
            all_inet.append(p)

    step = max(1, len(all_inet) // args.n_imagenet)
    inet_paths = all_inet[::step][: args.n_imagenet]
    print(f"ImageNet: sampling {len(inet_paths)} / {len(all_inet)} images")

    ac_inet = collect_ac(inet_paths, "ImageNet")
    results["imagenet"] = coverage_stats(ac_inet)
    results["imagenet"]["n_images"] = len(inet_paths)
    print(f"  ImageNet: {results['imagenet']['pct_outside']:.3f}% outside [-1.5, 1.5]")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AC coefficient coverage (proxy bin range: [-1.5, 1.5])")
    print("=" * 60)
    for dataset, stats in results.items():
        if dataset == "combined":
            continue
        print(f"\n  {dataset.upper()}")
        print(f"    Images:            {stats.get('n_images', '?')}")
        print(f"    Total AC coefs:    {stats['total_coefficients']:,}")
        print(f"    Outside [-1.5,1.5]: {stats['pct_outside']:.3f}%")
        print(f"    1st/5th percentile: {stats['p1']} / {stats['p5']}")
        print(f"    95th/99th pct:      {stats['p95']} / {stats['p99']}")
        print(f"    Range:             [{stats['min']}, {stats['max']}]")
        print(f"    Std dev:           {stats['std']}")

    print("\nImplementation note: out-of-range coefficients are soft-assigned")
    print("to the nearest boundary bin (Gaussian kernel weight decays to ~0")
    print("beyond 3σ from boundary, so boundary bins absorb tail weight).")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
