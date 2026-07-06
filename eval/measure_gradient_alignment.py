"""
measure_gradient_alignment.py — Empirically measure the cosine similarity
between the DCT proxy gradient and the JPEG rate gradient in AC-energy space.

JPEG uses the same 8x8 DCT-II structure as the proxy, making it the most
direct empirical comparison for gradient alignment.

For each image and each of the 63 AC frequency bands, compute:
  g_proxy[k] = ∂(proxy BPP) / ∂(energy in band k)  via centered FD
  g_jpeg[k]  = ∂(JPEG BPP)  / ∂(energy in band k)  via centered FD

Then report cosine similarity between the two 63-d vectors per image, plus
the per-band gradient ratio g_proxy[k] / g_jpeg[k].

Usage:
    python eval/measure_gradient_alignment.py \\
        --kodak_dir data/kodak \\
        --quality 50 \\
        --eps 0.3 \\
        --output results/gradient_alignment.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _EVAL_DIR.parent / "train"

# ---------------------------------------------------------------------------
# DCT helpers (inlined to avoid importing direct_optimize, which pulls lpips)
# ---------------------------------------------------------------------------


def _dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float32, device=device)
    m = torch.arange(n, dtype=torch.float32, device=device)
    D = torch.cos(math.pi / n * k.unsqueeze(1) * (m.unsqueeze(0) + 0.5))
    D[0] *= math.sqrt(1.0 / n)
    D[1:] *= math.sqrt(2.0 / n)
    return D


def _ac_band_positions(block_size: int = 8) -> List[tuple[int, int]]:
    """Raster-scan positions in block, excluding DC at (0,0). Length = bs²-1."""
    return [(i, j) for i in range(block_size) for j in range(block_size) if not (i == 0 and j == 0)]


# ---------------------------------------------------------------------------
# Proxy BPP (reimplemented to avoid importing lpips via direct_optimize)
# ---------------------------------------------------------------------------


def bpp_proxy_local(
    img: torch.Tensor,
    block_size: int = 8,
    n_bins: int = 64,
    sigma_bin: float | None = None,
) -> torch.Tensor:
    """Differentiable BPP proxy.  img: (B, C, H, W) in [0, 1]. Returns (B,)."""
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

    ac_mask = torch.ones(bs, bs, dtype=torch.bool, device=img.device)
    ac_mask[0, 0] = False
    ac = coeffs[:, ac_mask].reshape(B, C, -1)

    centers = torch.linspace(-DCT_RANGE, DCT_RANGE, n_bins, device=img.device)
    d_exp = ac.unsqueeze(-1)
    c_exp = centers.view(1, 1, 1, n_bins)
    weights = torch.exp(-0.5 * ((d_exp - c_exp) / sigma_bin) ** 2)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

    hist = weights.mean(dim=2).clamp(min=1e-9)
    entropy_bits = -(hist * hist.log()).sum(dim=-1) / math.log(2)
    return entropy_bits.mean(dim=1)


# Try importing bpp_proxy from train.direct_optimize; fall back to local.
try:
    sys.path.insert(0, str(_EVAL_DIR.parent))
    from ace import bpp_proxy as _bpp_proxy_imported  # type: ignore

    def bpp_proxy(img: torch.Tensor) -> torch.Tensor:
        return _bpp_proxy_imported(img)
except Exception:

    def bpp_proxy(img: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
        return bpp_proxy_local(img)


# ---------------------------------------------------------------------------
# JPEG BPP
# ---------------------------------------------------------------------------


def _jpeg_bpp(img_tensor: torch.Tensor, quality: int = 50) -> float:
    """Encode a (C, H, W) float tensor as JPEG (via Pillow), return BPP.

    JPEG uses the same 8x8 DCT-II structure as the proxy, making this the
    most direct empirical comparison for gradient alignment.
    """
    arr = img_tensor.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return (buf.tell() * 8) / (h * w)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(path: Path, max_side: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return T.ToTensor()(img)  # (3, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# DCT / IDCT block transforms on (C, H, W) images
# ---------------------------------------------------------------------------


def _to_blocks(
    img: torch.Tensor, D: torch.Tensor, bs: int
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """Return DCT blocks and padding info.

    Returns: (coeffs, block_means, Hp, Wp, nH, nW)
      coeffs     : (C*nH*nW, bs, bs) DCT-II coefficients (AC+DC)
      block_means: (C, nH, nW, 1, 1)
    """
    C, H, W = img.shape
    pad_h = (bs - H % bs) % bs
    pad_w = (bs - W % bs) % bs
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    Hp, Wp = img.shape[1], img.shape[2]
    nH, nW = Hp // bs, Wp // bs

    blocks = img.reshape(C, nH, bs, nW, bs).permute(0, 1, 3, 2, 4)
    # (C, nH, nW, bs, bs)
    means = blocks.mean(dim=(-2, -1), keepdim=True)
    blocks_dc = blocks - means
    flat = blocks_dc.reshape(-1, bs, bs)
    coeffs = D @ flat @ D.t()  # (C*nH*nW, bs, bs)
    return coeffs, means, Hp, Wp, nH, nW


def _from_blocks(
    coeffs: torch.Tensor,
    means: torch.Tensor,
    D: torch.Tensor,
    C: int,
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    nH: int,
    nW: int,
    bs: int,
) -> torch.Tensor:
    """Inverse: reconstruct (C, H, W) image from DCT blocks + means."""
    flat_rec = D.t() @ coeffs @ D  # IDCT: D^T @ coeff @ D
    blocks_rec = flat_rec.reshape(C, nH, nW, bs, bs) + means
    img_rec = blocks_rec.permute(0, 1, 3, 2, 4).reshape(C, Hp, Wp)[:, :H, :W]
    return img_rec


# ---------------------------------------------------------------------------
# Per-band perturbation and gradient estimation
# ---------------------------------------------------------------------------


def _perturb_image(
    img: torch.Tensor,
    band_idx: int,
    alpha: float,
    D: torch.Tensor,
    bs: int,
    band_positions: List[tuple[int, int]],
) -> torch.Tensor:
    """Scale AC band `band_idx` by `alpha`, return reconstructed image."""
    C, H, W = img.shape
    coeffs, means, Hp, Wp, nH, nW = _to_blocks(img, D, bs)
    bi, bj = band_positions[band_idx]
    coeffs = coeffs.clone()
    coeffs[:, bi, bj] = coeffs[:, bi, bj] * alpha
    img_rec = _from_blocks(coeffs, means, D, C, H, W, Hp, Wp, nH, nW, bs)
    return img_rec.clamp(0.0, 1.0)


def _band_energy(
    img: torch.Tensor,
    band_idx: int,
    D: torch.Tensor,
    bs: int,
    band_positions: List[tuple[int, int]],
) -> float:
    """Mean squared energy in AC band k across all blocks and channels."""
    coeffs, _, _, _, _, _ = _to_blocks(img, D, bs)
    bi, bj = band_positions[band_idx]
    return coeffs[:, bi, bj].pow(2).mean().item()


def compute_gradients(
    img: torch.Tensor,
    quality: int,
    eps: float,
    D: torch.Tensor,
    bs: int,
    band_positions: List[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute g_proxy and g_jpeg for all 63 AC bands via centered FD.

    Returns:
        g_proxy : (63,) tensor
        g_jpeg  : (63,) tensor
    """
    n_bands = len(band_positions)
    g_proxy = torch.zeros(n_bands)
    g_jpeg = torch.zeros(n_bands)

    for k in tqdm(range(n_bands), desc="  bands", leave=False):
        E_k = _band_energy(img, k, D, bs, band_positions)
        if E_k < 1e-12:
            # Band has effectively zero energy; gradient undefined/zero.
            continue

        img_plus = _perturb_image(img, k, 1.0 + eps, D, bs, band_positions)
        img_minus = _perturb_image(img, k, 1.0 - eps, D, bs, band_positions)

        # Proxy gradient (differentiable, computed in float)
        with torch.no_grad():
            bpp_plus_proxy = bpp_proxy(img_plus.unsqueeze(0)).item()
            bpp_minus_proxy = bpp_proxy(img_minus.unsqueeze(0)).item()

        # JPEG gradient (real non-differentiable codec, same 8x8 DCT structure)
        bpp_plus_jpeg = _jpeg_bpp(img_plus, quality=quality)
        bpp_minus_jpeg = _jpeg_bpp(img_minus, quality=quality)

        denom = 2.0 * eps * E_k

        if not (math.isnan(bpp_plus_proxy) or math.isnan(bpp_minus_proxy)):
            g_proxy[k] = (bpp_plus_proxy - bpp_minus_proxy) / denom

        if not (math.isnan(bpp_plus_jpeg) or math.isnan(bpp_minus_jpeg)):
            g_jpeg[k] = (bpp_plus_jpeg - bpp_minus_jpeg) / denom

    return g_proxy, g_jpeg


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return float("nan")
    return (a @ b / (norm_a * norm_b)).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure cosine similarity between DCT proxy and JPEG gradients."
    )
    ap.add_argument("--kodak_dir", required=True, help="Path to Kodak image directory.")
    ap.add_argument("--quality", type=int, default=50, help="JPEG quality (default 50).")
    ap.add_argument("--eps", type=float, default=0.3, help="FD step size (default 0.3).")
    ap.add_argument("--output", default="results/gradient_alignment.json")
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cpu")
    bs = args.block_size
    D = _dct_matrix(bs, device)
    band_positions = _ac_band_positions(bs)  # length 63
    n_bands = len(band_positions)

    kodak_dir = Path(args.kodak_dir)
    image_paths = sorted(
        list(kodak_dir.glob("*.png"))
        + list(kodak_dir.glob("*.PNG"))
        + list(kodak_dir.glob("*.jpg"))
        + list(kodak_dir.glob("*.jpeg"))
        + list(kodak_dir.glob("*.JPEG"))
    )
    if not image_paths:
        print(f"ERROR: no images found in {kodak_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {kodak_dir}")

    per_image_results = []
    all_cos_sims: list[float] = []
    all_g_proxy = torch.zeros(n_bands)
    all_g_jpeg = torch.zeros(n_bands)
    valid_counts = torch.zeros(n_bands)

    for path in tqdm(image_paths, desc="Images"):
        try:
            img = load_image(path, max_side=args.max_side)
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue

        g_proxy, g_jpeg = compute_gradients(
            img, args.quality, args.eps, D, bs, band_positions, device
        )

        cos_sim = cosine_similarity(g_proxy, g_jpeg)
        all_cos_sims.append(cos_sim)

        valid_mask = ~(g_proxy.isnan() | g_jpeg.isnan())
        all_g_proxy += torch.where(valid_mask, g_proxy, torch.zeros_like(g_proxy))
        all_g_jpeg += torch.where(valid_mask, g_jpeg, torch.zeros_like(g_jpeg))
        valid_counts += valid_mask.float()

        per_image_results.append(
            {
                "image": path.name,
                "cosine_similarity": cos_sim if not math.isnan(cos_sim) else None,
                "g_proxy": g_proxy.tolist(),
                "g_jpeg": g_jpeg.tolist(),
            }
        )
        print(f"  {path.name}: cos_sim = {cos_sim:.4f}")

    # Aggregate statistics
    valid_cos = [c for c in all_cos_sims if not math.isnan(c)]
    mean_cos = float(sum(valid_cos) / len(valid_cos)) if valid_cos else float("nan")
    var_cos = (
        float(sum((c - mean_cos) ** 2 for c in valid_cos) / len(valid_cos))
        if len(valid_cos) > 1
        else 0.0
    )
    std_cos = math.sqrt(var_cos)

    # Per-band mean gradient ratio
    mean_g_proxy = all_g_proxy / valid_counts.clamp(min=1)
    mean_g_jpeg = all_g_jpeg / valid_counts.clamp(min=1)

    ratio = torch.where(
        mean_g_jpeg.abs() > 1e-12,
        mean_g_proxy / mean_g_jpeg,
        torch.full_like(mean_g_proxy, float("nan")),
    )

    band_labels = [f"({i},{j})" for i, j in band_positions]
    per_band = [
        {
            "band": k,
            "position": band_labels[k],
            "mean_g_proxy": mean_g_proxy[k].item(),
            "mean_g_jpeg": mean_g_jpeg[k].item(),
            "ratio_proxy_over_jpeg": ratio[k].item(),
        }
        for k in range(n_bands)
    ]

    results = {
        "config": {
            "quality": args.quality,
            "eps": args.eps,
            "max_side": args.max_side,
            "block_size": bs,
            "n_bands": n_bands,
        },
        "summary": {
            "n_images": len(per_image_results),
            "mean_cosine_similarity": mean_cos,
            "std_cosine_similarity": std_cos,
            "sign_consistency_pct": float(
                ((mean_g_proxy > 0) == (mean_g_jpeg > 0)).float().mean().item() * 100
            ),
            "per_image_cosine_similarities": [r["cosine_similarity"] for r in per_image_results],
        },
        "per_band": per_band,
        "per_image": per_image_results,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Gradient alignment: proxy vs JPEG")
    print("=" * 60)
    print(f"Images:               {len(per_image_results)}")
    print(f"JPEG quality:         {args.quality}")
    print(f"FD epsilon:           {args.eps}")
    print(f"Mean cosine sim:      {mean_cos:.4f} ± {std_cos:.4f}")
    # Sign consistency: what fraction of bands have the same sign?
    signs_match = ((mean_g_proxy > 0) == (mean_g_jpeg > 0)).float().mean().item()
    print(f"Sign consistency:     {signs_match * 100:.1f}% of bands")
    print("\nPer-band ratio g_proxy / g_jpeg (first 10 bands):")
    for k in range(min(10, n_bands)):
        r_val = ratio[k].item()
        r_str = f"{r_val:.4f}" if not math.isnan(r_val) else "nan"
        print(f"  band {k:2d} {band_labels[k]:6s}: {r_str}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
