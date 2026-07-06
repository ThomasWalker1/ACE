"""PyTorch implementation of the AC-entropy (ACE) rate proxy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def dct_matrix(n: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the orthonormal DCT-II matrix of size ``n``."""
    k = torch.arange(n, dtype=dtype, device=device)
    m = torch.arange(n, dtype=dtype, device=device)
    matrix = torch.cos(math.pi / n * k.unsqueeze(1) * (m.unsqueeze(0) + 0.5))
    matrix[0] *= math.sqrt(1.0 / n)
    matrix[1:] *= math.sqrt(2.0 / n)
    return matrix


def bpp_proxy(
    image: Tensor,
    block_size: int = 8,
    n_bins: int = 64,
    sigma_bin: float | None = None,
    value_range: float = 1.5,
) -> Tensor:
    """Estimate coded rate from soft-histogram entropy of block-DCT AC coefficients.

    Args:
        image: Tensor shaped ``(batch, channels, height, width)`` in ``[0, 1]``.
        block_size: Spatial DCT block size. The paper uses 8.
        n_bins: Number of soft-histogram bins. The paper uses 64.
        sigma_bin: Gaussian assignment width. Defaults to ``3 * value_range / n_bins``.
        value_range: Symmetric histogram support ``[-value_range, value_range]``.

    Returns:
        One ACE value per image, shaped ``(batch,)``.
    """
    if image.ndim != 4:
        raise ValueError(f"expected a 4D BCHW tensor, got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor")
    if block_size < 2 or n_bins < 2:
        raise ValueError("block_size and n_bins must both be at least 2")

    if sigma_bin is None:
        sigma_bin = 2.0 * value_range / n_bins * 1.5

    batch, channels, height, width = image.shape
    pad_h = (-height) % block_size
    pad_w = (-width) % block_size
    if pad_h or pad_w:
        # Reflection requires padding smaller than the corresponding input axis.
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)

    n_h = image.shape[2] // block_size
    n_w = image.shape[3] // block_size
    blocks = image.reshape(batch, channels, n_h, block_size, n_w, block_size).permute(
        0, 1, 2, 4, 3, 5
    )
    blocks = blocks - blocks.mean(dim=(-2, -1), keepdim=True)

    matrix = dct_matrix(block_size, image.device, image.dtype)
    flat = blocks.reshape(-1, block_size, block_size)
    coefficients = matrix @ flat @ matrix.t()

    ac_mask = torch.ones(block_size, block_size, dtype=torch.bool, device=image.device)
    ac_mask[0, 0] = False
    ac = coefficients[:, ac_mask].reshape(batch, channels, -1)

    centers = torch.linspace(
        -value_range, value_range, n_bins, device=image.device, dtype=image.dtype
    )
    log_weights = -0.5 * ((ac.unsqueeze(-1) - centers) / sigma_bin).square()
    probabilities = log_weights.softmax(dim=-1).mean(dim=2).clamp_min(1e-9)
    entropy_bits = -(probabilities * probabilities.log2()).sum(dim=-1)
    return entropy_bits.mean(dim=1)
