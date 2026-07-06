"""
eval_proxy_calibration.py — Validate bpp_proxy against actual HEVC BPP on Kodak.

For each of the 24 Kodak images at CRFs {22,27,32,37,42,47}:
  - Compute bpp_proxy on the raw image
  - Encode with libx265 and measure actual BPP from file size

Reports Pearson r, RMSE, and saves scatter data to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

_FFMPEG = os.environ.get("FFMPEG_PATH", shutil.which("ffmpeg") or "ffmpeg")


def _load(path, device):
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def _hevc_bpp(img_path: Path, crf: int) -> float:
    uid = uuid.uuid4().hex
    enc = Path(f"/tmp/calib_{uid}.mkv")
    try:
        subprocess.run(
            [
                _FFMPEG,
                "-y",
                "-i",
                str(img_path),
                "-c:v",
                "libx265",
                "-x265-params",
                f"pools=none:frame-threads=1:crf={crf}",
                "-an",
                str(enc),
            ],
            check=True,
            capture_output=True,
        )
        size = enc.stat().st_size
        img = Image.open(img_path)
        w, h = img.size
        return (size * 8) / (h * w)
    finally:
        enc.unlink(missing_ok=True)


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (sx * sy + 1e-12)


def rmse(xs, ys):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kodak_dir", default="data/kodak")
    p.add_argument("--output_dir", default="results/proxy_calibration")
    p.add_argument("--crfs", nargs="+", type=int, default=[22, 27, 32, 37, 42, 47])
    p.add_argument("--bpp_bins", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(args.kodak_dir).glob("*.png"))
    print(f"Images: {len(img_paths)}, CRFs: {args.crfs}")

    records = []
    proxy_vals = []
    hevc_vals = []

    for img_path in tqdm(img_paths, desc="Images"):
        img = _load(img_path, args.device)
        with torch.no_grad():
            proxy = bpp_proxy(img, n_bins=args.bpp_bins).item()

        for crf in args.crfs:
            hevc = _hevc_bpp(img_path, crf)
            records.append(
                {
                    "image": img_path.name,
                    "crf": crf,
                    "proxy_bpp": round(proxy, 6),
                    "hevc_bpp": round(hevc, 6),
                }
            )
            proxy_vals.append(proxy)
            hevc_vals.append(hevc)
            print(f"  {img_path.name}  CRF={crf:2d}  proxy={proxy:.4f}  hevc={hevc:.4f}")

    r = pearson_r(proxy_vals, hevc_vals)
    rms = rmse(proxy_vals, hevc_vals)
    print(f"\nPearson r = {r:.4f},  RMSE = {rms:.4f} bpp")

    # Per-image summary (proxy is constant per image, hevc varies by CRF)
    per_image = {}
    for rec in records:
        name = rec["image"]
        if name not in per_image:
            per_image[name] = {"proxy_bpp": rec["proxy_bpp"], "hevc_bpp_by_crf": {}}
        per_image[name]["hevc_bpp_by_crf"][str(rec["crf"])] = rec["hevc_bpp"]

    summary = {
        "n_images": len(img_paths),
        "crfs": args.crfs,
        "pearson_r": round(r, 6),
        "rmse_bpp": round(rms, 6),
        "records": records,
        "per_image": per_image,
    }
    out_path = out / "proxy_calibration.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
