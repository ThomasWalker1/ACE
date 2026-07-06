"""
eval_split_inference.py — Experiment 3: Compressible Intermediate Features for Split Inference.

Evaluates accuracy vs. feature bits-per-spatial-position (bpsp) at a ResNet-50 split point.
Features are compressed via JPEG (channels normalized to [0,255], grouped in 3s) at varying
quality levels. Measures accuracy of the downstream head with compressed features.

Usage:
    # Baseline (pretrained, no regularization)
    python eval/eval_split_inference.py \
        --data_dir data/imagenet/val \
        --split_point layer2 --n_images 1000 --device cuda

    # Proxy-regularized checkpoint
    python eval/eval_split_inference.py \
        --data_dir data/imagenet/val \
        --checkpoint checkpoints/split_proxy/layer2_lam0.1/best.pth \
        --split_point layer2 --n_images 1000 --device cuda
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_split_proxy import SplitResNet50 as _SplitResNet50

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


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


# ── Model split ───────────────────────────────────────────────────────────────


class SplitResNet50:
    """Wraps ResNet-50 with a configurable split point. Encoder runs to split,
    decoder runs from split to logits."""

    def __init__(self, model: torch.nn.Module, split_point: str):
        self.model = model
        self.split_point = split_point

    def encode(self, x: Tensor) -> Tensor:
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        if self.split_point == "layer1":
            return x
        x = m.layer2(x)
        if self.split_point == "layer2":
            return x
        x = m.layer3(x)
        return x  # layer3

    def decode(self, x: Tensor) -> Tensor:
        m = self.model
        if self.split_point == "layer1":
            x = m.layer2(x)
        if self.split_point in ("layer1", "layer2"):
            x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = x.flatten(1)
        return m.fc(x)


# ── Feature compression ───────────────────────────────────────────────────────


def jpeg_compress_features(feat: Tensor, quality: int) -> tuple[Tensor, float]:
    """
    JPEG-compress feature maps by treating groups of 3 channels as an RGB image.
    Each channel is normalized to [0,255] independently (per-image per-channel min-max).

    Returns:
        compressed feature tensor (same shape as input),
        mean bpsp (bits per spatial position, averaged over batch).
    """
    B, C, H, W = feat.shape
    device = feat.device
    feat_np = feat.detach().cpu().numpy()

    out_feat = np.zeros_like(feat_np)
    total_bits = 0.0

    # Pad channels to multiple of 3
    pad = (3 - C % 3) % 3
    if pad:
        feat_np = np.concatenate([feat_np, np.zeros((B, pad, H, W), dtype=feat_np.dtype)], axis=1)

    for b in range(B):
        img_bits = 0
        for g in range(0, C, 3):
            group = feat_np[b, g : g + 3]  # (3, H, W)
            # Per-channel min-max normalize to [0, 255]
            mins = group.reshape(3, -1).min(axis=1)[:, None, None]
            maxs = group.reshape(3, -1).max(axis=1)[:, None, None]
            scale = maxs - mins
            scale[scale < 1e-8] = 1.0
            group_norm = np.clip((group - mins) / scale * 255, 0, 255).astype(np.uint8)

            pil_img = Image.fromarray(group_norm.transpose(1, 2, 0), mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            img_bits += buf.tell() * 8
            buf.seek(0)

            rec_np = np.array(Image.open(buf)).transpose(2, 0, 1).astype(np.float32)
            rec = rec_np / 255.0 * scale + mins  # denormalize

            end = min(g + 3, C)
            out_feat[b, g:end] = rec[: end - g]

        total_bits += img_bits

    # bpsp = total bits / (B * H * W)
    bpsp = total_bits / (B * H * W)
    return torch.from_numpy(out_feat[:, :C]).to(device), bpsp


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    split_model,
    loader: DataLoader,
    jpeg_qualities: list[int],
    device: torch.device,
    pre_normalised: bool = False,
) -> dict:
    """
    For each JPEG quality level, compress features and measure top-1 accuracy.
    Also measures uncompressed accuracy and per-quality bpsp.
    """
    results = {q: {"correct": 0, "total": 0, "bpsp": 0.0, "n_bpsp": 0} for q in jpeg_qualities}
    results["uncompressed"] = {"correct": 0, "total": 0}

    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        if not pre_normalised:
            x = (x - MEAN.to(device)[None, :, None, None]) / STD.to(device)[None, :, None, None]

        # Encode to feature space
        feat = split_model.encode(x)

        # Uncompressed accuracy
        logits = split_model.decode(feat)
        pred = logits.argmax(dim=1)
        results["uncompressed"]["correct"] += (pred == y).sum().item()
        results["uncompressed"]["total"] += y.size(0)

        # Compressed accuracy at each quality
        for q in jpeg_qualities:
            feat_comp, bpsp = jpeg_compress_features(feat, q)
            logits_comp = split_model.decode(feat_comp)
            pred_comp = logits_comp.argmax(dim=1)
            results[q]["correct"] += (pred_comp == y).sum().item()
            results[q]["total"] += y.size(0)
            results[q]["bpsp"] += bpsp
            results[q]["n_bpsp"] += 1

    # Convert to accuracy
    out = {
        "uncompressed_acc": results["uncompressed"]["correct"]
        / results["uncompressed"]["total"]
        * 100
    }
    for q in jpeg_qualities:
        r = results[q]
        out[f"q{q}"] = {
            "acc": r["correct"] / r["total"] * 100,
            "bpsp": r["bpsp"] / r["n_bpsp"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results/split_inference")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to proxy-regularized checkpoint. Omit for pretrained baseline.",
    )
    parser.add_argument("--split_point", default="layer2", choices=["layer1", "layer2", "layer3"])
    parser.add_argument("--n_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--jpeg_qualities", type=int, nargs="+", default=[10, 20, 30, 50, 70, 90])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="baseline", help="Label for this run in output file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Loading ResNet-50...")
    if args.checkpoint:
        split_model = _SplitResNet50(args.split_point)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        split_model.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        split_model = SplitResNet50(base, args.split_point)
    if args.checkpoint:
        split_model.eval().to(device)
    else:
        split_model.model.eval()
        split_model.model.to(device)

    if args.checkpoint:
        # Match the normalisation used during training
        import torchvision.datasets as tvds

        full_ds = tvds.ImageFolder(
            args.data_dir,
            transform=T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        )
        step = max(1, len(full_ds) // args.n_images)
        dataset = torch.utils.data.Subset(
            full_ds, list(range(0, len(full_ds), step))[: args.n_images]
        )
        pre_normalised = True
    else:
        dataset = ImageNetSubset(args.data_dir, args.n_images)
        pre_normalised = False

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Evaluating split at {args.split_point} with JPEG qualities {args.jpeg_qualities}...")
    results = evaluate(
        split_model, loader, args.jpeg_qualities, device, pre_normalised=pre_normalised
    )

    # Print table
    print(f"\nResults — split: {args.split_point}, tag: {args.tag}")
    print(f"Uncompressed accuracy: {results['uncompressed_acc']:.1f}%")
    print(f"\n{'Quality':>10} {'Acc (%)':>10} {'bpsp':>10}")
    print("-" * 35)
    for q in args.jpeg_qualities:
        r = results[f"q{q}"]
        print(f"{q:>10} {r['acc']:>10.1f} {r['bpsp']:>10.1f}")

    # Save
    out_path = output_dir / f"{args.split_point}_{args.tag}.json"
    payload = {
        "split_point": args.split_point,
        "tag": args.tag,
        "n_images": args.n_images,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
