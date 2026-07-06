"""Prepare COCO-val prompts and reference images for diffusion guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/coco_val")
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    real_dir = output_dir / "real_images"
    real_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    if args.n_images > len(dataset):
        raise ValueError(f"requested {args.n_images} images, dataset contains {len(dataset)}")

    prompts: list[str] = []
    for index in range(args.n_images):
        example = dataset[index]
        prompts.append(example["caption"])
        destination = real_dir / f"{index:05d}.png"
        if not destination.exists():
            image = example["image"].convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BICUBIC
            )
            image.save(destination)
        if (index + 1) % 500 == 0:
            print(f"{index + 1}/{args.n_images}", flush=True)

    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    print(f"COCO references are ready in {output_dir}")


if __name__ == "__main__":
    main()

