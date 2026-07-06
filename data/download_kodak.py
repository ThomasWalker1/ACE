"""Download the 24 lossless Kodak images used by the ACE calibration experiments."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://r0k.us/graphics/kodak/kodak/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/kodak")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 25):
        name = f"kodim{index:02d}.png"
        destination = output_dir / name
        if destination.exists():
            print(f"[skip] {destination}")
            continue
        print(f"[download] {name}")
        try:
            urllib.request.urlretrieve(BASE_URL + name, destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    images = sorted(output_dir.glob("kodim*.png"))
    if len(images) != 24:
        raise RuntimeError(f"expected 24 Kodak images, found {len(images)}")
    print(f"Kodak is ready in {output_dir}")


if __name__ == "__main__":
    main()

