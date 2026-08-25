# ACE: An Analytical Rate Proxy for Compression-aware Optimization

ACE is a training-free, differentiable estimate of transform-coded image rate. It computes the
soft-histogram entropy of the AC coefficients from an 8×8 block DCT-II, providing a lightweight
gradient signal for compression-aware optimization without a learned entropy model.

This repository contains the implementation and experiment code accompanying the paper. The
central implementation is [`ace/proxy.py`](ace/proxy.py); the remaining scripts reproduce the
paper's calibration, robustness, split-inference, and diffusion-guidance experiments.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the environment:

```bash
git clone https://github.com/ThomasWalker1/ACE.git
cd ACE
uv sync --extra dev
```

`uv sync` creates `.venv` using Python 3.11 (from `.python-version`). Run commands through
`uv run`; activating the environment is optional. Diffusion experiments need the larger optional
dependency set:

```bash
uv sync --extra dev --extra diffusion
```

Calibration and HEVC experiments also require an `ffmpeg` build with `libx265` support:

```bash
ffmpeg -hide_banner -encoders | grep libx265
```

If `ffmpeg` is not on `PATH`, set `FFMPEG_PATH=/absolute/path/to/ffmpeg`. A CUDA-capable GPU is
strongly recommended; pass `--device cpu` where supported for small smoke runs.

## Quick start

The public API accepts a float tensor in BCHW layout with values in `[0, 1]` and returns one ACE
value per image:

```python
import torch
from ace import bpp_proxy

image = torch.rand(2, 3, 224, 224, requires_grad=True)
rate = bpp_proxy(image)
rate.mean().backward()
print(rate, image.grad.shape)
```

Run the unit tests with:

```bash
uv run pytest
```

## Data

Datasets and checkpoints are ignored by Git. The scripts expect these standard layouts:

```text
data/
├── kodak/kodim01.png ... kodim24.png
├── imagenet/{train,val}/n01440764/*.JPEG
├── cifar-10-batches-py/{data_batch_1,...,test_batch}
├── CIFAR-10-C/{gaussian_noise.npy,...,labels.npy}
└── coco_val/{prompts.json,real_images/00000.png,...}
```

Download Kodak and prepare the 5,000-image COCO reference set as follows:

```bash
uv run python data/download_kodak.py
uv run --extra diffusion python data/prepare_coco.py --n_images 5000
```

ImageNet must be obtained separately. Download CIFAR-10 and CIFAR-10-C from their official
sources and extract the Python-format archives into the paths above.

## Reproducing the paper

All commands below run from the repository root and write into ignored `results/` or
`checkpoints/` directories. Replace the ImageNet paths if your data lives elsewhere.

### 1. Proxy validation

```bash
uv run python eval/eval_proxy_calibration.py --kodak_dir data/kodak
uv run python eval/measure_ac_coverage.py \
  --kodak_dir data/kodak --imagenet_dir data/imagenet/val
uv run python eval/measure_gradient_alignment.py --kodak_dir data/kodak
```

These commands reproduce the per-CRF correlation, coefficient-support coverage, and gradient
alignment results from the proxy analysis.

### 2. Adversarial purification

The main four-condition sweep used in the purification figure is:

```bash
uv run python eval/eval_lam010_sweep.py \
  --data_dir data/imagenet/val --n_images 1000
```

The supporting spectral and frequency-selection ablations are:

```bash
uv run python eval/eval_freq_analysis.py \
  --data_dir data/imagenet/val --n_images 200
uv run python eval/eval_hf_proxy_purification.py \
  --data_dir data/imagenet/val --n_images 500
```

`eval/eval_adv_purification.py`, `eval/eval_adv_strength_sweep.py`, and `eval/eval_bpda.py`
provide the individual baseline, strength-sweep, and adaptive-attack evaluations.

### 3. ACEAug on CIFAR-10-C

Train every condition for seeds 0, 1, and 2:

```bash
for condition in baseline proxy_aug augmix augmix_proxy; do
  for seed in 0 1 2; do
    uv run python eval/train_augmix_cifar.py \
      --data_dir data/cifar-10-batches-py \
      --condition "$condition" --seed "$seed"
  done
done
```

Evaluate a checkpoint pair (repeat for each seed and for the AugMix pair):

```bash
uv run python eval/eval_cifar_c.py \
  --checkpoint_baseline checkpoints/aceaug/baseline/seed0/best.pt \
  --checkpoint_proxy_aug checkpoints/aceaug/proxy_aug/seed0/best.pt \
  --cifar_dir data/cifar-10-batches-py \
  --cifar_c_dir data/CIFAR-10-C \
  --output results/aceaug/baseline_vs_proxy_seed0.json
```

### 4. Compression-surviving adversarial perturbations

```bash
uv run python eval/eval_adv_compression_survival.py \
  --data_dir data/imagenet/val --n_images 200 \
  --eps 4 8 --mu 0.3 1.0 3.0
```

### 5. Compressible features for split inference

Train the control and ACE-regularized models:

```bash
for lambda in 0.0 0.1; do
  uv run python train/train_split_proxy.py \
    --train_dir data/imagenet/train --val_dir data/imagenet/val \
    --split_point layer2 --lam "$lambda" --epochs 20
done
```

Then evaluate the pretrained baseline or a trained checkpoint:

```bash
uv run python eval/eval_split_inference.py \
  --data_dir data/imagenet/val --split_point layer2 \
  --checkpoint checkpoints/split_proxy/layer2_lam0.1_v2/best.pth \
  --tag proxy_lam0.1
```

### 6. Rate-constrained diffusion guidance

The definitive paper run uses Stable Diffusion 1.5, DDIM with 50 steps, guidance on the final 25
steps, 5,000 COCO-val prompts, and HEVC CRF 32:

```bash
uv run --extra diffusion python eval/eval_diffusion_guidance.py \
  --coco_dir data/coco_val \
  --output_dir results/diffusion_guidance \
  --lambdas 0 20 50 100 200 400 800 1600 \
  --n_images 5000 --batch_size 4 --num_steps 50 \
  --guide_last_k 25 --crf 32 --dtype float32
```

This is computationally expensive. Separate lambda values may be launched on different GPUs;
each setting writes to its own subdirectory and can resume completed images.

## Repository layout

```text
ace/       reusable ACE implementation
data/      download/preparation helpers (datasets remain untracked)
eval/      evaluation and application experiments
train/     split-inference training experiment
augmix/    AugMix operations used by ACEAug
tests/     lightweight correctness tests
paper/     manuscript source retained during repository preparation
```

## Citation

If you use this code, please cite *ACE: An Analytical Rate Proxy for Compression-aware
Optimization*. Full BibTeX metadata will be added with the public paper release.