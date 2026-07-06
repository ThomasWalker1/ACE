"""
eval_diffusion_guidance.py — Experiment 4: rate-constrained guidance in diffusion sampling.

Adds `lambda * grad_{x_t} dct_rate_proxy(x0_pred)` to each DDIM step of a
Stable Diffusion txt2img sampler, steering generated images toward lower
HEVC-coded bitrate at generation time (no training, no learned rate model).

Guidance mechanism (per denoising step, only within the last `--guide_last_k`
steps to bound cost — see the "Practical note" in EXPERIMENTS.md #4):
    1. x_t.requires_grad_(True)
    2. noise_pred = UNet(x_t, t, cond)   (CFG-combined)
    3. x0_pred = scheduler.step(noise_pred, t, x_t).pred_original_sample
    4. img = VAE.decode(x0_pred)                       (differentiable)
    5. rate = bpp_proxy(img)
    6. grad = d(rate)/d(x_t)
    7. x_{t-1} = scheduler.step(noise_pred, t, x_t).prev_sample - lambda * grad

Metrics per lambda, computed once all images are generated:
    - HEVC bpp at CRF=32 (ffmpeg/libx265) — does guidance reduce coding cost?
    - FID vs. real COCO reference images (clean-fid)
    - CLIP score (image-prompt cosine similarity, CLIP ViT-B/32)

Usage:
    python eval/eval_diffusion_guidance.py \\
        --coco_dir data/coco_val \\
        --output_dir results/diffusion_guidance \\
        --lambdas 0.0 0.01 0.05 0.10 0.20 \\
        --n_images 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace import bpp_proxy

_FFMPEG = os.environ.get("FFMPEG_PATH", shutil.which("ffmpeg") or "ffmpeg")
_MODEL_ID = "runwayml/stable-diffusion-v1-5"


# ---------------------------------------------------------------------------
# Guided DDIM sampling
# ---------------------------------------------------------------------------


class GuidedSDSampler:
    """Stable Diffusion txt2img with optional DCT-proxy rate guidance."""

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        self.device = device
        self.dtype = dtype
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.scaling = self.vae.config.scaling_factor
        for m in (self.vae, self.unet, self.text_encoder):
            for p in m.parameters():
                p.requires_grad_(False)

    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(tok.input_ids.to(self.device))[0]
        return embed

    @torch.no_grad()
    def _uncond_embed(self, batch_size: int) -> torch.Tensor:
        tok = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    def generate(
        self, prompts: list[str], lam: float, guide_last_k: int, seed: int, image_size: int = 512
    ) -> list[Image.Image]:
        B = len(prompts)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        text_embed = self._encode_prompts(prompts)
        uncond_embed = self._uncond_embed(B)
        embed = torch.cat([uncond_embed, text_embed], dim=0)

        latents = torch.randn(
            (B, self.unet.config.in_channels, image_size // 8, image_size // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        guided_steps = (
            set(self.scheduler.timesteps[-guide_last_k:].tolist())
            if lam > 0 and guide_last_k > 0
            else set()
        )

        for t in self.scheduler.timesteps:
            do_guide = lam > 0 and float(t) in guided_steps
            latents_in = latents.detach().requires_grad_(do_guide)

            model_in = torch.cat([latents_in] * 2)
            model_in = self.scheduler.scale_model_input(model_in, t)
            with torch.set_grad_enabled(do_guide):
                noise_pred = self.unet(model_in, t, encoder_hidden_states=embed).sample
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                step_out = self.scheduler.step(noise_pred, t, latents_in)
                prev_sample = step_out.prev_sample

                if do_guide:
                    x0_pred = step_out.pred_original_sample
                    img = self.vae.decode(x0_pred.float() / self.scaling).sample
                    img = (img + 1.0) / 2.0
                    rate = bpp_proxy(img.clamp(0, 1)).mean()
                    grad = torch.autograd.grad(rate, latents_in)[0]
                    prev_sample = prev_sample - lam * grad.to(prev_sample.dtype)

            latents = prev_sample.detach()

        with torch.no_grad():
            images = self.vae.decode(latents / self.scaling).sample
            images = ((images + 1.0) / 2.0).clamp(0, 1)
        return [
            Image.fromarray((img.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
            for img in images
        ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def hevc_bpp(img_path: Path, crf: int = 32) -> float:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.hevc"
        subprocess.run(
            [
                _FFMPEG,
                "-y",
                "-i",
                str(img_path),
                "-c:v",
                "libx265",
                "-crf",
                str(crf),
                "-preset",
                "medium",
                "-x265-params",
                "log-level=none",
                str(out),
            ],
            capture_output=True,
            check=False,
        )
        if not out.exists():
            return float("nan")
        return (out.stat().st_size * 8) / (h * w)


def clip_score(image_paths: list[Path], prompts: list[str], device: str) -> float:
    from transformers import CLIPModel, CLIPProcessor

    model = (
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        .to(device)
        .eval()
    )
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    bs = 16
    for i in range(0, len(image_paths), bs):
        batch_paths = image_paths[i : i + bs]
        batch_prompts = prompts[i : i + bs]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = proc(
            text=batch_prompts, images=images, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        img_emb = F.normalize(out.image_embeds, dim=-1)
        txt_emb = F.normalize(out.text_embeds, dim=-1)
        sims = (img_emb * txt_emb).sum(dim=-1)
        scores.extend(sims.cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return float(sum(scores) / len(scores)) * 100.0


def compute_fid(gen_dir: Path, real_dir: Path, device: str) -> float:
    from cleanfid import fid

    return float(
        fid.compute_fid(str(gen_dir), str(real_dir), device=device, mode="clean", verbose=False)
    )


def mean_proxy_bpp(image_paths: list[Path], device: str) -> float:
    import torchvision.transforms.functional as TF

    vals = []
    for p in image_paths:
        img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            vals.append(bpp_proxy(img).item())
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--coco_dir",
        required=True,
        help="Dir with prompts.json and real_images/ (see data/prepare_coco.py).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10, 0.20])
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument(
        "--guide_last_k",
        type=int,
        default=20,
        help="Only apply proxy guidance in the last K DDIM steps (cost control).",
    )
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--crf", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_dir = coco_dir / "real_images"

    with open(coco_dir / "prompts.json") as f:
        all_prompts = json.load(f)
    prompts = all_prompts[: args.n_images]
    print(f"Loaded {len(prompts)} prompts from {coco_dir}")

    print(f"Loading {_MODEL_ID} ({args.dtype}) ...")
    sampler = GuidedSDSampler(
        _MODEL_ID, args.device, dtype, num_steps=args.num_steps, guidance_scale=args.guidance_scale
    )

    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for lam in args.lambdas:
        tag = f"lam{lam:.3f}"
        img_dir = output_dir / tag / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(img_dir.glob("*.png"))

        if len(existing) < len(prompts):
            print(f"\n=== lambda={lam}  (guide_last_k={args.guide_last_k}) ===")
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i : i + args.batch_size]
                out_paths = [img_dir / f"{i + j:05d}.png" for j in range(len(batch))]
                if all(p.exists() for p in out_paths):
                    continue
                imgs = sampler.generate(
                    batch,
                    lam=lam,
                    guide_last_k=args.guide_last_k,
                    seed=args.seed + i,
                    image_size=args.image_size,
                )
                for path, img in zip(out_paths, imgs):
                    img.save(path)
                print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

        image_paths = sorted(img_dir.glob("*.png"))
        print(f"Scoring lambda={lam} ...")
        bpps = [hevc_bpp(p, crf=args.crf) for p in image_paths]
        bpps = [b for b in bpps if b == b]  # drop NaN
        mean_bpp = sum(bpps) / len(bpps)
        proxy_bpp = mean_proxy_bpp(image_paths, args.device)
        clip = clip_score(image_paths, prompts[: len(image_paths)], args.device)
        fid_val = compute_fid(img_dir, real_dir, args.device)

        results[tag] = {
            "lambda": lam,
            "n_images": len(image_paths),
            "hevc_bpp_mean": mean_bpp,
            "proxy_bpp_mean": proxy_bpp,
            "fid": fid_val,
            "clip_score": clip,
        }
        results_path.write_text(json.dumps(results, indent=2))
        print(
            f"  hevc_bpp={mean_bpp:.4f}  proxy_bpp={proxy_bpp:.4f}  "
            f"fid={fid_val:.2f}  clip={clip:.2f}"
        )

    print("\n--- Summary ---")
    print(f"{'lambda':>8}  {'hevc_bpp':>9}  {'proxy_bpp':>10}  {'fid':>8}  {'clip':>7}")
    for tag, r in sorted(results.items(), key=lambda kv: kv[1]["lambda"]):
        print(
            f"{r['lambda']:>8.3f}  {r['hevc_bpp_mean']:>9.4f}  "
            f"{r['proxy_bpp_mean']:>10.4f}  {r['fid']:>8.2f}  {r['clip_score']:>7.2f}"
        )
    print(f"\nSaved → {results_path}")


if __name__ == "__main__":
    main()
