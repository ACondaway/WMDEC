"""
SD3.5 Semantic Conditioning — Inference Script.

Phase 1: image → Qwen → Perceiver → MLP → SD3.5 → reconstructed image
Phase 2: image → Qwen → Perceiver → MLP + VAE latent → SD3.5 → reconstructed image

Usage
-----
Phase 1 inference:
    python inference/sample_sd35.py \
        --config    configs/sd35_phase1.yaml \
        --phase1_ckpt outputs/sd35_phase1/phase1_latest.pt \
        --input_dir  /path/to/images \
        --output_dir /path/to/outputs \
        --steps 28 --guidance 5.0

Phase 2 inference (with texture control):
    python inference/sample_sd35.py \
        --config       configs/sd35_phase2.yaml \
        --phase1_ckpt  outputs/sd35_phase2/phase2_latest.pt \
        --use_control \
        --input_dir    /path/to/images \
        --output_dir   /path/to/outputs \
        --steps 28 --guidance 5.0
"""

from __future__ import annotations

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.sd35_model import SD35SemanticModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_image(path: str, resolution: int) -> torch.Tensor:
    """Load a single image → (1, 3, H, W) normalised to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    return _TO_TENSOR(img).unsqueeze(0)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1] for saving."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SD3.5 semantic conditioning inference")
    parser.add_argument("--config",      required=True, help="Training config YAML")
    parser.add_argument("--phase1_ckpt", required=True, help="Checkpoint (phase1 or phase2)")
    parser.add_argument("--input_dir",   required=True, help="Directory of input images")
    parser.add_argument("--output_dir",  required=True, help="Directory to write output images")
    parser.add_argument("--use_control", action="store_true",
                        help="Enable Phase 2 texture control branch")
    parser.add_argument("--steps",     type=int,   default=28,  help="Denoising steps")
    parser.add_argument("--guidance",  type=float, default=5.0, help="CFG scale")
    parser.add_argument("--resolution", type=int,  default=768, help="Output resolution")
    parser.add_argument("--device",    type=str,   default="cuda")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    control_cfg = cfg.get("control", {}) if args.use_control else None

    model = SD35SemanticModel(
        transformer_path=cfg["transformer_path"],
        vae_path=cfg["vae_path"],
        qwen_encoder_ckpt=cfg["qwen_encoder_ckpt"],
        resampler_cfg=cfg.get("resampler", {}),
        adaptor_cfg=cfg.get("adaptor", {}),
        control_cfg=control_cfg,
        sem_loss_weight=0.0,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu", weights_only=False)
    model.resampler.load_state_dict(ckpt["resampler"])
    model.adaptor.load_state_dict(ckpt["adaptor"])
    if args.use_control and "control" in ckpt:
        model.control.load_state_dict(ckpt["control"])
    print(f"Loaded checkpoint from {args.phase1_ckpt}")

    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Process images
    # ------------------------------------------------------------------

    image_paths = sorted(
        p for p in Path(args.input_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"Processing {len(image_paths)} image(s) ...")

    for img_path in image_paths:

        image = load_image(str(img_path), args.resolution).to(device)

        # Extract Qwen features (runs frozen encoder)
        with torch.no_grad():
            qwen_features = model.qwen.encode_images(image)   # (1, 64, 2560)

        # Optional: VAE latent for Phase 2 texture control
        latent = None
        if args.use_control and model.control is not None:
            with torch.no_grad():
                latent = model.encode_image_to_latent(image)  # (1, 16, H/8, W/8)

        # Sample
        with torch.no_grad():
            output = model.sample(
                qwen_features=qwen_features,
                latent=latent,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            )  # (1, 3, H, W)

        # Save side-by-side: input | output
        out_path = os.path.join(args.output_dir, img_path.name)
        comparison = torch.cat([denorm(image.cpu()), denorm(output.cpu())], dim=-1)
        save_image(comparison, out_path)
        print(f"  {img_path.name} → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
