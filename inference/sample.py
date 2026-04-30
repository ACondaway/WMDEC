"""
Inference: reconstruct an image from a pre-extracted Qwen visual embedding.

Usage:
    python inference/sample.py \
        --config configs/base.yaml \
        --checkpoint /path/to/final.pt \
        --embedding /path/to/image_X.0.pt \
        --output output.png \
        --cfg_scale 2.0 \
        --steps 50
"""

import argparse
import os
import sys
import yaml
import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adapter import ImageAdapter
from models.unet import SDXLUNet
from models.vae import VAEWrapper
from diffusion.sampler import DDIMSampler
from training.cfg import build_uncond_context
from training.train import make_time_ids


def load_models(config: dict, checkpoint_path: str, device: torch.device):
    img_adapter = ImageAdapter(
        qwen_dim=config["model"]["qwen_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        pooled_proj_dim=config["model"]["pooled_proj_dim"],
        num_tokens=config["model"]["num_img_tokens"],
        num_heads=config["model"]["num_heads"],
    ).to(device)

    unet = SDXLUNet(
        model_name=config["model"]["unet_name"],
        gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(ckpt["unet"])
    img_adapter.load_state_dict(ckpt["img_adapter"])

    unet.eval()
    img_adapter.eval()
    return unet, img_adapter


@torch.no_grad()
def generate(
    embedding_path: str,
    config: dict,
    checkpoint_path: str,
    output_path: str = "output.png",
    cfg_scale: float = 2.0,
    num_steps: int = 50,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet, img_adapter = load_models(config, checkpoint_path, device)
    vae = VAEWrapper(config["model"]["unet_name"]).to(device)

    # Load pre-extracted patch embeddings
    data = torch.load(embedding_path, map_location="cpu", weights_only=True)
    z_img = data["z_img"].float().unsqueeze(0).to(device)   # (1, N_patches, D)

    # Conditioned tokens
    tokens, pooled = img_adapter(z_img)  # (1, N_tokens, 2048), (1, 1280)

    # Unconditional tokens for CFG
    uncond_tokens, uncond_pooled = build_uncond_context(
        1,
        config["model"]["num_img_tokens"],
        config["model"]["cross_attn_dim"],
        config["model"]["pooled_proj_dim"],
        device,
    )

    resolution = config["data"]["resolution"]
    time_ids = make_time_ids(1, resolution, device)
    latent_size = resolution // 8

    sampler = DDIMSampler(
        model_name=config["model"]["unet_name"],
        num_inference_steps=num_steps,
    )

    latent = sampler.sample(
        unet,
        shape=(1, 4, latent_size, latent_size),
        encoder_hidden_states=tokens,
        pooled_proj=pooled,
        time_ids=time_ids,
        cfg_scale=cfg_scale,
        uncond_hidden_states=uncond_tokens,
        uncond_pooled_proj=uncond_pooled,
        device=device,
    )

    image_out = vae.decode(latent)
    image_out = ((image_out + 1) / 2).clamp(0, 1)
    save_image(image_out, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate(
        embedding_path=args.embedding,
        config=config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        cfg_scale=args.cfg_scale,
        num_steps=args.steps,
    )
