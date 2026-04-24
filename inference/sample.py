import argparse
import os
import yaml
import torch
from PIL import Image
from torchvision.utils import save_image

from models.adapter import ImageAdapter, TextAdapter
from models.unet import ConditionalUNet
from models.vae import VAEWrapper
from models.siglip_encoder import SigLIPEncoder
from models.text_encoder import T5TextEncoder
from diffusion.scheduler import DDPMScheduler
from diffusion.sampler import DDIMSampler
from training.cfg import build_uncond_context


def load_models(config: dict, checkpoint_path: str, device: torch.device):
    img_adapter = ImageAdapter(
        siglip_dim=config["model"]["siglip_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        num_tokens=config["model"]["num_img_tokens"],
        num_layers=config["model"]["adapter_layers"],
    ).to(device)

    txt_adapter = TextAdapter(
        t5_dim=config["model"]["t5_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
    ).to(device)

    unet = ConditionalUNet(
        in_channels=4,
        model_channels=config["model"]["model_channels"],
        channel_mult=tuple(config["model"]["channel_mult"]),
        context_dim=config["model"]["cross_attn_dim"],
        num_heads=config["model"]["num_heads"],
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(ckpt["unet"])
    img_adapter.load_state_dict(ckpt["img_adapter"])
    txt_adapter.load_state_dict(ckpt["txt_adapter"])

    unet.eval()
    img_adapter.eval()
    txt_adapter.eval()

    return unet, img_adapter, txt_adapter


@torch.no_grad()
def generate(
    image_path: str,
    text: str,
    config: dict,
    checkpoint_path: str,
    output_path: str = "output.png",
    cfg_scale: float = 2.0,
    num_steps: int = 50,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    unet, img_adapter, txt_adapter = load_models(config, checkpoint_path, device)
    vae = VAEWrapper(config["model"]["vae_name"]).to(device)
    siglip = SigLIPEncoder(config["model"]["siglip_model"]).to(device)
    text_encoder = T5TextEncoder().to(device)

    # Encode input image
    image = Image.open(image_path).convert("RGB")
    z_img = siglip.encode_image_from_raw([image])  # (1, D)

    # Encode text
    if text:
        z_txt = text_encoder.encode([text])  # (1, T, C)
    else:
        z_txt = text_encoder.get_empty_embedding(1, device)

    # Adapt
    tokens_img = img_adapter(z_img)
    tokens_txt = txt_adapter(z_txt)
    context = torch.cat([tokens_img, tokens_txt], dim=1)

    # Unconditional context for CFG
    uncond_context = build_uncond_context(
        1, tokens_img.shape[1], tokens_txt.shape[1],
        config["model"]["cross_attn_dim"], device,
    )

    # Sample
    resolution = config["data"]["resolution"]
    latent_size = resolution // 8
    scheduler = DDPMScheduler(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["schedule"],
    )
    sampler = DDIMSampler(scheduler, num_inference_steps=num_steps)

    latent = sampler.sample(
        unet,
        shape=(1, 4, latent_size, latent_size),
        context=context,
        cfg_scale=cfg_scale,
        uncond_context=uncond_context,
        device=device,
    )

    # Decode
    image_out = vae.decode(latent)
    image_out = (image_out + 1) / 2  # [-1,1] -> [0,1]
    image_out = image_out.clamp(0, 1)

    save_image(image_out, output_path)
    print(f"Saved reconstructed image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--text", type=str, default="", help="Optional text description")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate(
        image_path=args.image,
        text=args.text,
        config=config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        cfg_scale=args.cfg_scale,
        num_steps=args.steps,
    )
