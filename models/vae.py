import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """
    Wrapper around a pretrained VAE for encoding/decoding images to/from latent space.

    Scaling factors (read automatically from model config):
        SD 1.x / 2.x : 0.18215
        SDXL          : 0.13025
    """

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.scaling_factor = self.vae.config.scaling_factor

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [-1, 1] → latent: (B, 4, H/8, W/8)"""
        return self.vae.encode(x).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (B, 4, H/8, W/8) → image: (B, 3, H, W) in [-1, 1]"""
        return self.vae.decode(latent / self.scaling_factor).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
