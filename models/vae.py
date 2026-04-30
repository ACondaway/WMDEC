import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """
    Wrapper around the SDXL VAE for encoding/decoding images to/from latent space.

    SDXL VAE scaling factor: 0.13025 (different from SD 1.x/2.x's 0.18215).
    """

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.scaling_factor = self.vae.config.scaling_factor  # 0.13025 for SDXL

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [-1, 1]
        Returns:
            latent: (B, 4, H/8, W/8)
        """
        posterior = self.vae.encode(x).latent_dist
        return posterior.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, H/8, W/8)
        Returns:
            image: (B, 3, H, W) in [-1, 1]
        """
        return self.vae.decode(latent / self.scaling_factor).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
