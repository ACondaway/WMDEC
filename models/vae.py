import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """Wrapper around a pretrained VAE for encoding/decoding images to/from latent space."""

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.scaling_factor = 0.18215

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            x: (B, 3, H, W) image tensor in [-1, 1]

        Returns:
            latent: (B, 4, H/8, W/8)
        """
        posterior = self.vae.encode(x).latent_dist
        latent = posterior.sample() * self.scaling_factor
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            latent: (B, 4, H/8, W/8)

        Returns:
            image: (B, 3, H, W) in [-1, 1]
        """
        latent = latent / self.scaling_factor
        image = self.vae.decode(latent).sample
        return image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
