"""
DDIM sampler using diffusers' DDIMScheduler for SDXL inference.
"""

import torch
from diffusers import DDIMScheduler


class DDIMSampler:
    """DDIM inference sampler wrapping diffusers' DDIMScheduler."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps)
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple,
        encoder_hidden_states: torch.Tensor,
        pooled_proj: torch.Tensor,
        time_ids: torch.Tensor,
        cfg_scale: float = 2.0,
        uncond_hidden_states: torch.Tensor = None,
        uncond_pooled_proj: torch.Tensor = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM with optional CFG.

        Args:
            model:                  SDXLUNet
            shape:                  (B, 4, H, W)
            encoder_hidden_states:  (B, S, 2048) conditioned tokens
            pooled_proj:            (B, 1280) conditioned pooled projection
            time_ids:               (B, 6)
            cfg_scale:              CFG guidance scale
            uncond_hidden_states:   (B, S, 2048) unconditional tokens (all-zeros)
            uncond_pooled_proj:     (B, 1280) unconditional pooled (all-zeros)
            device:                 target device

        Returns:
            x_0: (B, 4, H, W) denoised latent
        """
        if device is None:
            device = next(model.parameters()).device

        x = torch.randn(shape, device=device)

        for t in self.scheduler.timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            if uncond_hidden_states is not None and cfg_scale > 1.0:
                eps_cond = model(x, t_batch, encoder_hidden_states, pooled_proj, time_ids)
                eps_uncond = model(x, t_batch, uncond_hidden_states, uncond_pooled_proj, time_ids)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_batch, encoder_hidden_states, pooled_proj, time_ids)

            x = self.scheduler.step(eps, t, x, eta=self.eta).prev_sample

        return x
