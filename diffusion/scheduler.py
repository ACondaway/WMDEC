"""
DDPM scheduler wrapping diffusers' DDPMScheduler with SDXL's pretrained noise schedule.

Loading from the pretrained SDXL checkpoint ensures the noise schedule matches exactly
what the UNet was trained with.
"""

import torch
from diffusers import DDPMScheduler as _DDPMScheduler


class DDPMScheduler:
    """
    Thin wrapper around diffusers.DDPMScheduler.
    Loads the schedule from the SDXL pretrained checkpoint by default.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        # Fallback manual params (used only if model_name is None)
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule: str = "scaled_linear",
    ):
        if model_name is not None:
            self._sched = _DDPMScheduler.from_pretrained(
                model_name, subfolder="scheduler"
            )
        else:
            self._sched = _DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=schedule,
                clip_sample=False,
            )

        self.num_timesteps = self._sched.config.num_train_timesteps
        alphas_cumprod = self._sched.alphas_cumprod
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise at timestep t."""
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
