"""
DDPM scheduler wrapping diffusers' DDPMScheduler.

Supports two prediction types matching the two backbones:
    ε-prediction (backbone="sdxl") — standard noise prediction
    v-prediction (backbone="sd21") — SD 2.1-base's velocity prediction

v-prediction math:
    v = α_t * ε − σ_t * x_0       (training target)
    x_0 = α_t * x_t − σ_t * v     (x_0 reconstruction)
where α_t = sqrt(ᾱ_t), σ_t = sqrt(1 − ᾱ_t).
"""

import torch
from diffusers import DDPMScheduler as _DDPMScheduler


class DDPMScheduler:

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule: str = "scaled_linear",
    ):
        if model_name is not None:
            self._sched = _DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
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
        """Forward diffusion: x_t = α_t * x_0 + σ_t * ε"""
        alpha = self.sqrt_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        return alpha * x_0 + sigma * noise

    def get_v_target(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        v-prediction target for SD 2.1: v = α_t * ε − σ_t * x_0
        Use as training target instead of noise when backbone="sd21".
        """
        alpha = self.sqrt_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1)
        return alpha * noise - sigma * x_0

    def predict_x0_from_v(self, v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Recover x_0 from v-prediction: x_0 = α_t * x_t − σ_t * v
        Used for semantic loss and visualization when backbone="sd21".
        """
        alpha = self.sqrt_alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        return alpha * x_t - sigma * v_pred

    def predict_x0_from_eps(self, eps_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Recover x_0 from ε-prediction: x_0 = (x_t − σ_t * ε) / α_t
        Used for semantic loss and visualization when backbone="sdxl".
        """
        alpha = self.sqrt_alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        return (x_t - sigma * eps_pred) / alpha

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
