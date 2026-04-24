import torch
import numpy as np
from .scheduler import DDPMScheduler


class DDIMSampler:
    """DDIM sampler for fast inference."""

    def __init__(self, scheduler: DDPMScheduler, num_inference_steps: int = 50, eta: float = 0.0):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute DDIM timestep subsequence
        step_ratio = scheduler.num_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int64)
        self.timesteps = np.flip(self.timesteps).copy()

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple,
        context: torch.Tensor,
        cfg_scale: float = 2.0,
        uncond_context: torch.Tensor = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.

        Args:
            model: UNet model
            shape: (B, C, H, W) output shape
            context: (B, S, C) conditioning tokens
            cfg_scale: classifier-free guidance scale
            uncond_context: (B, S, C) unconditional context for CFG
            device: torch device

        Returns:
            x_0: (B, C, H, W) denoised latent
        """
        if device is None:
            device = next(model.parameters()).device

        x = torch.randn(shape, device=device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        for i, t in enumerate(self.timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            if uncond_context is not None and cfg_scale > 1.0:
                # CFG: run conditional and unconditional
                eps_cond = model(x, t_tensor, context)
                eps_uncond = model(x, t_tensor, uncond_context)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_tensor, context)

            # DDIM update
            alpha_t = alphas_cumprod[t]
            alpha_prev = alphas_cumprod[self.timesteps[i + 1]] if i + 1 < len(self.timesteps) else torch.tensor(1.0, device=device)

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev - self.eta ** 2 * (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) * eps

            if self.eta > 0 and i + 1 < len(self.timesteps):
                noise = torch.randn_like(x)
                sigma = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        return x
