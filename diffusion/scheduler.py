import torch
import numpy as np


class DDPMScheduler:
    """DDPM noise scheduler for training and sampling."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule: str = "scaled_linear",
    ):
        self.num_timesteps = num_timesteps

        if schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_0 at timestep t.

        Args:
            x_0: (B, C, H, W) clean latent
            t: (B,) timestep indices
            noise: (B, C, H, W) Gaussian noise

        Returns:
            x_t: (B, C, H, W) noisy latent
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def p_sample_step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Posterior mean
        coeff1 = beta_t * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - alpha_cumprod_t)
        coeff2 = (1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t)
        mean = coeff1 * pred_x0 + coeff2 * x_t

        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * noise
        return mean
