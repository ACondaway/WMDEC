"""
SD3.5 Medium Semantic Conditioning Pipeline.

Wires together:
    - Frozen Qwen Visual Encoder          → dense patch features (B, 64, 2560)
    - Trainable PerceiverResampler        → semantic latent tokens (B, 32, 1024)
    - Trainable ResidualMLPAdaptor        → pseudo text tokens (B, 32, 4096) + pooled (B, 2048)
    - Frozen SD3.5 MMDiT Transformer      → denoising (flow matching)
    - Frozen SD3.5 VAE                    → encode/decode images
    - Trainable TextureControlBranch      → AdaLN modulation via pooled_projections (Phase 2)

Training phases
---------------
Phase 1 — Semantic Alignment:
    Trainable : PerceiverResampler, ResidualMLPAdaptor
    Frozen    : everything else (incl. TextureControlBranch disabled)

Phase 2 — Texture Control:
    Trainable : TextureControlBranch + Phase 1 modules (continue training)
    Frozen    : SD3.5 MMDiT, VAE, Qwen

Loss
----
    L_diff  = MSE(model_pred, velocity_target)   flow-matching loss
    L_align = 1 - cos(z_qwen_pred, z_qwen_orig)  semantic alignment (optional)
    L       = L_diff + λ * L_align

SD3.5 dimensions
----------------
    VAE latent channels  : 16
    MMDiT hidden size    : 1536  (24 heads × 64 dim)
    joint_attention_dim  : 4096  (T5 text conditioning dim)
    pooled_projection_dim: 2048
    Flow timestep range  : [0, 1]  (scaled to [0, 1000] for transformer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from models.qwen_visual_encoder import QwenVisualEncoder
from models.perceiver_resampler import PerceiverResampler
from models.residual_mlp_adaptor import ResidualMLPAdaptor
from models.texture_control import TextureControlBranch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def _sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample continuous timesteps t ∈ [0, 1] for flow matching."""
    return torch.rand(batch_size, device=device, generator=generator)


def _add_flow_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Forward noising for flow matching:
        x_t = (1 - t) * x_0 + t * noise
    """
    t_ = t.view(-1, 1, 1, 1)
    return (1.0 - t_) * latents + t_ * noise


def _flow_velocity_target(
    latents: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Velocity target for flow matching:
        v = noise - x_0
    """
    return noise - latents


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SD35SemanticModel(nn.Module):
    """
    SD3.5 Medium with Qwen semantic + optional texture control conditioning.

    Args
    ----
    transformer_path   : HF path or local dir for SD3.5 MMDiT weights
    vae_path           : HF path or local dir for SD3.5 VAE weights
    qwen_encoder_ckpt  : path to standalone Qwen visual encoder .pt
    resampler_cfg      : kwargs forwarded to PerceiverResampler
    adaptor_cfg        : kwargs forwarded to ResidualMLPAdaptor
    control_cfg        : kwargs forwarded to TextureControlBranch (None = Phase 1)
    sem_loss_weight    : λ for semantic alignment loss (0.0 = disabled)
    """

    def __init__(
        self,
        transformer_path: str,
        vae_path: str,
        qwen_encoder_ckpt: str,
        resampler_cfg: dict | None = None,
        adaptor_cfg: dict | None = None,
        control_cfg: dict | None = None,
        sem_loss_weight: float = 0.0,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Frozen backbone
        # ------------------------------------------------------------------

        self.transformer = SD3Transformer2DModel.from_pretrained(
            transformer_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        _freeze(self.transformer)

        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        _freeze(self.vae)
        self.vae_scale = self.vae.config.scaling_factor

        self.qwen = QwenVisualEncoder.from_standalone(qwen_encoder_ckpt)
        _freeze(self.qwen)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            transformer_path,
            subfolder="scheduler",
        )

        # ------------------------------------------------------------------
        # Trainable semantic condition branch (Phase 1)
        # ------------------------------------------------------------------

        self.resampler = PerceiverResampler(**(resampler_cfg or {}))
        self.adaptor = ResidualMLPAdaptor(**(adaptor_cfg or {}))

        # ------------------------------------------------------------------
        # Optional texture control branch (Phase 2)
        # ------------------------------------------------------------------

        self.control: TextureControlBranch | None = None
        if control_cfg is not None:
            self.control = TextureControlBranch(**control_cfg)

        self.sem_loss_weight = sem_loss_weight

    # ------------------------------------------------------------------
    # Convenience: freeze / unfreeze helpers for phase switching
    # ------------------------------------------------------------------

    def set_phase1(self) -> None:
        """Freeze everything except Perceiver + Adaptor."""
        if self.control is not None:
            _freeze(self.control)
        for p in self.resampler.parameters():
            p.requires_grad_(True)
        for p in self.adaptor.parameters():
            p.requires_grad_(True)

    def set_phase2(self) -> None:
        """Unfreeze Phase 1 modules + TextureControlBranch."""
        if self.control is None:
            raise RuntimeError(
                "TextureControlBranch not initialised — pass control_cfg to constructor."
            )
        for p in self.resampler.parameters():
            p.requires_grad_(True)
        for p in self.adaptor.parameters():
            p.requires_grad_(True)
        for p in self.control.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a normalised image [-1, 1] to a scaled VAE latent.

        Args:
            image: (B, 3, H, W) in [-1, 1]
        Returns:
            latent: (B, 16, H/8, W/8)
        """
        dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        return dist.sample() * self.vae_scale

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a scaled latent to a normalised image [-1, 1]."""
        return self.vae.decode(latent / self.vae_scale).sample

    def encode_qwen(self, image_pixels: torch.Tensor) -> torch.Tensor:
        """
        Wrapper that calls QwenVisualEncoder.encode_images with no-grad.

        Args:
            image_pixels: (B, 3, H, W) in [-1, 1] — will be converted to PIL
                          inside QwenVisualEncoder
        Returns:
            patch_features: (B, 64, 2560)
        """
        with torch.no_grad():
            return self.qwen.encode_images(image_pixels)

    # ------------------------------------------------------------------
    # Condition computation (shared by train and inference)
    # ------------------------------------------------------------------

    def compute_condition(
        self,
        qwen_features: torch.Tensor,
        latent: torch.Tensor | None = None,
        timestep_t: torch.Tensor | None = None,
        force_drop_control: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (encoder_hidden_states, pooled_projections) for the MMDiT.

        Args:
            qwen_features       : (B, 64, 2560)  from Qwen encoder
            latent              : (B, 16, H/8, W/8)  VAE latent (Phase 2 only)
            timestep_t          : (B,) float [0, 1]  (Phase 2 only)
            force_drop_control  : zero out texture control (for uncond inference)

        Returns:
            enc_hidden  : (B, 32, 4096)   pseudo text tokens
            pooled_proj : (B, 2048)
        """
        tokens = self.resampler(qwen_features)         # (B, 32, 1024)
        enc_hidden, pooled = self.adaptor(tokens)       # (B, 32, 4096), (B, 2048)

        if self.control is not None and latent is not None and timestep_t is not None:
            ctrl_residual = self.control(
                latent, timestep_t, force_drop=force_drop_control
            )
            pooled = pooled + ctrl_residual

        return enc_hidden, pooled

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        qwen_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Single training step forward pass.

        Args:
            image          : (B, 3, H, W) normalised to [-1, 1]
            qwen_features  : (B, 64, 2560) pre-extracted Qwen patch features

        Returns:
            dict with keys:
                loss_diff  : flow matching MSE loss
                loss_sem   : semantic alignment loss (0 if disabled)
                loss       : total loss = loss_diff + λ * loss_sem
        """
        B, device = image.shape[0], image.device

        # Encode image to latent (frozen VAE)
        with torch.no_grad():
            latent = self.encode_image_to_latent(image)   # (B, 16, H/8, W/8)

        # Sample flow timestep
        t = _sample_flow_timesteps(B, device)             # (B,) in [0, 1]

        # Add noise
        noise = torch.randn_like(latent)
        noisy_latent = _add_flow_noise(latent, noise, t)
        velocity_target = _flow_velocity_target(latent, noise)

        # Condition tokens — CFG dropout applied externally in train script
        enc_hidden, pooled = self.compute_condition(
            qwen_features,
            latent=latent,
            timestep_t=t,
        )

        # SD3.5 transformer expects timestep in [0, 1000]
        timestep_scaled = (t * 1000.0).long()

        # Forward through frozen MMDiT
        model_pred = self.transformer(
            hidden_states=noisy_latent.to(self.transformer.dtype),
            timestep=timestep_scaled,
            encoder_hidden_states=enc_hidden.to(self.transformer.dtype),
            pooled_projections=pooled.to(self.transformer.dtype),
            return_dict=False,
        )[0]

        # Diffusion loss (flow matching MSE)
        loss_diff = F.mse_loss(
            model_pred.float(),
            velocity_target.float(),
        )

        # Semantic alignment loss (optional)
        loss_sem = torch.tensor(0.0, device=device)
        if self.sem_loss_weight > 0.0:
            # Reconstruct x_0 from predicted velocity
            t_ = t.view(-1, 1, 1, 1)
            pred_latent = noisy_latent - t_ * model_pred.float()
            pred_image = self.decode_latent(pred_latent.to(self.vae.dtype)).float()

            with torch.no_grad():
                z_pred = self.qwen.encode_images(
                    pred_image.clamp(-1, 1)
                ).float()  # (B, 64, 2560)

            z_orig = qwen_features.float()

            # Cosine similarity over flattened patch tokens
            z_pred_flat = z_pred.reshape(B, -1)
            z_orig_flat = z_orig.reshape(B, -1)
            loss_sem = 1.0 - F.cosine_similarity(z_pred_flat, z_orig_flat).mean()

        loss = loss_diff + self.sem_loss_weight * loss_sem

        return {
            "loss": loss,
            "loss_diff": loss_diff.detach(),
            "loss_sem": loss_sem.detach(),
        }

    # ------------------------------------------------------------------
    # Inference sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        qwen_features: torch.Tensor,
        latent: torch.Tensor | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 5.0,
        height: int = 768,
        width: int = 768,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Generate an image from Qwen semantic features.

        Args:
            qwen_features      : (B, 64, 2560)
            latent             : (B, 16, H/8, W/8) input latent for texture control
                                  (Phase 2 only; None = Phase 1)
            num_inference_steps: denoising steps
            guidance_scale     : CFG scale
            height, width      : output resolution
            generator          : optional RNG for reproducibility

        Returns:
            images: (B, 3, H, W) in [-1, 1]
        """
        device = qwen_features.device
        dtype = self.transformer.dtype
        B = qwen_features.shape[0]

        # Conditional tokens
        enc_cond, pooled_cond = self.compute_condition(
            qwen_features,
            latent=latent,
            timestep_t=torch.ones(B, device=device),  # t=1 for init
            force_drop_control=False,
        )

        # Unconditional tokens (zero)
        enc_uncond = torch.zeros_like(enc_cond)
        pooled_uncond = torch.zeros_like(pooled_cond)

        # Start from pure noise
        latent_h, latent_w = height // 8, width // 8
        noisy = torch.randn(
            B,
            self.transformer.config.in_channels,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for step_t in self.scheduler.timesteps:

            # Normalised t for timestep-aware strength
            t_norm = (step_t / 1000.0).clamp(0.0, 1.0)
            t_batch = t_norm.expand(B)

            # Update control with current timestep
            if self.control is not None and latent is not None:
                _, pooled_cond_t = self.compute_condition(
                    qwen_features,
                    latent=latent,
                    timestep_t=t_batch,
                    force_drop_control=False,
                )
            else:
                pooled_cond_t = pooled_cond

            timestep_tensor = step_t.expand(B)

            # Conditional prediction
            pred_cond = self.transformer(
                hidden_states=noisy,
                timestep=timestep_tensor,
                encoder_hidden_states=enc_cond,
                pooled_projections=pooled_cond_t,
                return_dict=False,
            )[0]

            # Unconditional prediction
            pred_uncond = self.transformer(
                hidden_states=noisy,
                timestep=timestep_tensor,
                encoder_hidden_states=enc_uncond,
                pooled_projections=pooled_uncond,
                return_dict=False,
            )[0]

            # CFG
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            noisy = self.scheduler.step(pred, step_t, noisy, return_dict=False)[0]

        return self.decode_latent(noisy)

    # ------------------------------------------------------------------
    # Parameter summary
    # ------------------------------------------------------------------

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
