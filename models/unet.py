"""
SDXL UNet wrapper using the pretrained UNet2DConditionModel from diffusers.

Replaces the custom UNet-from-scratch with a fine-tuned Stable Diffusion XL UNet.
Full fine-tune mode: all weights are trainable.

SDXL UNet conditioning interface:
    encoder_hidden_states : (B, S, 2048)   — sequence conditioning (our Qwen adapter tokens)
    added_cond_kwargs:
        text_embeds  : (B, 1280)            — pooled projection (from ImageAdapter)
        time_ids     : (B, 6)               — [orig_h, orig_w, crop_y, crop_x, tgt_h, tgt_w]
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class SDXLUNet(nn.Module):
    """Thin wrapper around the pretrained SDXL UNet2DConditionModel."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch.float32,
        )
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_proj: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:                     (B, 4, H, W) noisy latent
            t:                     (B,) timesteps
            encoder_hidden_states: (B, S, 2048) adapter tokens
            pooled_proj:           (B, 1280) pooled projection
            time_ids:              (B, 6) size conditioning

        Returns:
            eps_pred: (B, 4, H, W) predicted noise
        """
        out = self.unet(
            x,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": pooled_proj,
                "time_ids": time_ids,
            },
        )
        return out.sample
