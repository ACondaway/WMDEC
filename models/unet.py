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

# Default LoRA target modules for SDXL UNet attention layers.
_DEFAULT_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out"]


class SDXLUNet(nn.Module):
    """
    Thin wrapper around the pretrained SDXL UNet2DConditionModel.

    Supports two training modes, set via setup_lora() or left at default:
      full — all UNet weights are trainable (default)
      lora — base weights frozen; only LoRA delta weights are trainable
    """

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

    def setup_lora(
        self,
        rank: int = 64,
        alpha: int = 64,
        target_modules: list = None,
    ) -> None:
        """
        Freeze the base UNet and inject LoRA adapters into attention projections.
        Only the LoRA delta weights are trainable afterwards.
        Requires the `peft` package (pip install peft).
        """
        from peft import LoraConfig, get_peft_model

        for p in self.unet.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules or _DEFAULT_LORA_TARGETS,
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_cfg)
        trainable, total = self.unet.get_nb_trainable_parameters()
        print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total UNet params")

    def save_lora(self, output_dir: str) -> None:
        """Save only LoRA delta weights (far smaller than a full checkpoint)."""
        self.unet.save_pretrained(output_dir)

    def load_lora(self, lora_dir: str) -> None:
        """Load LoRA weights onto the frozen base model."""
        self.unet.load_adapter(lora_dir, adapter_name="default")

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
