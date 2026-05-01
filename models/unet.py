"""
UNet wrappers for two backbone options:

  SDXLUNet  — stabilityai/stable-diffusion-xl-base-1.0  (~2.6B params)
              Conditioning: encoder_hidden_states (B,S,2048) + pooled (B,1280) + time_ids (B,6)
              Prediction:   ε-prediction

  LDMUNet   — stabilityai/stable-diffusion-2-1-base     (~865M params)
              Conditioning: encoder_hidden_states (B,S,1024) only
              Prediction:   v-prediction
              Latent @ 448px: 56×56  (vs SDXL 128×128 at 1024px)

Select via config["model"]["backbone"]: "sdxl" | "sd21"
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

_SDXL_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out"]
_SD21_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# SDXL backbone (original)
# ---------------------------------------------------------------------------

class SDXLUNet(nn.Module):
    """
    SDXL UNet wrapper.
    forward(x, t, encoder_hidden_states, pooled_proj, time_ids) → ε_pred
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=torch.float32,
        )
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def setup_lora(self, rank=64, alpha=64, target_modules=None):
        from peft import LoraConfig, get_peft_model
        for p in self.unet.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=target_modules or _SDXL_LORA_TARGETS,
            lora_dropout=0.0, bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_cfg)
        trainable, total = self.unet.get_nb_trainable_parameters()
        print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total UNet params")

    def save_lora(self, output_dir):
        self.unet.save_pretrained(output_dir)

    def load_lora(self, lora_dir):
        self.unet.load_adapter(lora_dir, adapter_name="default")

    def forward(self, x, t, encoder_hidden_states, pooled_proj, time_ids):
        """
        Args:
            x:                     (B, 4, H, W)
            t:                     (B,)
            encoder_hidden_states: (B, S, 2048)
            pooled_proj:           (B, 1280)
            time_ids:              (B, 6)
        Returns: ε_pred (B, 4, H, W)
        """
        return self.unet(
            x, t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": pooled_proj, "time_ids": time_ids},
        ).sample


# ---------------------------------------------------------------------------
# SD 2.1-base backbone (lighter)
# ---------------------------------------------------------------------------

class LDMUNet(nn.Module):
    """
    SD 2.1-base UNet wrapper.
    forward(x, t, encoder_hidden_states) → v_pred
    No pooled embeddings, no time_ids.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1-base",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=torch.float32,
        )
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def setup_lora(self, rank=64, alpha=64, target_modules=None):
        from peft import LoraConfig, get_peft_model
        for p in self.unet.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=target_modules or _SD21_LORA_TARGETS,
            lora_dropout=0.0, bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_cfg)
        trainable, total = self.unet.get_nb_trainable_parameters()
        print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total UNet params")

    def save_lora(self, output_dir):
        self.unet.save_pretrained(output_dir)

    def load_lora(self, lora_dir):
        self.unet.load_adapter(lora_dir, adapter_name="default")

    def forward(self, x, t, encoder_hidden_states):
        """
        Args:
            x:                     (B, 4, H, W)
            t:                     (B,)
            encoder_hidden_states: (B, S, 1024)
        Returns: v_pred (B, 4, H, W)
        """
        return self.unet(x, t, encoder_hidden_states=encoder_hidden_states).sample


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_unet(config: dict, gradient_checkpointing: bool = True):
    """Return the correct UNet for config["model"]["backbone"]."""
    backbone = config["model"].get("backbone", "sdxl")
    model_name = config["model"]["unet_name"]
    if backbone == "sd21":
        return LDMUNet(model_name, gradient_checkpointing)
    return SDXLUNet(model_name, gradient_checkpointing)
