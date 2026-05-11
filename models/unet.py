"""
UNet wrappers for two backbone options:

  SDXLUNet  — stabilityai/stable-diffusion-xl-base-1.0  (~2.6B params)
              Conditioning: encoder_hidden_states (B,S,2048) + pooled (B,1280) + time_ids (B,6)
              Prediction:   ε-prediction

  LDMUNet   — stabilityai/stable-diffusion-2-1-base     (~865M params)
              Gated decoupled image cross-attention design:
                • encoder_hidden_states = frozen empty-text embedding (CLIP prior anchor)
                • image_tokens fed to UNetImageConditioner (gated per-block image cross-attn)
              Prediction:   v-prediction
              Latent @ 448px: 56×56

Select via config["model"]["backbone"]: "sdxl" | "sd21"
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

_SDXL_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out"]
_SD21_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# SDXL backbone (unchanged)
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
        return self.unet(
            x, t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": pooled_proj, "time_ids": time_ids},
        ).sample


# ---------------------------------------------------------------------------
# SD 2.1-base backbone — gated decoupled image cross-attention
# ---------------------------------------------------------------------------

class LDMUNet(nn.Module):
    """
    SD 2.1-base UNet with gated decoupled image cross-attention.

    Architecture:
      • base UNet: frozen SD 2.1 weights
      • text encoder: frozen CLIPTextModel, produces empty-text prior anchor
      • UNetImageConditioner: one GatedImageCrossAttention per transformer block
          gate=0 at init → model starts as pure SD 2.1, gates open gradually

    forward(x, t, image_tokens) → v_pred
      image_tokens: (B, N, cross_attn_dim) from ImageAdapter
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1-base",
        gradient_checkpointing: bool = True,
        image_tokens_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=torch.float32,
        )
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # ---- Frozen text encoder for empty-text prior anchor ----
        from transformers import CLIPTextModel, CLIPTokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # Pre-compute and cache empty text embedding: (1, 77, 1024)
        with torch.no_grad():
            empty_input = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
                truncation=True,
            )
            empty_emb = self.text_encoder(empty_input.input_ids).last_hidden_state
        self.register_buffer("empty_text_emb", empty_emb)   # (1, 77, 1024)

        # ---- Gated image cross-attention conditioner ----
        from models.image_cross_attention import UNetImageConditioner
        self.conditioner = UNetImageConditioner(
            self.unet,
            image_tokens_dim=image_tokens_dim,
            num_heads=num_heads,
        )
        print(
            f"LDMUNet: {self.conditioner.num_gated_layers()} gated layers, "
            f"{self.conditioner.trainable_param_count() / 1e6:.1f}M conditioner params"
        )

    # ------------------------------------------------------------------
    # LoRA methods (compatible with LoRA training mode on base UNet attn)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Frozen-mode helper
    # ------------------------------------------------------------------

    def freeze_base(self):
        """
        Freeze base UNet + text encoder.
        Conditioner (gated layers + gates) remains trainable.
        """
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # self.conditioner params stay trainable (default requires_grad=True)
        trainable = sum(p.numel() for p in self.conditioner.parameters())
        print(f"  freeze_base(): conditioner has {trainable / 1e6:.2f}M trainable params")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, t, image_tokens):
        """
        Args:
            x:            (B, 4, H, W)  noisy latent
            t:            (B,)          timestep
            image_tokens: (B, N, image_tokens_dim)  from ImageAdapter

        Returns:
            v_pred: (B, 4, H, W)
        """
        B = x.shape[0]
        empty_text = self.empty_text_emb.expand(B, -1, -1)
        # Inject image tokens; hooks add gated cross-attn during unet.forward()
        self.conditioner.set_image_tokens(image_tokens)
        pred = self.unet(x, t, encoder_hidden_states=empty_text).sample
        self.conditioner.set_image_tokens(None)
        return pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_unet(config: dict, gradient_checkpointing: bool = True):
    """Return the correct UNet for config["model"]["backbone"]."""
    backbone   = config["model"].get("backbone", "sdxl")
    model_name = config["model"]["unet_name"]
    if backbone == "sd21":
        return LDMUNet(
            model_name,
            gradient_checkpointing,
            image_tokens_dim=config["model"].get("cross_attn_dim", 1024),
            num_heads=config["model"].get("num_heads", 8),
        )
    return SDXLUNet(model_name, gradient_checkpointing)
