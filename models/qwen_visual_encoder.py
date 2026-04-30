"""
Frozen Qwen3.5 visual encoder (ViT backbone).

Extracts projected patch features from the Qwen3_5ForConditionalGeneration visual tower.

Architecture (from config.json)
--------------------------------
  model_type           : qwen3_5
  ViT hidden_size      : 1024   (internal transformer hidden dim)
  out_hidden_size      : 2560   (merger projection output; what gets stored)
  patch_size           : 16
  temporal_patch_size  : 2
  spatial_merge_size   : 2  →  196 tokens per 448×448 image
  depth                : 24

Token count for 448×448:
  raw patches  : (448/16)² = 784
  after 2×2 merge : 784/4 = 196 tokens

Stored shape per image: (196, 2560)  bfloat16

Usage
-----
# Load from standalone checkpoint (preferred):
enc = QwenVisualEncoder.from_standalone("/path/to/qwen3_5_visual_encoder_4b.pt")

# Or extract directly from the full model:
enc = QwenVisualEncoder.from_full_model("Qwen/Qwen3.5-4B")
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Union

# Fixed output shape contract:
#   input  : 448×448 RGB image
#   output : (196, 2560) patch features
# 448/16 = 28 patches per side → 28×28 = 784 → 2×2 spatial merge → 196 tokens
# 1024 ViT hidden dim → merger projects to out_hidden_size = 2560
_INPUT_SIZE = 448
_TOKENS_PER_IMAGE = 196
_FEATURE_DIM = 2560


class QwenVisualEncoder(nn.Module):
    """Frozen Qwen3.5 visual backbone producing projected patch features."""

    def __init__(self, visual_model, processor, hidden_size: int = _FEATURE_DIM):
        """Use from_full_model() or from_standalone() — do not call directly."""
        super().__init__()
        if hidden_size != _FEATURE_DIM:
            raise ValueError(
                f"hidden_size={hidden_size} but contract requires {_FEATURE_DIM}. "
                "Re-extract the visual encoder from the correct Qwen3.5-4B checkpoint."
            )
        self.visual = visual_model
        self.processor = processor
        self.hidden_size = hidden_size

        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_full_model(cls, model_name: str = "Qwen/Qwen3.5-4B"):
        """Extract the visual tower from the full Qwen3_5ForConditionalGeneration model."""
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

        print(f"Loading {model_name} ...")
        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        # Weight keys are "model.visual.*"; the visual tower lives at model.model.visual.
        visual = full_model.model.visual
        processor = AutoProcessor.from_pretrained(model_name)
        out_hidden_size = getattr(
            full_model.config.vision_config, "out_hidden_size", 2560
        )

        del full_model
        torch.cuda.empty_cache()

        return cls(visual, processor, out_hidden_size)

    @classmethod
    def from_standalone(cls, checkpoint_path: str):
        """
        Load from a standalone .pt saved by extract_qwen_visual_encoder.py.

        Reconstructs the visual model from the saved vision_config dict using
        Qwen3_5VisionModel so transformers handles all weight mapping.
        """
        from transformers import AutoProcessor

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hidden_size = ckpt.get("hidden_size", 2560)

        # Reconstruct vision model from the saved config dict.
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
            vision_cfg = Qwen3_5VisionConfig(**ckpt["vision_config"])
            visual = Qwen3_5VisionModel(vision_cfg)
        except (ImportError, KeyError, TypeError) as e:
            raise RuntimeError(
                f"Failed to reconstruct Qwen3_5VisionModel from checkpoint config: {e}. "
                "Ensure transformers>=4.57.0.dev0 is installed."
            ) from e

        visual.load_state_dict(ckpt["state_dict"])
        visual = visual.to(torch.bfloat16)

        processor = AutoProcessor.from_pretrained(ckpt["processor_name"])

        return cls(visual, processor, hidden_size)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], Image.Image],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Encode PIL images to projected patch features.

        Images are resized to exactly 448×448 before encoding so every call
        produces a fixed token count of 196 and the merger receives the correct
        grid_thw from the image processor.

        Args:
            images: single PIL image or list of PIL images
            device: target device (defaults to the visual model's device)

        Returns:
            patch_features: (B, 196, 2560)  in the dtype of the visual model
        """
        if isinstance(images, Image.Image):
            images = [images]

        B = len(images)

        if device is None:
            device = next(self.visual.parameters()).device

        dtype = next(self.visual.parameters()).dtype

        # Resize to exactly 448×448 so the image_processor produces the expected
        # grid_thw ([T, H, W] patch grid) that the visual merger requires.
        images_448 = [img.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.BICUBIC) for img in images]

        # Call the image_processor directly (bypasses text tokenizer) to get
        # pixel_values and image_grid_thw.
        proc_out = self.processor.image_processor(images=images_448, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(device, dtype=dtype)
        image_grid_thw = proc_out.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        out = self.visual(pixel_values, grid_thw=image_grid_thw)

        # Support both tensor return and dataclass return (BaseModelOutput etc.)
        features = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        # Expected: (B * 196, 2560)

        expected_tokens = B * _TOKENS_PER_IMAGE
        if features.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Visual encoder returned {features.shape[0]} total tokens "
                f"but expected {expected_tokens} (B={B} × {_TOKENS_PER_IMAGE} tokens/image). "
                f"image_grid_thw={image_grid_thw}. "
                f"Ensure all images are resized to 448×448 before encoding."
            )
        if features.shape[1] != self.hidden_size:
            raise RuntimeError(
                f"Visual encoder output dim {features.shape[1]} != expected "
                f"{self.hidden_size} (out_hidden_size). "
                f"The merger projection may not have run — check image_grid_thw."
            )

        return features.view(B, _TOKENS_PER_IMAGE, self.hidden_size)

    def forward(
        self,
        images: Union[List[Image.Image], Image.Image],
        device: torch.device = None,
    ) -> torch.Tensor:
        return self.encode_images(images, device)
