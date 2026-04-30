"""
Frozen Qwen3.5 visual encoder (ViT backbone).

Extracts projected patch features from the Qwen3_5ForConditionalGeneration visual tower.

Architecture (from config.json)
--------------------------------
  model_type           : qwen3_5
  ViT hidden_size      : 1024   (internal transformer hidden dim)
  out_hidden_size      : 2560   (projected output = text hidden_size; what we store)
  patch_size           : 16
  spatial_merge_size   : 2 × 2  →  196 tokens per 448×448 image
  temporal_patch_size  : 2
  depth                : 24
  deepstack_visual_indexes : []  (no DeepStack in this model)

Stored shape per image: (196, 2560)  bfloat16

Usage
-----
# One-time extraction from the full model:
enc = QwenVisualEncoder.from_full_model("Qwen/Qwen3.5-4B")

# Subsequent use from standalone checkpoint:
enc = QwenVisualEncoder.from_standalone("/path/to/qwen3_5_visual_encoder_4b.pt")
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Union

# Tokens per image for fixed 448×448 input:
#   (448/16)² patches = 784  →  784 / (2×2 spatial merge) = 196 visual tokens
_TOKENS_PER_IMAGE_448 = 196


class QwenVisualEncoder(nn.Module):
    """Frozen Qwen3.5 visual backbone producing projected patch features."""

    def __init__(self, visual_model, processor, hidden_size: int = 2560):
        """Use from_full_model() or from_standalone() — do not call directly."""
        super().__init__()
        self.visual = visual_model
        self.processor = processor
        self.hidden_size = hidden_size  # out_hidden_size = 2560

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
    def from_standalone(cls, checkpoint_dir: str):
        """
        Load from a HuggingFace-compatible directory saved by extract_qwen_visual_encoder.py.

        The directory contains config.json + model.safetensors written by
        visual.save_pretrained(), so AutoModel.from_pretrained() handles all
        config/weight reconstruction automatically without manual class imports.
        """
        import json
        import os
        from transformers import AutoModel, AutoProcessor

        # Read out_hidden_size from the metadata file written at extraction time.
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            hidden_size = meta.get("out_hidden_size", 2560)
        else:
            hidden_size = 2560

        visual = AutoModel.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(checkpoint_dir)

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

        All images are resized to 448×448 before encoding so every call
        returns a fixed token count of 196.

        Args:
            images: single PIL image or list of PIL images
            device: target device (defaults to the visual model's device)

        Returns:
            patch_features: (B, 196, 2560)  in the dtype of the visual model
        """
        if isinstance(images, Image.Image):
            images = [images]

        if device is None:
            device = next(self.visual.parameters()).device

        dtype = next(self.visual.parameters()).dtype

        inputs = self.processor.image_processor(
            images=images,
            return_tensors="pt",
            size={"shortest_edge": 448, "longest_edge": 448},
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        print(out.keys())  # debug

        # Handle both tensor return and dataclass return (e.g. BaseModelOutput)
        features = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        # features: (B * 196, out_hidden_size)

        # debug
        print(features.shape)

        B = len(images)
        N = features.shape[0] // B  # 196 for 448×448
        return features.view(B, N, self.hidden_size)

    def forward(
        self,
        images: Union[List[Image.Image], Image.Image],
        device: torch.device = None,
    ) -> torch.Tensor:
        return self.encode_images(images, device)
