"""
Frozen Qwen2.5-VL visual encoder (ViT backbone).

Extracts patch-level image features from the Qwen2.5-VL-3B-Instruct visual tower.
Output shape: (B, N_patches, D) where D=1152 for the 3B model.

Usage
-----
# Load from full Qwen model (first time / extraction)
enc = QwenVisualEncoder.from_full_model("Qwen/Qwen2.5-VL-3B-Instruct")

# Load from standalone checkpoint (after running extract_qwen_visual_encoder.py)
enc = QwenVisualEncoder.from_standalone("/path/to/qwen_visual_encoder.pt")
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Union


class QwenVisualEncoder(nn.Module):
    """Frozen Qwen2.5-VL visual backbone that produces patch embeddings."""

    HIDDEN_SIZE = {
        "Qwen/Qwen2.5-VL-3B-Instruct": 1152,
        "Qwen/Qwen2.5-VL-7B-Instruct": 1536,
    }

    def __init__(self, visual_model, processor, hidden_size: int = 1152):
        """Use from_full_model() or from_standalone() instead of calling directly."""
        super().__init__()
        self.visual = visual_model
        self.processor = processor
        self.hidden_size = hidden_size

        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()

    @classmethod
    def from_full_model(cls, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """Load the visual encoder from the full Qwen2.5-VL model."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"Loading full Qwen2.5-VL model: {model_name} ...")
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        visual = full_model.visual
        processor = AutoProcessor.from_pretrained(model_name)
        hidden_size = cls.HIDDEN_SIZE.get(model_name, visual.config.hidden_size)

        # Free the rest of the model
        del full_model
        torch.cuda.empty_cache()

        return cls(visual, processor, hidden_size)

    @classmethod
    def from_standalone(cls, checkpoint_path: str):
        """Load from a standalone .pt file saved by extract_qwen_visual_encoder.py."""
        from transformers import AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )
        from transformers import Qwen2_5_VLConfig

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        vision_cfg = Qwen2_5_VLConfig(**{"vision_config": ckpt["vision_config"]}).vision_config
        visual = Qwen2_5_VisionTransformerPretrainedModel(vision_cfg)
        visual.load_state_dict(ckpt["state_dict"])
        processor = AutoProcessor.from_pretrained(ckpt["processor_name"])
        hidden_size = ckpt.get("hidden_size", 1152)

        return cls(visual, processor, hidden_size)

    def _preprocess(self, images: List[Image.Image], device: torch.device) -> dict:
        """Preprocess PIL images using the Qwen processor."""
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            # Fixed size to get deterministic patch count
            size={"height": 448, "width": 448},
        )
        return {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, image_grid_thw=None) -> torch.Tensor:
        """
        Encode preprocessed pixel values to patch embeddings.

        Args:
            pixel_values: preprocessed tensor from Qwen processor
            image_grid_thw: grid shape tensor (required by Qwen2.5-VL)

        Returns:
            patch_embeds: (B, N_patches, D)
        """
        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        # out shape: (B * N_patches, D) -> reshape to (B, N_patches, D)
        return out

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], Image.Image],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Encode raw PIL images to patch embeddings.

        Args:
            images: single PIL image or list of PIL images
            device: target device

        Returns:
            patch_embeds: (B, N_patches, D)
        """
        if isinstance(images, Image.Image):
            images = [images]

        if device is None:
            device = next(self.visual.parameters()).device

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            size={"height": 448, "width": 448},
        )
        pixel_values = inputs["pixel_values"].to(device, dtype=next(self.visual.parameters()).dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        features = self.visual(pixel_values, grid_thw=image_grid_thw)
        # features: (total_patches, D). Reshape per image.
        # For fixed 448x448 input: N_patches = (448/14)^2 = 1024
        B = len(images)
        N = features.shape[0] // B
        return features.view(B, N, self.hidden_size)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw=None) -> torch.Tensor:
        return self.encode(pixel_values, image_grid_thw)
