import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor


class SigLIPEncoder(nn.Module):
    """Frozen SigLIP image encoder that produces normalized embeddings."""

    def __init__(self, model_name: str = "google/siglip-large-patch16-384"):
        super().__init__()
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.embed_dim = self.model.config.vision_config.hidden_size

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to normalized SigLIP embeddings.

        Args:
            pixel_values: (B, 3, H, W) preprocessed image tensor

        Returns:
            z_img: (B, D) normalized image embedding
        """
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        z_img = F.normalize(outputs, dim=-1)
        return z_img

    @torch.no_grad()
    def encode_image_from_raw(self, images) -> torch.Tensor:
        """Encode raw PIL images."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.model.parameters()).device)
        return self.encode_image(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_image(pixel_values)
