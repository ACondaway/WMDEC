import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class T5TextEncoder(nn.Module):
    """Frozen T5-XXL text encoder."""

    def __init__(self, model_name: str = "google/t5-xxl-lm-adapt", max_length: int = 77):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.max_length = max_length
        self.embed_dim = self.model.config.d_model

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text strings to T5 embeddings.

        Args:
            texts: list of B text strings

        Returns:
            z_txt: (B, T, C) text embeddings
        """
        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        z_txt = outputs.last_hidden_state.float()
        return z_txt

    @torch.no_grad()
    def encode_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode from pre-tokenized inputs."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.float()

    def get_empty_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return embedding for empty/padding text."""
        empty_texts = [""] * batch_size
        return self.encode(empty_texts).to(device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts)
