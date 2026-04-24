import torch
import torch.nn as nn


class ImageAdapter(nn.Module):
    """
    MLP adapter that projects SigLIP image embedding (B, D) into
    a sequence of tokens (B, N_img, C) for cross-attention conditioning.
    """

    def __init__(
        self,
        siglip_dim: int = 1152,
        cross_attn_dim: int = 1024,
        num_tokens: int = 8,
        num_layers: int = 3,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attn_dim = cross_attn_dim

        layers = []
        in_dim = siglip_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_tokens * cross_attn_dim))
        layers.append(nn.LayerNorm(num_tokens * cross_attn_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_img: (B, D) SigLIP image embedding

        Returns:
            tokens_img: (B, N_img, C)
        """
        out = self.mlp(z_img)  # (B, N_img * C)
        tokens_img = out.view(-1, self.num_tokens, self.cross_attn_dim)
        return tokens_img


class TextAdapter(nn.Module):
    """Linear projection of T5 text embeddings to cross-attention dimension."""

    def __init__(self, t5_dim: int = 4096, cross_attn_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(t5_dim, cross_attn_dim)
        self.norm = nn.LayerNorm(cross_attn_dim)

    def forward(self, z_txt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_txt: (B, T, C_t5) T5 text embeddings

        Returns:
            tokens_txt: (B, T, C)
        """
        return self.norm(self.proj(z_txt))
