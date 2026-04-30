"""
Image adapter for SDXL conditioning.

Converts Qwen patch embeddings (B, N_patches, 1152) into:
  1. tokens     : (B, N_tokens, 2048) — encoder_hidden_states for SDXL UNet cross-attention
  2. pooled_proj: (B, 1280)           — text_embeds for SDXL added_cond_kwargs

Uses a Perceiver-style cross-attention pooler so N_patches can vary at runtime.
"""

import torch
import torch.nn as nn


class CrossAttentionPooler(nn.Module):
    """
    N_tokens learned queries attend over N_patches patch embeddings.
    Output: (B, N_tokens, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.queries = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.norm_in(x)

        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        q = self.queries.expand(B, -1, -1)

        H, D = self.num_heads, self.head_dim
        q = q.view(B, self.num_tokens, H, D).permute(0, 2, 1, 3)
        k = k.view(B, -1, H, D).permute(0, 2, 1, 3)
        v = v.view(B, -1, H, D).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, self.num_tokens, H * D)
        return self.norm_out(self.out_proj(out))


class ImageAdapter(nn.Module):
    """
    Two outputs for SDXL conditioning:
      - tokens:      (B, N_tokens, 2048) via cross-attention pooler + FF
      - pooled_proj: (B, 1280) via mean-pool + MLP
    """

    def __init__(
        self,
        qwen_dim: int = 1152,
        cross_attn_dim: int = 2048,    # SDXL encoder_hidden_states dim
        pooled_proj_dim: int = 1280,   # SDXL text_embeds dim
        num_tokens: int = 16,
        num_heads: int = 8,
        ff_mult: int = 2,
    ):
        super().__init__()
        # Sequence conditioning path
        self.pooler = CrossAttentionPooler(qwen_dim, cross_attn_dim, num_tokens, num_heads)
        self.ff_norm = nn.LayerNorm(cross_attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(cross_attn_dim, cross_attn_dim * ff_mult),
            nn.GELU(),
            nn.Linear(cross_attn_dim * ff_mult, cross_attn_dim),
        )

        # Pooled projection path
        self.pooled_proj = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, pooled_proj_dim),
            nn.SiLU(),
            nn.Linear(pooled_proj_dim, pooled_proj_dim),
        )

    def forward(self, patch_embeds: torch.Tensor) -> tuple:
        """
        Args:
            patch_embeds: (B, N_patches, qwen_dim)

        Returns:
            tokens:      (B, N_tokens, 2048)
            pooled_proj: (B, 1280)
        """
        tokens = self.pooler(patch_embeds)                      # (B, N_tokens, 2048)
        tokens = tokens + self.ff(self.ff_norm(tokens))         # residual FF

        pooled = self.pooled_proj(patch_embeds.mean(dim=1))     # (B, 1280)
        return tokens, pooled
