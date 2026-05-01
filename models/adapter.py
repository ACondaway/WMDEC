"""
Image adapter — converts Qwen patch embeddings to UNet conditioning tokens.

Supports two backbones via the `backbone` parameter:

  backbone="sdxl"  — two outputs:
      tokens:      (B, N_tokens, 2048)   encoder_hidden_states for SDXL cross-attention
      pooled_proj: (B, 1280)             text_embeds for SDXL added_cond_kwargs

  backbone="sd21"  — one output:
      tokens:      (B, N_tokens, 1024)   encoder_hidden_states for SD 2.1 cross-attention
      pooled_proj: None                  (SD 2.1 has no pooled conditioning)

Call site always unpacks as:
    tokens, pooled = adapter(patch_embeds)
and passes pooled to the UNet only when it is not None.
"""

import torch
import torch.nn as nn


class CrossAttentionPooler(nn.Module):
    """N_tokens learned queries attend over N_patches patch embeddings → (B, N_tokens, out_dim)."""

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

        attn = (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, self.num_tokens, H * D)
        return self.norm_out(self.out_proj(out))


class ImageAdapter(nn.Module):

    def __init__(
        self,
        qwen_dim: int = 2560,
        cross_attn_dim: int = 2048,    # 2048 for SDXL, 1024 for SD 2.1
        pooled_proj_dim: int = 1280,   # ignored when backbone="sd21"
        num_tokens: int = 16,
        num_heads: int = 8,
        ff_mult: int = 2,
        backbone: str = "sdxl",        # "sdxl" | "sd21"
    ):
        super().__init__()
        self.backbone = backbone

        # Sequence conditioning path (shared)
        self.pooler = CrossAttentionPooler(qwen_dim, cross_attn_dim, num_tokens, num_heads)
        self.ff_norm = nn.LayerNorm(cross_attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(cross_attn_dim, cross_attn_dim * ff_mult),
            nn.GELU(),
            nn.Linear(cross_attn_dim * ff_mult, cross_attn_dim),
        )

        # Pooled projection path (SDXL only)
        if backbone == "sdxl":
            self.pooled_proj = nn.Sequential(
                nn.LayerNorm(qwen_dim),
                nn.Linear(qwen_dim, pooled_proj_dim),
                nn.SiLU(),
                nn.Linear(pooled_proj_dim, pooled_proj_dim),
            )

    def forward(self, patch_embeds: torch.Tensor):
        """
        Args:
            patch_embeds: (B, N_patches, qwen_dim)

        Returns:
            tokens:      (B, N_tokens, cross_attn_dim)
            pooled_proj: (B, pooled_proj_dim) for SDXL, or None for SD 2.1
        """
        tokens = self.pooler(patch_embeds)
        tokens = tokens + self.ff(self.ff_norm(tokens))

        pooled = self.pooled_proj(patch_embeds.mean(dim=1)) if self.backbone == "sdxl" else None
        return tokens, pooled
