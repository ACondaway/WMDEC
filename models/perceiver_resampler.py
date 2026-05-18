"""
Light Perceiver Resampler for SD3.5 semantic conditioning.

Compresses Qwen dense patch tokens into a small set of semantic latent tokens
via cross-attention, reducing attention complexity in downstream conditioning.

I/O contract
------------
    Input  : (B, N_in, in_dim)   e.g. (B, 64, 2560)  Qwen patch features
    Output : (B, N_queries, hidden_dim)  e.g. (B, 32, 1024)

Architecture (per layer)
------------------------
    learned queries  (B, N_queries, hidden_dim)
        → cross-attention over input tokens        [queries attend to context]
        → self-attention among queries             [queries interact]
        → feed-forward                             [per-token MLP]
    residual connections + pre-norm throughout

Recommended config (from CLAUDE.md)
------------------------------------
    num_layers  : 2–4
    num_queries : 32
    hidden_dim  : 1024
    num_heads   : 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FFN(nn.Module):
    """Pre-norm feed-forward block."""

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class _CrossAttention(nn.Module):
    """Pre-norm cross-attention: queries attend to context."""

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int):
        super().__init__()
        assert q_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q = nn.Linear(q_dim, q_dim, bias=False)
        self.k = nn.Linear(kv_dim, q_dim, bias=False)
        self.v = nn.Linear(kv_dim, q_dim, bias=False)
        self.out = nn.Linear(q_dim, q_dim, bias=False)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        B, Nq, _ = queries.shape
        _, Nc, _ = context.shape
        H, D = self.num_heads, self.head_dim

        q = self.q(self.norm_q(queries))
        k = self.k(self.norm_kv(context))
        v = self.v(self.norm_kv(context))

        q = q.view(B, Nq, H, D).permute(0, 2, 1, 3)
        k = k.view(B, Nc, H, D).permute(0, 2, 1, 3)
        v = v.view(B, Nc, H, D).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Nq, H * D)

        return queries + self.out(out)


class _SelfAttention(nn.Module):
    """Pre-norm self-attention among queries."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, H, D).permute(0, 2, 1, 3)
        k = k.view(B, N, H, D).permute(0, 2, 1, 3)
        v = v.view(B, N, H, D).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, H * D)

        return x + self.out(out)


class _PerceiverLayer(nn.Module):
    """Single Perceiver layer: cross-attn → self-attn → FFN."""

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        num_heads: int,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.cross_attn = _CrossAttention(hidden_dim, in_dim, num_heads)
        self.self_attn = _SelfAttention(hidden_dim, num_heads)
        self.ffn = _FFN(hidden_dim, ff_mult)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.cross_attn(queries, context)
        queries = self.self_attn(queries)
        queries = self.ffn(queries)
        return queries


class PerceiverResampler(nn.Module):
    """
    Light Perceiver Resampler.

    Args:
        in_dim      : feature dim of the Qwen patch tokens  (default 2560)
        hidden_dim  : internal query / output dim            (default 1024)
        num_queries : number of output semantic tokens        (default 32)
        num_heads   : attention heads                         (default 8)
        num_layers  : number of perceiver layers              (default 4)
        ff_mult     : FFN expansion factor                    (default 4)
    """

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 1024,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

        # Input projection (aligns Qwen dim to hidden_dim for KV)
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                _PerceiverLayer(hidden_dim, hidden_dim, num_heads, ff_mult)
                for _ in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_in, in_dim)  Qwen patch features

        Returns:
            tokens: (B, num_queries, hidden_dim)  semantic latent tokens
        """
        B = x.shape[0]

        # Project input to hidden_dim for KV
        context = self.input_proj(x)  # (B, N_in, hidden_dim)

        # Expand learned queries across batch
        queries = self.queries.expand(B, -1, -1)  # (B, num_queries, hidden_dim)

        for layer in self.layers:
            queries = layer(queries, context)

        return self.out_norm(queries)  # (B, num_queries, hidden_dim)
