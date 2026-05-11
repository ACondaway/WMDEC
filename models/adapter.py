"""
Image adapter — converts Qwen patch embeddings to UNet conditioning tokens.

Supports two backbones via the `backbone` parameter:

  backbone="sdxl"  — two outputs:
      tokens:      (B, N_tokens, 2048)   encoder_hidden_states for SDXL cross-attention
      pooled_proj: (B, 1280)             text_embeds for SDXL added_cond_kwargs

  backbone="sd21"  — one output (new gated image cross-attention design):
      tokens:      (B, N_tokens, cross_attn_dim)  fed to UNetImageConditioner
      pooled_proj: None

  null_image_tokens: learnable (1, N_tokens, cross_attn_dim) used for image CFG dropout.
  Obtained via img_adapter.null_image_tokens (raw model) or
  img_adapter.module.null_image_tokens (DDP-wrapped).

Call site always unpacks as:
    tokens, pooled = adapter(patch_embeds)
and passes pooled to the UNet only when it is not None (SDXL only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean-centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Residual pooler (multi-layer learned-query cross-attention)
# ---------------------------------------------------------------------------

class _PoolerLayer(nn.Module):
    """One layer of the residual pooler: cross-attn + optional self-attn + FFN."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        num_heads: int,
        ff_mult: int = 4,
        use_self_attn: bool = False,
    ):
        super().__init__()
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_kv  = nn.LayerNorm(in_dim)
        head_dim = dim // num_heads
        self._scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim  = head_dim

        self.to_q  = nn.Linear(dim,    dim,     bias=False)
        self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.norm_sa = nn.LayerNorm(dim)
            self.sa_q    = nn.Linear(dim, dim, bias=False)
            self.sa_kv   = nn.Linear(dim, dim * 2, bias=False)
            self.sa_out  = nn.Linear(dim, dim)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def _mha(self, q, k, v, B, S_q, S_kv):
        H, D = self.num_heads, self.head_dim
        q = q.view(B, S_q,  H, D).permute(0, 2, 1, 3)
        k = k.view(B, S_kv, H, D).permute(0, 2, 1, 3)
        v = v.view(B, S_kv, H, D).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1) * self._scale).softmax(-1)
        return (attn @ v).permute(0, 2, 1, 3).reshape(B, S_q, H * D)

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        B, S_q = queries.shape[:2]
        # Cross-attention (queries attend to patches)
        q    = self.to_q(self.norm_q(queries))
        kv   = self.to_kv(self.norm_kv(patches))
        k, v = kv.chunk(2, dim=-1)
        ca   = self._mha(q, k, v, B, S_q, patches.shape[1])
        queries = queries + self.out_proj(ca)

        # Optional self-attention among queries
        if self.use_self_attn:
            sq      = self.sa_q(self.norm_sa(queries))
            skv     = self.sa_kv(self.norm_sa(queries))
            sk, sv  = skv.chunk(2, dim=-1)
            sa      = self._mha(sq, sk, sv, B, S_q, S_q)
            queries = queries + self.sa_out(sa)

        # Feed-forward
        queries = queries + self.ff(self.norm_ff(queries))
        return queries


class ResidualPooler(nn.Module):
    """
    Multi-layer residual pooler: learned queries attend over patch embeddings.

    Architecture per layer:
      • Cross-attention (queries ← patches) + residual
      • Optional self-attention among queries + residual
      • Feed-forward + residual
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
        use_self_attn: bool = False,
    ):
        super().__init__()
        self.queries    = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)
        self.patch_norm = nn.LayerNorm(in_dim)
        self.patch_proj = nn.Linear(in_dim, out_dim, bias=False)

        self.layers = nn.ModuleList([
            _PoolerLayer(out_dim, out_dim, num_heads, use_self_attn=use_self_attn)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N_patches, in_dim)
        Returns:
            (B, num_tokens, out_dim)
        """
        B = patches.shape[0]
        kv_patches = self.patch_proj(self.patch_norm(patches))   # (B, N, out_dim)
        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, kv_patches)
        return self.out_norm(queries)


# ---------------------------------------------------------------------------
# Patch projector (input normalisation + MLP + learnable scale)
# ---------------------------------------------------------------------------

class PatchProjector(nn.Module):
    """RMSNorm → Linear → GELU → Linear → learnable scale."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm  = RMSNorm(in_dim)
        self.proj1 = nn.Linear(in_dim, out_dim * 2)
        self.proj2 = nn.Linear(out_dim * 2, out_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(F.gelu(self.proj1(self.norm(x)))) * self.scale


# ---------------------------------------------------------------------------
# Legacy cross-attention pooler (kept for SDXL path)
# ---------------------------------------------------------------------------

class CrossAttentionPooler(nn.Module):
    """N_tokens learned queries attend over N_patches patch embeddings → (B, N_tokens, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads  = num_heads
        self.head_dim   = out_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.queries  = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)
        self.kv_proj  = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm_in  = nn.LayerNorm(in_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]
        x  = self.norm_in(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        q = self.queries.expand(B, -1, -1)
        H, D = self.num_heads, self.head_dim
        q = q.view(B, self.num_tokens, H, D).permute(0, 2, 1, 3)
        k = k.view(B, -1, H, D).permute(0, 2, 1, 3)
        v = v.view(B, -1, H, D).permute(0, 2, 1, 3)
        attn = (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out  = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, self.num_tokens, H * D)
        return self.norm_out(self.out_proj(out))


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class ImageAdapter(nn.Module):
    """
    Converts Qwen patch embeddings to UNet conditioning tokens.

    SD 2.1 (new gated design):
      • PatchProjector normalises raw Qwen tokens
      • ResidualPooler (multi-layer) compresses N_patches → N_tokens
      • null_image_tokens: learnable uncond tokens for image CFG dropout

    SDXL (legacy):
      • Single-layer CrossAttentionPooler + FFN
      • Pooled projection head for added_cond_kwargs

    Returns:
        tokens:      (B, N_tokens, cross_attn_dim)
        pooled_proj: (B, pooled_proj_dim) for SDXL, or None for SD 2.1
    """

    def __init__(
        self,
        qwen_dim: int = 2560,
        cross_attn_dim: int = 1024,
        pooled_proj_dim: int = 1280,    # ignored for sd21
        num_tokens: int = 16,
        num_heads: int = 8,
        ff_mult: int = 2,               # kept for SDXL FFN path
        backbone: str = "sd21",
        num_pooler_layers: int = 2,
        pooler_self_attn: bool = False,
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "sd21":
            self.patch_projector = PatchProjector(qwen_dim, qwen_dim)
            self.pooler = ResidualPooler(
                in_dim=qwen_dim,
                out_dim=cross_attn_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                num_layers=num_pooler_layers,
                use_self_attn=pooler_self_attn,
            )
            # Learnable null tokens for image CFG dropout (1, N, D)
            self.null_image_tokens = nn.Parameter(
                torch.zeros(1, num_tokens, cross_attn_dim)
            )
        else:
            # SDXL legacy path
            self.pooler  = CrossAttentionPooler(qwen_dim, cross_attn_dim, num_tokens, num_heads)
            self.ff_norm = nn.LayerNorm(cross_attn_dim)
            self.ff = nn.Sequential(
                nn.Linear(cross_attn_dim, cross_attn_dim * ff_mult),
                nn.GELU(),
                nn.Linear(cross_attn_dim * ff_mult, cross_attn_dim),
            )
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
        if self.backbone == "sd21":
            projected = self.patch_projector(patch_embeds)
            tokens    = self.pooler(projected)
            return tokens, None
        else:
            tokens = self.pooler(patch_embeds)
            tokens = tokens + self.ff(self.ff_norm(tokens))
            pooled = self.pooled_proj(patch_embeds.mean(dim=1))
            return tokens, pooled
