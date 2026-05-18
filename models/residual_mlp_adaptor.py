"""
Residual Deep MLP Adaptor for SD3.5 semantic conditioning.

Projects semantic latent tokens from the Perceiver Resampler into the
SD3.5 MMDiT text token manifold (joint_attention_dim = 4096).

I/O contract
------------
    Input  : (B, N, hidden_dim)   e.g. (B, 32, 1024)  Perceiver output
    Output : (B, N, out_dim)      e.g. (B, 32, 4096)  SD3.5 pseudo text tokens

    Additionally returns a pooled projection:
    pooled : (B, pooled_dim)      e.g. (B, 2048)       SD3.5 pooled_projections

Architecture per residual block
--------------------------------
    x → LayerNorm → Linear(d, d*ff_mult) → GELU → Linear(d*ff_mult, d) → + x
        (with optional dimension-match projection for first block)

Recommended config (from CLAUDE.md)
------------------------------------
    num_blocks   : 4–8
    activation   : GELU
    normalization: LayerNorm
    residual     : true
"""

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * ff_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * ff_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ResidualMLPAdaptor(nn.Module):
    """
    Residual Deep MLP Adaptor.

    Aligns the Perceiver Resampler output (Qwen semantic manifold) with the
    SD3.5 MMDiT text conditioning manifold via a stack of residual MLP blocks
    followed by an output projection.

    Args:
        in_dim       : input feature dim from PerceiverResampler (default 1024)
        out_dim      : SD3.5 joint_attention_dim                 (default 4096)
        pooled_dim   : SD3.5 pooled_projection_dim               (default 2048)
        num_blocks   : number of residual MLP blocks             (default 6)
        ff_mult      : FFN expansion factor inside each block    (default 4)
    """

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 4096,
        pooled_dim: int = 2048,
        num_blocks: int = 6,
        ff_mult: int = 4,
    ):
        super().__init__()

        # Lift input to out_dim for residual processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

        self.blocks = nn.ModuleList(
            [_ResidualBlock(out_dim, ff_mult) for _ in range(num_blocks)]
        )

        self.out_norm = nn.LayerNorm(out_dim)

        # Pooled projection head: mean-pool over tokens → (B, pooled_dim)
        # This replaces CLIP pooled conditioning in SD3.5.
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, pooled_dim),
            nn.SiLU(),
            nn.Linear(pooled_dim, pooled_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, in_dim)  output from PerceiverResampler

        Returns:
            pseudo_tokens : (B, N, out_dim)   SD3.5 encoder_hidden_states
            pooled_proj   : (B, pooled_dim)   SD3.5 pooled_projections
        """
        # Pooled projection computed from input before expansion
        pooled = self.pooled_head(x.mean(dim=1))   # (B, pooled_dim)

        # Sequence projection
        h = self.input_proj(x)                      # (B, N, out_dim)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)                        # (B, N, out_dim)

        return h, pooled
