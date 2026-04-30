"""CFG dropout utilities for SDXL image-only conditioning."""

import torch


def apply_condition_dropout(
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    p_drop: float = 0.10,
) -> tuple:
    """
    Randomly zero out both the token sequence and pooled projection for the same samples.

    Args:
        tokens: (B, N_tokens, 2048)
        pooled: (B, 1280)
        p_drop: fraction of batch items to drop (default 10%)

    Returns:
        tokens, pooled with dropped entries zeroed
    """
    B = tokens.shape[0]
    drop_mask = torch.rand(B, device=tokens.device) < p_drop

    tokens = tokens.clone()
    pooled = pooled.clone()
    tokens[drop_mask] = 0.0
    pooled[drop_mask] = 0.0

    return tokens, pooled


def build_uncond_context(
    batch_size: int,
    num_tokens: int,
    cross_attn_dim: int,
    pooled_proj_dim: int,
    device: torch.device,
) -> tuple:
    """Build all-zeros unconditional context for CFG inference."""
    tokens = torch.zeros(batch_size, num_tokens, cross_attn_dim, device=device)
    pooled = torch.zeros(batch_size, pooled_proj_dim, device=device)
    return tokens, pooled
