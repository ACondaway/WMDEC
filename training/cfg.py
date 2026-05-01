"""CFG dropout utilities.  Works for both SDXL (tokens + pooled) and SD 2.1 (tokens only)."""

import torch


def apply_condition_dropout(
    tokens: torch.Tensor,
    pooled: torch.Tensor = None,
    p_drop: float = 0.10,
):
    """
    Randomly zero out conditioning for a random subset of batch items.

    Args:
        tokens: (B, N_tokens, cross_attn_dim)
        pooled: (B, pooled_dim) for SDXL, or None for SD 2.1
        p_drop: fraction of batch items to drop (default 10%)

    Returns:
        tokens, pooled  (pooled is None when not provided)
    """
    B = tokens.shape[0]
    drop_mask = torch.rand(B, device=tokens.device) < p_drop

    tokens = tokens.clone()
    tokens[drop_mask] = 0.0

    if pooled is not None:
        pooled = pooled.clone()
        pooled[drop_mask] = 0.0

    return tokens, pooled


def build_uncond_context(
    batch_size: int,
    num_tokens: int,
    cross_attn_dim: int,
    device: torch.device,
    pooled_proj_dim: int = 0,
):
    """
    Build all-zeros unconditional context for CFG inference.

    Returns:
        tokens: (B, N_tokens, cross_attn_dim)
        pooled: (B, pooled_proj_dim) if pooled_proj_dim > 0, else None
    """
    tokens = torch.zeros(batch_size, num_tokens, cross_attn_dim, device=device)
    pooled = torch.zeros(batch_size, pooled_proj_dim, device=device) if pooled_proj_dim > 0 else None
    return tokens, pooled
