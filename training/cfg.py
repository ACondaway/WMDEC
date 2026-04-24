import torch


def apply_condition_dropout(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    p_keep_all: float = 0.75,
    p_drop_text: float = 0.10,
    p_drop_image: float = 0.10,
    p_drop_all: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly drop conditions during training for classifier-free guidance.

    Probabilities:
        keep all:   75%
        drop text:  10%
        drop image: 10%
        drop all:   5%

    Args:
        z_img: (B, N, C) image condition tokens
        z_txt: (B, T, C) text condition tokens

    Returns:
        z_img, z_txt with some entries zeroed out
    """
    B = z_img.shape[0]
    device = z_img.device

    rand = torch.rand(B, device=device)

    # Cumulative thresholds
    thresh_keep = p_keep_all
    thresh_drop_text = thresh_keep + p_drop_text
    thresh_drop_image = thresh_drop_text + p_drop_image
    # rest is drop_all

    # Create masks
    drop_img_mask = ((rand >= thresh_drop_text) & (rand < thresh_drop_image)) | (rand >= thresh_drop_image)
    drop_txt_mask = ((rand >= thresh_keep) & (rand < thresh_drop_text)) | (rand >= thresh_drop_image)

    # Apply masks
    z_img = z_img.clone()
    z_txt = z_txt.clone()

    z_img[drop_img_mask] = 0.0
    z_txt[drop_txt_mask] = 0.0

    return z_img, z_txt


def build_uncond_context(batch_size: int, num_img_tokens: int, num_txt_tokens: int,
                         context_dim: int, device: torch.device) -> torch.Tensor:
    """Build unconditional context (all zeros) for CFG inference."""
    total_tokens = num_img_tokens + num_txt_tokens
    return torch.zeros(batch_size, total_tokens, context_dim, device=device)
