"""
Latent Texture Control Branch for SD3.5 semantic conditioning.

Encodes the input image's VAE latent into control features that modulate
the SD3.5 MMDiT hidden states via AdaLN.

Architecture
------------
    Input Image
        → SD3.5 VAE Encoder  (frozen)
        → Latent Prior  (B, 16, H/8, W/8)
        → LightControlEncoder (lightweight CNN)
        → control_features  (B, ctrl_dim)
        → ctrl_to_pooled  (B, 2048)
        → added (as residual) to pooled_projections before MMDiT

AdaLN formula (applied inside MMDiT via pooled_projections augmentation)
------------------------------------------------------------------------
    pooled_conditioned = pooled_text + strength(t) * ctrl_to_pooled(ctrl_features)

    where strength(t) = t  (strong at noisy early timesteps, weak at clean late ones)

LightControlEncoder
-------------------
    4 × [Conv2d → GroupNorm → SiLU → 2x downsample]
    Global average pooling → (B, ctrl_hidden)

    Input:  (B, 16, H/8, W/8)  — SD3.5 uses 16-channel latents
    Output: (B, ctrl_dim)

Stochastic Control Dropout (training only)
------------------------------------------
    Randomly zeroes ctrl_features with probability drop_prob (default 0.5)
    to prevent over-reliance on the latent prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv2d → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LightControlEncoder(nn.Module):
    """
    Lightweight CNN that encodes a VAE latent into a control feature vector.

    Args:
        latent_channels : VAE latent channels (16 for SD3.5, 4 for SDXL/SD2)
        ctrl_dim        : output feature dimension (default 512)
        base_channels   : first conv output channels; doubled each stage (default 64)
        num_stages      : number of downsample stages (default 4)
    """

    def __init__(
        self,
        latent_channels: int = 16,
        ctrl_dim: int = 512,
        base_channels: int = 64,
        num_stages: int = 4,
    ):
        super().__init__()

        channels = [latent_channels] + [
            base_channels * (2 ** i) for i in range(num_stages)
        ]

        stages = []
        for i in range(num_stages):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            stages.append(
                nn.Sequential(
                    _ConvBlock(in_ch, out_ch),
                    # 2× spatial downsample via strided conv
                    nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, bias=False),
                )
            )

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], ctrl_dim),
            nn.SiLU(),
            nn.Linear(ctrl_dim, ctrl_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 16, H/8, W/8)

        Returns:
            features: (B, ctrl_dim)
        """
        x = self.stages(latent)
        x = self.pool(x).flatten(1)   # (B, channels[-1])
        return self.proj(x)            # (B, ctrl_dim)


class TextureControlBranch(nn.Module):
    """
    Full texture control branch.

    Encodes a VAE latent into a residual contribution to the SD3.5 MMDiT's
    pooled_projections, which drives AdaLN modulation across all transformer blocks.

    The control strength is gated by the diffusion timestep t ∈ [0, 1]:
        strength(t) = t
    This makes control strong at early (noisy) timesteps (large t) where
    geometry and layout are established, and weak at late (clean) timesteps
    where the model adds fine-grained texture detail freely.

    Args:
        latent_channels : VAE latent channels (16 for SD3.5)
        ctrl_dim        : internal control feature dim    (default 512)
        pooled_dim      : SD3.5 pooled_projection_dim    (default 2048)
        drop_prob       : stochastic control dropout prob (default 0.5)
    """

    def __init__(
        self,
        latent_channels: int = 16,
        ctrl_dim: int = 512,
        pooled_dim: int = 2048,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.drop_prob = drop_prob

        self.encoder = LightControlEncoder(
            latent_channels=latent_channels,
            ctrl_dim=ctrl_dim,
        )

        self.ctrl_to_pooled = nn.Sequential(
            nn.LayerNorm(ctrl_dim),
            nn.Linear(ctrl_dim, pooled_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        timestep_t: torch.Tensor,
        force_drop: bool = False,
    ) -> torch.Tensor:
        """
        Compute the control residual to add to pooled_projections.

        Args:
            latent       : (B, 16, H/8, W/8)  VAE latent of the input image
            timestep_t   : (B,) float in [0, 1]  diffusion timestep (flow matching)
            force_drop   : if True, always zero the control (uncond inference)

        Returns:
            ctrl_residual : (B, pooled_dim)
        """
        ctrl_features = self.encoder(latent)   # (B, ctrl_dim)

        # Stochastic dropout during training
        if self.training and not force_drop:
            mask = (
                torch.rand(ctrl_features.shape[0], 1, device=ctrl_features.device)
                > self.drop_prob
            ).float()
            ctrl_features = ctrl_features * mask
        elif force_drop:
            ctrl_features = torch.zeros_like(ctrl_features)

        ctrl_pooled = self.ctrl_to_pooled(ctrl_features)   # (B, pooled_dim)

        # Timestep-aware strength: strong at large t (noisy/early), weak at small t (clean/late)
        strength = timestep_t.view(-1, 1).clamp(0.0, 1.0)  # (B, 1)

        return ctrl_pooled * strength   # (B, pooled_dim)
