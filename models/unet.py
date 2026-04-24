import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        H = self.heads

        q = self.to_q(x).view(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).view(B, -1, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).view(B, -1, H, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out) + residual


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim, heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)) + x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.self_attn = SelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, context_dim, heads)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        x = self.ff(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int,
                 has_attn: bool = True, num_heads: int = 8):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = TransformerBlock(out_ch, context_dim, num_heads) if has_attn else None
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor):
        h = self.res(x, t_emb)
        if self.attn is not None:
            B, C, H, W = h.shape
            h = h.view(B, C, H * W).permute(0, 2, 1)
            h = self.attn(h, context)
            h = h.permute(0, 2, 1).view(B, C, H, W)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int,
                 has_attn: bool = True, num_heads: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res = ResBlock(in_ch + out_ch, out_ch, time_dim)
        self.attn = TransformerBlock(out_ch, context_dim, num_heads) if has_attn else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.res(x, t_emb)
        if self.attn is not None:
            B, C, H, W = h.shape
            h = h.view(B, C, H * W).permute(0, 2, 1)
            h = self.attn(h, context)
            h = h.permute(0, 2, 1).view(B, C, H, W)
        return h


class MidBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.res1 = ResBlock(dim, dim, time_dim)
        self.attn = TransformerBlock(dim, context_dim, num_heads)
        self.res2 = ResBlock(dim, dim, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor):
        x = self.res1(x, t_emb)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.attn(x, context)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.res2(x, t_emb)
        return x


class ConditionalUNet(nn.Module):
    """
    UNet with cross-attention conditioning for latent diffusion.
    Operates on VAE latent space (4 channels).
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 256,
        channel_mult: tuple = (1, 2, 4, 4),
        context_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        time_dim = model_channels * 4
        self.time_embed = TimestepMLP(model_channels, time_dim)
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attn = i >= 1  # attention from second level
            self.down_blocks.append(DownBlock(ch, out_ch, time_dim, context_dim, has_attn, num_heads))
            ch = out_ch

        # Middle
        self.mid = MidBlock(ch, time_dim, context_dim, num_heads)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            has_attn = i >= 1
            self.up_blocks.append(UpBlock(ch, out_ch, time_dim, context_dim, has_attn, num_heads))
            ch = out_ch

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H, W) noisy latent
            t: (B,) timestep
            context: (B, S, C) conditioning tokens (concat of image + text tokens)

        Returns:
            eps_pred: (B, 4, H, W) predicted noise
        """
        t_emb = self.time_embed(t)
        h = self.conv_in(x)

        skips = []
        for block in self.down_blocks:
            h, skip = block(h, t_emb, context)
            skips.append(skip)

        h = self.mid(h, t_emb, context)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip, t_emb, context)

        h = self.act(self.norm_out(h))
        return self.conv_out(h)
