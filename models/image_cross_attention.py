"""
Gated decoupled image cross-attention for SD 2.1.

Adds a separate image conditioning branch to each BasicTransformerBlock of the
SD 2.1 UNet without touching the existing text cross-attention (attn2), which
continues to receive the empty-text prior anchor.

Per-block formula:
    output = block(hidden, empty_text) + tanh(gate) * ImageCrossAttn(hidden, image_tokens)

Gate initialises to zero → at step 0 the model is identical to vanilla SD 2.1.

Usage (inside LDMUNet):
    conditioner = UNetImageConditioner(self.unet, image_tokens_dim=1024, num_heads=8)
    ...
    conditioner.set_image_tokens(image_tokens)  # (B, N, D)
    output = self.unet(x, t, encoder_hidden_states=empty_text_emb)
    conditioner.set_image_tokens(None)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedImageCrossAttention(nn.Module):
    """
    Single-block gated image cross-attention.

    Q: from the transformer-block's pre-forward hidden states  (query_dim)
    K, V: from image tokens                                    (image_tokens_dim)
    Gate: learnable scalar, tanh-activated; initialised to 0.

    forward(query, image_tokens) → (B, S, query_dim)
    """

    def __init__(self, query_dim: int, image_tokens_dim: int, num_heads: int = 8):
        super().__init__()
        # Ensure divisibility; fall back to 1 head if query_dim is small
        while query_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.gate = nn.Parameter(torch.zeros(1))

        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(image_tokens_dim, query_dim, bias=False)
        self.to_v = nn.Linear(image_tokens_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

        # Init output projection to near-zero so gates starting at 0
        # are reinforced by small output magnitudes early in training.
        nn.init.zeros_(self.to_out.bias)
        nn.init.normal_(self.to_out.weight, std=0.01)

    def forward(self, query: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:        (B, S_q, query_dim)   — pre-block hidden states
            image_tokens: (B, N, image_tokens_dim)

        Returns:
            (B, S_q, query_dim) — gated cross-attention output
        """
        B, S_q, _ = query.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(self.norm(query))                   # (B, S_q, query_dim)
        k = self.to_k(image_tokens)                       # (B, N, query_dim)
        v = self.to_v(image_tokens)                       # (B, N, query_dim)

        # Reshape to multi-head format
        q = q.view(B, S_q, H, D).permute(0, 2, 1, 3)    # (B, H, S_q, D)
        k = k.view(B, -1, H, D).permute(0, 2, 1, 3)      # (B, H, N, D)
        v = v.view(B, -1, H, D).permute(0, 2, 1, 3)      # (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, S_q, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, S_q, H * D)
        out = self.to_out(out)

        return torch.tanh(self.gate) * out


class UNetImageConditioner(nn.Module):
    """
    Attaches one GatedImageCrossAttention to every BasicTransformerBlock in the
    UNet via PyTorch forward hooks.

    The conditioner is an nn.Module so its parameters appear in model.parameters()
    and can be passed to an optimiser.

    Workflow:
        conditioner.set_image_tokens(tokens)   # before UNet forward
        output = unet(x, t, encoder_hidden_states=empty_text)
        conditioner.set_image_tokens(None)     # clear (optional but clean)
    """

    def __init__(self, unet: nn.Module, image_tokens_dim: int = 1024, num_heads: int = 8):
        super().__init__()

        self._image_tokens: torch.Tensor | None = None
        self._pre_hidden: dict[str, torch.Tensor] = {}
        self._hook_handles = []

        # Discover all BasicTransformerBlock instances.
        # diffusers < 0.19: transformers.models.…; diffusers >= 0.19: same path.
        try:
            from diffusers.models.attention import BasicTransformerBlock
        except ImportError:
            from diffusers.models.transformer_2d import BasicTransformerBlock

        gated_layers: dict[str, GatedImageCrossAttention] = {}

        for name, module in unet.named_modules():
            if not isinstance(module, BasicTransformerBlock):
                continue
            # Derive query_dim from the cross-attention or self-attention layer.
            # attn1 is always present (self-attn); its to_q.in_features == residual stream dim.
            query_dim = module.attn1.to_q.in_features
            key = name.replace(".", "_")  # valid Python identifier for ModuleDict
            gated_layers[key] = GatedImageCrossAttention(query_dim, image_tokens_dim, num_heads)

            # Capture pre-block hidden state and inject gated output post-block.
            # Use default-arg binding (key=key) to close over the correct key.
            def _pre(mod, args, _key=key):
                # args[0] is hidden_states for BasicTransformerBlock.forward
                if isinstance(args, tuple) and len(args) >= 1:
                    self._pre_hidden[_key] = args[0]

            def _post(mod, args, output, _key=key):
                if self._image_tokens is None:
                    return output
                pre = self._pre_hidden.pop(_key, None)
                if pre is None:
                    return output
                img_tokens = self._image_tokens
                # Ensure dtype/device match (image tokens live on the UNet device)
                img_tokens = img_tokens.to(pre.dtype)
                gated_out = self.gated_layers[_key](pre, img_tokens)
                # BasicTransformerBlock.forward() returns hidden_states (a Tensor in
                # all diffusers versions we support).  Handle tuple just in case.
                if isinstance(output, tuple):
                    return (output[0] + gated_out,) + output[1:]
                return output + gated_out

            h_pre  = module.register_forward_pre_hook(_pre)
            h_post = module.register_forward_hook(_post)
            self._hook_handles.extend([h_pre, h_post])

        self.gated_layers = nn.ModuleDict(gated_layers)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_image_tokens(self, tokens: torch.Tensor | None) -> None:
        """Set image tokens before a UNet forward pass; pass None to clear."""
        self._image_tokens = tokens

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call before saving / pickling the UNet)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def num_gated_layers(self) -> int:
        return len(self.gated_layers)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
