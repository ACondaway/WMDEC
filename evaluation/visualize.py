"""
Image reconstruction visualisation utilities.

save_comparison_grid()
    Renders a side-by-side grid of original vs reconstructed images with
    per-pair PSNR / LPIPS annotations.  Accepts plain tensors so it can be
    reused from the VAE eval script, the diffusion validate loop, or any
    other evaluation pipeline.

Typical usage
-------------
    from evaluation.visualize import save_comparison_grid

    save_comparison_grid(
        originals=orig_list,          # list of (3, H, W) tensors in [-1, 1]
        reconstructions=recon_list,   # same shape
        psnr_values=psnr_list,        # list[float], dB
        lpips_values=lpips_list,      # list[float]
        output_path="vis/vae_recon.png",
        nrow=4,
        labels=ds_name_list,          # optional per-image caption
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) tensor in [-1, 1] to a (H, W, 3) uint8 numpy array."""
    t = t.detach().cpu().float().clamp(-1, 1)
    arr = ((t + 1.0) / 2.0 * 255.0).byte().numpy()  # (3, H, W)
    return arr.transpose(1, 2, 0)                     # (H, W, 3)


def save_comparison_grid(
    originals: List[torch.Tensor],
    reconstructions: List[torch.Tensor],
    psnr_values: List[float],
    lpips_values: List[float],
    output_path: str,
    nrow: int = 4,
    labels: Optional[List[str]] = None,
    figsize_per_pair: tuple[float, float] = (4.0, 4.8),
    dpi: int = 150,
    title: str = "VAE Reconstruction",
) -> None:
    """
    Save a PNG grid comparing original and reconstructed images.

    Layout
    ------
    Each row contains ``nrow`` (original | reconstruction) pairs.
    Pair header:   "Original"  |  "Reconstructed"
    Pair footer:   "PSNR XX.X dB  |  LPIPS 0.XXXX"
    Optional label (dataset name) in the top-left corner of each original.

    Args:
        originals:       List of (3, H, W) tensors in [-1, 1].
        reconstructions: List of (3, H, W) tensors in [-1, 1], same length.
        psnr_values:     Per-image PSNR in dB.
        lpips_values:    Per-image LPIPS scores.
        output_path:     File path for the saved PNG.
        nrow:            Number of (orig, recon) pairs per grid row.
        labels:          Optional list of per-image text labels shown on the
                         original image (e.g. dataset name).
        figsize_per_pair: (width, height) in inches for a single pair.
        dpi:             Output resolution.
        title:           Overall figure title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    n = len(originals)
    assert n == len(reconstructions) == len(psnr_values) == len(lpips_values), (
        "originals, reconstructions, psnr_values, lpips_values must all have the same length"
    )
    if labels is None:
        labels = [""] * n

    nrow  = min(nrow, n)
    ncols = nrow * 2          # each pair occupies 2 columns (orig + recon)
    nrows = math.ceil(n / nrow)

    fig_w = figsize_per_pair[0] * nrow
    fig_h = figsize_per_pair[1] * nrows + 0.6   # 0.6 for suptitle

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    # Column headers (printed once in the first row)
    for col in range(ncols):
        ax = axes[0][col]
        header = "Original" if col % 2 == 0 else "Reconstructed"
        ax.set_title(header, fontsize=9, pad=3, color="#333333")

    for idx in range(nrows * nrow):
        row = idx // nrow
        col_base = (idx % nrow) * 2   # leftmost column of this pair

        ax_orig  = axes[row][col_base]
        ax_recon = axes[row][col_base + 1]

        if idx < n:
            orig_arr  = _to_uint8(originals[idx])
            recon_arr = _to_uint8(reconstructions[idx])

            ax_orig.imshow(orig_arr)
            ax_recon.imshow(recon_arr)

            # Dataset label on original
            label = labels[idx]
            if label:
                ax_orig.text(
                    3, 3, label,
                    fontsize=6, color="white",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
                )

            # Metric annotation below the pair (centred between the two axes)
            metric_txt = (
                f"PSNR {psnr_values[idx]:.2f} dB   "
                f"LPIPS {lpips_values[idx]:.4f}"
            )
            # Draw the text in the coordinate space of ax_recon, outside the axes
            ax_recon.text(
                -0.05, -0.04,          # slightly left of recon, below both axes
                metric_txt,
                transform=ax_recon.transAxes,
                fontsize=7,
                ha="center", va="top",
                color="#222222",
            )
        else:
            # Pad empty cells with white
            ax_orig.set_visible(False)
            ax_recon.set_visible(False)

        for ax in (ax_orig, ax_recon):
            ax.axis("off")

    # Draw a thin vertical separator between each pair
    for row in range(nrows):
        for pair_idx in range(nrow):
            col_base = pair_idx * 2
            if col_base + 1 < ncols:
                ax = axes[row][col_base + 1]
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#cccccc")
                    spine.set_linewidth(0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
