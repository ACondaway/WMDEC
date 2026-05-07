"""
Reusable image quality metrics: PSNR and LPIPS.

All functions accept tensors in [-1, 1] range, matching the VAE / training
pipeline convention. Inputs are clamped before computing metrics so minor
out-of-range reconstructions don't corrupt scores.

Typical usage
-------------
    from evaluation.metrics import LPIPSMetric, compute_psnr, MetricAccumulator

    lpips_fn = LPIPSMetric(net="vgg", device=device)
    acc = MetricAccumulator()

    for pred, target in batches:
        psnr  = compute_psnr(pred, target)          # (B,)
        lpips = lpips_fn(pred, target)              # (B,)
        acc.update(psnr.mean().item(), lpips.mean().item(), n=len(pred))

    print(acc)   # PSNR=28.34 dB  LPIPS=0.0821  (n=1024)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """
    Per-image Peak Signal-to-Noise Ratio.

    Args:
        pred:       (B, C, H, W) in [-1, 1]
        target:     (B, C, H, W) in [-1, 1]
        data_range: value range of the signal.  2.0 for [-1, 1],
                    1.0 for [0, 1], 255.0 for uint8.

    Returns:
        psnr: (B,) tensor of PSNR values in dB.
              Returns +inf where MSE == 0 (identical images).
    """
    pred   = pred.clamp(-1, 1)
    target = target.clamp(-1, 1)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    # Guard against exact zeros — clamp to tiny epsilon to avoid -inf
    psnr = 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-12))
    return psnr


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

class LPIPSMetric:
    """
    LPIPS perceptual distance (lower = more perceptually similar).

    Wraps the ``lpips`` package and keeps the network frozen on the target
    device.  Import is deferred to __init__ so the module can be imported
    without installing lpips when only PSNR is needed.

    Args:
        net:    Backbone network — ``'vgg'`` (recommended, matches published
                LPIPS paper), ``'alex'`` (faster), or ``'squeeze'`` (lightest).
        device: Compute device.  Defaults to ``cuda`` if available.
    """

    def __init__(self, net: str = "vgg", device: Optional[torch.device] = None):
        try:
            import lpips as _lpips
        except ImportError as e:
            raise ImportError(
                "lpips is required: pip install lpips"
            ) from e

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self._loss_fn = _lpips.LPIPS(net=net).eval().to(device)
        for p in self._loss_fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-image LPIPS scores.

        Args:
            pred:   (B, 3, H, W) in [-1, 1]
            target: (B, 3, H, W) in [-1, 1]

        Returns:
            distances: (B,) tensor of LPIPS scores (lower = more similar).
        """
        pred   = pred.clamp(-1, 1).to(self.device)
        target = target.clamp(-1, 1).to(self.device)
        # lpips returns (B, 1, 1, 1); squeeze to (B,)
        return self._loss_fn(pred, target).reshape(-1)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """
    Running accumulator for PSNR and LPIPS across batches.

    Supports optional per-key (e.g. per-dataset) sub-accumulators that are
    updated alongside the global totals.

    Example::

        acc = MetricAccumulator()
        acc.update(psnr_batch_mean, lpips_batch_mean, n=batch_size, key="robobrain-dex")
        print(acc)
    """

    _psnr_sum:  float = field(default=0.0, repr=False)
    _lpips_sum: float = field(default=0.0, repr=False)
    _count:     int   = field(default=0,   repr=False)
    _per_key:   Dict[str, "MetricAccumulator"] = field(
        default_factory=dict, repr=False
    )

    def update(
        self,
        psnr: float,
        lpips_val: float,
        n: int = 1,
        key: Optional[str] = None,
    ) -> None:
        """
        Add a batch result.

        Args:
            psnr:      Mean PSNR over the batch (dB).
            lpips_val: Mean LPIPS over the batch.
            n:         Number of images in the batch.
            key:       Optional grouping key (e.g. dataset name).
        """
        self._psnr_sum  += psnr * n
        self._lpips_sum += lpips_val * n
        self._count     += n
        if key is not None:
            if key not in self._per_key:
                self._per_key[key] = MetricAccumulator()
            self._per_key[key].update(psnr, lpips_val, n)

    @property
    def mean_psnr(self) -> float:
        return self._psnr_sum / self._count if self._count else float("nan")

    @property
    def mean_lpips(self) -> float:
        return self._lpips_sum / self._count if self._count else float("nan")

    @property
    def count(self) -> int:
        return self._count

    @property
    def per_key(self) -> Dict[str, "MetricAccumulator"]:
        return self._per_key

    def to_dict(self) -> dict:
        """Serialisable summary dict (suitable for JSON output)."""
        result = {
            "psnr_db":  round(self.mean_psnr,  4),
            "lpips":    round(self.mean_lpips, 6),
            "n_images": self._count,
        }
        if self._per_key:
            result["per_dataset"] = {
                k: v.to_dict() for k, v in sorted(self._per_key.items())
            }
        return result

    def __repr__(self) -> str:
        lines = [
            f"PSNR={self.mean_psnr:.2f} dB  LPIPS={self.mean_lpips:.4f}"
            f"  (n={self._count:,})"
        ]
        for k, v in sorted(self._per_key.items()):
            lines.append(
                f"  [{k}]  PSNR={v.mean_psnr:.2f} dB  "
                f"LPIPS={v.mean_lpips:.4f}  (n={v.count:,})"
            )
        return "\n".join(lines)
