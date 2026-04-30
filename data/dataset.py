"""
Dataset classes for training.

EmbeddingDataset             — single dataset (used internally)
MultiDatasetEmbeddingDataset — combines N datasets with stats-aware rebalancing
DistributedWeightedSampler   — DDP-compatible weighted sampler

Rebalancing uses the verified frame counts from each dataset's stats.json
(written by the preprocessing pipeline) so the sampling probabilities reflect
the true dataset sizes rather than however many files happen to be on disk.
Falls back to file counting if stats.json is not present.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import List

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_stats(embedding_dir: str) -> dict | None:
    """
    Load stats.json from the embedding directory if it exists.
    stats.json is written by data/stats.py after preprocessing completes.
    """
    path = os.path.join(embedding_dir, "stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Single-dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted Qwen patch embeddings + corresponding raw images.

    Directory layout expected under embedding_dir:
        embedding_dir/
            stats.json               ← written by preprocessing pipeline
            {subdir}/.../{name}.pt   ← {"z_img": Tensor(N_patches, D), ...}

    Raw images (optional, used for on-the-fly VAE encoding):
        image_dir/{subdir}/.../{name}.jpg
    """

    def __init__(
        self,
        name: str,
        embedding_dir: str,
        image_dir: str = None,
        resolution: int = 1024,
    ):
        self.name = name
        self.embedding_dir = embedding_dir
        self.image_dir = image_dir
        self.resolution = resolution

        # Exclude stats.json itself from the glob
        self.samples = sorted(
            p for p in glob.glob(os.path.join(embedding_dir, "**", "*.pt"), recursive=True)
            if os.path.basename(p) != "stats.json"
        )
        if not self.samples:
            raise ValueError(f"No .pt embedding files found in {embedding_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, emb_path: str) -> str | None:
        if self.image_dir is None:
            return None
        rel = os.path.relpath(emb_path, self.embedding_dir)
        return os.path.join(self.image_dir, os.path.splitext(rel)[0] + ".jpg")

    def __getitem__(self, idx: int) -> dict:
        emb_path = self.samples[idx]
        data = torch.load(emb_path, map_location="cpu", weights_only=True)

        result = {
            "z_img": data["z_img"].float(),  # (N_patches, D)
            "dataset_name": self.name,
        }

        img_path = self._image_path(emb_path)
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            result["image"] = self.transform(image)
        else:
            result["image"] = torch.zeros(3, self.resolution, self.resolution)

        return result


# ---------------------------------------------------------------------------
# Multi-dataset with stats-aware rebalancing
# ---------------------------------------------------------------------------

class MultiDatasetEmbeddingDataset(Dataset):
    """
    Combines multiple EmbeddingDataset instances with temperature-scaled rebalancing.

    Sampling probability formula
    ----------------------------
        p_i  =  n_i^α  /  Σ_j  n_j^α

    where n_i is the **verified frame count** read from stats.json (produced by the
    preprocessing pipeline). This ensures the balance reflects the true dataset sizes
    rather than however many .pt files happen to be on disk at training time.

    α controls the balance:
        α = 1.0  →  proportional sampling (large datasets dominate)
        α = 0.5  →  square-root smoothing (moderate balance)
        α = 0.0  →  equal probability per dataset (smallest datasets oversampled)

    The per-sample weight assigned to sample j in dataset i is:
        w_j  =  p_i / |D_i|_disk

    where |D_i|_disk is the number of .pt files actually present on disk (which may
    be smaller than n_i during partial preprocessing runs).
    """

    def __init__(
        self,
        datasets_config: list[dict],
        resolution: int = 1024,
        rebalance_alpha: float = 0.7,
    ):
        """
        Args:
            datasets_config: list of dicts, each with:
                name          (str) — unique dataset identifier
                embedding_dir (str) — path to pre-extracted .pt files + stats.json
                image_dir     (str, optional) — path to raw images
            resolution:       output image resolution for the transform
            rebalance_alpha:  α in the sampling formula above (0 ≤ α ≤ 1)
        """
        self.datasets: list[EmbeddingDataset] = []
        self._stats: list[dict | None] = []       # stats.json contents per dataset
        self._stats_frame_counts: list[int] = []  # n_i from stats.json (or fallback)

        for cfg in datasets_config:
            ds = EmbeddingDataset(
                name=cfg["name"],
                embedding_dir=cfg["embedding_dir"],
                image_dir=cfg.get("image_dir"),
                resolution=resolution,
            )
            self.datasets.append(ds)

            stats = _load_stats(cfg["embedding_dir"])
            self._stats.append(stats)
            # Use stats.json frame count as the canonical size for weight computation.
            # Fall back to disk file count if stats.json is absent.
            if stats is not None:
                canonical = stats.get("total_frames", len(ds))
            else:
                canonical = len(ds)
            self._stats_frame_counts.append(canonical)

        # Build flat global index: [(dataset_idx, local_sample_idx), ...]
        self._index: list[tuple[int, int]] = []
        for ds_idx, ds in enumerate(self.datasets):
            for local_idx in range(len(ds)):
                self._index.append((ds_idx, local_idx))

        self._weights = self._compute_weights(rebalance_alpha)
        self._alpha = rebalance_alpha

    def _compute_weights(self, alpha: float) -> torch.Tensor:
        # Canonical sizes (from stats.json) drive the sampling probabilities
        canonical = torch.tensor(self._stats_frame_counts, dtype=torch.float64)
        p = canonical.pow(alpha)
        p = p / p.sum()  # normalise to valid probability distribution

        # Per-sample weight = p_i / |D_i|_disk
        per_sample_weights: list[float] = []
        for i, ds in enumerate(self.datasets):
            w = (p[i] / len(ds)).item()
            per_sample_weights.extend([w] * len(ds))

        return torch.tensor(per_sample_weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for DistributedWeightedSampler."""
        return self._weights

    def dataset_summary(self) -> str:
        """
        Human-readable table showing raw vs effective sampling rates.

        Example output:
            α = 0.70  (stats-based rebalancing)
            ┌─────────────────────────┬──────────────┬──────────────┬────────────┬────────────┐
            │ Dataset                 │ Stats frames │  Disk frames │  Raw share │  Eff share │
            ├─────────────────────────┼──────────────┼──────────────┼────────────┼────────────┤
            │ robobrain-dex           │  1,234,567   │  1,234,567   │    72.3 %  │    65.1 %  │
            │ my-new-dataset          │    472,890   │    472,890   │    27.7 %  │    34.9 %  │
            ├─────────────────────────┼──────────────┼──────────────┼────────────┼────────────┤
            │ TOTAL                   │  1,707,457   │  1,707,457   │   100.0 %  │   100.0 %  │
            └─────────────────────────┴──────────────┴──────────────┴────────────┴────────────┘
        """
        total_stats = sum(self._stats_frame_counts)
        total_disk = sum(len(ds) for ds in self.datasets)
        w = self._weights
        total_w = w.sum().item()

        stats_src = ["stats.json" if s is not None else "disk count" for s in self._stats]
        src_note = ", ".join(f"{ds.name}({src})" for ds, src in zip(self.datasets, stats_src))

        lines = [
            f"Rebalancing  α = {self._alpha:.2f}   (0=balanced, 1=proportional)",
            f"Frame counts from: {src_note}",
            f"{'─'*90}",
            f"  {'Dataset':<28}  {'Stats frames':>13}  {'Disk frames':>11}  {'Raw %':>7}  {'Effective %':>11}",
            f"{'─'*90}",
        ]

        offset = 0
        for ds, n_stats, n_disk in zip(self.datasets, self._stats_frame_counts, [len(d) for d in self.datasets]):
            eff = w[offset:offset + n_disk].sum().item() / total_w * 100
            raw = n_stats / total_stats * 100 if total_stats else 0
            lines.append(
                f"  {ds.name:<28}  {n_stats:>13,}  {n_disk:>11,}  {raw:>6.1f}%  {eff:>10.1f}%"
            )
            offset += n_disk

        lines.append(f"{'─'*90}")
        lines.append(
            f"  {'TOTAL':<28}  {total_stats:>13,}  {total_disk:>11,}  {'100.0%':>7}  {'100.0%':>10}"
        )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._index[idx]
        return self.datasets[ds_idx][local_idx]


# ---------------------------------------------------------------------------
# DDP-compatible weighted sampler
# ---------------------------------------------------------------------------

class DistributedWeightedSampler(Sampler):
    """
    Weighted random sampler that works with DistributedDataParallel.

    All ranks share the same Generator seed so they draw from the same global
    index sequence, then each rank takes every world_size-th element starting
    from its own rank index. No inter-rank communication required.

    Usage:
        sampler = DistributedWeightedSampler(
            weights=dataset.get_sample_weights(),
            num_samples_per_epoch=len(dataset),
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )
        loader = DataLoader(dataset, sampler=sampler, ...)

        for epoch in range(...):
            sampler.set_epoch(epoch)   # must call to shuffle each epoch
            for batch in loader: ...
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples_per_epoch: int,
        num_replicas: int,
        rank: int,
        seed: int = 42,
    ):
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        # Round up to nearest multiple of world_size
        self._total = math.ceil(num_samples_per_epoch / num_replicas) * num_replicas
        self.num_samples = self._total // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self._total, replacement=True, generator=g
        ).tolist()
        return iter(indices[self.rank : self._total : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples
