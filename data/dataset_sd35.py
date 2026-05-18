"""
Dataset for SD3.5 semantic conditioning training.

Wraps the existing EmbeddingDataset with SD3.5-specific settings:
    - Default resolution: 768×768
    - Always loads raw images (required for VAE encoding during training)
    - Returns z_img: (64, 2560) Qwen patch features pre-extracted offline

Usage
-----
    ds = SD35EmbeddingDataset(
        name="robobrain-dex",
        embedding_dir="/share/project/congsheng/robobrain-dex-qwen-embedding",
        image_dir="/share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex",
        resolution=768,
    )
    item = ds[0]
    # item["z_img"]  : (64, 2560)  float32
    # item["image"]  : (3, 768, 768) float32  normalised to [-1, 1]
"""

from __future__ import annotations

from torch.utils.data import Dataset

from data.dataset import (
    EmbeddingDataset,
    MultiDatasetEmbeddingDataset,
    DistributedWeightedSampler,
)


class SD35EmbeddingDataset(EmbeddingDataset):
    """
    Single-dataset loader for SD3.5 training.

    Inherits from EmbeddingDataset with resolution=768 default and enforces
    that image_dir is provided (images are required for VAE latent encoding).

    Args:
        name          : dataset identifier string
        embedding_dir : path to pre-extracted Qwen .pt embeddings
        image_dir     : path to corresponding raw images (required)
        resolution    : output image resolution (default 768)
    """

    def __init__(
        self,
        name: str,
        embedding_dir: str,
        image_dir: str,
        resolution: int = 768,
    ):
        super().__init__(
            name=name,
            embedding_dir=embedding_dir,
            image_dir=image_dir,
            resolution=resolution,
        )


class MultiDatasetSD35Dataset(MultiDatasetEmbeddingDataset):
    """
    Multi-dataset loader for SD3.5 training with stats-aware rebalancing.

    Each entry in datasets_config must include image_dir (required for VAE).

    Args:
        datasets_config : list of dicts:
            name          (str) — dataset identifier
            embedding_dir (str) — path to pre-extracted Qwen .pt embeddings
            image_dir     (str) — path to raw images
        resolution       : output resolution (default 768)
        rebalance_alpha  : α for temperature-scaled rebalancing (default 0.7)
    """

    def __init__(
        self,
        datasets_config: list[dict],
        resolution: int = 768,
        rebalance_alpha: float = 0.7,
    ):
        # Validate that image_dir is provided for all datasets
        for cfg in datasets_config:
            if "image_dir" not in cfg or cfg["image_dir"] is None:
                raise ValueError(
                    f"image_dir is required for SD3.5 training "
                    f"(missing for dataset '{cfg.get('name', '?')}')."
                )

        super().__init__(
            datasets_config=datasets_config,
            resolution=resolution,
            rebalance_alpha=rebalance_alpha,
        )


# Re-export sampler for convenience
__all__ = [
    "SD35EmbeddingDataset",
    "MultiDatasetSD35Dataset",
    "DistributedWeightedSampler",
]
