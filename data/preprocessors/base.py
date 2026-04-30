"""
Abstract base class for offline dataset preprocessing.

To add a new dataset:
1. Subclass BaseDatasetPreprocessor.
2. Implement `dataset_name`, `find_samples()`, and `load_image()`.
3. Register it in data/preprocessors/__init__.py under PREPROCESSORS.

The framework handles:
  - Distributed sharding across GPUs
  - Skip-already-processed logic (resume support)
  - Saving embeddings in a consistent .pt format
  - Statistics collection and reporting
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from PIL import Image


@dataclass
class SampleMeta:
    """
    Minimal descriptor for a single data sample.

    Attributes:
        rel_path:   Path relative to `image_root` (used to derive the output .pt path
                    by replacing the extension with .pt under `output_root`).
        extra_meta: Arbitrary key-value pairs stored alongside the embedding in the .pt file.
                    Use this for task name, episode, split, etc.
    """
    rel_path: Path
    extra_meta: dict = field(default_factory=dict)


class BaseDatasetPreprocessor(ABC):
    """
    Abstract interface for offline Qwen visual embedding extraction.

    Subclasses must implement:
        dataset_name  (property) — unique string identifier, e.g. "robobrain-dex"
        find_samples()           — return all SampleMeta for this dataset
        load_image()             — load a PIL.Image given a SampleMeta

    The base class provides:
        output_path()            — maps a sample to its .pt output path
        is_processed()           — True if the .pt already exists (for resuming)
        save_embedding()         — atomically writes the embedding .pt file
        process_batch()          — encode a batch and save all files
    """

    def __init__(self, image_root: str, output_root: str):
        self.image_root = Path(image_root)
        self.output_root = Path(output_root)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Unique identifier used as the top-level subdirectory under output_root."""

    @abstractmethod
    def find_samples(self) -> List[SampleMeta]:
        """
        Discover all processable samples in the dataset.

        Returns a list of SampleMeta sorted deterministically so every rank
        produces the same global ordering before sharding.
        """

    @abstractmethod
    def load_image(self, sample: SampleMeta) -> Image.Image:
        """Load and return a PIL RGB image for the given sample."""

    # ------------------------------------------------------------------
    # Provided helpers
    # ------------------------------------------------------------------

    def output_path(self, sample: SampleMeta) -> Path:
        """Derive the .pt output path from the sample's relative image path."""
        return self.output_root / self.dataset_name / sample.rel_path.with_suffix(".pt")

    def is_processed(self, sample: SampleMeta) -> bool:
        """Return True if the embedding file already exists (allows resuming)."""
        return self.output_path(sample).exists()

    def save_embedding(self, sample: SampleMeta, z_img: torch.Tensor) -> None:
        """Write embedding tensor + metadata to disk atomically via a temp file."""
        out = self.output_path(sample)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "z_img": z_img.cpu(),
            "dataset_name": self.dataset_name,
            **sample.extra_meta,
        }
        # Write to .tmp then rename — avoids corrupt files if interrupted
        tmp = out.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.rename(out)

    def process_batch(
        self,
        batch_samples: List[SampleMeta],
        encoder,
        device: torch.device,
    ) -> int:
        """
        Encode a batch of images and save their embeddings.

        Returns the number of samples actually written (skips already-processed).
        """
        to_process = [s for s in batch_samples if not self.is_processed(s)]
        if not to_process:
            return 0

        images = [self.load_image(s) for s in to_process]
        patch_embeds = encoder.encode_images(images, device=device).to(torch.bfloat16)

        for i, sample in enumerate(to_process):
            self.save_embedding(sample, patch_embeds[i])

        return len(to_process)

    # ------------------------------------------------------------------
    # Optional override: per-sample metadata for statistics
    # ------------------------------------------------------------------

    def stats_key(self, sample: SampleMeta) -> str:
        """
        Return a grouping key for statistics (e.g. task name, split).
        Override to get per-group breakdowns in the statistics report.
        Defaults to dataset_name (single group).
        """
        return sample.extra_meta.get("task_name", self.dataset_name)
