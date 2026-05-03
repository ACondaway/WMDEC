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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, FrozenSet, Optional

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
        build_processed_set()    — scan output dir once → frozenset of existing paths
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

    def build_processed_set(self) -> FrozenSet[str]:
        """
        Scan the output directory ONCE and return all existing .pt paths as a
        frozenset of absolute path strings.

        Use this before the main loop to avoid per-sample stat() calls.
        For millions of files, one os.walk pass is orders of magnitude faster
        than N individual Path.exists() calls.

        Pass the result to is_processed() and process_batch() as `processed_set`.
        """
        out_dir = self.output_root / self.dataset_name
        if not out_dir.exists():
            return frozenset()
        existing: set[str] = set()
        for dirpath, _, filenames in os.walk(out_dir):
            for fname in filenames:
                if fname.endswith(".pt"):
                    existing.add(os.path.join(dirpath, fname))
        return frozenset(existing)

    def output_path(self, sample: SampleMeta) -> Path:
        """Derive the .pt output path from the sample's relative image path."""
        return self.output_root / self.dataset_name / sample.rel_path.with_suffix(".pt")

    def is_processed(
        self,
        sample: SampleMeta,
        processed_set: Optional[FrozenSet[str]] = None,
    ) -> bool:
        """
        Return True if the embedding file already exists.

        If `processed_set` is provided (built via build_processed_set()), uses
        an O(1) set lookup instead of a filesystem stat() call.
        """
        if processed_set is not None:
            return str(self.output_path(sample)) in processed_set
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
        processed_set: Optional[FrozenSet[str]] = None,
        num_io_workers: int = 8,
    ) -> int:
        """
        Encode a batch of images and save their embeddings.

        Args:
            batch_samples:  Samples to process (already sharded to this rank).
            encoder:        QwenVisualEncoder instance.
            device:         CUDA device.
            processed_set:  Frozenset from build_processed_set() for O(1) skip checks.
                            Falls back to per-file stat() if None.
            num_io_workers: ThreadPoolExecutor workers for parallel image loading.

        Returns the number of samples actually written (skips already-processed).
        """
        to_process = [s for s in batch_samples if not self.is_processed(s, processed_set)]
        if not to_process:
            return 0

        # Create all unique parent directories in one pass — avoids one mkdir
        # syscall per sample when many samples share the same parent dir.
        seen_parents: set[Path] = set()
        for s in to_process:
            p = self.output_path(s).parent
            if p not in seen_parents:
                p.mkdir(parents=True, exist_ok=True)
                seen_parents.add(p)

        # Load images in parallel — image decoding is I/O + CPU bound; threading
        # overlaps disk reads and PIL decoding across multiple files at once.
        with ThreadPoolExecutor(max_workers=num_io_workers) as pool:
            images = list(pool.map(self.load_image, to_process))

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
