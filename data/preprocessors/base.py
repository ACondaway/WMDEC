"""
Abstract base class for offline dataset preprocessing.

To add a new dataset:
1. Subclass BaseDatasetPreprocessor.
2. Implement `dataset_name`, `iter_samples()`, and `load_image()`.
3. Register it in data/preprocessors/__init__.py under PREPROCESSORS.

The framework handles:
  - Distributed sharding across GPUs
  - Skip-already-processed logic (resume support)
  - Saving embeddings in a consistent .pt format
  - Statistics collection and reporting

Streaming design
----------------
iter_samples() is a generator — it yields SampleMeta objects one at a time
without ever materialising the full dataset list in memory.  For millions of
files this avoids gigabytes of RAM for SampleMeta objects.

iter_samples_for_rank() wraps iter_samples() with modulo interleaving so each
GPU rank receives only its share of the stream without any upfront shard list.

find_samples() is kept as a convenience wrapper (list(iter_samples())) for
callers that need a full list, but is no longer used by the main pipeline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Iterator, List, Optional

import torch
from PIL import Image


@dataclass
class SampleMeta:
    """
    Minimal descriptor for a single data sample.

    Attributes:
        rel_path:   Path relative to `image_root` (used to derive the output .pt path).
        extra_meta: Arbitrary key-value pairs stored in the .pt file alongside the embedding.
    """
    rel_path: Path
    extra_meta: dict = field(default_factory=dict)


class BaseDatasetPreprocessor(ABC):
    """
    Abstract interface for offline Qwen visual embedding extraction.

    Subclasses must implement:
        dataset_name  (property) — unique string identifier, e.g. "robobrain-dex"
        iter_samples()           — generator that yields SampleMeta one at a time
        load_image()             — load a PIL.Image given a SampleMeta

    The base class provides everything else.
    """

    def __init__(self, image_root: str, output_root: str):
        self.image_root  = Path(image_root)
        self.output_root = Path(output_root)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Unique identifier used as the top-level subdirectory under output_root."""

    @abstractmethod
    def iter_samples(self) -> Iterator[SampleMeta]:
        """
        Yield SampleMeta objects in a deterministic order, one at a time.

        Must produce the same ordering on every call and on every rank so that
        modulo interleaving in iter_samples_for_rank() partitions consistently.
        """

    @abstractmethod
    def load_image(self, sample: SampleMeta) -> Image.Image:
        """Load and return a PIL RGB image for the given sample."""

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def iter_samples_for_rank(
        self,
        rank: int,
        world_size: int,
    ) -> Iterator[SampleMeta]:
        """
        Yield only this rank's share of the global sample stream.

        Uses modulo interleaving: rank r receives samples at global indices
        [r, r + world_size, r + 2*world_size, ...].  Because iter_samples()
        produces a deterministic order, every rank gets a consistent, disjoint
        subset with no coordination needed between ranks.
        """
        for i, sample in enumerate(self.iter_samples()):
            if i % world_size == rank:
                yield sample

    def find_samples(self) -> List[SampleMeta]:
        """Convenience wrapper — materialises iter_samples() into a list."""
        return list(self.iter_samples())

    # ------------------------------------------------------------------
    # Output path / resume helpers
    # ------------------------------------------------------------------

    def build_processed_set(self) -> FrozenSet[str]:
        """
        Scan the output directory ONCE and return all existing .pt paths as a
        frozenset of absolute path strings.

        One os.walk pass is orders of magnitude faster than individual
        Path.exists() calls for millions of files.
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
        return self.output_root / self.dataset_name / sample.rel_path.with_suffix(".pt")

    def is_processed(
        self,
        sample: SampleMeta,
        processed_set: Optional[FrozenSet[str]] = None,
    ) -> bool:
        if processed_set is not None:
            return str(self.output_path(sample)) in processed_set
        return self.output_path(sample).exists()

    def save_embedding(self, sample: SampleMeta, z_img: torch.Tensor) -> None:
        """Write embedding tensor + metadata to disk atomically via a temp file."""
        out = self.output_path(sample)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"z_img": z_img.cpu(), "dataset_name": self.dataset_name, **sample.extra_meta}
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
        Encode a batch and save embeddings.  Returns number of samples written.

        Skips samples already in processed_set.  Parent dirs are created in a
        single batched pass.  Images are loaded in parallel via ThreadPoolExecutor.
        """
        to_process = [s for s in batch_samples if not self.is_processed(s, processed_set)]
        if not to_process:
            return 0

        seen_parents: set[Path] = set()
        for s in to_process:
            p = self.output_path(s).parent
            if p not in seen_parents:
                p.mkdir(parents=True, exist_ok=True)
                seen_parents.add(p)

        with ThreadPoolExecutor(max_workers=num_io_workers) as pool:
            images = list(pool.map(self.load_image, to_process))

        patch_embeds = encoder.encode_images(images, device=device).to(torch.bfloat16)
        for i, sample in enumerate(to_process):
            self.save_embedding(sample, patch_embeds[i])
        return len(to_process)

    # ------------------------------------------------------------------
    # Optional override
    # ------------------------------------------------------------------

    def stats_key(self, sample: SampleMeta) -> str:
        return sample.extra_meta.get("task_name", self.dataset_name)
