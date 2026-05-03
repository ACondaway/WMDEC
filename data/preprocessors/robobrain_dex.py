"""
Preprocessor implementation for the RoboBrain-Dex dataset.

Image structure:
    {image_root}/{Task_name}/videos/chunk-000/observation.images.image_top/
        episode_XXXXXX/image_X.0.jpg

Output .pt structure under {output_root}/robobrain-dex/:
    {Task_name}/episode_XXXXXX/image_X.0.pt
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PIL import Image

from .base import BaseDatasetPreprocessor, SampleMeta


class RoboBrainDexPreprocessor(BaseDatasetPreprocessor):

    @property
    def dataset_name(self) -> str:
        return "robobrain-dex"

    def find_samples(self) -> List[SampleMeta]:
        """
        Walk the dataset tree with os.scandir() instead of glob.glob().

        glob.glob() has per-path fnmatch overhead and builds the full path
        list in memory before returning.  os.scandir() returns DirEntry
        objects whose .path and .name are already computed by the OS during
        the directory listing — no extra stat() or string join per file.

        task_name and episode are resolved once per directory level, not
        once per frame.
        """
        samples: List[SampleMeta] = []
        obs_subpath = os.path.join("videos", "chunk-000", "observation.images.image_top")

        try:
            task_entries = sorted(os.scandir(self.image_root), key=lambda e: e.name)
        except FileNotFoundError:
            return samples

        for task_entry in task_entries:
            if not task_entry.is_dir():
                continue
            task_name = task_entry.name
            ep_base = os.path.join(task_entry.path, obs_subpath)
            if not os.path.isdir(ep_base):
                continue

            try:
                ep_entries = sorted(os.scandir(ep_base), key=lambda e: e.name)
            except OSError:
                continue

            for ep_entry in ep_entries:
                if not ep_entry.is_dir() or not ep_entry.name.startswith("episode_"):
                    continue
                episode = ep_entry.name

                try:
                    img_entries = sorted(os.scandir(ep_entry.path), key=lambda e: e.name)
                except OSError:
                    continue

                for img_entry in img_entries:
                    if not img_entry.name.endswith(".jpg"):
                        continue
                    # Slice off ".jpg" — faster than Path(name).stem for a known extension.
                    filename = img_entry.name[:-4]
                    flat_rel = Path(task_name) / episode / filename

                    samples.append(SampleMeta(
                        rel_path=flat_rel,
                        extra_meta={
                            "task_name":       task_name,
                            "episode":         episode,
                            "filename":        filename,
                            "_abs_image_path": img_entry.path,
                        },
                    ))

        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")

    def stats_key(self, sample: SampleMeta) -> str:
        return sample.extra_meta["task_name"]
