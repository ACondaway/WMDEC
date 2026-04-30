"""
Preprocessor implementation for the RoboBrain-Dex dataset.

Image structure:
    {image_root}/{Task_name}/videos/chunk-000/observation.images.image_top/
        episode_XXXXXX/image_X.0.jpg

Output .pt structure under {output_root}/robobrain-dex/:
    {Task_name}/episode_XXXXXX/image_X.0.pt
"""

from __future__ import annotations

import glob
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
        pattern = str(
            self.image_root
            / "*"
            / "videos"
            / "chunk-000"
            / "observation.images.image_top"
            / "episode_*"
            / "*.jpg"
        )
        image_paths = sorted(glob.glob(pattern))

        samples = []
        for path in image_paths:
            abs_path = Path(path)
            rel = abs_path.relative_to(self.image_root)
            parts = rel.parts
            # parts: (task_name, "videos", "chunk-000", "observation.images.image_top", episode, filename)
            task_name = parts[0]
            episode = parts[-2]
            filename = abs_path.stem  # image_X.0

            # Store a flat rel_path that mirrors what we want under output_root:
            # robobrain-dex/{task_name}/{episode}/{filename}.pt
            flat_rel = Path(task_name) / episode / filename

            samples.append(SampleMeta(
                rel_path=flat_rel,
                extra_meta={
                    "task_name": task_name,
                    "episode": episode,
                    "filename": filename,
                    "_abs_image_path": str(abs_path),  # cached for fast load
                },
            ))

        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")

    def stats_key(self, sample: SampleMeta) -> str:
        return sample.extra_meta["task_name"]
