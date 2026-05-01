"""
Preprocessor for LeRobot-format datasets.

Expected directory layout:
    {image_root}/
    ├── meta/
    │   ├── episodes.jsonl   # {"episode_index": N, "tasks": ["<instruction>"], "length": M}
    │   └── tasks.jsonl      # {"task_index": N, "task": "<instruction>"}
    └── videos/
        ├── chunk-000/
        │   └── observation.images.{camera_key}/
        │       ├── episode_000000/
        │       │   ├── image_0.0.jpg
        │       │   └── ...
        │       └── ...
        ├── chunk-001/
        └── ...

Output .pt structure under {output_root}/{dataset_name}/:
    episode_XXXXXX/image_X.0.pt
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .base import BaseDatasetPreprocessor, SampleMeta


def _load_episodes(meta_dir: Path) -> Dict[int, str]:
    """Return {episode_index: task_text} from episodes.jsonl."""
    episodes: Dict[int, str] = {}
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj["episode_index"]
            tasks = obj.get("tasks", [])
            episodes[idx] = tasks[0] if tasks else ""
    return episodes


class LeRobotPreprocessor(BaseDatasetPreprocessor):
    """
    Generic preprocessor for any LeRobot-format dataset.

    Parameters
    ----------
    image_root : str
        Root directory of the dataset (contains meta/ and videos/).
    output_root : str
        Root directory where .pt embedding files are written.
    name : str
        Dataset name used as the top-level output subdirectory, e.g. "my-dataset".
    camera_key : str
        Subdirectory name under each chunk, e.g. "image" or "image_top".
        Default: "image"
    """

    def __init__(
        self,
        image_root: str,
        output_root: str,
        name: str,
        camera_key: str = "image",
    ):
        super().__init__(image_root, output_root)
        self._name = name
        self._camera_key = camera_key

    @property
    def dataset_name(self) -> str:
        return self._name

    def find_samples(self) -> List[SampleMeta]:
        # Load episode → task mapping from meta/episodes.jsonl.
        meta_dir = self.image_root / "meta"
        episode_tasks = _load_episodes(meta_dir)

        # Glob all images across every chunk.
        pattern = str(
            self.image_root
            / "videos"
            / "chunk-*"
            / f"observation.images.{self._camera_key}"
            / "episode_*"
            / "*.jpg"
        )
        image_paths = sorted(glob.glob(pattern))

        samples = []
        for path in image_paths:
            abs_path = Path(path)
            episode_dir = abs_path.parent.name          # e.g. "episode_000000"
            filename    = abs_path.stem                 # e.g. "image_0.0"

            # Parse episode index from folder name.
            try:
                episode_index = int(episode_dir.split("_")[-1])
            except ValueError:
                continue

            task_text = episode_tasks.get(episode_index, "")

            # Flat rel_path: episode_XXXXXX/image_X.0
            flat_rel = Path(episode_dir) / filename

            samples.append(SampleMeta(
                rel_path=flat_rel,
                extra_meta={
                    "task_name":        task_text,
                    "episode":          episode_dir,
                    "episode_index":    episode_index,
                    "filename":         filename,
                    "_abs_image_path":  str(abs_path),
                },
            ))

        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")

    def stats_key(self, sample: SampleMeta) -> str:
        return sample.extra_meta.get("task_name") or self.dataset_name
