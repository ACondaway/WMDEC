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

import json
import os
from pathlib import Path
from typing import Dict, Iterator

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

    def iter_samples(self) -> Iterator[SampleMeta]:
        meta_dir = self.image_root / "meta"
        episode_tasks = _load_episodes(meta_dir)
        yield from self._iter_walk(episode_tasks)

    def _iter_walk(self, episode_tasks: Dict[int, str]) -> Iterator[SampleMeta]:
        """
        Stream SampleMeta objects one at a time via os.scandir().

        Yields immediately as each file is discovered — no list is built.
        Episode text is resolved once per episode directory, not per frame.
        """
        videos_dir = str(self.image_root / "videos")
        obs_dir_name = f"observation.images.{self._camera_key}"

        try:
            chunk_entries = sorted(os.scandir(videos_dir), key=lambda e: e.name)
        except FileNotFoundError:
            return

        for chunk_entry in chunk_entries:
            if not chunk_entry.is_dir() or not chunk_entry.name.startswith("chunk-"):
                continue
            obs_dir = os.path.join(chunk_entry.path, obs_dir_name)
            if not os.path.isdir(obs_dir):
                continue

            try:
                ep_entries = sorted(os.scandir(obs_dir), key=lambda e: e.name)
            except OSError:
                continue

            for ep_entry in ep_entries:
                if not ep_entry.is_dir() or not ep_entry.name.startswith("episode_"):
                    continue
                ep_name = ep_entry.name
                try:
                    episode_index = int(ep_name.rsplit("_", 1)[-1])
                except ValueError:
                    continue

                task_text = episode_tasks.get(episode_index, "")

                try:
                    img_entries = sorted(os.scandir(ep_entry.path), key=lambda e: e.name)
                except OSError:
                    continue

                for img_entry in img_entries:
                    if not img_entry.name.endswith(".jpg"):
                        continue
                    stem = img_entry.name[:-4]
                    yield SampleMeta(
                        rel_path=Path(ep_name) / stem,
                        extra_meta={
                            "task_name":       task_text,
                            "episode":         ep_name,
                            "episode_index":   episode_index,
                            "filename":        stem,
                            "_abs_image_path": img_entry.path,
                        },
                    )

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")

    def stats_key(self, sample: SampleMeta) -> str:
        return sample.extra_meta.get("task_name") or self.dataset_name


class LeRobotWithoutTextPreprocessor(LeRobotPreprocessor):
    """
    LeRobot-format dataset where no text extraction is needed.

    Identical to LeRobotPreprocessor except:
    - meta/episodes.jsonl is NOT read (may be absent).
    - task_name is always set to "" so no text encoder pass is triggered.

    Use type="lerobot_without_text" in preprocess_config.yaml.
    """

    def iter_samples(self) -> Iterator[SampleMeta]:
        yield from self._iter_walk(episode_tasks={})
