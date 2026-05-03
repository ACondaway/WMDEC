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

        return self._walk_samples(episode_tasks)

    def _walk_samples(self, episode_tasks: Dict[int, str]) -> List[SampleMeta]:
        """
        Walk the videos/ tree with os.listdir() instead of glob.glob().

        For millions of files this is significantly faster because:
        - os.listdir() avoids fnmatch pattern matching overhead on every path.
        - Episode text is resolved once per episode directory, not once per frame.
        - No large intermediate list of full path strings is materialised by glob.
        """
        videos_dir = self.image_root / "videos"
        obs_dir_name = f"observation.images.{self._camera_key}"

        samples: List[SampleMeta] = []

        try:
            chunk_names = sorted(os.listdir(videos_dir))
        except FileNotFoundError:
            return samples

        for chunk_name in chunk_names:
            if not chunk_name.startswith("chunk-"):
                continue
            obs_dir = videos_dir / chunk_name / obs_dir_name
            if not obs_dir.is_dir():
                continue

            try:
                episode_names = sorted(os.listdir(obs_dir))
            except OSError:
                continue

            for ep_name in episode_names:
                if not ep_name.startswith("episode_"):
                    continue
                try:
                    episode_index = int(ep_name.split("_")[-1])
                except ValueError:
                    continue

                # Resolve task text once per episode (not once per frame).
                task_text = episode_tasks.get(episode_index, "")
                ep_dir = obs_dir / ep_name

                try:
                    filenames = sorted(os.listdir(ep_dir))
                except OSError:
                    continue

                for fname in filenames:
                    if not fname.endswith(".jpg"):
                        continue
                    abs_path = ep_dir / fname
                    stem = Path(fname).stem          # e.g. "image_0.0"
                    flat_rel = Path(ep_name) / stem  # episode_000000/image_0.0

                    samples.append(SampleMeta(
                        rel_path=flat_rel,
                        extra_meta={
                            "task_name":       task_text,
                            "episode":         ep_name,
                            "episode_index":   episode_index,
                            "filename":        stem,
                            "_abs_image_path": str(abs_path),
                        },
                    ))

        return samples

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

    def find_samples(self) -> List[SampleMeta]:
        return self._walk_samples(episode_tasks={})
