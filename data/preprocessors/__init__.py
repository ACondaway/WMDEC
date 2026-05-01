from functools import partial

from .base import BaseDatasetPreprocessor, SampleMeta
from .robobrain_dex import RoboBrainDexPreprocessor
from .lerobot import LeRobotPreprocessor

# ---------------------------------------------------------------------------
# Dataset registry
#
# Maps dataset_name → callable(image_root, output_root) → preprocessor.
#
# For datasets using the LeRobot schema (meta/episodes.jsonl + videos/chunk-*/
# observation.images.{camera_key}/episode_*/image_*.jpg), add entries via
# functools.partial:
#
#   "my-dataset": partial(LeRobotPreprocessor, name="my-dataset", camera_key="image"),
#
# The `name` must match the registry key so that output_root/{name}/ is used.
# Change camera_key if the dataset uses a different camera (e.g. "image_top").
# ---------------------------------------------------------------------------

PREPROCESSORS: dict[str, type[BaseDatasetPreprocessor]] = {
    # Original RoboBrain-Dex (per-task subdirs, image_top camera)
    "robobrain-dex": RoboBrainDexPreprocessor,

    # LeRobot-format datasets (flat episode dirs, single camera)
    # Rename the keys to match your actual dataset names and set camera_key accordingly.
    "lerobot-dataset-a": partial(LeRobotPreprocessor, name="lerobot-dataset-a", camera_key="image"),
    "lerobot-dataset-b": partial(LeRobotPreprocessor, name="lerobot-dataset-b", camera_key="image"),
}
