from .base import BaseDatasetPreprocessor, SampleMeta
from .robobrain_dex import RoboBrainDexPreprocessor
from .lerobot import LeRobotPreprocessor

# ---------------------------------------------------------------------------
# Schema registry
#
# Keys are *schema types* (the `type` field in preprocess_config.yaml),
# NOT dataset names.  The actual dataset name comes from the YAML `name` field
# and is passed to the constructor at runtime.
#
# To add a new schema, subclass BaseDatasetPreprocessor and add an entry here.
# To add a new dataset of an existing schema, just add it to preprocess_config.yaml
# with the appropriate `type` — no code change needed.
#
# For schemas where dataset_name is fixed (e.g. RoboBrainDexPreprocessor),
# `type` and `name` in the YAML can be the same value.
# ---------------------------------------------------------------------------

PREPROCESSORS: dict[str, type[BaseDatasetPreprocessor]] = {
    # type="robobrain-dex" → name is fixed inside the class
    "robobrain-dex": RoboBrainDexPreprocessor,

    # type="lerobot" → name comes from the YAML `name` field at runtime
    "lerobot": LeRobotPreprocessor,
}
