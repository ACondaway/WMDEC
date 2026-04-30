from .base import BaseDatasetPreprocessor, SampleMeta
from .robobrain_dex import RoboBrainDexPreprocessor

# Registry: add new dataset preprocessors here
PREPROCESSORS: dict[str, type[BaseDatasetPreprocessor]] = {
    "robobrain-dex": RoboBrainDexPreprocessor,
}
