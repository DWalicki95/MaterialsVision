"""
Training-time access to the frozen dataset split.

``split_io`` reads the split table produced by
``scripts/create_dataset_split.py`` and guards the test set;
``sampling`` decides the order in which TRAIN images reach the
optimizer.
"""
from materials_vision.data.sampling import (ProportionalImageSampler,
                                            sampler_run_metadata)
from materials_vision.data.split_io import (LockedTestSetError, SplitLoadError,
                                            SplitSubset, load_split)

__all__ = [
    "ProportionalImageSampler",
    "SplitLoadError",
    "SplitSubset",
    "LockedTestSetError",
    "load_split",
    "sampler_run_metadata",
]
