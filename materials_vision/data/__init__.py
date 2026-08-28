"""
Training-time access to the frozen dataset split.

``split_io`` reads the split table produced by
``scripts/create_dataset_split.py`` and guards the test set;
``sampling`` decides the order in which TRAIN images reach the
optimizer.
"""
from materials_vision.data.instances import CroppedSample, apply_content_crop
from materials_vision.data.masks import MaskLoadError, load_instance_mask
from materials_vision.data.samples import (PreparedSample, SampleRecord,
                                           SampleSource, SampleSourceError,
                                           read_manifest)
from materials_vision.data.sampling import (ProportionalImageSampler,
                                            sampler_run_metadata)
from materials_vision.data.split_io import (LockedTestSetError, SplitLoadError,
                                            SplitSubset, load_split)

__all__ = [
    "CroppedSample",
    "LockedTestSetError",
    "MaskLoadError",
    "PreparedSample",
    "ProportionalImageSampler",
    "SampleRecord",
    "SampleSource",
    "SampleSourceError",
    "SplitLoadError",
    "SplitSubset",
    "apply_content_crop",
    "load_instance_mask",
    "load_split",
    "read_manifest",
    "sampler_run_metadata",
]
