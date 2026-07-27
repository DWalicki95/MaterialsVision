"""
Standalone, model-agnostic image/mask augmentation.

This package is a data-prep stage driven by
``side_scripts/augment_dataset.py``; it is not imported by any training
or inference code in ``materials_vision/experiments/``. Model-specific
preprocessing belongs in that experiment's own directory instead of here
-- see ``experiments/peft_sam/dataset.py`` (``RawTrafo``,
``PerObjectDistanceTransform``) and ``experiments/cellpose/utils.py``
(``precompute_flows_batched``) for the established pattern.
"""

from materials_vision.image_preprocessing.image_transformation import (
    Augmentor
)
from materials_vision.image_preprocessing.transform import augment_dataset

__all__ = ['Augmentor', 'augment_dataset']
