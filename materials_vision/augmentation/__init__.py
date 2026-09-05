"""
Online augmentation of image/label pairs for instance segmentation.

The package produces training samples that vary the things a real
acquisition varies - orientation, tonal response, sharpness, scale -
without ever changing what the annotation says. Everything happens in
memory as a sample is drawn; nothing is written to disk, so a policy
change costs no dataset rebuild and two policies can be compared step
for step rather than image for image.

Three properties hold across the package and the rest of the pipeline
depends on all three.

*The annotation stays exact.* A mask is never interpolated, never left
holding an id nobody annotated, and never quietly emptied. Where a
transformation legitimately changes it, the change is stated and
checked; where it does not, the mask is verified to be untouched.

*A sample is reproducible from a seed.* Randomness comes from a
generator seeded per sample and separate from every other source in the
process, so the same run reproduces the same samples no matter how many
worker processes build them or in what order they finish.

A transformation that cannot find a valid draw gives up in a controlled way and
says so; a transformation that produces an invalid sample stops the run.
The two are never confused, because one is expected and the other is a defect.
"""
from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL,
                                                  MASK_CHANGING_FAMILIES,
                                                  BlurConfig, MaskAwareConfig,
                                                  OrientationConfig,
                                                  PolicyConfig, ScaleConfig,
                                                  SeptumConfig, TonalConfig,
                                                  enabled_families,
                                                  policy_run_metadata)
from materials_vision.augmentation.integrity import (IntegrityError,
                                                     check_connectivity,
                                                     check_labels_preserved,
                                                     check_mask_untouched,
                                                     check_sample)
from materials_vision.augmentation.mask_aware import (PoreBrightnessField,
                                                      PoreDarkening)
from materials_vision.augmentation.policy import AugmentationPolicy
from materials_vision.augmentation.records import (AugmentationRecord,
                                                   AugmentedSample,
                                                   TransformRecord)
from materials_vision.augmentation.scale import MultiScaleCrop
from materials_vision.augmentation.structural import SyntheticSeptum
from materials_vision.augmentation.walls import (WallSample, WallSummary,
                                                 measure_walls,
                                                 summarize_walls)

__all__ = [
    "FAMILY_BLUR",
    "FAMILY_MASK_AWARE",
    "FAMILY_ORIENTATION",
    "FAMILY_SCALE",
    "FAMILY_SEPTUM",
    "FAMILY_TONAL",
    "MASK_CHANGING_FAMILIES",
    "AugmentationPolicy",
    "AugmentationRecord",
    "AugmentedSample",
    "BlurConfig",
    "IntegrityError",
    "MaskAwareConfig",
    "MultiScaleCrop",
    "OrientationConfig",
    "PolicyConfig",
    "PoreBrightnessField",
    "PoreDarkening",
    "ScaleConfig",
    "SeptumConfig",
    "SyntheticSeptum",
    "TonalConfig",
    "TransformRecord",
    "WallSample",
    "WallSummary",
    "check_connectivity",
    "check_labels_preserved",
    "check_mask_untouched",
    "check_sample",
    "enabled_families",
    "measure_walls",
    "policy_run_metadata",
    "summarize_walls",
]
