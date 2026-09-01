"""
Measuring how well a segmentation model reproduces the annotation.

Everything in this package compares a predicted instance mask against
the annotated one and reports on **the model**. The quantitative
characterisation of the material itself lives in
``materials_vision.quantitative_analysis`` and is a separate concern:
shape descriptors computed here answer "did the model reproduce the
outline it was shown", never "what is this foam like".
"""
from materials_vision.evaluation.boundary import (BOUNDARY_SCALES,
                                                  DECISION_SCALE,
                                                  BoundaryScore,
                                                  boundary_scores)
from materials_vision.evaluation.matching import (InstanceMatch, MatchedPair,
                                                  match_instances,
                                                  pore_count_error)
from materials_vision.evaluation.shape import (ANGLE_ELONGATION_THRESHOLD,
                                               InstanceShapes, PairShapeError,
                                               ShapeErrors, instance_shapes,
                                               shape_errors)
from materials_vision.evaluation.size_bins import (SIZE_BIN_LABELS, SizeBins,
                                                   SizeBinRecall,
                                                   SizeBinsLoadError,
                                                   calibrate_size_bins,
                                                   instance_areas_um2,
                                                   load_size_bins,
                                                   recall_per_size_bin)

__all__ = [
    "ANGLE_ELONGATION_THRESHOLD",
    "BOUNDARY_SCALES",
    "DECISION_SCALE",
    "SIZE_BIN_LABELS",
    "BoundaryScore",
    "InstanceMatch",
    "InstanceShapes",
    "MatchedPair",
    "PairShapeError",
    "ShapeErrors",
    "SizeBinRecall",
    "SizeBins",
    "SizeBinsLoadError",
    "boundary_scores",
    "calibrate_size_bins",
    "instance_areas_um2",
    "instance_shapes",
    "load_size_bins",
    "match_instances",
    "pore_count_error",
    "recall_per_size_bin",
    "shape_errors",
]
