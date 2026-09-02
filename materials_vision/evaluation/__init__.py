"""
Measuring how well a segmentation model reproduces the annotation.

Everything in this package compares a predicted instance mask against
the annotated one and reports on **the model**. The quantitative
characterisation of the material itself lives in
``materials_vision.quantitative_analysis`` and is a separate concern:
shape descriptors computed here answer "did the model reproduce the
outline it was shown", never "what is this foam like".
"""
from materials_vision.evaluation.aggregate import (CROSS_SECTION_KEYS,
                                                   AggregateResult,
                                                   ImageEvaluation, aggregate,
                                                   cross_sections,
                                                   evaluate_image,
                                                   scale_outlier_report)
from materials_vision.evaluation.boundary import (BOUNDARY_SCALES,
                                                  DECISION_SCALE,
                                                  BoundaryScore,
                                                  boundary_scores)
from materials_vision.evaluation.matching import (InstanceMatch, MatchedPair,
                                                  match_instances,
                                                  pore_count_error)
from materials_vision.evaluation.materials import (AreaNumberDensity,
                                                   DiameterDistributionError,
                                                   OrientationDistribution,
                                                   PorosityError,
                                                   area_number_density,
                                                   diameter_distribution_error,
                                                   orientation_distribution,
                                                   porosity_error)
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
    "CROSS_SECTION_KEYS",
    "DECISION_SCALE",
    "SIZE_BIN_LABELS",
    "AggregateResult",
    "AreaNumberDensity",
    "BoundaryScore",
    "DiameterDistributionError",
    "ImageEvaluation",
    "InstanceMatch",
    "InstanceShapes",
    "MatchedPair",
    "OrientationDistribution",
    "PairShapeError",
    "PorosityError",
    "ShapeErrors",
    "SizeBinRecall",
    "SizeBins",
    "SizeBinsLoadError",
    "aggregate",
    "area_number_density",
    "boundary_scores",
    "calibrate_size_bins",
    "cross_sections",
    "diameter_distribution_error",
    "evaluate_image",
    "instance_areas_um2",
    "instance_shapes",
    "load_size_bins",
    "match_instances",
    "orientation_distribution",
    "pore_count_error",
    "porosity_error",
    "recall_per_size_bin",
    "scale_outlier_report",
    "shape_errors",
]
