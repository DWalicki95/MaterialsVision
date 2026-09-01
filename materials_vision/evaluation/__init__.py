"""
Measuring how well a segmentation model reproduces the annotation.

Everything in this package compares a predicted instance mask against
the annotated one and reports on **the model**. The quantitative
characterisation of the material itself lives in
``materials_vision.quantitative_analysis`` and is a separate concern:
shape descriptors computed here answer "did the model reproduce the
outline it was shown", never "what is this foam like".
"""
from materials_vision.evaluation.matching import (InstanceMatch, MatchedPair,
                                                  match_instances,
                                                  pore_count_error)

__all__ = [
    "InstanceMatch",
    "MatchedPair",
    "match_instances",
    "pore_count_error",
]
