"""
Looking at what a trained model does, image by image.

``materials_vision.evaluation`` turns predictions into the numbers a
run reports. This package is for everything around those numbers that
a person reads rather than a comparison decides on: overlays, figures,
browsers, qualitative inspection of where and how the model errs.

**Dependencies run one way only.** Modules here may import from
``evaluation``, ``data`` and ``training``; nothing outside this package
imports from it. A visualisation can therefore be added, changed or
deleted without any possibility of moving a reported figure, and a
test enforces the rule.

**Inference goes through the scoring path.** Models are loaded and
post-processed by the same functions the evaluation script uses, so
what is looked at here is what was scored, not a near copy of it.
"""
from materials_vision.inspection.browser import (ALL_MATERIALS,
                                                 PredictionBrowser,
                                                 image_choices)
from materials_vision.inspection.figures import comparison_figure, describe
from materials_vision.inspection.metric_charts import (METRICS, MetricSpec,
                                                       metric_figures,
                                                       metrics_table)
from materials_vision.inspection.model import (MODEL_TYPE,
                                               PRETRAINED_MODEL_TYPE,
                                               load_pretrained_segmenter,
                                               load_segmenter, predict)
from materials_vision.inspection.overlay import (MatchSummary, outcome_overlay,
                                                 pair_palette, paired_colours,
                                                 summarise_match)
from materials_vision.inspection.scoring import (MaterialDrift, SizeBinBias,
                                                 SubsetScores, material_drift,
                                                 score_model,
                                                 scores_by_material,
                                                 size_bias_by_bin)
from materials_vision.inspection.training_curves import (
    RunCurves, Series, load_run_curves, training_curves_figure)

__all__ = [
    "ALL_MATERIALS",
    "METRICS",
    "MODEL_TYPE",
    "PRETRAINED_MODEL_TYPE",
    "MatchSummary",
    "MaterialDrift",
    "MetricSpec",
    "PredictionBrowser",
    "RunCurves",
    "Series",
    "SizeBinBias",
    "SubsetScores",
    "comparison_figure",
    "describe",
    "image_choices",
    "load_pretrained_segmenter",
    "load_run_curves",
    "load_segmenter",
    "material_drift",
    "metric_figures",
    "metrics_table",
    "outcome_overlay",
    "pair_palette",
    "paired_colours",
    "predict",
    "score_model",
    "scores_by_material",
    "size_bias_by_bin",
    "summarise_match",
    "training_curves_figure",
]
