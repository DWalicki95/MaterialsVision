"""
Scoring a model over a set of images, for reading rather than deciding.

Every figure comes out of the evaluation package unchanged: the images
are segmented and measured by ``score_settings`` and folded by
``aggregate`` and ``cross_sections``, exactly as the scoring script
does it. Nothing here selects a model or a setting; the result is for
figures and tables.

**The material metrics are also given relative to the annotation.**
The evaluation reports them in their natural units - percentage points
of porosity, micrometres of diameter - which cannot be compared across
quantities or materials. Here each drift is also divided by the size of
what it drifted from, so that "the median diameter moved by 3 um" reads
as "by 2 percent of the annotated median". The denominators are
computed from the same instances the drift was: the annotated diameters
of non-border pores, over the images the subset kept.

**Number density is not given separately.** Pooled over a subset it is
the pore count divided by the total frame area on both sides, so its
relative error is the pore count error to the last digit.

**The size bias of matched pores is also split by pore size.** A pair
is placed in the size class of its annotated pore, on the same frozen
class edges the evaluation reports recall on, so the two read side by
side: how often a pore of that size is found, and how large it is drawn
when it is. Only matched pairs can be split this way - a prediction
with no annotated counterpart has no annotated size to be classed by -
and pairs touching the frame edge are left out as they are from every
shape metric, because the frame cut their diameter.
"""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from materials_vision.data.samples import SampleSource
from materials_vision.evaluation.aggregate import (OVERALL_LABEL,
                                                   AggregateResult,
                                                   ImageEvaluation, aggregate,
                                                   cross_sections)
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.inference import score_settings
from materials_vision.evaluation.materials import (interquartile_range,
                                                   measurable_diameters_um)
from materials_vision.evaluation.size_bins import SIZE_BIN_LABELS, SizeBins
from materials_vision.evaluation.watershed import (DEPLOYED_POSTPROCESSING,
                                                   WatershedParams)

logger = logging.getLogger(__name__)

PERCENT = 100.0

MATERIAL_KEY = "material"


@dataclass(frozen=True)
class SizeBinBias:
    """How large the matched pores of one size class are drawn.

    Parameters
    ----------
    label : str
        The size class, ``Q1`` for the smallest pores.
    n_pairs : int
        Matched non-border pairs whose annotated pore is in the class.
    median_pct : float
        Median of ``d_pred / d_gt - 1`` in percent, taken as the median
        of the log ratio and converted back, as the evaluation does it;
        positive when pores are drawn too large. ``nan`` when the class
        holds no pair.
    """

    label: str
    n_pairs: int
    median_pct: float


@dataclass(frozen=True)
class MaterialDrift:
    """Material metrics as a signed percentage of the annotated value.

    Parameters
    ----------
    pore_count_pct : float
        ``100 (n_pred - n_gt) / n_gt``, pooled.
    porosity_pct : float
        Mean per-image porosity drift over the mean annotated porosity.
    median_diameter_pct : float
        Shift of the median equivalent diameter over the annotated
        median.
    iqr_diameter_pct : float
        Change of the interquartile range over the annotated one.
    wasserstein_pct : float
        Wasserstein distance between the diameter distributions over
        the annotated median diameter. Unsigned: a distance has no
        direction.

    Every field is ``nan`` where its denominator is zero or undefined.
    """

    pore_count_pct: float
    porosity_pct: float
    median_diameter_pct: float
    iqr_diameter_pct: float
    wasserstein_pct: float


@dataclass(frozen=True)
class SubsetScores:
    """What one subset of images scored.

    Parameters
    ----------
    result : AggregateResult
        The evaluation's figures, unchanged.
    drift : MaterialDrift
        The material metrics relative to the annotation.
    size_bias : tuple of SizeBinBias
        Size bias of matched pores per size class, smallest first.
    """

    result: AggregateResult
    drift: MaterialDrift
    size_bias: tuple[SizeBinBias, ...]


def score_model(
    segmenter: Any,
    source: SampleSource,
    size_bins: SizeBins,
    settings: WatershedParams = DEPLOYED_POSTPROCESSING,
    positions: Optional[Sequence[int]] = None,
) -> list[ImageEvaluation]:
    """Segment and measure every image of a source.

    Boundary agreement is scored at the decision tolerance only, as the
    scoring script does for every snapshot.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
    source : SampleSource
    size_bins : SizeBins
        The frozen pore size classes.
    settings : WatershedParams, optional
        Defaults to the post-processing the deployed model runs with.
    positions : sequence of int, optional
        Which images to measure; all of them by default.

    Returns
    -------
    list of ImageEvaluation
        In the order measured.
    """
    chosen = (
        list(range(len(source))) if positions is None else list(positions)
    )
    measured = score_settings(
        segmenter, source, chosen, size_bins, (settings,),
        boundary_scales=(DECISION_SCALE,),
    )
    return measured[settings]


def scores_by_material(
    evaluations: Sequence[ImageEvaluation],
    size_bins: SizeBins,
) -> dict[str, SubsetScores]:
    """Scores of all images together and of each material apart.

    Parameters
    ----------
    evaluations : sequence of ImageEvaluation
    size_bins : SizeBins
        The frozen pore size classes the images were scored with.

    Returns
    -------
    dict of str to SubsetScores
        ``"overall"`` first, then one entry per material in name order.
    """
    groups: dict[str, list[ImageEvaluation]] = {OVERALL_LABEL: list(
        evaluations
    )}
    for evaluation in evaluations:
        groups.setdefault(
            str(getattr(evaluation.record, MATERIAL_KEY)), []
        ).append(evaluation)
    results = {OVERALL_LABEL: aggregate(evaluations)}
    results.update({
        result.label.split("=", 1)[1]: result
        for result in cross_sections(evaluations, MATERIAL_KEY)
    })
    return {
        name: SubsetScores(
            result=result,
            drift=material_drift(result, groups[name]),
            size_bias=size_bias_by_bin(groups[name], size_bins),
        )
        for name, result in results.items()
    }


def size_bias_by_bin(
    evaluations: Sequence[ImageEvaluation], size_bins: SizeBins
) -> tuple[SizeBinBias, ...]:
    """Size bias of matched pores, per size class of the annotated pore.

    Parameters
    ----------
    evaluations : sequence of ImageEvaluation
        Close-ups are left out, as ``aggregate`` leaves them out.
    size_bins : SizeBins

    Returns
    -------
    tuple of SizeBinBias
        One per class, smallest first.
    """
    log_ratios: list[list[float]] = [[] for _ in SIZE_BIN_LABELS]
    for item in evaluations:
        if item.is_scale_outlier or not item.shape.pairs:
            continue
        classes = size_bins.assign(_pair_gt_areas_um2(item))
        for size_class, pair in zip(classes, item.shape.pairs):
            log_ratios[int(size_class)].append(pair.diameter_log_ratio)
    return tuple(
        SizeBinBias(
            label=label,
            n_pairs=len(values),
            median_pct=_median_pct(np.asarray(values, dtype=float)),
        )
        for label, values in zip(SIZE_BIN_LABELS, log_ratios)
    )


def _pair_gt_areas_um2(evaluation: ImageEvaluation) -> np.ndarray:
    """Area of the annotated pore of every shape pair, in um2.

    Recovered from the equivalent diameter, which the evaluation derives
    from the pixel count, and rounded back to whole pixels, so that a
    pore lying exactly on a class edge is classed as it is for recall
    rather than wherever floating-point round-off puts it.
    """
    shapes = evaluation.gt_shapes
    pixel_area_um2 = evaluation.record.pixel_size_um ** 2
    area_px = np.rint(
        np.pi * (shapes.equivalent_diameter_um / 2.0) ** 2 / pixel_area_um2
    )
    by_label = dict(zip(shapes.label_ids.tolist(), area_px.tolist()))
    return np.array(
        [by_label[pair.gt_id] for pair in evaluation.shape.pairs],
        dtype=float,
    ) * pixel_area_um2


def _median_pct(log_ratios: np.ndarray) -> float:
    """Median log ratio as a percentage, ``nan`` for no finite value."""
    finite = log_ratios[np.isfinite(log_ratios)]
    if finite.size == 0:
        return float("nan")
    return PERCENT * float(np.expm1(np.median(finite)))


def material_drift(
    result: AggregateResult, evaluations: Sequence[ImageEvaluation]
) -> MaterialDrift:
    """Express a subset's material metrics relative to the annotation.

    Parameters
    ----------
    result : AggregateResult
        The subset's figures, as ``aggregate`` returned them.
    evaluations : sequence of ImageEvaluation
        The images that subset was aggregated from. Close-ups are left
        out here as ``aggregate`` left them out, so that numerator and
        denominator describe the same images.

    Returns
    -------
    MaterialDrift
    """
    kept = [item for item in evaluations if not item.is_scale_outlier]
    gt_diameters_um = _pooled_gt_diameters_um(kept)
    gt_median_um = (
        float(np.median(gt_diameters_um)) if gt_diameters_um.size
        else float("nan")
    )
    gt_iqr_um = (
        interquartile_range(gt_diameters_um) if gt_diameters_um.size
        else float("nan")
    )
    gt_porosity = (
        float(np.mean([item.porosity.gt for item in kept])) if kept
        else float("nan")
    )
    return MaterialDrift(
        pore_count_pct=PERCENT * result.pore_count_error,
        # pp divided by a fraction is already a percentage of it.
        porosity_pct=_relative(result.mean_porosity_error_pp, gt_porosity,
                               scale=1.0),
        median_diameter_pct=_relative(result.median_diameter_drift_um,
                                      gt_median_um),
        iqr_diameter_pct=_relative(result.iqr_drift_um, gt_iqr_um),
        wasserstein_pct=_relative(result.wasserstein_um, gt_median_um),
    )


def _pooled_gt_diameters_um(
    evaluations: Sequence[ImageEvaluation],
) -> np.ndarray:
    """Annotated diameters of non-border pores across the images."""
    parts = [measurable_diameters_um(item.gt_shapes) for item in evaluations]
    if not parts:
        return np.empty(0, dtype=float)
    return np.concatenate(parts)


def _relative(
    difference: float, reference: float, scale: float = PERCENT
) -> float:
    """A difference over its reference, ``nan`` where that is undefined."""
    if not np.isfinite(reference) or reference == 0:
        return float("nan")
    return scale * difference / reference
