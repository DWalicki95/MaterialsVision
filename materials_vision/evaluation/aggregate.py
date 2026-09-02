"""
Turning per-image measurements into the numbers a run reports.

The modules beside this one each answer one question about one image.
This one runs them all over an image, and then folds many images into
the figures a comparison is actually decided on - overall, and broken
down by formulation, scale, material and microscope.

**Micro and macro are different questions, so both are reported.** A
micro figure pools the raw counts across images and divides once, so
an image with 200 pores weighs four times an image with 50; it answers
"across everything evaluated, what fraction was right". A macro figure
scores each image and averages the scores, so every image weighs the
same; it answers "on a typical image, how well does this do". The
primary metric is the micro one, but a policy that improves the micro
figure while lowering the macro one has helped the crowded images at
the expense of the sparse ones, and that is worth seeing.

An image where a metric is undefined - no annotation, no prediction -
contributes nothing to the micro pool and drops out of the macro
average rather than entering it as a zero, which would assert a
failure that was never measured.

**Close-up images are excluded from every subset automatically.** Six
images in the dataset were taken at three to thirteen times the usual
magnification and are a visually different imaging task; they stay in
training but are not evaluated. Leaving that to the caller would make
it a step someone can forget, so it happens here and the count of what
was dropped travels with the result.

**Distribution metrics are pooled, not averaged.** The Wasserstein
distance of a formulation is computed between all its annotated
diameters and all its predicted ones, not as the mean of its
per-image distances: the question is what the size report for that
formulation would look like, and that report is built from every pore
at once. The same holds for the orientation statistics. Per-image
shape errors are pooled as a median over all pairs, for the same
reason.
"""
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.boundary import (BOUNDARY_SCALES,
                                                  BoundaryScore,
                                                  boundary_scores)
from materials_vision.evaluation.matching import (InstanceMatch,
                                                  match_instances,
                                                  pore_count_error)
from materials_vision.evaluation.materials import (AreaNumberDensity,
                                                   OrientationDistribution,
                                                   PorosityError,
                                                   area_number_density,
                                                   diameter_distribution_error,
                                                   orientation_distribution,
                                                   porosity_error)
from materials_vision.evaluation.shape import (InstanceShapes, ShapeErrors,
                                               instance_shapes, shape_errors)
from materials_vision.evaluation.size_bins import (SIZE_BIN_LABELS, SizeBins,
                                                   SizeBinRecall,
                                                   recall_per_size_bin)

logger = logging.getLogger(__name__)

SCALE_OUTLIER_BIN = "outlier"

CROSS_SECTION_KEYS = ("formulation", "material", "microscope", "scale_bin")

OVERALL_LABEL = "overall"


@dataclass(frozen=True)
class ImageEvaluation:
    """Every measurement taken on one image.

    The instance shapes of both sides are kept rather than discarded
    after the per-image figures are computed, because the
    distribution metrics of a subset are pooled from the individual
    pores, and an average of per-image summaries is not the same
    number.

    Parameters
    ----------
    record : SampleRecord
        Identity and provenance of the image: which formulation,
        material, microscope and scale bin it belongs to.
    match : InstanceMatch
    boundary : dict
        One :class:`BoundaryScore` per tolerance, keyed by scale.
    size_bins : tuple of SizeBinRecall
    shape : ShapeErrors
    porosity : PorosityError
    density : AreaNumberDensity
    gt_shapes : InstanceShapes
    pred_shapes : InstanceShapes
    """

    record: SampleRecord
    match: InstanceMatch
    boundary: dict[float, BoundaryScore]
    size_bins: tuple[SizeBinRecall, ...]
    shape: ShapeErrors
    porosity: PorosityError
    density: AreaNumberDensity
    gt_shapes: InstanceShapes
    pred_shapes: InstanceShapes

    @property
    def is_scale_outlier(self) -> bool:
        """Whether this image is a close-up left out of evaluation.

        Returns
        -------
        bool
        """
        return self.record.scale_bin == SCALE_OUTLIER_BIN


@dataclass(frozen=True)
class AggregateResult:
    """The figures reported for one subset of images.

    Parameters
    ----------
    label : str
        Which subset this is, e.g. ``"material=AS"``.
    n_images : int
        Images the figures were computed on, after close-ups were
        dropped. Reported beside every result, because some subsets of
        this dataset are thin enough that the count changes how the
        number should be read.
    n_scale_outliers_excluded : int
        Close-up images dropped from this subset.
    n_gt, n_pred : int
        Annotated and predicted instances pooled over the subset.
    n_true_positives, n_false_positives, n_false_negatives : int
        The pooled counts the micro figures below are derived from,
        reported in their own right so that a reader can recompute
        any of them.
    precision, recall, f1 : float
        Micro figures: counts pooled, then divided once.
    macro_precision, macro_recall, macro_f1 : float
        Per-image figures averaged, skipping images where undefined.
    mean_pair_iou : float
        Mean overlap of the matched pairs, pooled.
    n_merges, n_splits : int
        Predictions that ran pores together, and pores cut apart.
    merges_per_100_gt, splits_per_100_gt : float
        The same relative to the annotation, so subsets of different
        pore density can be compared.
    pore_count_error : float
        Signed, pooled: ``(sum n_pred - sum n_gt) / sum n_gt``.
    macro_abs_pore_count_error : float
        Mean of the per-image absolute errors. This is the one the
        checkpoint tie-break reads, since a subset whose images err in
        opposite directions would look error-free pooled.
    boundary_f1, macro_boundary_f1 : dict
        Boundary agreement per tolerance, pooled over outline pixels
        and averaged over images respectively.
    size_bin_recall : tuple of SizeBinRecall
        Recall per pore size class, pooled.
    median_diameter_error, median_elongation_error : float
    median_angle_error_deg : float
        Per-pair shape errors, pooled as medians over every usable
        pair in the subset.
    n_shape_pairs : int
        Pairs those medians rest on, after border pairs were dropped.
    wasserstein_um : float
        Distance between the pooled annotated and predicted diameter
        distributions, in micrometres.
    median_diameter_drift_um, iqr_drift_um : float
        Signed shift and spread of the same pooled distributions.
    mean_porosity_error_pp : float
        Mean signed areal porosity error, in percentage points.
    gt_per_mm2, pred_per_mm2 : float
        Pore number density, pooled as total count over total area.
    orientation : OrientationDistribution
        Angle statistics of the pooled instances.
    """

    label: str
    n_images: int
    n_scale_outliers_excluded: int
    n_gt: int
    n_pred: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_pair_iou: float
    n_merges: int
    n_splits: int
    merges_per_100_gt: float
    splits_per_100_gt: float
    pore_count_error: float
    macro_abs_pore_count_error: float
    boundary_f1: dict[float, float]
    macro_boundary_f1: dict[float, float]
    size_bin_recall: tuple[SizeBinRecall, ...]
    median_diameter_error: float
    median_elongation_error: float
    median_angle_error_deg: float
    n_shape_pairs: int
    wasserstein_um: float
    median_diameter_drift_um: float
    iqr_drift_um: float
    mean_porosity_error_pp: float
    gt_per_mm2: float
    pred_per_mm2: float
    orientation: OrientationDistribution


def evaluate_image(
    record: SampleRecord,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    size_bins: SizeBins,
    boundary_scales: tuple[float, ...] = BOUNDARY_SCALES,
) -> ImageEvaluation:
    """Run every metric over one annotated and predicted pair.

    Parameters
    ----------
    record : SampleRecord
        The image's row of the frozen split, carrying the physical
        pixel size and the keys every cross-section is sliced on.
    gt_labels : np.ndarray
        Ground-truth instance labels at content resolution, i.e. after
        the deterministic panel crop.
    pred_labels : np.ndarray
        Predicted labels on the same frame.
    size_bins : SizeBins
        Frozen pore size classes, read from their artifact rather than
        recomputed, so that two runs describe the same classes.
    boundary_scales : tuple of float, optional
        Boundary tolerances to score at.

    Returns
    -------
    ImageEvaluation
    """
    pixel_size_um = record.pixel_size_um
    match = match_instances(gt_labels, pred_labels)
    gt_shapes = instance_shapes(gt_labels, pixel_size_um=pixel_size_um)
    pred_shapes = instance_shapes(pred_labels, pixel_size_um=pixel_size_um)

    matched_gt_ids = np.array([pair.gt_id for pair in match.pairs],
                              dtype=np.int64)
    return ImageEvaluation(
        record=record,
        match=match,
        boundary=boundary_scores(gt_labels, pred_labels,
                                 scales=boundary_scales),
        size_bins=recall_per_size_bin(gt_labels, matched_gt_ids, size_bins,
                                      pixel_size_um=pixel_size_um),
        shape=shape_errors(match, gt_shapes, pred_shapes),
        porosity=porosity_error(gt_labels, pred_labels),
        density=area_number_density(gt_labels, pred_labels,
                                    pixel_size_um=pixel_size_um),
        gt_shapes=gt_shapes,
        pred_shapes=pred_shapes,
    )


def aggregate(
    evaluations: Iterable[ImageEvaluation],
    *,
    label: str = OVERALL_LABEL,
) -> AggregateResult:
    """Fold per-image measurements into one subset's figures.

    Close-up images are dropped here rather than by the caller, and
    the number dropped is reported, so that the exclusion cannot be
    forgotten in one place and applied in another.

    Parameters
    ----------
    evaluations : iterable of ImageEvaluation
    label : str, optional
        Name for the subset, used in reports.

    Returns
    -------
    AggregateResult
    """
    kept, n_dropped = _drop_scale_outliers(evaluations)
    if not kept:
        logger.warning(
            "Subset %r holds no evaluable image (%d close-up(s) "
            "dropped); its figures are all undefined.", label, n_dropped,
        )
    return _assemble(label, kept, n_dropped)


def cross_sections(
    evaluations: Iterable[ImageEvaluation], key: str
) -> tuple[AggregateResult, ...]:
    """Aggregate separately for each value of one grouping key.

    Parameters
    ----------
    evaluations : iterable of ImageEvaluation
    key : str
        One of ``formulation``, ``material``, ``microscope`` or
        ``scale_bin``.

    Returns
    -------
    tuple of AggregateResult
        One per value present, ordered by value.

    Raises
    ------
    ValueError
        If the key is not one the split can be sliced on.
    """
    if key not in CROSS_SECTION_KEYS:
        raise ValueError(
            f"{key!r} is not a cross-section key; expected one of "
            f"{CROSS_SECTION_KEYS}"
        )

    groups: dict[str, list[ImageEvaluation]] = {}
    for evaluation in evaluations:
        value = str(getattr(evaluation.record, key))
        groups.setdefault(value, []).append(evaluation)

    return tuple(
        aggregate(groups[value], label=f"{key}={value}")
        for value in sorted(groups)
    )


def scale_outlier_report(
    evaluations: Iterable[ImageEvaluation],
) -> AggregateResult:
    """Figures for the close-up images, which no subset includes.

    They are reported separately rather than silently dropped: an
    excluded image should be visible as excluded, with its own
    numbers, not absent.

    Parameters
    ----------
    evaluations : iterable of ImageEvaluation

    Returns
    -------
    AggregateResult
    """
    outliers = [item for item in evaluations if item.is_scale_outlier]
    return _assemble("scale_outlier", outliers, 0)


def _drop_scale_outliers(
    evaluations: Iterable[ImageEvaluation],
) -> tuple[list[ImageEvaluation], int]:
    """Split off the close-ups, returning what remains and how many."""
    kept = []
    n_dropped = 0
    for evaluation in evaluations:
        if evaluation.is_scale_outlier:
            n_dropped += 1
        else:
            kept.append(evaluation)
    return kept, n_dropped


def _assemble(
    label: str,
    evaluations: Sequence[ImageEvaluation],
    n_dropped: int,
) -> AggregateResult:
    """Compute every reported figure for an already-filtered subset."""
    n_gt = sum(item.match.n_gt for item in evaluations)
    n_pred = sum(item.match.n_pred for item in evaluations)
    true_positive = sum(item.match.true_positives for item in evaluations)
    false_positive = sum(item.match.false_positives for item in evaluations)
    false_negative = sum(item.match.false_negatives for item in evaluations)

    n_merges = sum(item.match.n_merges for item in evaluations)
    n_splits = sum(item.match.n_splits for item in evaluations)
    signed_count_error, _ = pore_count_error(n_gt, n_pred)

    pooled_gt, pooled_pred = _pooled_shapes(evaluations)

    distribution = diameter_distribution_error(pooled_gt, pooled_pred)
    pairs = [pair for item in evaluations for pair in item.shape.pairs]

    return AggregateResult(
        label=label,
        n_images=len(evaluations),
        n_scale_outliers_excluded=n_dropped,
        n_gt=n_gt,
        n_pred=n_pred,
        n_true_positives=true_positive,
        n_false_positives=false_positive,
        n_false_negatives=false_negative,
        precision=_ratio(true_positive, true_positive + false_positive),
        recall=_ratio(true_positive, true_positive + false_negative),
        f1=_f1(true_positive, false_positive, false_negative),
        macro_precision=_mean_over_images(evaluations,
                                          lambda e: e.match.precision),
        macro_recall=_mean_over_images(evaluations,
                                       lambda e: e.match.recall),
        macro_f1=_mean_over_images(evaluations, lambda e: e.match.f1),
        mean_pair_iou=_mean_of([pair.iou for item in evaluations
                                for pair in item.match.pairs]),
        n_merges=n_merges,
        n_splits=n_splits,
        merges_per_100_gt=_per_hundred(n_merges, n_gt),
        splits_per_100_gt=_per_hundred(n_splits, n_gt),
        pore_count_error=signed_count_error,
        macro_abs_pore_count_error=_mean_over_images(
            evaluations,
            lambda e: pore_count_error(e.match.n_gt, e.match.n_pred)[1],
        ),
        boundary_f1=_pooled_boundary(evaluations),
        macro_boundary_f1=_macro_boundary(evaluations),
        size_bin_recall=_pooled_size_bins(evaluations),
        median_diameter_error=_median_of(
            [pair.diameter_error for pair in pairs]
        ),
        median_elongation_error=_median_of(
            [pair.elongation_error for pair in pairs]
        ),
        median_angle_error_deg=_median_of(
            [pair.angle_error_deg for pair in pairs]
        ),
        n_shape_pairs=len(pairs),
        wasserstein_um=distribution.wasserstein_um,
        median_diameter_drift_um=distribution.median_error_um,
        iqr_drift_um=distribution.iqr_error_um,
        mean_porosity_error_pp=_mean_over_images(
            evaluations, lambda e: e.porosity.error_pp
        ),
        gt_per_mm2=_density(evaluations, "n_gt"),
        pred_per_mm2=_density(evaluations, "n_pred"),
        orientation=orientation_distribution(pooled_gt, pooled_pred),
    )


def _pooled_shapes(
    evaluations: Sequence[ImageEvaluation],
) -> tuple[InstanceShapes, InstanceShapes]:
    """Concatenate both sides' instances across the subset.

    Labels lose their meaning once images are pooled - the same label
    occurs on every image - so they are renumbered consecutively.
    Nothing downstream of pooling looks an instance up by label; the
    distribution metrics only read the value arrays.
    """
    return (_concatenate([item.gt_shapes for item in evaluations]),
            _concatenate([item.pred_shapes for item in evaluations]))


def _concatenate(parts: Sequence[InstanceShapes]) -> InstanceShapes:
    """Join several images' shapes into one set."""
    if not parts:
        empty = np.empty(0, dtype=float)
        return InstanceShapes(
            label_ids=np.empty(0, dtype=np.int64),
            equivalent_diameter_um=empty,
            elongation=empty,
            angle_deg=empty,
            border=np.empty(0, dtype=bool),
        )
    diameters = np.concatenate(
        [part.equivalent_diameter_um for part in parts]
    )
    return InstanceShapes(
        label_ids=np.arange(1, diameters.size + 1, dtype=np.int64),
        equivalent_diameter_um=diameters,
        elongation=np.concatenate([part.elongation for part in parts]),
        angle_deg=np.concatenate([part.angle_deg for part in parts]),
        border=np.concatenate([part.border for part in parts]),
    )


def _pooled_boundary(
    evaluations: Sequence[ImageEvaluation],
) -> dict[float, float]:
    """Boundary F1 per tolerance, over outline pixels of every image."""
    scores: dict[float, float] = {}
    for scale in _boundary_scales(evaluations):
        true_positive = sum(item.boundary[scale].true_positive_px
                            for item in evaluations)
        false_positive = sum(item.boundary[scale].false_positive_px
                             for item in evaluations)
        false_negative = sum(item.boundary[scale].false_negative_px
                             for item in evaluations)
        scores[scale] = _f1(true_positive, false_positive, false_negative)
    return scores


def _macro_boundary(
    evaluations: Sequence[ImageEvaluation],
) -> dict[float, float]:
    """Boundary F1 per tolerance, averaged over images."""
    return {
        scale: _mean_over_images(
            evaluations, lambda e, s=scale: e.boundary[s].f1
        )
        for scale in _boundary_scales(evaluations)
    }


def _boundary_scales(
    evaluations: Sequence[ImageEvaluation],
) -> tuple[float, ...]:
    """Tolerances every image in the subset was scored at.

    Raises
    ------
    ValueError
        If the images were not all scored at the same tolerances,
        which would make any pooled figure a mixture.
    """
    if not evaluations:
        return ()
    scales = tuple(sorted(evaluations[0].boundary))
    for item in evaluations[1:]:
        if tuple(sorted(item.boundary)) != scales:
            raise ValueError(
                "Images in this subset were scored at different "
                "boundary tolerances; they cannot be pooled"
            )
    return scales


def _pooled_size_bins(
    evaluations: Sequence[ImageEvaluation],
) -> tuple[SizeBinRecall, ...]:
    """Recall per size class, over every instance in the subset."""
    pooled = []
    for index, label in enumerate(SIZE_BIN_LABELS):
        n_gt = sum(item.size_bins[index].n_gt for item in evaluations)
        n_matched = sum(item.size_bins[index].n_matched
                        for item in evaluations)
        pooled.append(SizeBinRecall(
            label=label, n_gt=n_gt, n_matched=n_matched,
            recall=_ratio(n_matched, n_gt),
        ))
    return tuple(pooled)


def _density(evaluations: Sequence[ImageEvaluation], side: str) -> float:
    """Instances per square millimetre, pooled over the subset.

    A density over several frames is the total count over the total
    area; averaging per-image densities would weigh a small frame like
    a large one.
    """
    total = sum(getattr(item.density, side) for item in evaluations)
    area_mm2 = sum(item.density.frame_area_mm2 for item in evaluations)
    if area_mm2 <= 0:
        return float("nan")
    return total / area_mm2


def _mean_over_images(
    evaluations: Sequence[ImageEvaluation], select
) -> float:
    """Average a per-image figure, skipping where it is undefined."""
    return _mean_of([select(item) for item in evaluations])


def _mean_of(values: Sequence[float]) -> float:
    """Mean that ignores undefined entries instead of spreading them."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.nanmean(array[np.isfinite(array)]))


def _median_of(values: Sequence[float]) -> float:
    """Median that ignores undefined entries."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.median(array[np.isfinite(array)]))


def _ratio(numerator: int, denominator: int) -> float:
    """Guarded ratio; an absent denominator gives ``nan``, not zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _f1(
    true_positive: int, false_positive: int, false_negative: int
) -> float:
    """F1 from counts, which stays defined when precision is not."""
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return float("nan")
    return 2 * true_positive / denominator


def _per_hundred(count: int, n_gt: int) -> float:
    """An event count relative to the annotation it happened in."""
    if n_gt == 0:
        return float("nan")
    return 100.0 * count / n_gt
