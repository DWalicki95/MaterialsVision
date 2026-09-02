"""
How closely the predicted pore outlines follow the annotated ones.

Instance matching says whether the model found a pore. It says very
little about whether it drew it in the right place: two masks can
overlap well past the matching threshold and still disagree about
where the wall runs. This module scores that disagreement, and it is
the first tie-break when two policies score alike on detection.

**Tolerance is relative, not a pixel count.** A boundary score needs a
distance within which two outlines count as the same, and here that
distance is a fraction of the mean annotated pore diameter of the
image rather than a fixed number of pixels. The dataset was acquired
on two microscopes calibrated at 3.24 and 2.48 um/px, so the same
physical pore covers more pixels on one than on the other; a fixed
tolerance of, say, 2 px would mean 6.48 um on one microscope and
4.96 um on the other, and would therefore be physically stricter for
one of them. Since microscope and foam family coincide in this
dataset, that would push a measurement artefact straight into the
per-family comparison, where it could not be told apart from a real
material effect. A relative tolerance removes the confound by
construction.

The tolerance is derived from the **ground truth** on purpose. Taking
it from the prediction would let a model that runs pores together earn
itself a larger tolerance, which is the wrong incentive.

**Outlines on the frame edge are removed before scoring.** Where a
pore runs off the side of the image, its outline follows the edge of
the frame, and that line is an artefact of cropping rather than a pore
wall - both masks are guaranteed to agree on it, for no merit of the
model. On this dataset those pixels are 9.3% of all outline pixels on
average and up to 18.9% on a single image, and leaving them in
inflates the score by about +0.027, at times +0.066: larger than the
difference between augmentation policies that this metric exists to
resolve. The exclusion is therefore not a refinement, it is a
correction without which the number would not mean what it says.

Everything else follows ``cellpose.metrics.boundary_scores`` exactly -
the same outline extraction, the same disk, the same counting - and
the test suite pins it to that reference on masks where no pore
touches the frame, i.e. wherever the two are supposed to agree.

Instances are scored together as one outline map rather than pair by
pair. A pooled score forgives a merge, whose false wall may sit on a
neighbour's real one, but merges are counted directly and separately
in ``matching``; keeping boundary geometry and instance topology in
separate metrics is clearer than one number that half-measures both.
"""
import logging
from dataclasses import dataclass

import numpy as np
from cellpose import utils
from scipy.ndimage import convolve

logger = logging.getLogger(__name__)

BOUNDARY_SCALES = (0.05, 0.1, 0.2)

DECISION_SCALE = 0.1

MIN_TOLERANCE_PX = 1.0


@dataclass(frozen=True)
class BoundaryScore:
    """Agreement of two outline maps at one tolerance.

    Parameters
    ----------
    scale : float
        Tolerance as a fraction of the mean annotated pore diameter.
    tolerance_px : float
        The distance that fraction worked out to on this image, in
        pixels of the frame as scored.
    precision : float
        Share of predicted outline that lies within tolerance of the
        annotated one. ``nan`` when the prediction has no outline.
    recall : float
        Share of annotated outline covered by the prediction. ``nan``
        when the annotation has no outline.
    f1 : float
        Harmonic mean of the two; ``nan`` when either is undefined.
    true_positive_px : int
        Pixels within tolerance of both outlines. Carried so that
        several images can be pooled into one score, which averaging
        their F1 values cannot reconstruct.
    false_positive_px : int
        Pixels near the predicted outline but not the annotated one.
    false_negative_px : int
        Pixels near the annotated outline but not the predicted one.
    """

    scale: float
    tolerance_px: float
    precision: float
    recall: float
    f1: float
    true_positive_px: int
    false_positive_px: int
    false_negative_px: int


def boundary_scores(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    scales: tuple[float, ...] = BOUNDARY_SCALES,
) -> dict[float, BoundaryScore]:
    """Score predicted outlines against annotated ones on one frame.

    Parameters
    ----------
    gt_labels : np.ndarray
        Ground-truth instance labels, ``(H, W)``, 0 as background.
    pred_labels : np.ndarray
        Predicted instance labels, same shape and convention.
    scales : tuple of float, optional
        Tolerances to score at, each a fraction of the mean annotated
        pore diameter. The full curve is reported so that the result's
        sensitivity to the tolerance is visible rather than asserted;
        ``DECISION_SCALE`` is the one the checkpoint tie-break reads.

    Returns
    -------
    dict
        One :class:`BoundaryScore` per requested scale, keyed by it.

    Raises
    ------
    ValueError
        If the two label images describe different frames, are not
        2-D integer arrays, or a scale is not positive.
    """
    _validate(gt_labels, pred_labels, scales)

    gt_outline = _outline_without_frame_edge(gt_labels)
    pred_outline = _outline_without_frame_edge(pred_labels)
    mean_diameter_px = _mean_annotated_diameter_px(gt_labels)

    return {
        scale: _score_at_scale(
            gt_outline, pred_outline, scale, mean_diameter_px
        )
        for scale in scales
    }


def _score_at_scale(
    gt_outline: np.ndarray,
    pred_outline: np.ndarray,
    scale: float,
    mean_diameter_px: float,
) -> BoundaryScore:
    """Compare two outline maps allowing one tolerance of slack.

    Each outline is thickened by a disk of the tolerance radius, so a
    predicted outline pixel counts as correct when an annotated one
    lies within that distance. This is the step
    ``cellpose.metrics.boundary_scores`` performs, reproduced here so
    that the frame edge can be removed first.

    Parameters
    ----------
    gt_outline : np.ndarray
        Boolean outline map of the annotation.
    pred_outline : np.ndarray
        Boolean outline map of the prediction.
    scale : float
    mean_diameter_px : float
        Mean annotated pore diameter; 0 when nothing is annotated.

    Returns
    -------
    BoundaryScore
    """
    if mean_diameter_px <= 0:
        return BoundaryScore(
            scale=float(scale), tolerance_px=float("nan"),
            precision=float("nan"), recall=float("nan"), f1=float("nan"),
            true_positive_px=0, false_positive_px=0, false_negative_px=0,
        )

    tolerance_px = max(MIN_TOLERANCE_PX, scale * mean_diameter_px)
    near_gt = _within_tolerance(gt_outline, tolerance_px)
    near_pred = _within_tolerance(pred_outline, tolerance_px)

    true_positive = int(np.count_nonzero(near_gt & near_pred))
    false_positive = int(np.count_nonzero(~near_gt & near_pred))
    false_negative = int(np.count_nonzero(near_gt & ~near_pred))

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return BoundaryScore(
        scale=float(scale),
        tolerance_px=float(tolerance_px),
        precision=precision,
        recall=recall,
        f1=_harmonic_mean(precision, recall),
        true_positive_px=true_positive,
        false_positive_px=false_positive,
        false_negative_px=false_negative,
    )


def _mean_annotated_diameter_px(gt_labels: np.ndarray) -> float:
    """Mean diameter of the annotated pores, the tolerance's yardstick.

    ``cellpose.utils.diameters`` averages over the instances present
    and would average an empty sequence on a frame with no annotation,
    so that case is answered here instead of letting numpy raise a
    warning and return ``nan`` from inside the library.

    Parameters
    ----------
    gt_labels : np.ndarray

    Returns
    -------
    float
        Mean equivalent diameter in pixels, or 0.0 when the frame
        carries no annotated instance.
    """
    if not np.any(gt_labels):
        logger.warning(
            "Frame carries no annotated instance, so no tolerance can "
            "be derived from it; reporting nan boundary scores."
        )
        return 0.0
    return float(utils.diameters(gt_labels)[0])


def _outline_without_frame_edge(labels: np.ndarray) -> np.ndarray:
    """Trace instance outlines, dropping those on the frame edge.

    Where an instance runs off the side of the image its outline
    follows the edge of the frame. That line is where the crop fell,
    not where a pore wall is, and both masks reproduce it identically,
    so counting it rewards the model for the shape of the image.

    Parameters
    ----------
    labels : np.ndarray

    Returns
    -------
    np.ndarray
        Boolean outline map with the first and last row and column
        cleared. A fresh array; the caller's labels are untouched.
    """
    outline = np.array(utils.masks_to_outlines(labels), copy=True)
    outline[0, :] = False
    outline[-1, :] = False
    outline[:, 0] = False
    outline[:, -1] = False
    return outline


def _within_tolerance(
    outline: np.ndarray, tolerance_px: float
) -> np.ndarray:
    """Mark every pixel lying within ``tolerance_px`` of an outline.

    Convolving a boolean map with a boolean disk keeps the result
    boolean, so the count of neighbouring outline pixels collapses to
    "at least one" - which is the question being asked. This mirrors
    how Cellpose performs the same step.
    """
    radius = int(np.ceil(tolerance_px))
    distances = utils.circleMask([radius, radius])[0]
    disk = (distances <= tolerance_px).astype(np.float32)
    return convolve(outline, disk)


def _validate(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    scales: tuple[float, ...],
) -> None:
    """Reject inputs the boundary score cannot be applied to."""
    for name, labels in (("gt_labels", gt_labels),
                         ("pred_labels", pred_labels)):
        if labels.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D, got shape {labels.shape}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                f"{name} must hold integer labels, got {labels.dtype}"
            )
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"gt_labels {gt_labels.shape} and pred_labels "
            f"{pred_labels.shape} describe different frames"
        )
    if not scales:
        raise ValueError("at least one scale is required")
    if any(scale <= 0 for scale in scales):
        raise ValueError(f"scales must be positive, got {scales}")


def _ratio(numerator: int, denominator: int) -> float:
    """Guarded ratio; an absent denominator gives ``nan``, not zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _harmonic_mean(precision: float, recall: float) -> float:
    """Harmonic mean that stays ``nan`` when either side is."""
    total = precision + recall
    if not np.isfinite(total) or total == 0:
        return float("nan")
    return 2 * precision * recall / total
