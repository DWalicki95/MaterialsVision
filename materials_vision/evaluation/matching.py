"""
Matching predicted pore instances against the annotated ones.

Everything measured downstream of a segmentation - how many pores were
found, how faithfully their shapes came out, how far the reported size
distribution drifts - rests on one question: which predicted instance
corresponds to which annotated one. This module answers that, and
nothing else.

Two instances correspond when their masks overlap enough, and "enough"
is mask IoU >= 0.5. The overlap is measured on pixels rather than on
bounding boxes: pores are frequently elongated and lie at an angle, so
a box drawn round one is much larger than the pore, and box overlap
would flatter the result exactly where the shape matters most.

The correspondence is resolved globally with
``scipy.optimize.linear_sum_assignment``, in the cost form the
segmentation ecosystem agrees on - Cellpose, elf and StarDist all use
it: maximise the number of pairs above the threshold and let IoU break
ties only. At a threshold of 0.5 the rule turns out not to matter, as a
prediction can exceed IoU 0.5 with at most one annotated pore, so any
sensible rule returns the same pairs; the ecosystem form costs one
scipy call and keeps the numbers comparable with published ones.

Beside the pairing, two failure modes are counted straight from the
overlap table, because a single F1 hides them and they are what
specific augmentation families are meant to fix:

- a **merge** is one prediction covering at least half of each of two
  or more different annotated pores - the model ran pores together;
- a **split** is one annotated pore at least half-covered by each of
  two or more different predictions - the model cut one pore in two.

The two are mirror images: exchanging the ground truth with the
prediction turns every merge into a split and back again. That
symmetry is the sharpest available check that neither is implemented
back to front.

Instances touching the frame edge are **kept** here. Their outline is
truncated, so their shape cannot be measured - but they are still
objects the model was supposed to find, and detection is what this
module scores. Excluding them belongs to the shape and distribution
metrics, not to the pairing.
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential

logger = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD = 0.5

MERGE_SPLIT_THRESHOLD = 0.5


@dataclass(frozen=True)
class MatchedPair:
    """One annotated pore and the prediction that corresponds to it.

    Parameters
    ----------
    gt_id : int
        Label of the instance in the ground-truth image, in its
        original numbering.
    pred_id : int
        Label of the instance in the predicted image, in its original
        numbering.
    iou : float
        Mask intersection over union of the two, at or above the
        matching threshold.
    """

    gt_id: int
    pred_id: int
    iou: float


@dataclass(frozen=True)
class InstanceMatch:
    """Correspondence between predicted and annotated instances.

    Counts follow the convention frozen for empty frames: a metric
    whose denominator is zero is ``nan`` rather than a made-up zero, so
    that averaging across images can skip it instead of being dragged
    towards a value nobody measured.

    Parameters
    ----------
    pairs : tuple of MatchedPair
        Instances that correspond, ordered by ground-truth label.
    n_gt : int
        Annotated instances in the frame.
    n_pred : int
        Predicted instances in the frame.
    unmatched_gt_ids : np.ndarray
        Annotated instances with no correspondence, in their original
        numbering. These are the false negatives.
    unmatched_pred_ids : np.ndarray
        Predicted instances with no correspondence, in their original
        numbering. These are the false positives.
    merged_pred_ids : np.ndarray
        Predictions that cover at least half of each of two or more
        annotated pores.
    split_gt_ids : np.ndarray
        Annotated pores at least half-covered by each of two or more
        predictions.
    iou_threshold : float
        Overlap a pair had to reach to count as a correspondence.
    """

    pairs: tuple[MatchedPair, ...]
    n_gt: int
    n_pred: int
    unmatched_gt_ids: np.ndarray
    unmatched_pred_ids: np.ndarray
    merged_pred_ids: np.ndarray
    split_gt_ids: np.ndarray
    iou_threshold: float

    @property
    def true_positives(self) -> int:
        """Predictions correctly matched to an annotated instance.

        Returns
        -------
        int
        """
        return len(self.pairs)

    @property
    def false_positives(self) -> int:
        """Predictions with no annotated counterpart.

        Returns
        -------
        int
        """
        return self.n_pred - self.true_positives

    @property
    def false_negatives(self) -> int:
        """Annotated instances the model did not find.

        Returns
        -------
        int
        """
        return self.n_gt - self.true_positives

    @property
    def precision(self) -> float:
        """Share of predictions that were correct.

        Returns
        -------
        float
            ``nan`` when nothing was predicted, since the question does
            not arise rather than being answered with zero.
        """
        return _ratio(self.true_positives, self.n_pred)

    @property
    def recall(self) -> float:
        """Share of annotated instances that were found.

        Returns
        -------
        float
            ``nan`` when the frame holds no annotation.
        """
        return _ratio(self.true_positives, self.n_gt)

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall.

        Computed from the counts as ``2 TP / (2 TP + FP + FN)``, which
        stays defined when precision or recall is not: a frame with
        annotations and no predictions scores 0, whereas a frame with
        neither scores ``nan`` and drops out of any average.

        Returns
        -------
        float
        """
        denominator = 2 * self.true_positives + (
            self.false_positives + self.false_negatives
        )
        if denominator == 0:
            return float("nan")
        return 2 * self.true_positives / denominator

    @property
    def mean_pair_iou(self) -> float:
        """Average overlap over the matched pairs.

        Returns
        -------
        float
            ``nan`` when nothing matched.
        """
        if not self.pairs:
            return float("nan")
        return float(np.mean([pair.iou for pair in self.pairs]))

    @property
    def n_merges(self) -> int:
        """Predictions that ran two or more annotated pores together.

        Returns
        -------
        int
        """
        return int(self.merged_pred_ids.size)

    @property
    def n_splits(self) -> int:
        """Annotated pores the model cut into two or more pieces.

        Returns
        -------
        int
        """
        return int(self.split_gt_ids.size)

    def per_hundred_gt(self, count: int) -> float:
        """Express a count relative to the annotation it happened in.

        Raw merge and split counts are not comparable between frames of
        different density: a typical AS image holds around 200 pores
        and a VAB image around 44, so the same eight merges are an
        error on 4% of instances in one and on 18% in the other.

        Parameters
        ----------
        count : int
            Raw number of events, e.g. ``n_merges``.

        Returns
        -------
        float
            Events per 100 annotated instances, ``nan`` for an empty
            annotation.
        """
        if self.n_gt == 0:
            return float("nan")
        return 100.0 * count / self.n_gt


def match_instances(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> InstanceMatch:
    """Pair predicted instances with annotated ones on one frame.

    Parameters
    ----------
    gt_labels : np.ndarray
        Ground-truth instance labels, ``(H, W)``, 0 as background.
        Labels need not be dense; they are reported back in their
        original numbering.
    pred_labels : np.ndarray
        Predicted instance labels, same shape and convention.
    iou_threshold : float, optional
        Smallest mask IoU that counts as a correspondence. Frozen at
        0.5 for this project; the parameter exists so the threshold can
        be swept in diagnostics, not so it can drift between runs.

    Returns
    -------
    InstanceMatch

    Raises
    ------
    ValueError
        If the two label images describe different frames, are not
        2-D integer arrays, hold negative labels, or the threshold
        falls outside ``(0, 1]``.
    """
    _validate(gt_labels, pred_labels, iou_threshold)

    gt_dense, gt_ids = _densify(gt_labels)
    pred_dense, pred_ids = _densify(pred_labels)
    n_gt, n_pred = gt_ids.size, pred_ids.size

    overlap, gt_areas, pred_areas = _contingency(gt_dense, pred_dense,
                                                 n_gt, n_pred)
    iou = _iou_from_overlap(overlap, gt_areas, pred_areas)
    gt_rows, pred_cols = _assign(iou, iou_threshold)

    pairs = tuple(
        MatchedPair(gt_id=int(gt_ids[row]),
                    pred_id=int(pred_ids[col]),
                    iou=float(iou[row, col]))
        for row, col in zip(gt_rows, pred_cols)
    )
    merged, split = _merges_and_splits(overlap, gt_areas, pred_areas)

    return InstanceMatch(
        pairs=pairs,
        n_gt=int(n_gt),
        n_pred=int(n_pred),
        unmatched_gt_ids=np.delete(gt_ids, gt_rows),
        unmatched_pred_ids=np.delete(pred_ids, pred_cols),
        merged_pred_ids=pred_ids[merged],
        split_gt_ids=gt_ids[split],
        iou_threshold=float(iou_threshold),
    )


def pore_count_error(n_gt: int, n_pred: int) -> tuple[float, float]:
    """Relative error in the number of pores found.

    The signed form says which way the model errs - too many
    instances usually means splitting, too few means merging or
    missed pores - and the absolute form is what the checkpoint
    tie-break orders on.

    Parameters
    ----------
    n_gt : int
        Annotated instances.
    n_pred : int
        Predicted instances.

    Returns
    -------
    tuple of float
        ``(signed, absolute)``; both ``nan`` when the frame holds no
        annotation, since there is nothing to be relative to.
    """
    if n_gt == 0:
        logger.warning(
            "Pore count error is undefined for a frame with no "
            "annotated instances; reporting nan."
        )
        return float("nan"), float("nan")
    signed = (n_pred - n_gt) / n_gt
    return float(signed), float(abs(signed))


def _validate(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float,
) -> None:
    """Reject inputs the matching cannot be applied to."""
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
        if labels.size and labels.min() < 0:
            raise ValueError(
                f"{name} holds negative labels; 0 is background and "
                f"instances are positive"
            )
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"gt_labels {gt_labels.shape} and pred_labels "
            f"{pred_labels.shape} describe different frames"
        )
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(
            f"iou_threshold must lie in (0, 1], got {iou_threshold}"
        )


def _densify(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber labels to ``1..n`` and report the original numbering.

    The contingency table below is indexed by label, so a labelling
    with gaps would size it by the largest label rather than by the
    instance count. Ground-truth masks are dense by construction, but
    a prediction from the instance decoder carries no such promise.

    The renumbering itself is
    ``skimage.segmentation.relabel_sequential``; all that is left here
    is unwrapping its ``ArrayMap`` return into a plain array of the
    original labels, so that results can be reported in the numbering
    the caller passed in.

    Parameters
    ----------
    labels : np.ndarray

    Returns
    -------
    tuple of np.ndarray
        Densely numbered labels, and the original label of each
        instance indexed by ``new_id - 1``.
    """
    relabelled, _, inverse_map = relabel_sequential(labels, offset=1)
    return relabelled, np.asarray(inverse_map)[1:]


def _contingency(
    gt_dense: np.ndarray,
    pred_dense: np.ndarray,
    n_gt: int,
    n_pred: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count shared pixels for every instance pair, in one pass.

    Flattening the label pair into a single index and counting it is
    what makes this a single pass over the frame instead of one pass
    per instance pair; at several hundred instances per image the
    difference is between milliseconds and minutes.

    Parameters
    ----------
    gt_dense : np.ndarray
        Ground-truth labels numbered ``1..n_gt``.
    pred_dense : np.ndarray
        Predicted labels numbered ``1..n_pred``.
    n_gt : int
    n_pred : int

    Returns
    -------
    tuple of np.ndarray
        Overlap matrix ``(n_gt, n_pred)`` without the background row
        and column, and the instance areas on each side.
    """
    width = n_pred + 1
    flat = gt_dense.ravel().astype(np.int64) * width + pred_dense.ravel()
    table = np.bincount(
        flat, minlength=(n_gt + 1) * width
    ).reshape(n_gt + 1, width)

    gt_areas = table.sum(axis=1)[1:]
    pred_areas = table.sum(axis=0)[1:]
    return table[1:, 1:], gt_areas, pred_areas


def _iou_from_overlap(
    overlap: np.ndarray,
    gt_areas: np.ndarray,
    pred_areas: np.ndarray,
) -> np.ndarray:
    """Turn shared-pixel counts into intersection over union."""
    if overlap.size == 0:
        return np.zeros(overlap.shape, dtype=float)
    union = gt_areas[:, None] + pred_areas[None, :] - overlap
    return np.divide(
        overlap, union, out=np.zeros(overlap.shape, dtype=float),
        where=union > 0,
    )


def _assign(
    iou: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the correspondence that maximises pairs above threshold.

    The cost mirrors Cellpose, elf and StarDist: a flat reward for
    clearing the threshold, plus a fraction of the IoU that can only
    settle ties between assignments with equally many valid pairs.

    Parameters
    ----------
    iou : np.ndarray
        Overlap matrix, instances only.
    iou_threshold : float

    Returns
    -------
    tuple of np.ndarray
        Row and column indices of the accepted pairs.
    """
    n_matchable = min(iou.shape)
    empty = np.empty(0, dtype=np.int64)
    if n_matchable == 0 or not np.any(iou >= iou_threshold):
        return empty, empty

    costs = -(iou >= iou_threshold).astype(float) - iou / (2 * n_matchable)
    rows, cols = linear_sum_assignment(costs)
    accepted = iou[rows, cols] >= iou_threshold
    return rows[accepted], cols[accepted]


def _merges_and_splits(
    overlap: np.ndarray,
    gt_areas: np.ndarray,
    pred_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find predictions that ran pores together and pores cut apart.

    A merge is read against the annotated areas (how much of each pore
    a prediction swallowed) and a split against the predicted ones (how
    much of each prediction one pore accounts for). The two criteria
    are transposes of one another, which is why exchanging the ground
    truth with the prediction exchanges merges with splits.

    Parameters
    ----------
    overlap : np.ndarray
        Shared-pixel counts, ``(n_gt, n_pred)``.
    gt_areas : np.ndarray
    pred_areas : np.ndarray

    Returns
    -------
    tuple of np.ndarray
        Indices of merging predictions and of split annotated
        instances, both zero-based into the dense numbering.
    """
    empty = np.empty(0, dtype=np.int64)
    if overlap.size == 0:
        return empty, empty

    covered_gt = _fraction(overlap, gt_areas[:, None])
    covered_pred = _fraction(overlap, pred_areas[None, :])

    merges = np.nonzero(
        (covered_gt >= MERGE_SPLIT_THRESHOLD).sum(axis=0) >= 2
    )[0]
    splits = np.nonzero(
        (covered_pred >= MERGE_SPLIT_THRESHOLD).sum(axis=1) >= 2
    )[0]
    return merges, splits


def _fraction(overlap: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Divide overlaps by areas, leaving 0 where the area is 0."""
    return np.divide(
        overlap, areas, out=np.zeros(overlap.shape, dtype=float),
        where=areas > 0,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Guarded ratio; an absent denominator gives ``nan``, not zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator
