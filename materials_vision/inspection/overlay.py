"""
Colouring a prediction and its annotation so that the errors show.

A plain label overlay colours each instance at random. That shows where
one pore ends and the next begins, but not which predicted pore belongs
to which annotated one: the same pore comes out green on one side and
purple on the other, and a missed or a spurious pore looks like any
other. Here the colours carry the matching the metrics are computed
from. A predicted pore and the annotated pore it was paired with share
one colour on both sides, and an instance with no counterpart is drawn
in a single reserved colour on whichever side it lives. A missed or a
spurious pore then reads as a lone reserved-colour one.

**A merge or a split does not always show as reserved colour alone.**
Pairing asks only for an overlap of one half, so a prediction running
two equal pores together still pairs with one of them: the merge reads
as a coloured pair with a reserved-colour neighbour on the annotated
side. Only a merge of three or more pores, or of pores too unequal for
either to reach that overlap, leaves every piece unpaired. The counts
of merges and splits come with the colouring for that reason.

**The pair colours never approach the reserved one.** Their hues are
spread over a band that leaves red out at both ends of the hue circle,
so a paired pore cannot be taken for an unpaired one however many pairs
an image holds.

**Outlines opaque, fills translucent.** Two unpaired instances side by
side share a colour; without an outline they would read as one region,
which is exactly the merge the picture is meant to expose.
"""
import logging
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.segmentation import find_boundaries

from materials_vision.evaluation.matching import InstanceMatch

logger = logging.getLogger(__name__)

UNPAIRED_RGB = np.array([0.95, 0.10, 0.10])

# Red sits at both ends of the hue circle, so the band is cut from each.
PAIR_HUE_RANGE = (0.12, 0.88)

PAIR_SATURATION = 0.85

PAIR_VALUE = 0.95

FILL_ALPHA = 0.4

RGB_CHANNELS = 3


@dataclass(frozen=True)
class MatchSummary:
    """What one image contributes to the pooled counts.

    Parameters
    ----------
    n_gt, n_pred : int
        Annotated and predicted instances.
    n_true_positives, n_false_positives, n_false_negatives : int
        Paired instances, unpaired predictions, unpaired annotations.
    n_merges, n_splits : int
        Predictions covering several annotated pores, and annotated
        pores cut by several predictions.
    f1 : float
        Of this image alone; ``nan`` when the frame holds no instance
        on either side. The F1 of a set of images pools the counts
        first, so it is not the mean of these.
    """

    n_gt: int
    n_pred: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    n_merges: int
    n_splits: int
    f1: float


def summarise_match(match: InstanceMatch) -> MatchSummary:
    """Count the outcomes of one image's matching.

    Parameters
    ----------
    match : InstanceMatch

    Returns
    -------
    MatchSummary
    """
    n_tp = len(match.pairs)
    n_fp = len(match.unmatched_pred_ids)
    n_fn = len(match.unmatched_gt_ids)
    denominator = 2 * n_tp + n_fp + n_fn
    return MatchSummary(
        n_gt=match.n_gt,
        n_pred=match.n_pred,
        n_true_positives=n_tp,
        n_false_positives=n_fp,
        n_false_negatives=n_fn,
        n_merges=len(match.merged_pred_ids),
        n_splits=len(match.split_gt_ids),
        f1=2 * n_tp / denominator if denominator else float("nan"),
    )


def pair_palette(n_pairs: int, seed: int = 0) -> np.ndarray:
    """One colour per matched pair, none of them close to red.

    The hues are shuffled because pairs are ordered by annotated label,
    which roughly follows position in the frame: in order, neighbouring
    pores would get nearly the same colour.

    Parameters
    ----------
    n_pairs : int
    seed : int, optional
        Fixes the shuffle, so an image is drawn the same way each time.

    Returns
    -------
    np.ndarray
        ``(n_pairs, 3)`` RGB in ``[0, 1]``.

    Raises
    ------
    ValueError
        For a negative count.
    """
    if n_pairs < 0:
        raise ValueError(f"Pair count must not be negative, got {n_pairs}.")
    if n_pairs == 0:
        return np.zeros((0, RGB_CHANNELS))
    hues = np.linspace(*PAIR_HUE_RANGE, n_pairs)
    np.random.default_rng(seed).shuffle(hues)
    hsv = np.column_stack([
        hues,
        np.full(n_pairs, PAIR_SATURATION),
        np.full(n_pairs, PAIR_VALUE),
    ])
    return hsv_to_rgb(hsv)


def paired_colours(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    match: InstanceMatch,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour both sides so that paired instances share a colour.

    Parameters
    ----------
    gt_labels, pred_labels : np.ndarray
        ``(H, W)`` instance labels of one frame, background zero; the
        pair ``match`` was computed from. Not modified.
    match : InstanceMatch
    seed : int, optional
        Passed to :func:`pair_palette`.

    Returns
    -------
    tuple of np.ndarray
        ``(gt_rgb, pred_rgb)``, each ``(H, W, 3)`` in ``[0, 1]``:
        a pair's colour on both of its instances, ``UNPAIRED_RGB`` on
        every other instance, zero on the background.

    Raises
    ------
    ValueError
        If the two label images differ in shape or hold negative
        labels.
    """
    _check_labels(gt_labels, pred_labels)
    palette = pair_palette(len(match.pairs), seed)
    gt_ids = np.array([pair.gt_id for pair in match.pairs], dtype=np.int64)
    pred_ids = np.array(
        [pair.pred_id for pair in match.pairs], dtype=np.int64
    )
    return (
        _colour_by_id(gt_labels, gt_ids, palette),
        _colour_by_id(pred_labels, pred_ids, palette),
    )


def outcome_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    colours: np.ndarray,
    alpha: float = FILL_ALPHA,
) -> np.ndarray:
    """Lay coloured instances over a monochrome image.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)``, integer on its type's full range or float in
        ``[0, 1]``. Not modified.
    labels : np.ndarray
        ``(H, W)`` instance labels, background zero. Not modified.
    colours : np.ndarray
        ``(H, W, 3)`` per-pixel colour, as :func:`paired_colours`
        returns it.
    alpha : float, optional
        Opacity of the fill; outlines are always opaque.

    Returns
    -------
    np.ndarray
        A new ``(H, W, 3)`` float RGB image in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the shapes disagree, the image is not single-channel, or
        ``alpha`` lies outside ``[0, 1]``.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected one channel of shape (H, W), got {image.shape}."
        )
    if labels.shape != image.shape or colours.shape != (
        *image.shape, RGB_CHANNELS
    ):
        raise ValueError(
            f"Image {image.shape}, labels {labels.shape} and colours "
            f"{colours.shape} do not describe one frame."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Opacity must lie in [0, 1], got {alpha}.")
    grey = np.repeat(_unit_grey(image)[..., None], RGB_CHANNELS, axis=-1)
    inside = (labels > 0)[..., None]
    filled = np.where(inside, (1.0 - alpha) * grey + alpha * colours, grey)
    outline = find_boundaries(labels, mode="inner")[..., None]
    return np.where(outline, colours, filled)


def _colour_by_id(
    labels: np.ndarray, paired_ids: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """Paint every instance through one lookup table indexed by label."""
    lookup = np.tile(UNPAIRED_RGB, (int(labels.max()) + 1, 1))
    lookup[0] = 0.0
    lookup[paired_ids] = palette
    return lookup[labels]


def _check_labels(gt_labels: np.ndarray, pred_labels: np.ndarray) -> None:
    """Reject a pair of label images that cannot share one frame."""
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"Annotation {gt_labels.shape} and prediction "
            f"{pred_labels.shape} are not the same frame."
        )
    for name, labels in (("annotation", gt_labels),
                         ("prediction", pred_labels)):
        if labels.size and labels.min() < 0:
            raise ValueError(f"The {name} holds negative labels.")


def _unit_grey(image: np.ndarray) -> np.ndarray:
    """Put a monochrome image on ``[0, 1]`` without stretching it.

    Scaled by the range of its type rather than by its own maximum, so
    a dark image stays dark and two images keep their relative
    brightness.
    """
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float64) / np.iinfo(image.dtype).max
    return np.clip(image.astype(np.float64), 0.0, 1.0)
