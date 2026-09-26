"""
Quantities read from the annotation alone, before any model is run.

Part of the post-processing calibration needs numbers that do not
depend on the model: how porous each material really is, how thin the
walls between pores are, and how large the smallest real pores are.
They anchor the settings being calibrated to something physical rather
than to the metric being optimized, so they are measured once, on the
training annotation, and read against everything that follows.

**Walls are measured where two pores face each other.** The background
of a foam image holds two kinds of solid: thin walls separating two
neighbouring pores, and thick struts and nodes where several meet. Only
the first kind limits how much the foreground map may be blurred - a
blur wider than a wall floods it and merges the pores on either side -
so only the first kind is measured. Every pixel is assigned to its
nearest pore; wherever two adjacent pixels belong to different pores,
the line between them crosses a wall, and the wall is as thick as the
sum of their distances to their own pores. The distances are Euclidean,
so an oblique wall is measured across its thickness rather than along
an image axis. Two pores annotated as touching, with no background
between them, yield a wall of zero, which is a property of how the
annotation was drawn rather than of the material; how often that
happens is reported on its own.

**A wall floods at its thinnest point.** Besides the thickness at every
crossing, each wall - each pair of pores that face each other - is
summarized by its minimum, because that is where a blur would breach
it first. Both distributions are reported.

**Border instances are left out of the area distribution.** A pore cut
by the frame can have any area at all in the image, so it says nothing
about how small a real pore can be. Porosity, by contrast, counts every
pore pixel, exactly as the evaluation does.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class WallCrossings:
    """Wall thickness sampled where adjacent pixels face different pores.

    Parameters
    ----------
    thickness_px : np.ndarray
        One value per crossing, in pixels of the frame.
    per_wall_min_px : np.ndarray
        The thinnest crossing of each pair of facing pores.
    n_touching_walls : int
        Walls whose thinnest crossing is zero: two pores annotated as
        touching, with no background between them.
    """

    thickness_px: np.ndarray
    per_wall_min_px: np.ndarray
    n_touching_walls: int


def areal_porosity(labels: np.ndarray) -> float:
    """Fraction of the frame covered by pores.

    Parameters
    ----------
    labels : np.ndarray
        ``(H, W)`` instance labels, 0 as background.

    Returns
    -------
    float
        In ``[0, 1]``.

    Raises
    ------
    ValueError
        For an empty frame, whose porosity is undefined.
    """
    if labels.size == 0:
        raise ValueError("An empty frame has no porosity.")
    return float(np.count_nonzero(labels)) / labels.size


def nearest_pore(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign every pixel to its nearest pore.

    Parameters
    ----------
    labels : np.ndarray
        ``(H, W)`` instance labels, 0 as background.

    Returns
    -------
    nearest : np.ndarray
        Label of the nearest pore; a pore pixel is its own.
    distance_px : np.ndarray
        Euclidean distance to that pore, zero on pore pixels.

    Raises
    ------
    ValueError
        If the frame holds no pore to be nearest to.
    """
    if not np.any(labels):
        raise ValueError("The frame holds no pore.")
    distance_px, indices = ndimage.distance_transform_edt(
        labels == 0, return_indices=True
    )
    nearest = labels[indices[0], indices[1]]
    return nearest, distance_px


def wall_crossings(labels: np.ndarray) -> WallCrossings:
    """Sample wall thickness across every pair of facing pores.

    Parameters
    ----------
    labels : np.ndarray
        ``(H, W)`` instance labels, 0 as background.

    Returns
    -------
    WallCrossings
        Empty when fewer than two pores face each other.
    """
    if np.unique(labels[labels > 0]).size < 2:
        return _no_walls()
    nearest, distance_px = nearest_pore(labels)

    pairs_a, pairs_b, thickness = [], [], []
    for axis in (0, 1):
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        left, right = nearest[tuple(head)], nearest[tuple(tail)]
        crossing = left != right
        pairs_a.append(np.minimum(left, right)[crossing])
        pairs_b.append(np.maximum(left, right)[crossing])
        thickness.append(
            (distance_px[tuple(head)] + distance_px[tuple(tail)])[crossing]
        )
    first = np.concatenate(pairs_a)
    second = np.concatenate(pairs_b)
    thickness_px = np.concatenate(thickness)
    if thickness_px.size == 0:
        return _no_walls()

    per_wall_min_px = _minimum_per_pair(first, second, thickness_px)
    return WallCrossings(
        thickness_px=thickness_px,
        per_wall_min_px=per_wall_min_px,
        n_touching_walls=int(np.count_nonzero(per_wall_min_px == 0)),
    )


def _minimum_per_pair(
    first: np.ndarray, second: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Smallest value for every distinct (first, second) pair."""
    keys = first.astype(np.int64) * (int(second.max()) + 1) + second
    order = np.lexsort((values, keys))
    sorted_keys = keys[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    )
    return values[order][starts]


def _no_walls() -> WallCrossings:
    empty = np.empty(0, dtype=float)
    return WallCrossings(empty, empty, 0)


def interior_instance_areas_px2(
    labels: np.ndarray, border_instance: np.ndarray
) -> np.ndarray:
    """Areas of the instances the frame does not cut.

    Parameters
    ----------
    labels : np.ndarray
        ``(H, W)`` instance labels numbered ``1..n``, 0 as background.
    border_instance : np.ndarray
        Boolean per instance, indexed by ``id - 1``.

    Returns
    -------
    np.ndarray
        Areas in square pixels, in label order.

    Raises
    ------
    ValueError
        If the flags do not cover the labels exactly.
    """
    n_instances = int(labels.max()) if labels.size else 0
    if border_instance.size != n_instances:
        raise ValueError(
            f"{border_instance.size} border flag(s) for {n_instances} "
            f"instance(s)."
        )
    areas = np.bincount(labels.ravel(), minlength=n_instances + 1)[1:]
    return areas[~border_instance].astype(float)


def encoder_grid_period_px(content_shape: tuple[int, int]) -> float:
    """Period of the block pattern the decoder leaves, in frame pixels.

    The encoder sees the image scaled so that its longer side is 1024
    pixels and cuts it into patches of 16. The decoder's output is
    scaled back to the frame before it is thresholded, so the pattern
    it carries repeats every 16 encoder pixels, which is this many
    pixels of the frame.

    Parameters
    ----------
    content_shape : tuple of int
        ``(H, W)`` of the frame.

    Returns
    -------
    float
    """
    encoder_long_side_px = 1024
    patch_px = 16
    return patch_px * max(content_shape) / encoder_long_side_px
