"""
How thick a real wall is, and how much brighter than the pore beside it.

A synthetic wall drawn across a pore has to look like the walls already
in the picture. Two numbers decide that: how many pixels across it is,
and how far its brightness sits from the interior it divides. Both are
measured here, on annotated images, so that the wall the augmentation
draws is a copy of a measured one rather than a guess.

**What counts as a wall.** Not every pixel outside a pore is one. The
frame's outer margin, an unannotated corner and a dent in a single
pore's outline are all "not inside a pore" and none of them is a wall.
A wall is background lying between *two different* pores, and that is
the definition used: for every pixel, the nearest annotated instance
is known, and a pixel qualifies only where its immediate neighbourhood
disagrees about which instance that is. The disagreement happens
exactly along the line equidistant from two pores, which is where a
wall's middle is.

**Why the middle and not the whole wall.** Thickness is read off the
medial axis, where the distance to the nearest pore is half the local
width; anywhere else that distance says how far the pixel is from one
side, not how wide the wall is. Brightness is read there too, for a
different reason: the middle of a wall is the part of it least mixed
with the pores on either side, so it is the honest figure to reproduce
at the centre of a synthetic one.

**Why brightness is recorded as a contrast, not as a grey level.** The
images come from two microscopes exposed differently. A wall at grey
level 180 in one series and 140 in the other are the same wall; what
they share is how far they sit above the pore interior next to them,
measured against the tonal range of their own image.
"""
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import (distance_transform_edt, maximum_filter,
                           minimum_filter)
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

# The intensities taken to bound an image's tonal range. The extremes
# themselves are left out so that one blown-out speck or a single dead
# pixel cannot set the scale for the whole image.
TONAL_PERCENTILES = (5.0, 95.0)

# Where the frozen thickness range is read off the measured
# distribution: its thin half, not its middle.
#
# The measured widths are strongly skewed. A quarter of all wall-centre
# pixels sit at the thinnest width that can be resolved at all, while
# the top tenth is several times that - those are the struts and the
# junctions where three walls meet, which are structural members rather
# than the membrane between two neighbouring pores. A synthetic wall
# stands for the membrane, so it is drawn from the half of the
# distribution the membranes occupy. The thin end is also the case
# worth training on: two pores divided by a wall wide enough to be
# obvious are already told apart.
THICKNESS_PERCENTILES = (10.0, 50.0)

REPORTED_PERCENTILES = (10.0, 25.0, 50.0, 75.0, 90.0)


@dataclass(frozen=True)
class WallSample:
    """What one image says about the walls it contains.

    Parameters
    ----------
    thickness_px : np.ndarray
        Local wall width in source pixels, one value per pixel of the
        wall network's medial axis.
    wall_intensity : float
        Median intensity along that axis.
    pore_intensity : float
        Median intensity inside the annotated pores.
    tonal_span : float
        Width of the image's tonal range.
    """

    thickness_px: np.ndarray
    wall_intensity: float
    pore_intensity: float
    tonal_span: float

    @property
    def contrast(self) -> float:
        """How far a wall sits above its pores, as a share of the range.

        Returns
        -------
        float
            Positive when walls are the brighter side, which is the
            usual case; ``nan`` when the image offered nothing to
            measure.
        """
        if self.tonal_span <= 0.0 or not np.isfinite(
            self.wall_intensity
        ):
            return float("nan")
        return (
            self.wall_intensity - self.pore_intensity
        ) / self.tonal_span


@dataclass(frozen=True)
class WallSummary:
    """The frozen figures, and the distribution they were read from.

    Parameters
    ----------
    thickness_px : tuple of float
        The range a synthetic wall's width is drawn from.
    contrast : float
        How far a synthetic wall's centre sits above the pore it
        divides, as a share of that image's tonal range.
    thickness_percentiles : dict
        The measured distribution of real wall widths, so the frozen
        range can be seen in the context it came from.
    n_images : int
    n_ridge_px : int
        Wall-centre pixels the figures were measured on.
    """

    thickness_px: tuple[float, float]
    contrast: float
    thickness_percentiles: dict[str, float]
    n_images: int
    n_ridge_px: int


def measure_walls(image: np.ndarray, labels: np.ndarray) -> WallSample:
    """Measure the walls of one annotated image.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` working channel.
    labels : np.ndarray
        ``(H, W)`` instance labels, 0 outside the pores.

    Returns
    -------
    WallSample
        Empty of thickness values, and carrying ``nan`` intensities,
        when the image holds no wall between two different pores.
    """
    ridge, distance = _wall_ridge(labels)
    intensities = image[ridge]
    inside = labels > 0
    low, high = np.percentile(image, TONAL_PERCENTILES)
    return WallSample(
        thickness_px=2.0 * distance[ridge],
        wall_intensity=(
            float(np.median(intensities)) if intensities.size
            else float("nan")
        ),
        pore_intensity=(
            float(np.median(image[inside])) if inside.any()
            else float("nan")
        ),
        tonal_span=float(high) - float(low),
    )


def summarize_walls(samples: Iterable[WallSample]) -> WallSummary:
    """Reduce many images' walls to the figures a wall is drawn from.

    Thicknesses are pooled across images before the percentiles are
    taken, since each one is a measurement of a wall and they are all
    equally walls. Contrast is averaged the other way round, one figure
    per image: it is a property of how an image was exposed, so an
    image with more wall in it should not count for more.

    Parameters
    ----------
    samples : Iterable of WallSample

    Returns
    -------
    WallSummary

    Raises
    ------
    ValueError
        If no image yielded a wall to measure.
    """
    collected = list(samples)
    thicknesses = [
        sample.thickness_px for sample in collected
        if sample.thickness_px.size
    ]
    contrasts = [
        sample.contrast for sample in collected
        if np.isfinite(sample.contrast)
    ]
    if not thicknesses:
        raise ValueError(
            "no image held a wall between two different pores; there "
            "is nothing to calibrate a synthetic wall against"
        )

    pooled = np.concatenate(thicknesses)
    low, high = np.percentile(pooled, THICKNESS_PERCENTILES)
    return WallSummary(
        thickness_px=(float(low), float(high)),
        contrast=float(np.median(contrasts)) if contrasts else 0.0,
        thickness_percentiles=_percentiles(pooled),
        n_images=len(thicknesses),
        n_ridge_px=int(pooled.size),
    )


def _wall_ridge(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the middle of every wall lying between two pores.

    Returns the ridge as a boolean image together with the distance
    from every pixel to the nearest annotated pore, so that a caller
    reading the ridge gets half the local wall width for free.
    """
    outside = labels == 0
    if not outside.any() or not (labels > 0).any():
        return np.zeros(labels.shape, dtype=bool), np.zeros(
            labels.shape, dtype=float
        )

    distance, indices = distance_transform_edt(
        outside, return_indices=True
    )
    nearest = labels[indices[0], indices[1]]
    divides_two_pores = (
        maximum_filter(nearest, size=3)
        != minimum_filter(nearest, size=3)
    )
    ridge = skeletonize(outside) & divides_two_pores
    return ridge, np.asarray(distance)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    """Describe a distribution at the percentiles worth reporting."""
    measured = np.percentile(values, REPORTED_PERCENTILES)
    return {
        f"p{int(percentile)}": float(value)
        for percentile, value in zip(REPORTED_PERCENTILES, measured)
    }
