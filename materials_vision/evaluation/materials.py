"""
How far segmentation error carries into the numbers actually reported.

Instance F1 says how many pores the model got right one at a time.
It says nothing about whether the pore size distribution a reader
would be handed is right - and that distribution, not the F1, is the
product of this project. A model can score 0.90 while systematically
losing the smallest pores, and the median diameter in the report
climbs without the headline metric moving much.

This module measures that drift. Every figure here is still a
statement about **the model**, not about the foam: the question is
"if manual annotation were swapped for this model, how far would the
reported number move", and the reference is always the annotation on
the same image.

Four quantities, each with a different reason for being here:

- **the equivalent-diameter distribution**, compared with the
  Wasserstein distance. Unlike the per-pair shape errors in ``shape``,
  a distribution sees instances that were never matched at all, and
  those are exactly how a model biases a size report: a missed small
  pore never appears in a per-pair average. The distance is the mean
  displacement needed to turn one distribution into the other, so it
  comes out in micrometres and reads directly. Median and
  interquartile drift are reported beside it, because a single number
  cannot say whether the distribution shifted or spread;
- **areal porosity**, as the fraction of the frame the annotation
  covers. This is a pixel quantity, not an instance one, so no
  per-object decision enters it and border instances need no special
  handling. Be warned that on this dataset it sits near 0.95 - the
  annotation nearly tiles the frame - so its discriminating power is
  likely weak, and it serves better as a health check than as a
  decision metric;
- **areal number density**, in pores per square millimetre. For a
  single image this is the pore count error in different clothes, the
  frame area cancelling out; it earns its place only once images of
  the two content geometries and the two scales are pooled. It is
  **not** an estimator of volumetric number density, which a single
  section cannot give;
- **the orientation distribution**, compared through two concentration
  statistics. This is the targeted probe for the orientation
  augmentation family, which otherwise has none.

Orientation needs two statistics rather than one, and **neither is
readable without the other**. Angles are axial - a pore turned by half
a turn is the same pore - so they are averaged by doubling them first,
which is what ``calculate_anisotropy`` in the material analysis
already does and what is reused here. That doubled resultant detects
**one** preferred axis. But the failure this probe was built for is a
model that has learnt the pixel grid, and such a model concentrates on
**two perpendicular** axes, near 0 and 90 degrees. Doubling sends
those to 0 and 180 degrees, which are opposite directions and cancel
exactly: a perfectly grid-aligned prediction scores an axial resultant
of zero, indistinguishable from angles spread evenly. Doubling a
second time - a fourth-order resultant - brings them back together.

The fourth-order resultant on its own is not the answer either,
because it reaches 1 for a **single** preferred axis just as readily
as for a perpendicular pair: it measures concentration modulo 90
degrees, and one direction is a degenerate case of that. Only the two
together identify anything:

===============  ==============  ===================================
axial resultant  grid resultant  reading
===============  ==============  ===================================
high             high            one preferred axis
**low**          **high**        two perpendicular axes: grid
low              low             angles spread
===============  ==============  ===================================
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from materials_vision.evaluation.shape import (ANGLE_ELONGATION_THRESHOLD,
                                               InstanceShapes)
from materials_vision.quantitative_analysis.quantitative_analysis import (
    GlobalMicrostructureDescriptors)

logger = logging.getLogger(__name__)

UM2_PER_MM2 = 1e6

PERCENTAGE_POINTS = 100.0


@dataclass(frozen=True)
class DiameterDistributionError:
    """Drift between the annotated and predicted size distributions.

    Border instances are excluded from both sides: a pore truncated by
    the frame has a diameter that measures the crop rather than the
    pore. The exclusion biases both distributions the same way - large
    pores reach the edge more often than small ones - so the
    comparison stays fair even though neither distribution is an
    unbiased picture of the material on its own.

    Parameters
    ----------
    wasserstein_um : float
        Mean displacement needed to turn one distribution into the
        other, in micrometres. ``nan`` when either side is empty.
    median_error_um : float
        Predicted median minus annotated median; signed, so it says
        which way the report would move.
    iqr_error_um : float
        Predicted interquartile range minus the annotated one; signed.
        Separates a distribution that spread from one that shifted.
    n_gt : int
        Annotated instances the comparison used.
    n_pred : int
        Predicted instances the comparison used.
    """

    wasserstein_um: float
    median_error_um: float
    iqr_error_um: float
    n_gt: int
    n_pred: int


@dataclass(frozen=True)
class PorosityError:
    """Drift in the fraction of the frame covered by pores.

    Parameters
    ----------
    gt : float
        Annotated areal fraction, between 0 and 1.
    pred : float
        Predicted areal fraction.
    error_pp : float
        ``pred - gt`` in percentage points; signed.
    abs_error_pp : float
        Its magnitude.
    """

    gt: float
    pred: float
    error_pp: float
    abs_error_pp: float


@dataclass(frozen=True)
class AreaNumberDensity:
    """Pores per square millimetre, on each side of the comparison.

    Counts and frame area are carried separately so that images can be
    pooled correctly: a density over several images is the total count
    over the total area, not the mean of the per-image densities.

    Parameters
    ----------
    n_gt : int
    n_pred : int
    frame_area_mm2 : float
        Area of the frame as evaluated, after the content crop.
    gt_per_mm2 : float
    pred_per_mm2 : float
    """

    n_gt: int
    n_pred: int
    frame_area_mm2: float
    gt_per_mm2: float
    pred_per_mm2: float


@dataclass(frozen=True)
class OrientationDistribution:
    """Axial angle statistics on each side, and their difference.

    Only pores elongated enough for an orientation to mean anything
    take part, and border instances are excluded, for the same reasons
    they are in the per-pair angle error.

    Parameters
    ----------
    gt_mean_angle_deg : float
        Axial mean orientation of the annotation, in ``[0, 180)``.
    pred_mean_angle_deg : float
        The same for the prediction.
    gt_resultant : float
        Axial resultant length of the annotation, from 0 for angles
        spread evenly to 1 for perfect alignment. A low value means
        the mean angle above is not worth reading.
    pred_resultant : float
        The same for the prediction. Rising well above the annotated
        value means the prediction favours a single direction the
        annotation does not have.
    gt_grid_resultant : float
        Fourth-order resultant length of the annotation, i.e.
        concentration of the angles modulo 90 degrees. Zero for angles
        spread evenly, one for angles confined to a pair of
        perpendicular directions - and equally one for angles confined
        to a **single** direction, which is the degenerate case of
        that pair. It therefore says nothing on its own; pair it with
        ``gt_resultant``.
    pred_grid_resultant : float
        The same for the prediction. Together with ``pred_resultant``
        it registers a pixel-grid shortcut, which the axial resultant
        alone cannot see because two perpendicular axes cancel when
        doubled once.
    mean_angle_difference_deg : float
        Axial difference between the two mean angles, in ``[0, 90]``.
    resultant_difference : float
        ``pred_resultant - gt_resultant``; signed.
    grid_resultant_difference : float
        ``pred_grid_resultant - gt_grid_resultant``; signed. The
        grid-shortcut signature is this rising **while**
        ``resultant_difference`` does not: both rising together is a
        single preferred axis, not a grid.
    n_gt : int
        Annotated instances that qualified.
    n_pred : int
        Predicted instances that qualified.
    """

    gt_mean_angle_deg: float
    pred_mean_angle_deg: float
    gt_resultant: float
    pred_resultant: float
    gt_grid_resultant: float
    pred_grid_resultant: float
    mean_angle_difference_deg: float
    resultant_difference: float
    grid_resultant_difference: float
    n_gt: int
    n_pred: int


def diameter_distribution_error(
    gt_shapes: InstanceShapes, pred_shapes: InstanceShapes
) -> DiameterDistributionError:
    """Compare the annotated and predicted size distributions.

    Parameters
    ----------
    gt_shapes : InstanceShapes
        Shapes of the annotation, from
        :func:`materials_vision.evaluation.shape.instance_shapes`.
    pred_shapes : InstanceShapes
        Shapes of the prediction on the same image.

    Returns
    -------
    DiameterDistributionError
    """
    gt = _measurable_diameters(gt_shapes)
    pred = _measurable_diameters(pred_shapes)

    if gt.size == 0 or pred.size == 0:
        logger.warning(
            "One side has no measurable instance (annotation %d, "
            "prediction %d); the size distributions cannot be "
            "compared.", gt.size, pred.size,
        )
        return DiameterDistributionError(
            wasserstein_um=float("nan"),
            median_error_um=float("nan"),
            iqr_error_um=float("nan"),
            n_gt=int(gt.size), n_pred=int(pred.size),
        )

    return DiameterDistributionError(
        wasserstein_um=float(wasserstein_distance(gt, pred)),
        median_error_um=float(np.median(pred) - np.median(gt)),
        iqr_error_um=float(_iqr(pred) - _iqr(gt)),
        n_gt=int(gt.size), n_pred=int(pred.size),
    )


def porosity_error(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> PorosityError:
    """Compare the annotated and predicted areal pore fraction.

    Every pixel is counted once, with no instance filtering. Porosity
    is a pixel quantity, so there is no per-object decision to make;
    excluding border instances would remove their area from the
    numerator while leaving it in the frame, which is not a bias that
    can be reasoned about but simply a mismatch.

    Parameters
    ----------
    gt_labels : np.ndarray
        Ground-truth instance labels, ``(H, W)``, 0 as background.
    pred_labels : np.ndarray
        Predicted labels, same frame.

    Returns
    -------
    PorosityError

    Raises
    ------
    ValueError
        If the two label images describe different frames.
    """
    _validate_pair(gt_labels, pred_labels)

    gt = float(np.count_nonzero(gt_labels)) / gt_labels.size
    pred = float(np.count_nonzero(pred_labels)) / pred_labels.size
    error_pp = (pred - gt) * PERCENTAGE_POINTS
    return PorosityError(
        gt=gt, pred=pred,
        error_pp=error_pp, abs_error_pp=abs(error_pp),
    )


def area_number_density(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    pixel_size_um: float,
) -> AreaNumberDensity:
    """Count pores per square millimetre on each side.

    Parameters
    ----------
    gt_labels : np.ndarray
    pred_labels : np.ndarray
    pixel_size_um : float
        Physical size of one pixel, in micrometres.

    Returns
    -------
    AreaNumberDensity

    Raises
    ------
    ValueError
        If the frames disagree or the pixel size is not positive.
    """
    _validate_pair(gt_labels, pred_labels)
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            f"pixel_size_um must be positive, got {pixel_size_um}"
        )

    area_mm2 = gt_labels.size * pixel_size_um ** 2 / UM2_PER_MM2
    n_gt = _n_instances(gt_labels)
    n_pred = _n_instances(pred_labels)
    return AreaNumberDensity(
        n_gt=n_gt, n_pred=n_pred, frame_area_mm2=area_mm2,
        gt_per_mm2=n_gt / area_mm2, pred_per_mm2=n_pred / area_mm2,
    )


def orientation_distribution(
    gt_shapes: InstanceShapes,
    pred_shapes: InstanceShapes,
    *,
    min_elongation: float = ANGLE_ELONGATION_THRESHOLD,
) -> OrientationDistribution:
    """Compare how the two sides distribute pore orientation.

    Parameters
    ----------
    gt_shapes : InstanceShapes
    pred_shapes : InstanceShapes
    min_elongation : float, optional
        Elongation a pore must reach for its orientation to be
        counted. Below it the angle is dominated by rounding, and two
        clouds of arbitrary angles would compare as similar no matter
        what the model did.

    Returns
    -------
    OrientationDistribution
    """
    gt_angles = _orientable_angles(gt_shapes, min_elongation)
    pred_angles = _orientable_angles(pred_shapes, min_elongation)

    gt_mean, gt_resultant = _axial_statistics(gt_angles)
    pred_mean, pred_resultant = _axial_statistics(pred_angles)
    gt_grid = _grid_resultant(gt_angles)
    pred_grid = _grid_resultant(pred_angles)

    return OrientationDistribution(
        gt_mean_angle_deg=gt_mean,
        pred_mean_angle_deg=pred_mean,
        gt_resultant=gt_resultant,
        pred_resultant=pred_resultant,
        gt_grid_resultant=gt_grid,
        pred_grid_resultant=pred_grid,
        mean_angle_difference_deg=_axial_difference(gt_mean, pred_mean),
        resultant_difference=pred_resultant - gt_resultant,
        grid_resultant_difference=pred_grid - gt_grid,
        n_gt=int(gt_angles.size), n_pred=int(pred_angles.size),
    )


def _measurable_diameters(shapes: InstanceShapes) -> np.ndarray:
    """Diameters of the instances whose size the frame did not cut."""
    keep = ~shapes.border & np.isfinite(shapes.equivalent_diameter_um)
    return shapes.equivalent_diameter_um[keep]


def _orientable_angles(
    shapes: InstanceShapes, min_elongation: float
) -> np.ndarray:
    """Angles of instances elongated enough to have an orientation."""
    keep = (
        ~shapes.border
        & np.isfinite(shapes.angle_deg)
        & np.isfinite(shapes.elongation)
        & (shapes.elongation >= min_elongation)
    )
    return shapes.angle_deg[keep]


def _axial_statistics(angles_deg: np.ndarray) -> tuple[float, float]:
    """Axial mean orientation and resultant length of a set of angles.

    An orientation is defined modulo 180 degrees, so angles at 1 and
    179 degrees are nearly the same axis and their arithmetic mean of
    90 would be perpendicular to both. Doubling the angles maps the
    axial circle onto a full one, where ordinary circular statistics
    apply, and halving the result maps it back.

    The resultant length is delegated to
    ``GlobalMicrostructureDescriptors.calculate_anisotropy``, the
    implementation the material analysis already uses, so that the
    two never drift apart. The mean direction is computed here,
    because that method reports only the length.

    Parameters
    ----------
    angles_deg : np.ndarray
        Orientations in degrees; any consistent range works, since
        doubling makes the result independent of the choice.

    Returns
    -------
    tuple of float
        Mean orientation in ``[0, 180)`` and resultant length in
        ``[0, 1]``; both ``nan`` when there is nothing to average.
    """
    if angles_deg.size == 0:
        return float("nan"), float("nan")

    doubled = np.deg2rad(angles_deg * 2.0)
    mean_angle = float(
        np.rad2deg(np.arctan2(np.sin(doubled).sum(),
                              np.cos(doubled).sum())) / 2.0
    ) % 180.0

    descriptors = GlobalMicrostructureDescriptors(
        mask=np.zeros((1, 1), dtype=np.int32),
        morphology_results=[
            {"ellipse_angle": float(angle)} for angle in angles_deg
        ],
        pixel_size=1.0,
    )
    return mean_angle, float(descriptors.calculate_anisotropy())


def _grid_resultant(angles_deg: np.ndarray) -> float:
    """Concentration of the angles modulo 90 degrees.

    The axial resultant answers "is there one preferred direction",
    and for a pixel-grid shortcut the answer is no: such a model
    favours two directions 90 degrees apart, which doubling maps to
    opposite points of the circle, where they cancel to zero. Doubling
    a second time brings them together again, so the resultant of
    ``4 * angle`` is what survives a perpendicular pair.

    It does not distinguish a perpendicular pair from a single
    direction - both are concentrated modulo 90 - so it is only
    interpretable next to the axial resultant, which separates them.

    Parameters
    ----------
    angles_deg : np.ndarray
        Finite orientations in degrees; any consistent range works.

    Returns
    -------
    float
        Between 0 for angles spread evenly over the half turn and 1
        for angles confined to one direction or to a perpendicular
        pair; ``nan`` when empty.
    """
    if angles_deg.size == 0:
        return float("nan")
    quadrupled = np.deg2rad(angles_deg * 4.0)
    return float(np.hypot(np.cos(quadrupled).mean(),
                          np.sin(quadrupled).mean()))


def _axial_difference(first_deg: float, second_deg: float) -> float:
    """Angular distance between two axes, in ``[0, 90]`` degrees."""
    if not np.isfinite(first_deg) or not np.isfinite(second_deg):
        return float("nan")
    difference = abs(second_deg - first_deg) % 180.0
    return float(min(difference, 180.0 - difference))


def _iqr(values: np.ndarray) -> float:
    """Interquartile range, the spread the median does not show."""
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _n_instances(labels: np.ndarray) -> int:
    """Instances present, counted without assuming dense numbering."""
    return int(np.unique(labels[labels > 0]).size)


def _validate_pair(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> None:
    """Reject label images that do not describe the same frame."""
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
