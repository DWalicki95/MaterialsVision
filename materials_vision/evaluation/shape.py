"""
How faithfully a matched prediction reproduces the pore's shape.

Instance matching accepts a prediction at mask IoU 0.5, which is a
loose agreement: a model can clear it while rounding off every
elongated pore or shaving a tenth off every diameter. This module
measures what survives that threshold, pore by pore, over the pairs
matching has already established.

Three errors are reported per pair, all relative to the annotation:

- **diameter error** - how far the equivalent diameter drifted;
- **elongation error** - how far the ratio of the fitted ellipse's
  axes drifted, i.e. whether the model kept the pore as elongated as
  it was drawn;
- **angle error** - how far the ellipse's orientation turned.

Two things about these numbers need saying plainly, because they are
easy to over-read.

**They describe the model, not the foam.** A pore's apparent
elongation and orientation on a two-dimensional section through a
three-dimensional foam are an artefact of where the section fell: a
sphere cut off-centre is a circle, an ellipsoid's apparent axis ratio
depends on the cutting plane. As material descriptors they would be
indefensible. Here they are not material descriptors - both sides are
measured on the same image and compared with each other, so the
reference is the annotation, not the foam. The sentence "the model
reproduces the annotated shape to within X%" is supported; "this foam
has anisotropy R" is not, and this project does not make it.

**The angle is only meaningful on pores that have one.** As a pore
approaches a circle its orientation becomes arbitrary, and the
difference between two arbitrary angles is noise. The angle error is
therefore reported only for pairs whose annotated elongation reaches
``ANGLE_ELONGATION_THRESHOLD``; below it the value would say more
about rounding than about the model.

**Border instances are excluded here**, unlike in detection. A pore
running off the frame has a truncated outline, so its diameter, axis
ratio and orientation are measurements of the crop rather than of the
pore. A pair is dropped when either side touches the edge - dropping
only one side would compare a truncated shape with a complete one.

Shapes come from ``PoreMorphologyMetrics``, the same class the
material report uses. That is deliberate: if the segmentation metric
and the downstream report disagreed about what "equivalent diameter"
means, the sentence "the model reproduces diameters to within X%"
would not transfer to the report the reader actually receives.
"""
import logging
from dataclasses import dataclass

import numpy as np
from skimage.measure import regionprops

from materials_vision.data.instances import border_instance_labels
from materials_vision.evaluation.matching import InstanceMatch
from materials_vision.quantitative_analysis.quantitative_analysis import (
    PoreMorphologyMetrics)

logger = logging.getLogger(__name__)

ANGLE_ELONGATION_THRESHOLD = 1.2

DEGREES_PER_HALF_TURN = 180.0


@dataclass(frozen=True)
class InstanceShapes:
    """Shape descriptors of every instance in one label image.

    Parameters
    ----------
    label_ids : np.ndarray
        Label of each instance, in the numbering of the source image.
        The arrays below are aligned with it.
    equivalent_diameter_um : np.ndarray
        Diameter of the circle with the same area, in micrometres.
    elongation : np.ndarray
        Ratio of the fitted ellipse's major to minor axis, at least 1.
        ``nan`` for an instance too degenerate to fit an ellipse to,
        such as a one-pixel-wide sliver.
    angle_deg : np.ndarray
        Orientation of the fitted ellipse, normalised to ``[0, 180)``
        because orientation is axial: a pore turned by 180 degrees is
        the same pore. ``nan`` where the elongation is.
    border : np.ndarray
        Boolean per instance: whether it touches the frame edge and so
        has a truncated, unmeasurable shape.
    """

    label_ids: np.ndarray
    equivalent_diameter_um: np.ndarray
    elongation: np.ndarray
    angle_deg: np.ndarray
    border: np.ndarray

    @property
    def n_instances(self) -> int:
        """Instances described.

        Returns
        -------
        int
        """
        return int(self.label_ids.size)

    def index_of(self, label: int) -> int:
        """Position of one label in these arrays.

        Parameters
        ----------
        label : int

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the label is not present in the image these shapes came
            from.
        """
        found = np.flatnonzero(self.label_ids == label)
        if found.size == 0:
            raise KeyError(f"label {label} is absent from these shapes")
        return int(found[0])


@dataclass(frozen=True)
class PairShapeError:
    """Shape disagreement on one matched pair.

    Parameters
    ----------
    gt_id : int
    pred_id : int
    diameter_error : float
        ``|d_pred - d_gt| / d_gt``, dimensionless.
    elongation_error : float
        ``|R_pred - R_gt| / R_gt``, dimensionless; ``nan`` when either
        side is too degenerate for an ellipse.
    angle_error_deg : float
        Axial angular difference in degrees, in ``[0, 90]``; ``nan``
        when the annotated pore is too round for its orientation to
        mean anything.
    """

    gt_id: int
    pred_id: int
    diameter_error: float
    elongation_error: float
    angle_error_deg: float


@dataclass(frozen=True)
class ShapeErrors:
    """Shape disagreement over all usable matched pairs.

    Parameters
    ----------
    pairs : tuple of PairShapeError
        One entry per pair that survived the border exclusion.
    n_matched_pairs : int
        Pairs the matching produced, before any exclusion.
    n_excluded_border : int
        Pairs dropped because a side touched the frame edge.
    n_undefined_elongation : int
        Kept pairs whose elongation could not be measured on one side.
    n_angle_eligible : int
        Kept pairs whose annotated pore was elongated enough for its
        orientation to be meaningful.
    """

    pairs: tuple[PairShapeError, ...]
    n_matched_pairs: int
    n_excluded_border: int
    n_undefined_elongation: int
    n_angle_eligible: int

    @property
    def median_diameter_error(self) -> float:
        """Typical relative error in pore diameter.

        The median rather than the mean, because a single badly
        matched pair can move a mean of ratios a long way.

        Returns
        -------
        float
            ``nan`` when no pair was usable.
        """
        return self._median("diameter_error")

    @property
    def median_elongation_error(self) -> float:
        """Typical relative error in pore elongation.

        Returns
        -------
        float
        """
        return self._median("elongation_error")

    @property
    def median_angle_error_deg(self) -> float:
        """Typical angular error over the pairs where it applies.

        Returns
        -------
        float
        """
        return self._median("angle_error_deg")

    def _median(self, field: str) -> float:
        """Median of one field, ignoring the pairs it is undefined on."""
        if not self.pairs:
            return float("nan")
        values = np.array([getattr(pair, field) for pair in self.pairs])
        if not np.any(np.isfinite(values)):
            return float("nan")
        return float(np.nanmedian(values))


def instance_shapes(
    labels: np.ndarray, *, pixel_size_um: float
) -> InstanceShapes:
    """Measure the shape of every instance in a label image.

    Parameters
    ----------
    labels : np.ndarray
        Instance label image, ``(H, W)``, 0 as background. Labels need
        not be dense; results are reported against the labels present.
    pixel_size_um : float
        Physical size of one pixel, in micrometres.

    Returns
    -------
    InstanceShapes

    Raises
    ------
    ValueError
        If the labels are not a 2-D integer image, or the pixel size
        is not positive.
    """
    _validate(labels, pixel_size_um)

    regions = regionprops(labels)
    on_border = set(border_instance_labels(labels).tolist())

    label_ids = np.array([region.label for region in regions],
                         dtype=np.int64)
    diameters = np.empty(len(regions), dtype=float)
    elongations = np.empty(len(regions), dtype=float)
    angles = np.empty(len(regions), dtype=float)

    n_degenerate = 0
    for index, region in enumerate(regions):
        metrics = PoreMorphologyMetrics(region, pixel_size_um)
        diameters[index] = metrics.calculate_equivalent_diameter()[
            "equivalent_diameter"
        ]
        if region.axis_minor_length <= 0:
            # A sliver one pixel wide has no minor axis, and the
            # ellipse ratio would divide by zero. Predictions from a
            # watershed can look like this; annotations rarely do.
            elongations[index] = float("nan")
            angles[index] = float("nan")
            n_degenerate += 1
            continue
        ellipse = metrics.calculate_ellipse_metrics()
        elongations[index] = ellipse["aspect_ratio"]
        angles[index] = ellipse["ellipse_angle"] % DEGREES_PER_HALF_TURN

    if n_degenerate:
        logger.debug(
            "%d instance(s) had no measurable minor axis and carry no "
            "elongation or angle.", n_degenerate,
        )

    return InstanceShapes(
        label_ids=label_ids,
        equivalent_diameter_um=diameters,
        elongation=elongations,
        angle_deg=angles,
        border=np.array([region.label in on_border for region in regions],
                        dtype=bool),
    )


def shape_errors(
    match: InstanceMatch,
    gt_shapes: InstanceShapes,
    pred_shapes: InstanceShapes,
) -> ShapeErrors:
    """Compare the shape of every matched pair with its annotation.

    Parameters
    ----------
    match : InstanceMatch
        Correspondence produced by
        :func:`materials_vision.evaluation.matching.match_instances`.
    gt_shapes : InstanceShapes
        Shapes of the ground-truth image the match was computed on.
    pred_shapes : InstanceShapes
        Shapes of the predicted image the match was computed on.

    Returns
    -------
    ShapeErrors

    Raises
    ------
    KeyError
        If a matched label is absent from the shapes given, which
        means the shapes were measured on a different image than the
        match.
    """
    errors: list[PairShapeError] = []
    n_border = 0
    n_undefined = 0
    n_angle = 0

    for pair in match.pairs:
        gt_index = gt_shapes.index_of(pair.gt_id)
        pred_index = pred_shapes.index_of(pair.pred_id)
        if gt_shapes.border[gt_index] or pred_shapes.border[pred_index]:
            n_border += 1
            continue

        elongation_error = _relative_error(
            gt_shapes.elongation[gt_index],
            pred_shapes.elongation[pred_index],
        )
        if not np.isfinite(elongation_error):
            n_undefined += 1

        angle_error = _axial_angle_error(
            gt_shapes.angle_deg[gt_index],
            pred_shapes.angle_deg[pred_index],
            gt_shapes.elongation[gt_index],
        )
        if np.isfinite(angle_error):
            n_angle += 1

        errors.append(PairShapeError(
            gt_id=pair.gt_id,
            pred_id=pair.pred_id,
            diameter_error=_relative_error(
                gt_shapes.equivalent_diameter_um[gt_index],
                pred_shapes.equivalent_diameter_um[pred_index],
            ),
            elongation_error=elongation_error,
            angle_error_deg=angle_error,
        ))

    return ShapeErrors(
        pairs=tuple(errors),
        n_matched_pairs=len(match.pairs),
        n_excluded_border=n_border,
        n_undefined_elongation=n_undefined,
        n_angle_eligible=n_angle,
    )


def _relative_error(reference: float, measured: float) -> float:
    """Magnitude of the drift from a reference, as a fraction of it."""
    if not np.isfinite(reference) or not np.isfinite(measured):
        return float("nan")
    if reference == 0:
        return float("nan")
    return float(abs(measured - reference) / reference)


def _axial_angle_error(
    gt_angle_deg: float, pred_angle_deg: float, gt_elongation: float
) -> float:
    """Angular difference between two axes, in degrees.

    Orientation is axial rather than directional - an axis at 10 and
    one at 170 degrees are 20 degrees apart, not 160 - so the
    difference wraps at 180 and folds at 90.

    Returns ``nan`` when the annotated pore is too round for its
    orientation to be meaningful, since the difference between two
    arbitrary angles measures rounding rather than the model.
    """
    if not np.isfinite(gt_elongation):
        return float("nan")
    if gt_elongation < ANGLE_ELONGATION_THRESHOLD:
        return float("nan")
    if not np.isfinite(gt_angle_deg) or not np.isfinite(pred_angle_deg):
        return float("nan")

    difference = abs(pred_angle_deg - gt_angle_deg) % DEGREES_PER_HALF_TURN
    return float(min(difference, DEGREES_PER_HALF_TURN - difference))


def _validate(labels: np.ndarray, pixel_size_um: float) -> None:
    """Reject inputs whose shapes cannot be measured."""
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"labels must hold integer labels, got {labels.dtype}"
        )
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            f"pixel_size_um must be positive, got {pixel_size_um}"
        )
