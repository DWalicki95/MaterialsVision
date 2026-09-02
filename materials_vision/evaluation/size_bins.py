"""
Splitting recall by how large the pore was.

A single recall figure hides the failure this project cares about
most: a model can find nearly every large pore, miss a good share of
the small ones, and still look respectable. Small pores are also
exactly what one augmentation family is meant to rescue, so the effect
has to be visible somewhere. Here it is - annotated pores are sorted
into four size classes and recall is reported in each.

Only recall is reported. Precision has no meaning per size class,
because a prediction with no counterpart has no annotated size to be
classified by.

**Class edges are quartiles of the annotated areas, measured in square
micrometres.** The unit is the whole point and not a formality. The
dataset was acquired on two microscopes calibrated at 3.24 and
2.48 um/px, so a pore of a given physical size covers a different
number of pixels depending on which one took the picture. Measured on
TRAIN, the quartile of instance area is 9481 px^2 on the coarse scale
and 12621 px^2 on the fine one - a 33% difference that is pure
resolution. Convert to micrometres and the ordering even reverses.
With edges in pixels a class would therefore mean "pore of about this
size **or** picture from that microscope", and since microscope and
foam family coincide in this dataset, recall in the smallest class
would blend a size effect with a material one. In micrometres a class
means a size.

The edges are calibrated once on TRAIN and frozen, in the same spirit
as ``A_min_fragment``: they must not shift between runs, or the
per-class numbers of two policies would not be describing the same
classes. Calibration and use are separate functions here for that
reason - the value is produced by a script and then read back, never
recomputed inside an evaluation.

**Instances touching the frame edge are included.** Their outline is
truncated, so their shape cannot be measured - but recall is a
detection metric, and detection counts them like any other object. If
they were dropped, the classes would stop decomposing overall recall
and the two figures could no longer be reconciled. Measured on TRAIN
they are 19% of instances and spread across all four classes (16% of
the smallest, 24% of the largest), so they do not pile up in one
class and distort it.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SIZE_BIN_LABELS = ("Q1", "Q2", "Q3", "Q4")

N_SIZE_BINS = len(SIZE_BIN_LABELS)

QUARTILE_PERCENTILES = (25.0, 50.0, 75.0)


class SizeBinsLoadError(RuntimeError):
    """Raised when the frozen size classes cannot be read back."""


@dataclass(frozen=True)
class SizeBins:
    """Frozen size classes for annotated pores.

    Parameters
    ----------
    edges_um2 : tuple of float
        The three quartile boundaries, in square micrometres, in
        ascending order. An area below the first belongs to ``Q1``, an
        area at or above the last to ``Q4``.
    n_calibration_instances : int
        Annotated instances the quartiles were measured on. Recorded so
        that a later recalibration on a different population is
        visible rather than silent.
    """

    edges_um2: tuple[float, float, float]
    n_calibration_instances: int

    def __post_init__(self) -> None:
        edges = self.edges_um2
        if len(edges) != N_SIZE_BINS - 1:
            raise ValueError(
                f"expected {N_SIZE_BINS - 1} edges, got {len(edges)}"
            )
        if any(edge <= 0 for edge in edges):
            raise ValueError(f"edges must be positive, got {edges}")
        if list(edges) != sorted(edges):
            raise ValueError(f"edges must ascend, got {edges}")

    def assign(self, areas_um2: np.ndarray) -> np.ndarray:
        """Sort areas into size classes.

        Parameters
        ----------
        areas_um2 : np.ndarray
            Instance areas in square micrometres.

        Returns
        -------
        np.ndarray
            Class index per area, 0 for ``Q1`` up to 3 for ``Q4``.
        """
        return np.digitize(areas_um2, np.asarray(self.edges_um2))

    def as_metadata(self) -> dict:
        """Describe the classes for a run's metadata.

        Returns
        -------
        dict
        """
        return {
            "unit": "um2",
            "labels": list(SIZE_BIN_LABELS),
            "edges_um2": list(self.edges_um2),
            "n_calibration_instances": self.n_calibration_instances,
        }


@dataclass(frozen=True)
class SizeBinRecall:
    """Recall within one size class.

    Parameters
    ----------
    label : str
        Class name, ``Q1`` to ``Q4``.
    n_gt : int
        Annotated instances falling in this class.
    n_matched : int
        How many of them the model found.
    recall : float
        ``n_matched / n_gt``, or ``nan`` when the class is empty on
        this image - an empty class is not a failure to detect
        anything, and averaging it as zero would say it was.
    """

    label: str
    n_gt: int
    n_matched: int
    recall: float


def calibrate_size_bins(areas_um2: np.ndarray) -> SizeBins:
    """Measure the quartile boundaries of an area distribution.

    Intended to be run once, on TRAIN, by the calibration script. An
    evaluation reads the frozen result instead of calling this, so that
    two runs cannot end up describing different classes by the same
    names.

    Parameters
    ----------
    areas_um2 : np.ndarray
        Annotated instance areas in square micrometres.

    Returns
    -------
    SizeBins

    Raises
    ------
    ValueError
        If the sample is empty, holds a non-positive area, or is too
        small for its quartiles to mean anything.
    """
    areas = np.asarray(areas_um2, dtype=float).ravel()
    if areas.size < N_SIZE_BINS:
        raise ValueError(
            f"need at least {N_SIZE_BINS} instances to form "
            f"{N_SIZE_BINS} classes, got {areas.size}"
        )
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError(
            "areas must be finite and positive; an instance with no "
            "pixels is not an instance"
        )

    edges = np.percentile(areas, QUARTILE_PERCENTILES)
    if len(set(edges.tolist())) != edges.size:
        raise ValueError(
            f"quartiles of this sample are not distinct ({edges}); the "
            f"area distribution is too concentrated to split in four"
        )

    logger.info(
        "Size classes calibrated on %d instance(s): edges %.0f / %.0f "
        "/ %.0f um^2", areas.size, edges[0], edges[1], edges[2],
    )
    return SizeBins(
        edges_um2=(float(edges[0]), float(edges[1]), float(edges[2])),
        n_calibration_instances=int(areas.size),
    )


def load_size_bins(path: Path) -> SizeBins:
    """Read the frozen size classes back from their artifact.

    This is the half of the contract that keeps the classes frozen:
    an evaluation loads the boundaries that were measured once rather
    than recomputing them, so two runs cannot end up reporting
    different classes under the same names.

    Parameters
    ----------
    path : Path
        Artifact written by ``scripts/calibrate_size_bins.py``.

    Returns
    -------
    SizeBins

    Raises
    ------
    SizeBinsLoadError
        If the file is missing, unreadable, or does not describe four
        classes in square micrometres.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except OSError as error:
        raise SizeBinsLoadError(
            f"Cannot read frozen size classes from {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise SizeBinsLoadError(
            f"{path} is not valid JSON: {error}"
        ) from error

    try:
        bins = record["bins"]
        unit = bins["unit"]
        edges = bins["edges_um2"]
        n_instances = bins["n_calibration_instances"]
    except (KeyError, TypeError) as error:
        raise SizeBinsLoadError(
            f"{path} does not describe frozen size classes: missing "
            f"{error}"
        ) from error

    if unit != "um2":
        raise SizeBinsLoadError(
            f"{path} records edges in {unit!r}; evaluation requires "
            f"square micrometres, since pixel areas encode which "
            f"microscope took the picture as well as the pore size"
        )
    try:
        low, middle, high = (float(edge) for edge in edges)
        return SizeBins(
            edges_um2=(low, middle, high),
            n_calibration_instances=int(n_instances),
        )
    except (TypeError, ValueError) as error:
        raise SizeBinsLoadError(
            f"{path} holds unusable class boundaries: {error}"
        ) from error


def instance_areas_um2(
    labels: np.ndarray, *, pixel_size_um: float
) -> tuple[np.ndarray, np.ndarray]:
    """Area of every instance in a label image, in square micrometres.

    Labels need not be dense. Ground-truth masks are numbered ``1..n``
    by the loader, but a prediction is under no such obligation and a
    caller may hand over an annotation it has already filtered, so the
    labels present are reported alongside their areas rather than
    assumed to be positions.

    Parameters
    ----------
    labels : np.ndarray
        Instance label image, 0 as background.
    pixel_size_um : float
        Physical size of one pixel, in micrometres.

    Returns
    -------
    tuple of np.ndarray
        The labels present, in ascending order, and the area of each.

    Raises
    ------
    ValueError
        If the pixel size is not positive.
    """
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            f"pixel_size_um must be positive, got {pixel_size_um}"
        )
    counts = np.bincount(labels.ravel())
    present = np.nonzero(counts)[0]
    present = present[present > 0]
    areas = counts[present].astype(float) * pixel_size_um ** 2
    return present, areas


def recall_per_size_bin(
    gt_labels: np.ndarray,
    matched_gt_ids: np.ndarray,
    bins: SizeBins,
    *,
    pixel_size_um: float,
) -> tuple[SizeBinRecall, ...]:
    """Report recall separately in each size class.

    Parameters
    ----------
    gt_labels : np.ndarray
        Ground-truth instance labels, 0 as background. Dense
        numbering is not required.
    matched_gt_ids : np.ndarray
        Labels of the annotated instances the model found, i.e. the
        ``gt_id`` of every matched pair.
    bins : SizeBins
        Frozen class boundaries.
    pixel_size_um : float
        Physical size of one pixel, in micrometres.

    Returns
    -------
    tuple of SizeBinRecall
        One entry per class, ordered from smallest to largest.

    Raises
    ------
    ValueError
        If a matched label does not exist in the ground truth, which
        means the match was computed against a different image.
    """
    present, areas = instance_areas_um2(
        gt_labels, pixel_size_um=pixel_size_um
    )

    matched = np.asarray(matched_gt_ids, dtype=np.int64).ravel()
    position = np.searchsorted(present, matched)
    if matched.size and (
        np.any(position >= present.size)
        or np.any(present[np.minimum(position, present.size - 1)] != matched)
    ):
        raise ValueError(
            "matched ids include labels that are absent from "
            "gt_labels"
        )

    found = np.zeros(present.size, dtype=bool)
    found[position] = True
    assignment = bins.assign(areas)

    return tuple(
        _bin_recall(SIZE_BIN_LABELS[index], found[assignment == index])
        for index in range(N_SIZE_BINS)
    )


def _bin_recall(label: str, found_in_bin: np.ndarray) -> SizeBinRecall:
    """Turn the hit-or-miss flags of one class into its recall."""
    n_gt = int(found_in_bin.size)
    n_matched = int(np.count_nonzero(found_in_bin))
    recall = float("nan") if n_gt == 0 else n_matched / n_gt
    return SizeBinRecall(
        label=label, n_gt=n_gt, n_matched=n_matched, recall=recall
    )
