"""Finding the classical predictions on disk and reading them as labels.

The classical segmentations were produced outside this repository, in
Fiji, one image at a time, and saved under a nested tree::

    ROOT/<group>/<sample>/predicted_image/<variant>/<sample>*.tif

``<sample>`` is the name the image carried when it was exported for
annotation, e.g. ``AS1_40_14_jpg.rf.675e59e3...``: the dataset's
``image_id`` followed by the suffix the export tool appended. That
prefix is what ties a prediction to its row of the split, so the
ground truth can be read through the same sample source the models
were scored with.

Every variant must be present for every sample. A variant missing on
one image would score the variants on different populations, and
their figures could then not be set side by side.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import relabel_sequential

logger = logging.getLogger(__name__)

PREDICTION_SUBDIR = "predicted_image"

MASK_EXTENSIONS = (".tif", ".tiff")

_EXPORT_SUFFIX = re.compile(
    r"_(jpg|jpeg|png|tif|tiff)\.rf\.[0-9a-f]+$", re.IGNORECASE
)


class PredictionLoadError(ValueError):
    """Raised when the predictions on disk cannot be read unambiguously
    as one instance label image per sample and variant."""


@dataclass(frozen=True)
class ClassicalSample:
    """One image's classical predictions.

    Parameters
    ----------
    image_id : str
        Identifier of the image in the manifest and the split.
    sample_dir : Path
    mask_paths : dict of str to Path
        Variant name to the label image it produced.
    """

    image_id: str
    sample_dir: Path
    mask_paths: dict[str, Path]


def image_id_from_sample_dir(name: str) -> str:
    """Recover the dataset's ``image_id`` from a sample directory name.

    Parameters
    ----------
    name : str
        E.g. ``"AS1_40_14_jpg.rf.675e59e3641e9b985aa3282dfc6f641f"``.

    Returns
    -------
    str
        E.g. ``"AS1_40_14"``. A name without the export suffix is
        taken to be the identifier already.
    """
    return _EXPORT_SUFFIX.sub("", name)


def find_samples(
    predictions_root: Path, variants: tuple[str, ...]
) -> tuple[ClassicalSample, ...]:
    """Locate every sample and the mask of each variant.

    Parameters
    ----------
    predictions_root : Path
        Directory holding the group directories.
    variants : tuple of str
        Variant subdirectory names to read.

    Returns
    -------
    tuple of ClassicalSample
        Ordered by ``image_id``.

    Raises
    ------
    PredictionLoadError
        If the root holds no sample, a variant is missing or ambiguous
        for some sample, or two directories map to the same image.
    """
    if not predictions_root.is_dir():
        raise PredictionLoadError(
            f"Predictions root not found: {predictions_root}"
        )
    samples: dict[str, ClassicalSample] = {}
    for sample_dir in _sample_dirs(predictions_root):
        image_id = image_id_from_sample_dir(sample_dir.name)
        if image_id in samples:
            raise PredictionLoadError(
                f"{sample_dir} and {samples[image_id].sample_dir} both "
                f"hold predictions for {image_id}"
            )
        samples[image_id] = ClassicalSample(
            image_id=image_id,
            sample_dir=sample_dir,
            mask_paths={
                variant: _variant_mask(sample_dir, variant)
                for variant in variants
            },
        )
    if not samples:
        raise PredictionLoadError(
            f"No sample directory under {predictions_root}"
        )
    logger.info(
        "Found %d sample(s) with %d variant(s) under %s.",
        len(samples), len(variants), predictions_root,
    )
    return tuple(samples[key] for key in sorted(samples))


def load_prediction_labels(path: Path) -> np.ndarray:
    """Read one predicted label image.

    Parameters
    ----------
    path : Path

    Returns
    -------
    np.ndarray
        ``int32`` labels numbered ``1..n``, 0 = background.

    Raises
    ------
    PredictionLoadError
        If the file does not hold a usable label image.
    """
    try:
        return as_instance_labels(tifffile.imread(str(path)))
    except PredictionLoadError as error:
        raise PredictionLoadError(f"{path}: {error}") from error


def as_instance_labels(array: np.ndarray) -> np.ndarray:
    """Turn a stored label image into dense integer labels.

    Fiji saves label images as ``float32``. They are accepted only if
    every value is a non-negative whole number: a fractional value
    would mean the file is an intensity image rather than labels, and
    rounding it would invent instance boundaries.

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    np.ndarray
        ``int32`` labels numbered ``1..n``, 0 = background. A new
        array; the input is not modified.

    Raises
    ------
    PredictionLoadError
        If the array is not 2-D, or holds non-finite, negative or
        fractional values, or labels beyond the ``int32`` range.
    """
    if array.ndim != 2:
        raise PredictionLoadError(
            f"expected a 2-D label image, got shape {array.shape}"
        )
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise PredictionLoadError("label image holds non-finite values")
    if values.min(initial=0) < 0:
        raise PredictionLoadError("label image holds negative values")
    if np.any(values != np.floor(values)):
        raise PredictionLoadError(
            "label image holds fractional values; it is not a label "
            "image"
        )
    if values.max(initial=0) > np.iinfo(np.int32).max:
        raise PredictionLoadError("labels exceed the int32 range")
    labels, _, _ = relabel_sequential(values.astype(np.int32))
    return labels


def _sample_dirs(predictions_root: Path) -> list[Path]:
    """Sample directories, two levels below the root, in name order."""
    return [
        sample_dir
        for group_dir in sorted(predictions_root.iterdir())
        if group_dir.is_dir()
        for sample_dir in sorted(group_dir.iterdir())
        if sample_dir.is_dir()
    ]


def _variant_mask(sample_dir: Path, variant: str) -> Path:
    """The single mask a variant produced for one sample."""
    variant_dir = sample_dir / PREDICTION_SUBDIR / variant
    if not variant_dir.is_dir():
        raise PredictionLoadError(
            f"{sample_dir.name}: no {variant} prediction "
            f"({variant_dir} is missing)"
        )
    candidates = sorted(
        path for path in variant_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in MASK_EXTENSIONS
        and path.name.startswith(sample_dir.name)
    )
    if len(candidates) != 1:
        raise PredictionLoadError(
            f"{variant_dir}: expected one mask, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]
