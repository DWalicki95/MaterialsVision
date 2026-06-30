"""Filesystem traversal and matching of prediction samples.

Predictions live under a nested directory tree::

    ROOT/<group>/<sample>/image/<sample>.tif
    ROOT/<group>/<sample>/predicted_image/<variant>/<sample>*.tif

This module locates originals and prediction masks, and matches each
sample to its ground-truth entry.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import tifffile
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_SUBDIR = "image"
_PREDICTION_SUBDIR = "predicted_image"
_IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def iter_sample_dirs(predictions_root: Path) -> Iterator[Path]:
    """
    Yield sample directories (two levels below the root).

    Parameters
    ----------
    predictions_root : Path
        Root directory holding group directories.

    Yields
    ------
    Path
        Each sample directory (``ROOT/<group>/<sample>``).
    """
    for group_dir in sorted(p for p in predictions_root.iterdir()
                            if p.is_dir()):
        for sample_dir in sorted(p for p in group_dir.iterdir()
                                 if p.is_dir()):
            yield sample_dir


def _find_file_with_prefix(
    directory: Path, prefix: str, extensions: tuple
) -> Optional[Path]:
    """Return the first file matching prefix and an allowed extension."""
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in extensions:
            continue
        if candidate.name.startswith(prefix):
            return candidate
    return None


def find_original_image(sample_dir: Path) -> Optional[Path]:
    """
    Locate the original image for a sample.

    The original may be stored as ``.tif``, ``.jpg`` or ``.png``.

    Parameters
    ----------
    sample_dir : Path
        Sample directory.

    Returns
    -------
    Optional[Path]
        Path to the original image, or None when missing.
    """
    image_dir = sample_dir / _IMAGE_SUBDIR
    found = _find_file_with_prefix(
        image_dir, sample_dir.name, _IMAGE_EXTENSIONS
    )
    if found is None:
        logger.error("No original image in %s", image_dir)
    return found


def find_prediction_masks(
    sample_dir: Path, variants: List[str]
) -> Dict[str, Path]:
    """
    Locate prediction mask files for each watershed variant.

    Parameters
    ----------
    sample_dir : Path
        Sample directory.
    variants : List[str]
        Variant subdirectory names (e.g. ``"interactive_watershed"``).

    Returns
    -------
    Dict[str, Path]
        Mapping ``variant -> mask_path`` for variants that were found.
    """
    masks = {}
    for variant in variants:
        variant_dir = sample_dir / _PREDICTION_SUBDIR / variant
        found = _find_file_with_prefix(
            variant_dir, sample_dir.name, (".tif", ".tiff")
        )
        if found is None:
            logger.error("No prediction mask in %s", variant_dir)
            continue
        masks[variant] = found
    return masks


def match_sample_to_gt(
    sample_dir_name: str, gt_index: Dict[str, dict]
) -> Optional[dict]:
    """
    Match a sample directory name to a ground-truth entry.

    Tries an exact stem match first, then a ``startswith`` fallback to
    tolerate differing ``_image`` suffixes.

    Parameters
    ----------
    sample_dir_name : str
        Name of the sample directory.
    gt_index : Dict[str, dict]
        Ground-truth index keyed by canonical stem.

    Returns
    -------
    Optional[dict]
        Ground-truth entry, or None when no match is found.
    """
    if sample_dir_name in gt_index:
        return gt_index[sample_dir_name]
    for stem, entry in gt_index.items():
        if stem.startswith(sample_dir_name) or \
                sample_dir_name.startswith(stem):
            return entry
    logger.error("Sample %s not found in JSON annotations",
                 sample_dir_name)
    return None


def load_tif(path: Path) -> np.ndarray:
    """
    Load a .tif file as a NumPy array.

    Parameters
    ----------
    path : Path
        Path to the .tif file.

    Returns
    -------
    np.ndarray
        Image or mask array.
    """
    return tifffile.imread(str(path))


def load_image(path: Path) -> np.ndarray:
    """
    Load an image of any supported format as a NumPy array.

    Uses ``tifffile`` for TIFF files and Pillow otherwise.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array.
    """
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    with Image.open(path) as image:
        return np.asarray(image)
