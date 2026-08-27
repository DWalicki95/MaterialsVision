"""
Reading instance mask files at training and evaluation time.

Masks are built once, ahead of training, by
``scripts/build_instance_masks.py``: every annotated pore becomes a
region of pixels carrying one positive integer, background is 0. This
module only reads them back, and checks the few properties the rest of
the pipeline is entitled to assume - that the mask lines up with its
image, and that instance ids run densely from 1 with no gaps.

The dense-numbering check is worth its cost. Later stages index
per-instance arrays by ``id - 1`` and count instances as ``labels.max()``;
a gap in the numbering makes those two disagree silently, producing
targets that are quietly misaligned rather than obviously broken.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


class MaskLoadError(ValueError):
    """Raised when a mask file is missing or fails its checks."""


def load_instance_mask(
    path: Path,
    *,
    expected_shape: Optional[tuple[int, int]] = None,
    check_dense_ids: bool = True,
) -> np.ndarray:
    """Read one instance label image.

    Parameters
    ----------
    path : Path
        Mask file written by the mask builder.
    expected_shape : tuple of int, optional
        ``(height, width)`` the mask must have, normally taken from
        the image it belongs to.
    check_dense_ids : bool, optional
        Verify that instance ids form ``1..n`` with no gaps.

    Returns
    -------
    np.ndarray
        Label image, 0 = background.

    Raises
    ------
    MaskLoadError
        If the file is missing, is not a 2-D label image, disagrees
        with ``expected_shape``, holds negative values, or - when
        ``check_dense_ids`` is set - has gaps in its numbering.
    """
    if not path.exists():
        raise MaskLoadError(
            f"Mask not found: {path}. Instance masks are built ahead "
            f"of training by scripts/build_instance_masks.py."
        )

    labels = tifffile.imread(path)
    if labels.ndim != 2:
        raise MaskLoadError(
            f"Mask {path} has shape {labels.shape}; a label image must "
            f"be 2-D"
        )
    if expected_shape is not None and labels.shape != expected_shape:
        raise MaskLoadError(
            f"Mask {path} has shape {labels.shape} but its image is "
            f"{expected_shape}"
        )
    if labels.min() < 0:
        raise MaskLoadError(
            f"Mask {path} holds negative labels; 0 is background and "
            f"instances are positive"
        )
    if check_dense_ids:
        _check_dense_ids(labels, path)
    return labels


def _check_dense_ids(labels: np.ndarray, path: Path) -> None:
    """Verify instance ids run 1..n without gaps.

    Parameters
    ----------
    labels : np.ndarray
    path : Path
        Only used in the error message.

    Raises
    ------
    MaskLoadError
        If any id between 1 and the maximum is unused.
    """
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return
    expected = np.arange(1, present.size + 1)
    if not np.array_equal(present, expected):
        missing = sorted(set(expected.tolist()) - set(present.tolist()))
        raise MaskLoadError(
            f"Mask {path} numbers {present.size} instance(s) up to id "
            f"{int(present.max())}, leaving gaps at {missing[:10]}; "
            f"ids must run 1..n so that instance count and maximum id "
            f"agree"
        )
