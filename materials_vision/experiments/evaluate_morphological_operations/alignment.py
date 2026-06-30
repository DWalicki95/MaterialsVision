"""Resolution alignment between ground-truth and prediction masks.

IoU and boundary metrics require ground-truth and prediction masks to
share the same pixel grid. Ground-truth is rasterized at the original
image shape, so predictions are resized to match when needed.
"""

import logging
from typing import Tuple

import numpy as np
from skimage.transform import resize

logger = logging.getLogger(__name__)


def align_mask_to_shape(
    mask: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize a labeled mask to the target shape preserving labels.

    Uses nearest-neighbor interpolation (``order=0``) so instance labels
    are not blended.

    Parameters
    ----------
    mask : np.ndarray
        Labeled instance mask.
    target_shape : Tuple[int, int]
        Desired shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        Mask resized to ``target_shape`` with ``uint16`` labels.
    """
    if mask.shape[:2] == target_shape:
        return mask.astype(np.uint16)
    logger.debug("Resizing mask from %s to %s", mask.shape[:2],
                 target_shape)
    resized = resize(
        mask,
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.uint16)
