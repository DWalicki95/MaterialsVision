"""
Single-pass image file reading: dimensions, channels, hash, and the
grayscale working copy used by the non-image region detector.

Reading the file exactly once matters at this dataset's scale (707
images): hashing and grayscale conversion both require the full pixel
data, so they are derived from one decode instead of two.
"""
import hashlib
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from data_prep.inventory.models import ImageProperties

logger = logging.getLogger(__name__)


def read_image_properties(path: Path) -> ImageProperties:
    """Read dimensions, channel/format metadata, hash and a grayscale
    working copy from an image file, in a single read.

    Parameters
    ----------
    path : Path
        Image file path.

    Returns
    -------
    ImageProperties

    Raises
    ------
    OSError
        If the file cannot be read or decoded (surfaced by the caller
        as the ``image_unreadable`` fatal issue).
    """
    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        file_format = img.format
        width_px, height_px = img.size
        arr = np.array(img)
        gray = np.array(img.convert("L"))

    if arr.ndim == 2:
        n_channels = 1
        channels_identical = None
    else:
        n_channels = arr.shape[2]
        channels_identical = bool(
            all(
                np.array_equal(arr[..., 0], arr[..., c])
                for c in range(1, n_channels)
            )
        )

    bit_depth = arr.dtype.itemsize * 8

    return ImageProperties(
        width_px=width_px,
        height_px=height_px,
        file_format=file_format,
        bit_depth=bit_depth,
        n_channels=n_channels,
        channels_identical=channels_identical,
        file_hash=file_hash,
        gray=gray,
    )
