"""Deterministic detector for the bottom-of-frame non-image band (SEM
data panel / scale bar), shared by both a bright panel with dark text
(TM3000) and a dark strip with bright text (SU8000).

No learning, no randomness: a row belongs to the band when the fraction
of near-black or near-white pixels in it clears a threshold, scanning
from the bottom until a row fails the test or a safety cap on the band
height is hit (protecting against a false detection eating the whole
image).
"""
import logging

import numpy as np

from data_prep.inventory.models import NonImageRegion

logger = logging.getLogger(__name__)

DETECTOR_VERSION = "bottom_bar_v1"

_NEAR_BLACK = 2
_NEAR_WHITE = 253


def detect_nonimage_region(
    gray: np.ndarray,
    *,
    extreme_fraction: float,
    max_band_fraction: float,
) -> NonImageRegion:
    """Detect a non-image band at the bottom of the frame.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, shape ``(height, width)``.
    extreme_fraction : float
        Minimum fraction of near-black/near-white pixels for a row to
        count as part of the band.
    max_band_fraction : float
        Maximum fraction of image height the band is allowed to claim;
        if the scan does not find the band's real top edge within this
        many rows, no band is reported at all (rather than a truncated
        one), to avoid a false positive consuming the whole image.

    Returns
    -------
    NonImageRegion
    """
    height, width = gray.shape[:2]
    limit_rows = int(height * max_band_fraction)

    band_rows = 0
    found_end = False
    for row in range(height - 1, -1, -1):
        if band_rows >= limit_rows:
            break
        line = gray[row]
        frac = float(
            np.mean((line <= _NEAR_BLACK) | (line >= _NEAR_WHITE))
        )
        if frac >= extreme_fraction:
            band_rows += 1
        else:
            found_end = True
            break

    if not found_end or band_rows == 0:
        return NonImageRegion(
            present=False,
            bbox=None,
            content_bbox=(0, 0, width, height),
            detector_version=DETECTOR_VERSION,
        )

    band_top = height - band_rows
    return NonImageRegion(
        present=True,
        bbox=(0, band_top, width, height),
        content_bbox=(0, 0, width, band_top),
        detector_version=DETECTOR_VERSION,
    )
