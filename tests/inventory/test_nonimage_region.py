"""Tests for data_prep.inventory.nonimage_region.

The detector operates purely on an in-memory grayscale array, so test
inputs are built directly with numpy rather than reading fixture image
files from disk.
"""
import numpy as np

from data_prep.inventory.nonimage_region import (DETECTOR_VERSION,
                                                 detect_nonimage_region)

_KWARGS = dict(extreme_fraction=0.90, max_band_fraction=0.35)
_WIDTH, _HEIGHT = 64, 48


def _image_with_band(band_rows: int, band_value: int) -> np.ndarray:
    """64x48 grayscale array of mid-gray noise, with the bottom
    ``band_rows`` set to a uniform near-extreme ``band_value``."""
    rng = np.random.default_rng(0)
    gray = rng.integers(
        60, 200, size=(_HEIGHT, _WIDTH), dtype=np.uint8
    )
    if band_rows > 0:
        gray[-band_rows:, :] = band_value
    return gray


def test_no_band():
    gray = _image_with_band(band_rows=0, band_value=0)
    region = detect_nonimage_region(gray, **_KWARGS)
    assert region.present is False
    assert region.bbox is None
    assert region.content_bbox == (0, 0, _WIDTH, _HEIGHT)
    assert region.detector_version == DETECTOR_VERSION


def test_dark_band():
    gray = _image_with_band(band_rows=8, band_value=0)
    region = detect_nonimage_region(gray, **_KWARGS)
    assert region.present is True
    assert region.bbox == (0, 40, _WIDTH, _HEIGHT)
    assert region.content_bbox == (0, 0, _WIDTH, 40)


def test_bright_band():
    gray = _image_with_band(band_rows=8, band_value=255)
    region = detect_nonimage_region(gray, **_KWARGS)
    assert region.present is True
    assert region.bbox == (0, 40, _WIDTH, _HEIGHT)
    assert region.content_bbox == (0, 0, _WIDTH, 40)


def test_band_wider_than_max_fraction_not_detected():
    # 20px band on a 48px-tall image with max_band_fraction=0.35
    # (limit=16 rows) exceeds the cap -> must not be reported at all.
    gray = _image_with_band(band_rows=20, band_value=0)
    region = detect_nonimage_region(gray, **_KWARGS)
    assert region.present is False
    assert region.content_bbox == (0, 0, _WIDTH, _HEIGHT)
