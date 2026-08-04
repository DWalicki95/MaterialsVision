"""Tests for data_prep.inventory.annotation_stats."""
import math

import numpy as np
import pytest

from data_prep.inventory.annotation_stats import (compute_instance_stats,
                                                  rasterize_annotation)

_SHAPE = (40, 40)


def _square(x0, y0, size):
    """10-point-free axis-aligned square polygon, (x, y) columns."""
    x1, y1 = x0 + size - 1, y0 + size - 1
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float
    )


def test_two_disjoint_squares_no_overlap():
    polygons = [_square(2, 2, 10), _square(20, 20, 10)]
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    assert n_degenerate == 0
    assert labels.max() == 2

    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], _SHAPE[0]),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    assert stats.n_instances == 2
    assert stats.overlap_px_fraction == 0.0
    assert stats.n_border_instances == 0


def test_equivalent_diameter_of_10x10_square():
    polygons = [_square(2, 2, 10)]
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], _SHAPE[0]),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    expected = 2 * math.sqrt(100 / math.pi)
    _, median_diam, _ = stats.equivalent_diameter_px
    assert abs(median_diam - expected) < 0.01


def test_overlapping_squares_share_pixels():
    # Two 10x10 squares overlapping by a 5x10 strip.
    polygons = [_square(0, 0, 10), _square(5, 0, 10)]
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], _SHAPE[0]),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    assert stats.overlap_px_fraction > 0.0
    # Below the 1% significance threshold used elsewhere would be false
    # for this deliberately large overlap - just check it's a small
    # fraction of covered pixels here, not near 1.0 (whole-image
    # overlap).
    assert stats.overlap_px_fraction < 1.0


def test_last_polygon_wins_at_overlap():
    polygons = [_square(0, 0, 10), _square(5, 0, 10)]
    labels, _, _ = rasterize_annotation(polygons, _SHAPE)
    # Pixel (row=5, col=7) is in both squares' rasterized region;
    # the later polygon (id=2) must win there.
    assert labels[5, 7] == 2


def test_square_touching_content_edge_is_border():
    polygons = [_square(0, 5, 10)]  # touches left edge (x=0)
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], _SHAPE[0]),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    assert stats.n_instances == 1
    assert stats.n_border_instances == 1


def test_square_beyond_content_bbox_bottom_is_border():
    # content_bbox excludes the bottom 10 rows (simulating a VAB-style
    # data panel); a square dipping past that line must count as
    # border even though it does not touch the true image edge.
    polygons = [_square(15, 25, 10)]  # rows 25..34, image height 40
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], 30),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    assert stats.n_instances == 1
    assert stats.n_border_instances == 1


def test_square_fully_inside_content_bbox_not_border():
    polygons = [_square(15, 5, 10)]  # rows 5..14, well within (0,30)
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], 30),
        pixel_size_um=None, n_degenerate_polygons=n_degenerate,
    )
    assert stats.n_border_instances == 0


def test_degenerate_polygon_too_few_points():
    polygons = [
        _square(2, 2, 10),
        np.array([[1.0, 1.0], [2.0, 2.0]]),  # 2 points, degenerate
    ]
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    assert n_degenerate == 1
    assert labels.max() == 1  # degenerate polygon got no ID


def test_degenerate_polygon_zero_area_after_clip():
    # A polygon entirely outside the raster shape has >= 3 points but
    # rasterizes to zero pixels.
    polygons = [
        _square(2, 2, 10),
        np.array([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0]]),
    ]
    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, _SHAPE
    )
    assert n_degenerate == 1


def test_no_instances_returns_zeroed_stats():
    labels = np.zeros(_SHAPE, dtype=np.int32)
    coverage = np.zeros(_SHAPE, dtype=np.uint8)
    stats = compute_instance_stats(
        labels, coverage,
        content_bbox=(0, 0, _SHAPE[1], _SHAPE[0]),
        pixel_size_um=None, n_degenerate_polygons=0,
    )
    assert stats.n_instances == 0
    assert stats.n_border_instances == 0
    assert stats.overlap_px_fraction == 0.0
    assert stats.equivalent_diameter_px == (0.0, 0.0, 0.0)


def test_content_bbox_must_originate_at_zero():
    labels = np.zeros(_SHAPE, dtype=np.int32)
    coverage = np.zeros(_SHAPE, dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_instance_stats(
            labels, coverage,
            content_bbox=(5, 0, _SHAPE[1], _SHAPE[0]),
            pixel_size_um=None, n_degenerate_polygons=0,
        )
