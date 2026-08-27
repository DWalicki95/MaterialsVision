"""Tests for turning annotation polygons into an instance label image."""
import numpy as np
import pytest

from data_prep.masks.rasterize import MASK_DTYPE, rasterize_instances

SHAPE = (24, 24)


def _rectangle(x0, y0, x1, y1):
    """Closed polygon of an axis-aligned rectangle, (x, y) columns."""
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float
    )


def test_disjoint_polygons_become_separate_instances():
    polygons = [
        _rectangle(2, 2, 6, 6),
        _rectangle(12, 12, 18, 18),
    ]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_instances == 2
    assert mask.n_polygons == 2
    assert mask.n_vanished_polygons == 0
    assert mask.overlap_px == 0
    assert mask.labels.dtype == MASK_DTYPE


def test_ids_run_densely_from_one():
    polygons = [
        _rectangle(1, 1, 4, 4),
        _rectangle(8, 1, 11, 4),
        _rectangle(16, 1, 19, 4),
    ]

    mask = rasterize_instances(polygons, SHAPE)

    present = np.unique(mask.labels)
    assert present.tolist() == [0, 1, 2, 3]


def test_later_polygon_wins_the_contested_pixels():
    polygons = [
        _rectangle(2, 2, 10, 10),
        _rectangle(6, 2, 14, 10),      # overlaps the first
    ]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_instances == 2
    assert mask.overlap_px > 0
    # every contested pixel carries the second polygon's id
    assert int((mask.labels == 2).sum()) > int((mask.labels == 1).sum())


def test_polygon_painted_over_completely_is_reported():
    polygons = [
        _rectangle(4, 4, 8, 8),
        _rectangle(2, 2, 12, 12),      # swallows the first
    ]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_instances == 1
    assert mask.n_polygons == 2
    assert mask.n_vanished_polygons == 1


def test_instance_pinched_in_two_keeps_only_its_larger_half():
    """A later polygon can cut a pore's neck and leave two pieces.

    The training targets are built per instance, so a pore in two
    pieces would teach the model to split a pore the annotation says
    is whole.
    """
    polygons = [
        _rectangle(2, 2, 20, 20),
        _rectangle(2, 8, 20, 10),      # a bar straight across it
    ]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_repaired_instances == 1
    assert mask.n_pieces_removed == 1
    assert mask.discarded_px > 0
    first = mask.labels == 1
    assert first.sum() > 0
    rows = np.unique(np.nonzero(first)[0])
    # what remains is one contiguous band, not two
    assert rows.max() - rows.min() + 1 == rows.size


def test_repair_keeps_the_larger_piece():
    polygons = [
        _rectangle(2, 2, 20, 20),
        _rectangle(2, 6, 20, 8),
    ]

    mask = rasterize_instances(polygons, SHAPE)

    kept_rows = np.unique(np.nonzero(mask.labels == 1)[0])
    assert kept_rows.min() > 8          # the taller piece is below


def test_untouched_instances_are_not_counted_as_repaired():
    polygons = [_rectangle(2, 2, 8, 8), _rectangle(12, 12, 18, 18)]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_repaired_instances == 0
    assert mask.n_pieces_removed == 0
    assert mask.discarded_px == 0


def test_empty_annotation_gives_an_empty_mask():
    mask = rasterize_instances([], SHAPE)

    assert mask.n_instances == 0
    assert mask.labels.shape == SHAPE
    assert mask.labels.max() == 0
    assert mask.covered_px == 0


def test_degenerate_polygon_counts_as_vanished():
    polygons = [
        _rectangle(2, 2, 8, 8),
        np.array([[1.0, 1.0], [2.0, 2.0]]),     # only two points
    ]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.n_instances == 1
    assert mask.n_vanished_polygons == 1


def test_covered_and_overlap_pixel_counts():
    polygons = [_rectangle(0, 0, 9, 9), _rectangle(5, 0, 14, 9)]

    mask = rasterize_instances(polygons, SHAPE)

    assert mask.covered_px == int((mask.labels > 0).sum()) + \
        mask.discarded_px
    assert 0 < mask.overlap_px < mask.covered_px


def test_too_many_instances_for_the_label_type_is_refused(monkeypatch):
    polygons = [_rectangle(2, 2, 6, 6)]
    monkeypatch.setattr(
        np, "iinfo", lambda _dtype: type("I", (), {"max": 0})()
    )

    with pytest.raises(ValueError, match="exceed"):
        rasterize_instances(polygons, SHAPE)
