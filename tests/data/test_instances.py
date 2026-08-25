"""Tests for the deterministic content crop and instance rebuild."""
import numpy as np
import pytest

from materials_vision.data.instances import apply_content_crop, parse_crop_bbox

FRAME = (20, 16)          # (height, width)
PANEL_CROP = (0, 0, 16, 14)


def _frame(instances):
    """Build a label image from ``{id: (rows_slice, cols_slice)}``."""
    labels = np.zeros(FRAME, dtype=np.int32)
    for instance_id, (rows, cols) in instances.items():
        labels[rows, cols] = instance_id
    return labels


def _image_like(labels):
    return np.full(labels.shape, 128, dtype=np.uint8)


def test_image_and_labels_are_cropped_together():
    labels = _frame({1: (slice(2, 6), slice(2, 6))})
    image = _image_like(labels)

    result = apply_content_crop(
        image, labels, PANEL_CROP, min_fragment_area_px2=1
    )

    assert result.image.shape == (14, 16)
    assert result.labels.shape == (14, 16)


def test_instance_fully_inside_survives_unchanged():
    labels = _frame({7: (slice(2, 6), slice(2, 6))})

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 1
    assert result.original_ids.tolist() == [7]
    assert int((result.labels == 1).sum()) == 16
    assert not result.border_instance[0]


def test_instance_crossing_the_crop_edge_is_cut_not_discarded():
    labels = _frame({3: (slice(12, 18), slice(4, 8))})

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 1
    assert result.original_ids.tolist() == [3]
    assert int((result.labels == 1).sum()) == 2 * 4
    assert result.border_instance[0]
    assert result.n_dropped_below_min_area == 0


def test_fragment_below_the_minimum_area_is_removed():
    labels = _frame({3: (slice(13, 18), slice(4, 8))})

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=8,
    )

    assert result.n_instances == 0
    assert result.n_cut_by_crop == 1
    assert result.n_dropped_below_min_area == 1


def test_instance_entirely_inside_the_panel_is_dropped():
    labels = _frame({
        1: (slice(2, 6), slice(2, 6)),
        2: (slice(15, 19), slice(2, 6)),
    })

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 1
    assert result.original_ids.tolist() == [1]
    assert result.n_dropped_outside == 1
    assert result.n_input_instances == 2


def test_ids_are_renumbered_densely():
    labels = _frame({
        5: (slice(1, 3), slice(1, 3)),
        9: (slice(5, 7), slice(5, 7)),
        40: (slice(9, 11), slice(9, 11)),
    })

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    present = np.unique(result.labels)
    assert present.tolist() == [0, 1, 2, 3]
    assert sorted(result.original_ids.tolist()) == [5, 9, 40]


def test_small_instance_untouched_by_the_crop_survives():
    """A_min_fragment filters manufactured fragments, not annotations.

    It is the P1 of the instance area distribution, so applying it to
    intact instances would delete the bottom percentile of the ground
    truth on every image - including images with no crop at all.
    """
    labels = _frame({1: (slice(2, 4), slice(2, 4))})   # 4 px

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1000,
    )

    assert result.n_instances == 1
    assert result.n_cut_by_crop == 0
    assert result.n_dropped_below_min_area == 0


def test_disconnected_instance_untouched_by_the_crop_stays_whole():
    """Overlap resolution can split an instance before any cropping.

    Both pieces keep the single ID they were annotated with; only a
    cut instance loses its smaller pieces.
    """
    labels = np.zeros(FRAME, dtype=np.int32)
    labels[2:8, 2:5] = 4
    labels[2:4, 9:11] = 4

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 1
    assert int((result.labels == 1).sum()) == 18 + 4
    assert result.n_cut_by_crop == 0
    assert result.n_dropped_disconnected == 0


def test_only_the_largest_piece_of_a_cut_instance_is_kept():
    """Two prongs joined below the crop line fall apart when cut."""
    labels = np.zeros(FRAME, dtype=np.int32)
    labels[10:18, 2:5] = 6        # wide prong: 4 rows x 3 inside
    labels[10:18, 9:11] = 6       # narrow prong: 4 rows x 2 inside
    labels[16:18, 2:11] = 6       # joiner, entirely below the crop

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 1
    assert result.n_cut_by_crop == 1
    assert int((result.labels == 1).sum()) == 12
    assert result.n_dropped_disconnected == 1


def test_border_flag_covers_every_edge_of_the_crop():
    labels = _frame({
        1: (slice(0, 3), slice(4, 7)),      # top
        2: (slice(4, 7), slice(0, 3)),      # left
        3: (slice(4, 7), slice(13, 16)),    # right
        4: (slice(12, 16), slice(6, 9)),    # cut bottom edge
        5: (slice(5, 8), slice(6, 9)),      # interior
    })

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    flags = dict(zip(result.original_ids.tolist(),
                     result.border_instance.tolist()))
    assert flags == {1: True, 2: True, 3: True, 4: True, 5: False}
    assert result.n_border_instances == 4


def test_caller_arrays_are_not_modified():
    labels = _frame({3: (slice(12, 18), slice(4, 8))})
    image = _image_like(labels)
    labels_before = labels.copy()
    image_before = image.copy()

    result = apply_content_crop(
        image, labels, PANEL_CROP, min_fragment_area_px2=1
    )
    result.image[0, 0] = 255
    result.labels[0, 0] = 99

    assert np.array_equal(labels, labels_before)
    assert np.array_equal(image, image_before)


def test_full_frame_bbox_is_a_no_op_on_instances():
    labels = _frame({1: (slice(2, 6), slice(2, 6))})

    result = apply_content_crop(
        _image_like(labels), labels, (0, 0, 16, 20),
        min_fragment_area_px2=1,
    )

    assert result.labels.shape == FRAME
    assert result.n_instances == 1
    assert result.n_dropped_outside == 0


def test_empty_crop_yields_no_instances():
    labels = _frame({1: (slice(15, 19), slice(2, 6))})

    result = apply_content_crop(
        _image_like(labels), labels, PANEL_CROP,
        min_fragment_area_px2=1,
    )

    assert result.n_instances == 0
    assert result.n_dropped_outside == 1
    assert result.labels.max() == 0


def test_rgb_image_is_cropped_on_the_spatial_axes():
    labels = _frame({1: (slice(2, 6), slice(2, 6))})
    image = np.zeros((*FRAME, 3), dtype=np.uint8)

    result = apply_content_crop(
        image, labels, PANEL_CROP, min_fragment_area_px2=1
    )

    assert result.image.shape == (14, 16, 3)


@pytest.mark.parametrize("bbox", [
    (0, 0, 17, 14),          # wider than the frame
    (0, 0, 16, 21),          # taller than the frame
    (-1, 0, 16, 14),         # negative origin
])
def test_bbox_outside_the_frame_is_refused(bbox):
    labels = _frame({1: (slice(2, 6), slice(2, 6))})

    with pytest.raises(ValueError, match="outside"):
        apply_content_crop(
            _image_like(labels), labels, bbox,
            min_fragment_area_px2=1,
        )


def test_degenerate_bbox_is_refused():
    labels = _frame({1: (slice(2, 6), slice(2, 6))})

    with pytest.raises(ValueError, match="degenerate"):
        apply_content_crop(
            _image_like(labels), labels, (5, 5, 5, 12),
            min_fragment_area_px2=1,
        )


def test_shape_mismatch_is_refused():
    labels = np.zeros(FRAME, dtype=np.int32)
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="different frames"):
        apply_content_crop(
            image, labels, PANEL_CROP, min_fragment_area_px2=1
        )


def test_parse_crop_bbox_round_trip():
    assert parse_crop_bbox("0,0,1280,890") == (0, 0, 1280, 890)
    assert parse_crop_bbox(None) is None


def test_parse_crop_bbox_rejects_malformed_input():
    with pytest.raises(ValueError, match="Malformed"):
        parse_crop_bbox("0,0,1280")
