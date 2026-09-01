"""Tests for per-pair shape errors.

Two properties get the most attention here. The angle is axial, so a
pore turned by 180 degrees must count as unturned and the error must
never exceed 90; and a degenerate prediction must produce a missing
value rather than a crash, because the existing ellipse code divides
by the minor axis without guarding it.
"""
import numpy as np
import pytest

from materials_vision.evaluation.matching import (InstanceMatch, MatchedPair,
                                                  match_instances)
from materials_vision.evaluation.shape import (ANGLE_ELONGATION_THRESHOLD,
                                               InstanceShapes,
                                               instance_shapes, shape_errors)

FRAME = (80, 80)

PIXEL_SIZE_UM = 2.0


def _blank():
    return np.zeros(FRAME, dtype=np.int32)


def _rectangle(labels, label, rows, cols):
    labels[rows, cols] = label
    return labels


def _centred_square(label=1, half=10):
    centre = FRAME[0] // 2
    return _rectangle(
        _blank(), label,
        slice(centre - half, centre + half),
        slice(centre - half, centre + half),
    )


def _shapes(labels):
    return instance_shapes(labels, pixel_size_um=PIXEL_SIZE_UM)


def _errors(gt, pred):
    return shape_errors(match_instances(gt, pred), _shapes(gt), _shapes(pred))


def test_equivalent_diameter_matches_the_circle_of_equal_area():
    labels = _centred_square(half=10)

    shapes = _shapes(labels)

    area_um2 = 20 * 20 * PIXEL_SIZE_UM ** 2
    expected = 2 * np.sqrt(area_um2 / np.pi)
    assert shapes.equivalent_diameter_um[0] == pytest.approx(expected)


def test_a_square_is_reported_as_barely_elongated():
    shapes = _shapes(_centred_square(half=10))

    assert shapes.elongation[0] == pytest.approx(1.0, abs=0.01)


def test_an_elongated_rectangle_reports_its_axis_ratio():
    labels = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))

    shapes = _shapes(labels)

    assert shapes.elongation[0] == pytest.approx(2.0, rel=0.05)


def test_angles_are_normalised_to_half_a_turn():
    labels = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))

    shapes = _shapes(labels)

    assert 0.0 <= shapes.angle_deg[0] < 180.0


def test_identical_masks_have_no_shape_error():
    gt = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))

    errors = _errors(gt, gt.copy())

    assert len(errors.pairs) == 1
    assert errors.pairs[0].diameter_error == pytest.approx(0.0)
    assert errors.pairs[0].elongation_error == pytest.approx(0.0)
    assert errors.pairs[0].angle_error_deg == pytest.approx(0.0)


def test_a_shrunken_prediction_reports_a_diameter_error():
    gt = _centred_square(half=10)
    pred = _centred_square(half=9)

    errors = _errors(gt, pred)

    assert errors.pairs[0].diameter_error == pytest.approx(0.1, abs=0.01)


def test_a_rounded_prediction_reports_an_elongation_error():
    gt = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))
    pred = _centred_square(half=14)

    errors = _errors(gt, pred)

    assert errors.pairs[0].elongation_error > 0.3


def test_the_angle_error_is_axial_so_half_a_turn_is_no_turn():
    gt = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))
    pred = np.rot90(gt, 2).copy()

    errors = _errors(gt, pred)

    assert errors.pairs[0].angle_error_deg == pytest.approx(0.0, abs=1e-6)


def _shapes_at(angle_deg, elongation=2.0):
    """One synthetic instance with a chosen orientation."""
    return InstanceShapes(
        label_ids=np.array([1]),
        equivalent_diameter_um=np.array([10.0]),
        elongation=np.array([elongation]),
        angle_deg=np.array([angle_deg % 180.0]),
        border=np.array([False]),
    )


def _one_pair_match():
    return InstanceMatch(
        pairs=(MatchedPair(gt_id=1, pred_id=1, iou=0.9),),
        n_gt=1, n_pred=1,
        unmatched_gt_ids=np.empty(0, dtype=np.int64),
        unmatched_pred_ids=np.empty(0, dtype=np.int64),
        merged_pred_ids=np.empty(0, dtype=np.int64),
        split_gt_ids=np.empty(0, dtype=np.int64),
        iou_threshold=0.5,
    )


@pytest.mark.parametrize("gt_angle, pred_angle, expected", [
    (10.0, 10.0, 0.0),
    (10.0, 190.0, 0.0),      # half a turn is no turn for an axis
    (10.0, 100.0, 90.0),     # a quarter turn is the largest error
    (10.0, 170.0, 20.0),     # wraps the short way round, not 160
    (170.0, 10.0, 20.0),     # and symmetrically
    (0.0, 45.0, 45.0),
])
def test_the_angle_error_folds_like_an_axis(gt_angle, pred_angle, expected):
    errors = shape_errors(_one_pair_match(), _shapes_at(gt_angle),
                          _shapes_at(pred_angle))

    assert errors.pairs[0].angle_error_deg == pytest.approx(expected)


def test_the_angle_error_never_exceeds_a_quarter_turn():
    match = _one_pair_match()

    worst = max(
        shape_errors(match, _shapes_at(gt), _shapes_at(pred))
        .pairs[0].angle_error_deg
        for gt in range(0, 180, 7) for pred in range(0, 360, 11)
    )

    assert worst <= 90.0 + 1e-9


def test_the_angle_of_a_round_pore_is_not_reported():
    gt = _centred_square(half=10)
    pred = _centred_square(half=10)

    shapes = _shapes(gt)
    errors = _errors(gt, pred)

    assert shapes.elongation[0] < ANGLE_ELONGATION_THRESHOLD
    assert np.isnan(errors.pairs[0].angle_error_deg)
    assert errors.n_angle_eligible == 0


def test_a_degenerate_prediction_gives_a_missing_value_not_a_crash():
    """A sliver one pixel wide has no minor axis to divide by.

    ``calculate_ellipse_metrics`` raises ``ZeroDivisionError`` on such
    an instance, and a watershed can easily produce one, so the guard
    is checked rather than assumed.
    """
    pred = _rectangle(_blank(), 1, slice(39, 40), slice(20, 60))

    shapes = _shapes(pred)

    assert np.isnan(shapes.elongation[0])
    assert np.isnan(shapes.angle_deg[0])


def test_a_degenerate_side_is_counted_and_left_undefined():
    gt = _rectangle(_blank(), 1, slice(30, 50), slice(20, 60))
    thin = _rectangle(_blank(), 1, slice(39, 40), slice(20, 60))

    errors = shape_errors(match_instances(gt, gt.copy()),
                          _shapes(gt), _shapes(thin))

    assert errors.n_undefined_elongation == 1
    assert np.isnan(errors.pairs[0].elongation_error)
    assert np.isfinite(errors.pairs[0].diameter_error)


def test_a_pair_touching_the_frame_edge_is_dropped():
    gt = _rectangle(_blank(), 1, slice(0, 20), slice(20, 60))

    errors = _errors(gt, gt.copy())

    assert errors.n_matched_pairs == 1
    assert errors.n_excluded_border == 1
    assert errors.pairs == ()


def test_a_pair_is_dropped_when_only_the_prediction_touches_the_edge():
    gt = _rectangle(_blank(), 1, slice(2, 22), slice(20, 60))
    pred = _rectangle(_blank(), 1, slice(0, 22), slice(20, 60))

    errors = _errors(gt, pred)

    assert errors.n_excluded_border == 1
    assert errors.pairs == ()


def test_medians_ignore_the_pairs_a_value_is_missing_on():
    gt = _blank()
    gt = _rectangle(gt, 1, slice(10, 30), slice(5, 45))
    gt = _rectangle(gt, 2, slice(45, 65), slice(5, 45))

    errors = _errors(gt, gt.copy())

    assert len(errors.pairs) == 2
    assert errors.median_diameter_error == pytest.approx(0.0)
    assert errors.median_elongation_error == pytest.approx(0.0)


def test_medians_are_undefined_when_no_pair_survives():
    gt = _rectangle(_blank(), 1, slice(0, 20), slice(20, 60))

    errors = _errors(gt, gt.copy())

    assert np.isnan(errors.median_diameter_error)
    assert np.isnan(errors.median_angle_error_deg)


def test_shapes_report_labels_as_given_even_when_sparse():
    labels = _blank()
    labels = _rectangle(labels, 42, slice(10, 30), slice(5, 45))
    labels = _rectangle(labels, 7, slice(45, 65), slice(5, 45))

    shapes = _shapes(labels)

    assert sorted(shapes.label_ids.tolist()) == [7, 42]
    assert shapes.index_of(42) in (0, 1)


def test_asking_for_an_absent_label_is_an_error():
    shapes = _shapes(_centred_square(label=3))

    with pytest.raises(KeyError, match="absent"):
        shapes.index_of(99)


@pytest.mark.parametrize("labels, pixel_size_um, message", [
    (np.zeros((4, 4, 2), np.int32), 1.0, "must be 2-D"),
    (np.zeros((4, 4), np.float32), 1.0, "integer labels"),
    (np.zeros((4, 4), np.int32), 0.0, "pixel_size_um"),
    (np.zeros((4, 4), np.int32), -1.0, "pixel_size_um"),
])
def test_malformed_input_is_rejected(labels, pixel_size_um, message):
    with pytest.raises(ValueError, match=message):
        instance_shapes(labels, pixel_size_um=pixel_size_um)
