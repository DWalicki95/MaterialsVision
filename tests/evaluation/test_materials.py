"""Tests for the propagation metrics.

The property worth guarding hardest is the one that motivates the
whole module: a distribution comparison must notice instances that
were never matched, because that is precisely how a model biases a
size report without the per-pair errors moving.
"""
import numpy as np
import pytest

from materials_vision.evaluation.materials import (area_number_density,
                                                   diameter_distribution_error,
                                                   orientation_distribution,
                                                   porosity_error)
from materials_vision.evaluation.shape import InstanceShapes

FRAME = (60, 60)


def _shapes(diameters=(), angles=(), elongations=None, border=None):
    """Synthetic shapes with whichever fields a test needs."""
    n = max(len(diameters), len(angles))
    if elongations is None:
        elongations = [2.0] * n
    if border is None:
        border = [False] * n
    return InstanceShapes(
        label_ids=np.arange(1, n + 1),
        equivalent_diameter_um=np.array(diameters or [10.0] * n,
                                        dtype=float),
        elongation=np.array(elongations, dtype=float),
        angle_deg=np.array(angles or [0.0] * n, dtype=float),
        border=np.array(border, dtype=bool),
    )


def _labels(instances):
    labels = np.zeros(FRAME, dtype=np.int32)
    for instance_id, (rows, cols) in instances.items():
        labels[rows, cols] = instance_id
    return labels


def test_identical_distributions_have_no_distance():
    shapes = _shapes(diameters=[10.0, 20.0, 30.0, 40.0])

    result = diameter_distribution_error(shapes, shapes)

    assert result.wasserstein_um == pytest.approx(0.0)
    assert result.median_error_um == pytest.approx(0.0)
    assert result.iqr_error_um == pytest.approx(0.0)


def test_a_uniform_shift_equals_the_shift_itself():
    gt = _shapes(diameters=[10.0, 20.0, 30.0, 40.0])
    pred = _shapes(diameters=[13.0, 23.0, 33.0, 43.0])

    result = diameter_distribution_error(gt, pred)

    assert result.wasserstein_um == pytest.approx(3.0)
    assert result.median_error_um == pytest.approx(3.0)
    assert result.iqr_error_um == pytest.approx(0.0)


def test_a_spread_shows_in_the_iqr_and_not_in_the_median():
    gt = _shapes(diameters=[20.0, 30.0, 40.0, 50.0])
    pred = _shapes(diameters=[10.0, 30.0, 40.0, 60.0])

    result = diameter_distribution_error(gt, pred)

    assert result.median_error_um == pytest.approx(0.0)
    assert result.iqr_error_um > 0.0


def test_losing_the_small_pores_moves_the_distribution():
    """The failure a per-pair average cannot see.

    Every surviving pore is reproduced exactly, so a per-pair error
    would be zero; the size report would still be wrong.
    """
    gt = _shapes(diameters=[5.0, 6.0, 7.0, 40.0, 50.0, 60.0])
    pred = _shapes(diameters=[40.0, 50.0, 60.0])

    result = diameter_distribution_error(gt, pred)

    assert result.wasserstein_um > 15.0
    assert result.median_error_um > 0.0
    assert result.n_gt == 6 and result.n_pred == 3


def test_border_instances_leave_the_distribution():
    gt = _shapes(diameters=[10.0, 20.0, 999.0],
                 border=[False, False, True])
    pred = _shapes(diameters=[10.0, 20.0])

    result = diameter_distribution_error(gt, pred)

    assert result.n_gt == 2
    assert result.wasserstein_um == pytest.approx(0.0)


def test_an_empty_side_leaves_the_distance_undefined():
    gt = _shapes(diameters=[10.0, 20.0])
    empty = _shapes(diameters=[30.0], border=[True])

    result = diameter_distribution_error(gt, empty)

    assert np.isnan(result.wasserstein_um)
    assert result.n_pred == 0


def test_porosity_counts_every_pixel_once():
    gt = _labels({1: (slice(0, 30), slice(0, 60))})

    result = porosity_error(gt, gt.copy())

    assert result.gt == pytest.approx(0.5)
    assert result.error_pp == pytest.approx(0.0)


def test_porosity_error_is_signed_in_percentage_points():
    gt = _labels({1: (slice(0, 30), slice(0, 60))})
    pred = _labels({1: (slice(0, 24), slice(0, 60))})

    result = porosity_error(gt, pred)

    assert result.pred == pytest.approx(0.4)
    assert result.error_pp == pytest.approx(-10.0)
    assert result.abs_error_pp == pytest.approx(10.0)


def test_porosity_includes_instances_on_the_frame_edge():
    """A pixel quantity has no per-object decision to make."""
    gt = _labels({1: (slice(0, 10), slice(0, 60))})

    result = porosity_error(gt, gt.copy())

    assert result.gt == pytest.approx(10 * 60 / (60 * 60))


def test_number_density_converts_to_square_millimetres():
    gt = _labels({
        1: (slice(2, 10), slice(2, 10)),
        2: (slice(20, 28), slice(20, 28)),
    })
    pred = _labels({1: (slice(2, 10), slice(2, 10))})

    result = area_number_density(gt, pred, pixel_size_um=10.0)

    expected_area_mm2 = 60 * 60 * 100.0 / 1e6
    assert result.frame_area_mm2 == pytest.approx(expected_area_mm2)
    assert result.n_gt == 2 and result.n_pred == 1
    assert result.gt_per_mm2 == pytest.approx(2 / expected_area_mm2)


def test_number_density_counts_sparse_labels_correctly():
    gt = _labels({
        7: (slice(2, 10), slice(2, 10)),
        900: (slice(20, 28), slice(20, 28)),
    })

    result = area_number_density(gt, gt.copy(), pixel_size_um=1.0)

    assert result.n_gt == 2


def test_identical_orientations_agree_completely():
    shapes = _shapes(angles=[10.0, 20.0, 30.0])

    result = orientation_distribution(shapes, shapes)

    assert result.mean_angle_difference_deg == pytest.approx(0.0)
    assert result.resultant_difference == pytest.approx(0.0)
    assert result.gt_resultant == result.pred_resultant


def test_the_axial_mean_does_not_average_across_the_wrap():
    """Angles at 1 and 179 degrees are nearly the same axis.

    Their arithmetic mean would be 90 degrees, perpendicular to both.
    """
    shapes = _shapes(angles=[1.0, 179.0])

    result = orientation_distribution(shapes, shapes)

    assert result.gt_mean_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_evenly_spread_angles_give_a_resultant_near_zero():
    shapes = _shapes(angles=list(np.arange(0.0, 180.0, 5.0)))

    result = orientation_distribution(shapes, shapes)

    assert result.gt_resultant == pytest.approx(0.0, abs=1e-6)


def test_aligned_angles_give_a_resultant_of_one():
    shapes = _shapes(angles=[45.0] * 10)

    result = orientation_distribution(shapes, shapes)

    assert result.gt_resultant == pytest.approx(1.0)


def test_the_axial_resultant_is_blind_to_a_perpendicular_pair():
    """Why one concentration statistic is not enough.

    Angles at 0 and 90 degrees are opposite directions once doubled,
    so they cancel and the axial resultant reads zero - the same as
    angles spread evenly. A grid-aligned prediction would slip past
    this statistic unnoticed.
    """
    perpendicular = _shapes(angles=[0.0, 0.0, 90.0, 90.0])

    result = orientation_distribution(perpendicular, perpendicular)

    assert result.gt_resultant == pytest.approx(0.0, abs=1e-9)


def test_a_grid_biased_prediction_shows_in_the_fourth_order_resultant():
    """The signature this probe exists to catch."""
    gt = _shapes(angles=list(np.arange(0.0, 180.0, 5.0)))
    pred = _shapes(angles=[0.0, 0.0, 90.0, 90.0, 0.0, 90.0])

    result = orientation_distribution(gt, pred)

    assert result.gt_grid_resultant < 0.1
    assert result.pred_grid_resultant > 0.9
    assert result.grid_resultant_difference > 0.8


def test_the_fourth_order_resultant_also_catches_the_diagonals():
    """A watershed can align to 45 degrees as readily as to 0."""
    diagonal = _shapes(angles=[45.0, 45.0, 135.0, 135.0])

    result = orientation_distribution(diagonal, diagonal)

    assert result.gt_grid_resultant == pytest.approx(1.0)


def test_a_single_preferred_axis_shows_in_both_statistics():
    aligned = _shapes(angles=[30.0] * 8)

    result = orientation_distribution(aligned, aligned)

    assert result.gt_resultant == pytest.approx(1.0)
    assert result.gt_grid_resultant == pytest.approx(1.0)


def test_the_grid_resultant_alone_cannot_tell_one_axis_from_two():
    """Why the two statistics are only readable as a pair.

    A single direction is the degenerate case of a perpendicular
    pair, so the fourth-order resultant reaches 1 for both. Only the
    axial resultant separates them, and the grid signature is
    therefore a high grid resultant beside a *low* axial one.
    """
    one_axis = orientation_distribution(_shapes(angles=[30.0] * 8),
                                        _shapes(angles=[30.0] * 8))
    perpendicular = _shapes(angles=[0.0, 90.0] * 4)
    two_axes = orientation_distribution(perpendicular, perpendicular)

    assert one_axis.gt_grid_resultant == pytest.approx(1.0)
    assert two_axes.gt_grid_resultant == pytest.approx(1.0)

    assert one_axis.gt_resultant == pytest.approx(1.0)
    assert two_axes.gt_resultant == pytest.approx(0.0, abs=1e-9)


def test_the_grid_resultant_matches_the_direct_definition():
    """Pinned to the textbook form, not to this implementation."""
    angles = np.array([3.0, 17.0, 61.0, 88.0, 129.0, 174.0])
    shapes = _shapes(angles=list(angles))

    result = orientation_distribution(shapes, shapes)

    expected = abs(np.exp(1j * np.deg2rad(angles * 4)).mean())
    assert result.gt_grid_resultant == pytest.approx(expected)


def test_the_grid_resultant_does_not_depend_on_how_the_frame_is_turned():
    base = np.array([0.0, 90.0, 10.0, 100.0])

    values = [
        orientation_distribution(_shapes(angles=list(base + turn)),
                                 _shapes(angles=list(base + turn)))
        .gt_grid_resultant
        for turn in (0.0, 7.0, 23.0, 61.0, 180.0)
    ]

    assert values == pytest.approx([values[0]] * len(values))


def test_round_pores_are_left_out_of_the_orientation_comparison():
    shapes = _shapes(angles=[10.0, 20.0, 30.0],
                     elongations=[2.0, 1.0, 1.1])

    result = orientation_distribution(shapes, shapes)

    assert result.n_gt == 1


def test_border_instances_are_left_out_of_the_orientation_comparison():
    shapes = _shapes(angles=[10.0, 20.0], border=[False, True])

    result = orientation_distribution(shapes, shapes)

    assert result.n_gt == 1


def test_orientation_is_undefined_when_nothing_qualifies():
    shapes = _shapes(angles=[10.0], elongations=[1.0])

    result = orientation_distribution(shapes, shapes)

    assert np.isnan(result.gt_mean_angle_deg)
    assert np.isnan(result.gt_resultant)
    assert np.isnan(result.gt_grid_resultant)
    assert np.isnan(result.mean_angle_difference_deg)
    assert result.n_gt == 0


def test_normalising_angles_to_half_a_turn_changes_nothing():
    """Doubling makes the statistics blind to the chosen range."""
    negative = _shapes(angles=[-80.0, -10.0, 30.0])
    positive = _shapes(angles=[100.0, 170.0, 30.0])

    from_negative = orientation_distribution(negative, negative)
    from_positive = orientation_distribution(positive, positive)

    assert from_negative.gt_mean_angle_deg == pytest.approx(
        from_positive.gt_mean_angle_deg
    )
    assert from_negative.gt_resultant == pytest.approx(
        from_positive.gt_resultant
    )


@pytest.mark.parametrize("gt, pred, message", [
    (np.zeros((4, 4), np.int32), np.zeros((5, 5), np.int32),
     "different frames"),
    (np.zeros((4, 4, 2), np.int32), np.zeros((4, 4, 2), np.int32),
     "must be 2-D"),
    (np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32),
     "integer labels"),
])
def test_malformed_input_is_rejected(gt, pred, message):
    with pytest.raises(ValueError, match=message):
        porosity_error(gt, pred)


@pytest.mark.parametrize("pixel_size_um", [0.0, -1.0, np.nan])
def test_number_density_rejects_an_impossible_pixel_size(pixel_size_um):
    frame = np.zeros((4, 4), np.int32)

    with pytest.raises(ValueError, match="pixel_size_um"):
        area_number_density(frame, frame.copy(),
                            pixel_size_um=pixel_size_um)
