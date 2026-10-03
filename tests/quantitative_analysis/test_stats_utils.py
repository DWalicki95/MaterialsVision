"""Tests for the weighted statistics behind the edge-corrected metrics.

The corrected and uncorrected statistics are reported side by side, so
with equal weights they must agree exactly - otherwise a difference in a
report could come from the estimator rather than from the correction.
"""
import numpy as np
import pytest

from materials_vision.quantitative_analysis.stats_utils import (
    calculate_statistics, calculate_weighted_statistics, weighted_quantile)

VALUES = np.array([3.0, 1.0, 7.0, 2.0, 10.0, 4.0])


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
def test_equal_weights_match_numpy_percentile(q):
    weights = np.full_like(VALUES, 2.5)

    assert weighted_quantile(VALUES, weights, q) == pytest.approx(
        np.percentile(VALUES, q * 100)
    )


def test_heavier_large_values_raise_the_median():
    weights = np.where(VALUES > 5, 10.0, 1.0)

    assert weighted_quantile(VALUES, weights, 0.5) > np.median(VALUES)


def test_single_value_is_every_quantile():
    assert weighted_quantile(np.array([4.0]), np.array([3.0]), 0.9) == 4.0


def test_empty_sample_is_nan():
    assert np.isnan(weighted_quantile(np.array([]), np.array([]), 0.5))


@pytest.mark.parametrize(
    "weights, q",
    [
        (np.ones(6), 1.5),
        (np.ones(5), 0.5),
        (-np.ones(6), 0.5),
        (np.zeros(6), 0.5),
    ],
)
def test_invalid_input_raises(weights, q):
    with pytest.raises(ValueError):
        weighted_quantile(VALUES, weights, q)


def test_corrected_statistics_equal_raw_ones_for_equal_weights():
    stats = calculate_statistics(list(VALUES), weights=[1.0] * 6)

    assert stats["mean_corrected"] == pytest.approx(stats["mean"])
    assert stats["median_corrected"] == pytest.approx(stats["median"])
    assert stats["std_corrected"] == pytest.approx(stats["std"])


def test_statistics_without_weights_have_no_corrected_keys():
    assert not any(
        key.endswith("_corrected") for key in calculate_statistics(VALUES)
    )


def test_nan_value_makes_corrected_statistics_nan():
    stats = calculate_weighted_statistics([1.0, np.nan, 3.0], [1, 1, 1])

    assert all(np.isnan(value) for value in stats.values())


def test_weight_of_the_largest_value_counts():
    """Midpoint positions give the last observation its full say."""
    values = np.array([10.0, 20.0, 30.0, 40.0, 200.0])

    assert weighted_quantile(values, np.array([1, 1, 1, 1, 50]), 0.5) > 40


def test_quantile_is_invariant_to_weight_scale():
    weights = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0])

    assert weighted_quantile(VALUES, weights, 0.3) == pytest.approx(
        weighted_quantile(VALUES, weights * 7.5, 0.3)
    )


def test_zero_weight_observation_is_ignored():
    with_zero = weighted_quantile(
        np.array([1.0, 2.0, 100.0]), np.array([1.0, 1.0, 0.0]), 0.5
    )

    assert with_zero == pytest.approx(1.5)
