"""Tests for the rules that choose post-processing settings.

The seeding rule was pre-registered as "the setting in use holds unless
something beats it by more than one noise band", so the tests pin both
sides of that band. The foreground rule reads where a curve crosses
zero, so the tests pin the interpolation and the case where the curve
never crosses inside the grid, which must be reported rather than
clamped to the grid's edge.
"""
from types import SimpleNamespace

import pytest

from materials_vision.evaluation.postprocessing_calibration import (
    choose_within_band, f1_noise, on_grid_edge, rank, seeding_decision,
    zero_crossing)
from materials_vision.evaluation.watershed import WatershedParams


def _result(f1, count_error=0.0, merges=2.0, splits=2.0, n_gt=4700):
    return SimpleNamespace(
        f1=f1, pore_count_error=count_error, merges_per_100_gt=merges,
        splits_per_100_gt=splits, boundary_f1={}, n_gt=n_gt,
    )


REFERENCE = WatershedParams(0.30, 0.40)


class TestSeedingDecision:
    def test_holds_when_nothing_clears_the_band(self):
        band = f1_noise(_result(0.85))
        results = {
            REFERENCE: _result(0.85),
            WatershedParams(0.25, 0.40): _result(0.85 + 0.9 * band),
            WatershedParams(0.35, 0.40): _result(0.84),
        }
        decision = seeding_decision(results, REFERENCE)
        assert decision["holds"] is True
        assert decision["chosen"] == REFERENCE

    def test_replaced_when_something_clears_the_band(self):
        band = f1_noise(_result(0.85))
        winner = WatershedParams(0.25, 0.40)
        results = {
            REFERENCE: _result(0.85),
            winner: _result(0.85 + 1.5 * band),
            WatershedParams(0.35, 0.40): _result(0.84),
        }
        decision = seeding_decision(results, REFERENCE)
        assert decision["holds"] is False
        assert decision["chosen"] == winner
        assert decision["on_edge"] == ["center_distance_threshold"]

    def test_unscored_reference_is_an_error(self):
        with pytest.raises(KeyError):
            seeding_decision({WatershedParams(0.25, 0.4): _result(0.8)},
                             REFERENCE)


def test_a_setting_within_one_band_of_the_best_is_tied_with_it():
    # The two leaders of the first knob A grid: 0.0023 apart in F1 at a
    # band of 0.0047. Fixed intervals put them in different bins and
    # never compared their counts; within-band ties compare them.
    top = WatershedParams(0.20, 0.50, distance_smoothing=1.0)
    runner_up = WatershedParams(0.25, 0.45, distance_smoothing=2.4)
    results = {
        top: _result(0.8968, count_error=-0.014),
        runner_up: _result(0.8945, count_error=0.009),
        WatershedParams(0.35, 0.40): _result(0.880, count_error=0.0),
    }
    chosen, tied = choose_within_band(results, band=0.0047)
    assert chosen == runner_up
    assert tied == [runner_up, top]


def test_the_setting_in_use_is_kept_when_it_holds_even_if_not_best():
    band = f1_noise(_result(0.85))
    results = {
        REFERENCE: _result(0.85, count_error=0.05),
        WatershedParams(0.25, 0.40): _result(0.85 + 0.5 * band,
                                             count_error=0.0),
    }
    assert seeding_decision(results, REFERENCE)["chosen"] == REFERENCE


def test_ties_in_a_band_go_to_the_better_count():
    results = {
        WatershedParams(0.25, 0.40): _result(0.851, count_error=0.08),
        WatershedParams(0.30, 0.40): _result(0.850, count_error=0.01),
    }
    assert rank(results, tolerance=0.005)[0] == WatershedParams(0.30, 0.40)


def test_edge_is_read_per_varied_parameter():
    grid = [
        WatershedParams(c, b) for c in (0.2, 0.3, 0.4) for b in (0.35, 0.4)
    ]
    assert on_grid_edge(WatershedParams(0.3, 0.35), grid) == [
        "boundary_distance_threshold"
    ]
    # Distance smoothing is not varied, so it is never an edge.
    assert "distance_smoothing" not in on_grid_edge(
        WatershedParams(0.2, 0.4), grid
    )


class TestZeroCrossing:
    def test_interpolates_between_bracketing_samples(self):
        assert zero_crossing([0.4, 0.5, 0.6], [0.02, 0.01, -0.03]) == (
            pytest.approx(0.525)
        )

    def test_exact_zero_is_returned(self):
        assert zero_crossing([0.4, 0.5, 0.6], [0.02, 0.0, -0.01]) == 0.5

    def test_no_crossing_inside_the_grid_is_none(self):
        assert zero_crossing([0.4, 0.5, 0.6], [0.03, 0.02, 0.01]) is None

    def test_missing_values_are_skipped(self):
        nan = float("nan")
        assert zero_crossing([0.4, 0.5, 0.6], [0.02, nan, -0.02]) == (
            pytest.approx(0.5)
        )
