"""Tests for the decision rule of ``scripts/peft_sam/compare_paired_runs.py``.

The MLflow store is not touched: the rule is checked on figures handed
in directly, and the report parsing on a report shaped like the ones
``evaluate_checkpoint.py`` writes.
"""
import pytest

import scripts.peft_sam.compare_paired_runs as cli

WINDOW = [7, 8, 9, 10, 11, 12]


def _curve(values, start=7):
    return {start + i: v for i, v in enumerate(values)}


REFERENCE = {
    "f1": _curve([0.86, 0.87, 0.85, 0.86, 0.87, 0.86]),
    "F1 K": _curve([0.78] * 6),
}


def _judged(candidate):
    comparison = cli.compare("run", REFERENCE, candidate, WINDOW)
    comparison.verdict = cli.judge(comparison)
    return comparison


def test_a_consistent_gain_wins():
    comparison = _judged({
        "f1": {e: v + 0.007 for e, v in REFERENCE["f1"].items()},
        "F1 K": _curve([0.78] * 6),
    })

    assert comparison.mean_delta_pp == pytest.approx(0.7)
    assert comparison.n_positive == 6
    assert comparison.verdict == cli.VERDICT_WINS


def test_a_gain_carried_by_one_epoch_does_not_win():
    """The same mean, but the difference is positive in a single epoch."""
    f1 = {e: v - 0.001 for e, v in REFERENCE["f1"].items()}
    f1[12] = REFERENCE["f1"][12] + 0.047
    comparison = _judged({"f1": f1, "F1 K": _curve([0.78] * 6)})

    assert comparison.mean_delta_pp >= cli.MIN_GAIN_PP
    assert comparison.verdict == cli.VERDICT_NO_GAIN


def test_a_gain_bought_on_a_weak_material_fails_its_guard():
    comparison = _judged({
        "f1": {e: v + 0.01 for e, v in REFERENCE["f1"].items()},
        "F1 K": _curve([0.75] * 6),
    })

    assert comparison.guard_deltas_pp["F1 K"] == pytest.approx(-3.0)
    assert comparison.verdict == cli.VERDICT_GUARD


def test_a_guard_that_cannot_be_computed_does_not_decide():
    comparison = _judged({
        "f1": {e: v + 0.01 for e, v in REFERENCE["f1"].items()},
        "F1 K": _curve([0.75] * 3),
    })

    assert comparison.guard_deltas_pp["F1 K"] is None
    assert comparison.verdict == cli.VERDICT_WINS


def test_a_run_short_of_the_window_is_not_judged_on_fewer_epochs():
    comparison = _judged({"f1": _curve([0.95] * 3)})

    assert comparison.missing_epochs == [10, 11, 12]
    assert comparison.verdict == cli.VERDICT_INCOMPLETE


def test_recall_on_the_smallest_pores_is_read_from_the_report():
    def entry(recall):
        return {"overall": {"size_bin_recall": [
            {"label": "Q1", "recall": recall},
            {"label": "Q2", "recall": 0.9},
        ]}}

    reports = [{
        "run/epoch-7@frozen": entry(0.70),
        "run/epoch-7@cdt=0.25": entry(0.10),
        "run/epoch-8": entry(0.72),
        "run/epoch-9[apg]@frozen": entry(0.99),
        "other/epoch-7@frozen": entry(0.50),
    }]

    assert cli.size_bin_recall_by_epoch(reports, "run", "frozen") == {
        7: 0.70, 8: 0.72,
    }


def test_the_setting_is_named_as_the_scorer_logs_it():
    args = cli.parse_args(["--reference", "r", "--center-threshold", "0.25"])
    setting = cli.resolve_setting(args)

    assert cli.postprocessing_id(setting) == "c0.25_b0.4_f0.5_fs1_d1.6_m0"
    assert setting.label() == "cdt=0.25"


def test_the_default_window_is_the_late_half_of_a_twelve_epoch_run():
    assert cli.parse_args(["--reference", "r"]).epochs == WINDOW
