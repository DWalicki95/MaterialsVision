"""Tests for reading the step 7 reports into the screening rule.

The reports are keyed the way the checkpoint scorer names a setting,
so the test builds its keys through the same ``label()`` the scorer
uses: a reader that guessed the keys would silently find nothing.
"""
import json

import pytest

import scripts.screen_hparam_arms as screen


def _record(f1, k_f1=0.86):
    overall = {
        "f1": f1, "merges_per_100_gt": 3.0, "splits_per_100_gt": 2.0,
        "pore_count_error": -0.01, "boundary_f1": {"0.1": 0.8},
        "median_diameter_log_ratio": 0.01,
    }
    per_material = [
        {"label": f"material={m}", "f1": k_f1 if m == "K" else f1,
         "pore_count_error": 0.0}
        for m in screen.MATERIALS
    ]
    return {"overall": overall, "per_material": per_material,
            "per_scale_bin": []}


def _report(run, f1_calibrated, f1_frozen):
    report = {}
    for epoch in range(1, 13):
        for key, setting in screen.SETTINGS.items():
            f1 = f1_frozen if key == "frozen" else f1_calibrated
            report[f"{run}/epoch-{epoch}@{setting.label()}"] = _record(f1)
    return report


@pytest.fixture
def eval_dir(tmp_path):
    for run, f1 in zip(screen.REFERENCE_RUNS, (0.880, 0.884, 0.878)):
        (tmp_path / f"{run}_val.json").write_text(
            json.dumps(_report(run, f1, f1 - 0.004))
        )
    return tmp_path


def test_band_is_fixed_from_the_reference_alone(eval_dir):
    decision = screen.decide(screen.read_all(eval_dir))
    assert decision["band"] == pytest.approx(0.006)
    assert decision["arms"] == {}
    assert "branch" not in decision
    for gain in decision["part_one_gain_on_val"].values():
        assert gain == pytest.approx(0.004)


def test_both_arms_are_screened_and_branched(eval_dir):
    for arm, f1 in (("a1_r32", 0.884), ("a2_lr1e-3", 0.892)):
        run = screen.ARM_RUNS[arm]
        (eval_dir / f"{run}_val.json").write_text(
            json.dumps(_report(run, f1, f1))
        )
    decision = screen.decide(screen.read_all(eval_dir))
    assert decision["arms"]["a1_r32"]["passes"] is False
    assert decision["arms"]["a2_lr1e-3"]["passes"] is True
    assert decision["branch"] == "rate"


def test_an_incomplete_reference_decides_nothing(tmp_path):
    run = screen.REFERENCE_RUNS[0]
    (tmp_path / f"{run}_val.json").write_text(
        json.dumps(_report(run, 0.88, 0.88))
    )
    decision = screen.decide(screen.read_all(tmp_path))
    assert decision["status"] == "reference incomplete"
