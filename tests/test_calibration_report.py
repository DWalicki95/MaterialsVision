"""A calibration report is read back as exactly the setting it chose.

Per-fold calibration hands its choice to the evaluator through a file:
``calibrate_watershed.py`` writes it, ``evaluate_checkpoint.py
--also-calibration`` and ``summarize_cv.py`` read it. These tests write
the report with the calibration script's own writer and read it with
the shared loader, so a change to either side that breaks the hand-off
fails here rather than after a night of folds.
"""
import json
from types import SimpleNamespace

import pytest

import scripts.peft_sam.calibrate_watershed as calibrate
import scripts.peft_sam.evaluate_checkpoint as evaluate
from materials_vision.evaluation import (DECISION_SCALE,
                                         CalibrationReportError,
                                         WatershedParams,
                                         load_calibrated_setting)
from materials_vision.evaluation.watershed import FROZEN_WATERSHED


def _figures(f1):
    return SimpleNamespace(
        f1=f1, boundary_f1={DECISION_SCALE: 0.8}, splits_per_100_gt=2.0,
        merges_per_100_gt=3.0, pore_count_error=0.01,
    )


@pytest.fixture
def report(tmp_path):
    """A report as the calibration writes it, best setting first."""
    best = WatershedParams(
        center_distance_threshold=0.35, boundary_distance_threshold=0.5,
    )
    other = WatershedParams(
        center_distance_threshold=0.25, boundary_distance_threshold=0.4,
    )
    destination = tmp_path / "calibration" / "fold3.json"
    calibrate.write_report(
        destination, tmp_path / "epoch-24.pt", tmp_path / "fold3.csv",
        [0, 12, 24], 0.004, [(best, _figures(0.87)), (other, _figures(0.86))],
    )
    return destination, best


def test_the_chosen_setting_is_read_back_unchanged(report):
    path, best = report

    assert load_calibrated_setting(path) == best


def test_the_report_records_where_it_was_calibrated(report, tmp_path):
    path, _ = report
    payload = json.loads(path.read_text())

    assert payload["split"] == str(tmp_path / "fold3.csv")
    assert payload["subset"] == "train"


def test_the_evaluator_scores_the_calibrated_setting_alongside(report):
    path, best = report
    args = evaluate.parse_args([
        "--checkpoint", "x.pt", "--also-calibration", str(path),
        "--center-threshold-sweep", "0.25", "0.30", "0.35",
    ])

    settings = evaluate.resolve_settings(args)

    assert best in settings
    assert FROZEN_WATERSHED in settings
    assert len(settings) == 4


def test_a_calibration_equal_to_a_swept_setting_is_scored_once(tmp_path):
    destination = tmp_path / "calibration.json"
    calibrate.write_report(
        destination, tmp_path / "e.pt", tmp_path / "s.csv", [0], 0.004,
        [(FROZEN_WATERSHED, _figures(0.87))],
    )
    args = evaluate.parse_args([
        "--checkpoint", "x.pt", "--also-calibration", str(destination),
        "--center-threshold-sweep", "0.25", "0.30", "0.35",
    ])

    assert len(evaluate.resolve_settings(args)) == 3


def test_an_older_report_keeps_the_frozen_value_of_what_it_lacks(tmp_path):
    destination = tmp_path / "old.json"
    chosen = WatershedParams(center_distance_threshold=0.45).to_kwargs()
    del chosen["min_instance_area_um2"]
    destination.write_text(json.dumps({"chosen": chosen}))

    setting = load_calibrated_setting(destination)

    assert setting.center_distance_threshold == 0.45
    assert setting.min_instance_area_um2 == (
        FROZEN_WATERSHED.min_instance_area_um2
    )


@pytest.mark.parametrize("content, message", [
    (None, "Cannot read"),
    ("not json", "Cannot read"),
    (json.dumps({"grid": []}), "no chosen setting"),
    (json.dumps({"chosen": {"centre": 0.3}}), "centre"),
])
def test_an_unusable_report_is_refused(tmp_path, content, message):
    destination = tmp_path / "report.json"
    if content is not None:
        destination.write_text(content)

    with pytest.raises(CalibrationReportError, match=message):
        load_calibrated_setting(destination)


def test_the_evaluator_stops_before_loading_a_model(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(evaluate, "build_pretrained_segmenter", fail)
    monkeypatch.setattr(evaluate, "prepare_geometry", fail)

    code = evaluate.main([
        "--pretrained", "vit_l_lm", "--no-mlflow",
        "--also-calibration", str(tmp_path / "missing.json"),
    ])

    assert code == evaluate.EXIT_FAILED
