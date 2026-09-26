"""Tests for the lock on TEST in the checkpoint scorer.

TEST is scored once, on snapshots chosen on VALIDATION beforehand. The
two ways of breaking that are both quiet: scoring TEST without meaning
to, and scoring several epochs of one run so that TEST ends up choosing
among them. Both must be refused before any model is loaded, so the
tests below make any model work raise, and check that a refused request
never gets that far while an allowed one does.
"""
import pytest

import scripts.evaluate_checkpoint as cli


class ReachedModelWork(Exception):
    """Raised when a request gets as far as loading or scoring a model."""


@pytest.fixture
def no_model_work(monkeypatch):
    def refuse(*args, **kwargs):
        raise ReachedModelWork

    monkeypatch.setattr(cli, "prepare_geometry", refuse)
    monkeypatch.setattr(cli, "score_checkpoint", refuse)


def _snapshots(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    return paths


def _argv(paths, *extra):
    argv = []
    for path in paths:
        argv += ["--checkpoint", str(path)]
    return argv + list(extra)


def test_test_without_the_flag_is_refused_before_any_work(
    tmp_path, no_model_work
):
    paths = _snapshots(tmp_path, "run_a/epoch-3.pt")

    status = cli.main(_argv(paths, "--subset", "test"))

    assert status == cli.EXIT_FAILED


def test_several_snapshots_of_one_run_are_refused_on_test(
    tmp_path, no_model_work
):
    paths = _snapshots(tmp_path, "run_a/epoch-3.pt", "run_a/epoch-8.pt")

    status = cli.main(_argv(paths, "--subset", "test", "--unlock-test"))

    assert status == cli.EXIT_FAILED


def test_one_snapshot_per_run_gets_past_the_lock(tmp_path, no_model_work):
    paths = _snapshots(tmp_path, "run_a/epoch-3.pt", "run_b/epoch-8.pt")

    with pytest.raises(ReachedModelWork):
        cli.main(_argv(paths, "--subset", "test", "--unlock-test"))


def test_a_whole_curve_is_still_scored_on_validation(
    tmp_path, no_model_work
):
    paths = _snapshots(tmp_path, "run_a/epoch-3.pt", "run_a/epoch-8.pt")

    with pytest.raises(ReachedModelWork):
        cli.main(_argv(paths, "--subset", "val"))


def test_the_reason_names_the_run_with_several_snapshots(tmp_path):
    paths = _snapshots(tmp_path, "run_a/epoch-3.pt", "run_a/epoch-8.pt")
    args = cli.parse_args(_argv(paths, "--subset", "test", "--unlock-test"))
    args.checkpoint = cli.resolve_checkpoints(args)

    reason = cli.locked_test_reason(args)

    assert reason is not None
    assert "run_a" in reason


def test_the_default_postprocessing_is_the_attribution_studys():
    from materials_vision.evaluation.watershed import FROZEN_WATERSHED

    args = cli.parse_args(["--checkpoint", "x.pt"])
    assert cli.resolve_settings(args) == (FROZEN_WATERSHED,)


def test_a_named_postprocessing_is_the_base_single_flags_override():
    from materials_vision.evaluation.watershed import CALIBRATED_2026_09_24

    args = cli.parse_args([
        "--checkpoint", "x.pt",
        "--postprocessing", "calibrated_2026_09_24",
    ])
    assert cli.resolve_settings(args) == (CALIBRATED_2026_09_24,)

    args = cli.parse_args([
        "--checkpoint", "x.pt",
        "--postprocessing", "calibrated_2026_09_24",
        "--center-distance-threshold", "0.3",
    ])
    (setting,) = cli.resolve_settings(args)
    assert setting.center_distance_threshold == 0.3
    assert setting.distance_smoothing == (
        CALIBRATED_2026_09_24.distance_smoothing
    )
    assert setting.min_instance_area_um2 == (
        CALIBRATED_2026_09_24.min_instance_area_um2
    )


def test_another_named_postprocessing_is_scored_in_the_same_pass():
    from materials_vision.evaluation.watershed import (CALIBRATED_2026_09_24,
                                                       FROZEN_WATERSHED)

    args = cli.parse_args([
        "--checkpoint", "x.pt",
        "--postprocessing", "calibrated_2026_09_24",
        "--center-threshold-sweep", "0.20", "0.25", "0.30",
        "--also-postprocessing", "frozen",
    ])
    settings = cli.resolve_settings(args)
    assert [s.center_distance_threshold for s in settings[:3]] == [
        0.20, 0.25, 0.30,
    ]
    assert settings[1] == CALIBRATED_2026_09_24
    assert settings[3] == FROZEN_WATERSHED
    assert len(settings) == 4
