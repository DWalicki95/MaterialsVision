"""Tests for the MLflow record of training runs and their evaluations.

The record matters for one reason above the others: the watershed
settings are about to be recalibrated, and a figure read under the old
settings must never share a curve with one read under the new. The
naming tests pin that. The remaining tests run the trainer's logger
against a real MLflow store, because a logger tested apart from the
store it writes to is exactly the kind of isolated check that let
defects through in this repository before.
"""
import math
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from mlflow import MlflowClient

from materials_vision import tracking
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   WatershedParams)
from materials_vision.tracking import (CHECKPOINT_DIR_TAG, RunTracking,
                                       checkpoint_dir_tag,
                                       evaluation_metric_name,
                                       evaluation_metrics,
                                       find_run_for_checkpoint_dir,
                                       late_epoch_summary, log_metrics_at,
                                       postprocessing_id, section_metrics)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway MLflow store, so no test touches the project's."""
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    monkeypatch.setattr(tracking, "DEFAULT_ARTIFACT_ROOT", tmp_path / "art")
    tracking.configure_tracking()
    yield tmp_path
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _history(run_id, key):
    return sorted(
        (entry.step, entry.value)
        for entry in MlflowClient().get_metric_history(run_id, key)
    )


class TestPostprocessingId:
    def test_frozen_setting_spells_out_every_value(self):
        assert postprocessing_id(FROZEN_WATERSHED) == (
            "c0.3_b0.4_f0.5_fs1_d1.6_m0"
        )

    def test_does_not_depend_on_what_is_frozen(self):
        # A setting that differs from the frozen one in a value the
        # label() helper would omit must still get its own name.
        other = WatershedParams(
            center_distance_threshold=0.30,
            boundary_distance_threshold=0.40,
            foreground_threshold=0.45,
        )
        assert postprocessing_id(other) != postprocessing_id(
            FROZEN_WATERSHED
        )

    def test_is_a_legal_metric_name_component(self):
        name = evaluation_metric_name(
            "val", postprocessing_id(FROZEN_WATERSHED), "f1",
            section="material=K",
        )
        assert "=" not in name
        assert name == "val/c0.3_b0.4_f0.5_fs1_d1.6_m0/material_K/f1"


class TestEvaluationMetrics:
    def _result(self, **overrides):
        result = {
            "label": "overall", "f1": 0.85, "precision": 0.8,
            "recall": 0.9, "n_gt": 100, "n_pred": 110,
            "boundary_f1": {"0.1": 0.77}, "wasserstein_um": float("nan"),
        }
        result.update(overrides)
        return result

    def test_reads_json_form_with_string_scale_keys(self):
        named = evaluation_metrics(self._result(), "val", "pp")
        assert named["val/pp/f1"] == pytest.approx(0.85)
        assert named["val/pp/boundary_f1"] == pytest.approx(0.77)
        assert named["val/pp/n_pred"] == 110

    def test_reads_float_scale_keys(self):
        named = evaluation_metrics(
            self._result(boundary_f1={0.1: 0.5}), "val", "pp"
        )
        assert named["val/pp/boundary_f1"] == pytest.approx(0.5)

    def test_leaves_out_non_finite_and_missing(self):
        named = evaluation_metrics(self._result(), "val", "pp")
        assert "val/pp/wasserstein_um" not in named
        assert "val/pp/macro_f1" not in named

    def test_sections_are_named_by_their_label(self):
        named = section_metrics(
            [self._result(label="material=AS", f1=0.88),
             self._result(label="material=K", f1=0.78)],
            "val", "pp",
        )
        assert named["val/pp/material_AS/f1"] == pytest.approx(0.88)
        assert named["val/pp/material_K/f1"] == pytest.approx(0.78)


class TestLateEpochSummary:
    def test_mean_of_the_last_three_and_the_peak_beside_it(self):
        curve = {epoch: 0.80 for epoch in range(1, 13)}
        curve.update({6: 0.90, 10: 0.84, 11: 0.85, 12: 0.86})
        summary = late_epoch_summary(curve)
        assert summary["late_mean"] == pytest.approx(0.85)
        assert summary["peak"] == pytest.approx(0.90)
        assert summary["peak_epoch"] == 6

    def test_no_late_mean_over_an_incomplete_tail(self):
        summary = late_epoch_summary({1: 0.8, 10: 0.84, 11: 0.85})
        assert "late_mean" not in summary
        assert summary["peak_epoch"] == 11

    def test_empty_curve(self):
        assert late_epoch_summary({}) == {}


class TestCheckpointDirTag:
    def test_inside_the_repository_is_relative(self):
        inside = tracking.REPO_ROOT / "checkpoints" / "x" / "run"
        assert checkpoint_dir_tag(inside) == "checkpoints/x/run"

    def test_relative_and_absolute_spellings_agree(self, monkeypatch):
        monkeypatch.chdir(tracking.REPO_ROOT)
        assert checkpoint_dir_tag(Path("checkpoints/x/run")) == (
            checkpoint_dir_tag(tracking.REPO_ROOT / "checkpoints/x/run")
        )


class TestStoreRoundTrip:
    def test_training_run_is_found_from_its_checkpoint_dir(self, store):
        from materials_vision.training import tracked_run

        checkpoint_dir = store / "checkpoints" / "run_a"
        spec = RunTracking(
            experiment="test_training", tags={"arm": "a1_r32"},
            params={"seed": 1},
        )
        with tracked_run("run_a", checkpoint_dir, spec, {"lora_rank": 32}):
            pass

        run_id = find_run_for_checkpoint_dir(checkpoint_dir)
        assert run_id is not None
        run = MlflowClient().get_run(run_id)
        assert run.data.tags["arm"] == "a1_r32"
        assert run.data.tags[CHECKPOINT_DIR_TAG] == checkpoint_dir_tag(
            checkpoint_dir
        )
        assert run.data.params["lora_rank"] == "32"
        assert run.data.params["seed"] == "1"

    def test_evaluation_lands_on_the_epoch_axis(self, store):
        experiment = tracking.ensure_experiment("test_eval")
        with mlflow.start_run(experiment_id=experiment) as run:
            run_id = run.info.run_id
        log_metrics_at(run_id, {"val/pp/f1": 0.81}, step=3)
        log_metrics_at(run_id, {"val/pp/f1": 0.84}, step=4)
        assert _history(run_id, "val/pp/f1") == [(3, 0.81), (4, 0.84)]

    def test_unknown_checkpoint_dir_finds_nothing(self, store):
        tracking.ensure_experiment("test_empty")
        assert find_run_for_checkpoint_dir(store / "nowhere") is None


class _TinyDecoder(torch.nn.Module):
    """Three sigmoid maps, the shape the instance decoder outputs.

    Refuses an image in [0, 1] the way the real decoder does, because
    the library's own logger rescales the image in place before ours
    sees it, and a decoder that accepted anything would hide that.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        if x.max() <= 1.0:
            raise ValueError("input looks rescaled to [0, 1]")
        return torch.sigmoid(self.conv(x))


class _FakeTrainer:
    def __init__(self, save_root):
        self.name = "fake"
        self.log_image_interval = 10_000
        self._epoch = 0
        self.device = "cpu"
        self.unetr = _TinyDecoder()
        self.optimizer = torch.optim.AdamW([
            {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": 3e-4},
            {"params": list(self.unetr.parameters()), "lr": 1e-5},
        ])
        self.save_root = save_root


def _batch():
    x = torch.rand(1, 3, 16, 20) * 255
    y = torch.zeros(1, 1, 16, 20)
    y[0, 0, 2:8, 3:9] = 1
    y[0, 0, 9:14, 10:18] = 2
    return x, y


# The interactive branch's sampled masks, which the library's logger
# draws into TensorBoard at validation and on every image step.
SAMPLES = torch.zeros(1, 1, 16, 20)


class TestTrainerLogger:
    def _logger(self, store):
        from materials_vision.training_logger import MlflowJointSamLogger

        trainer = _FakeTrainer(str(store))
        return trainer, MlflowJointSamLogger(trainer, str(store))

    def _step(self, logger, step, loss):
        x, y = _batch()
        logger.log_train(
            step, torch.tensor(loss), 3e-4, x, y, SAMPLES,
            torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.7),
            torch.tensor(loss / 2),
        )

    def test_epoch_is_recorded_on_both_axes(self, store):
        trainer, logger = self._logger(store)
        experiment = tracking.ensure_experiment("test_logger")
        with mlflow.start_run(experiment_id=experiment) as run:
            for step, loss in enumerate((1.0, 2.0, 3.0), start=1):
                self._step(logger, step, loss)
            x, y = _batch()
            logger.log_validation(
                3, 0.5, 0.4, x, y, SAMPLES, 0.1, 0.2, 0.7, 0.3,
            )
            mlflow.flush_async_logging()
            run_id = run.info.run_id

        assert [value for _, value in _history(run_id, "train/loss")] == [
            1.0, 2.0, 3.0,
        ]
        assert _history(run_id, "train/lr_lora")[0][1] == pytest.approx(3e-4)
        assert _history(run_id, "train/lr_decoder")[0][1] == pytest.approx(
            1e-5
        )
        # Epoch-level figures sit at the one-based epoch, the axis the
        # instance metrics are logged on afterwards.
        assert _history(run_id, "epoch/train_loss") == [(1, 2.0)]
        assert _history(run_id, "epoch/val_loss") == [(1, pytest.approx(0.4))]
        assert _history(run_id, "epoch/lr_decoder")[0][1] == pytest.approx(
            1e-5
        )
        assert logger._failures == 0

        images = [
            item.path for item in
            MlflowClient().list_artifacts(run_id, "images")
        ]
        assert any("val_decoder_maps" in path for path in images)

    def test_epoch_means_restart_every_epoch(self, store):
        trainer, logger = self._logger(store)
        experiment = tracking.ensure_experiment("test_logger_reset")
        with mlflow.start_run(experiment_id=experiment) as run:
            x, y = _batch()
            self._step(logger, 1, 1.0)
            logger.log_validation(1, 0.5, 0.4, x, y, SAMPLES, 0, 0, 0, 0)
            trainer._epoch = 1
            self._step(logger, 2, 5.0)
            logger.log_validation(2, 0.5, 0.4, x, y, SAMPLES, 0, 0, 0, 0)
            mlflow.flush_async_logging()
            run_id = run.info.run_id
        assert _history(run_id, "epoch/train_loss") == [(1, 1.0), (2, 5.0)]

    def test_missing_run_is_counted_not_raised(self, store):
        _, logger = self._logger(store)
        self._step(logger, 1, 1.0)
        assert logger._failures == 1


def test_decoder_maps_figure_is_an_rgb_image():
    from materials_vision.training_logger import decoder_maps_figure

    instances = np.zeros((16, 20), dtype=np.int64)
    instances[2:8, 3:9] = 1
    figure = decoder_maps_figure(
        np.random.rand(3, 16, 20), instances, np.random.rand(3, 16, 20),
        "t",
    )
    assert figure.ndim == 3 and figure.shape[2] == 3
    assert figure.dtype == np.uint8
    assert not math.isnan(float(figure.mean()))
