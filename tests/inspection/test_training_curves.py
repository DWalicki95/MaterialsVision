"""Tests for reading training curves back from a store and drawing them.

Each test writes a small run into a temporary MLflow store under the
names the training logger and the scoring script use, so that a change
in either naming would break the reading here.
"""
import time

import mlflow
import numpy as np
import pytest
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient

from materials_vision import tracking
from materials_vision.evaluation.watershed import (CALIBRATED_2026_09_24,
                                                   FROZEN_WATERSHED)
from materials_vision.inspection.training_curves import (
    load_run_curves, training_curves_figure)

EXPERIMENT = "peft_sam_training"

FROZEN_ID = tracking.postprocessing_id(FROZEN_WATERSHED)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store the loader is pointed at through the environment."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "DEFAULT_ARTIFACT_ROOT", tmp_path / "art")
    tracking.configure_tracking()
    tracking.ensure_experiment(EXPERIMENT)
    return uri


def _log_run(name, train_subset="train", test_epoch=None, rescored=False):
    """Twelve epochs of losses and validation F1, optionally a TEST point."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name, tags={"train_subset": train_subset}):
        for epoch in range(1, 13):
            mlflow.log_metrics({
                "epoch/train_instance_loss": 0.08 - 0.002 * epoch,
                "epoch/val_instance_loss": 0.07 - 0.001 * epoch,
            }, step=epoch)
            if train_subset == "train":
                mlflow.log_metric(
                    f"val/{FROZEN_ID}/f1", 0.80 + 0.005 * epoch, step=epoch
                )
        if rescored:
            # Stamped a second later, so that "logged last" does not
            # depend on two writes landing in different milliseconds.
            MlflowClient().log_metric(
                mlflow.active_run().info.run_id, f"val/{FROZEN_ID}/f1",
                0.99, timestamp=int(time.time() * 1000) + 1000, step=5,
            )
        if test_epoch is not None:
            mlflow.log_metric(f"test/{FROZEN_ID}/f1", 0.85, step=test_epoch)


def test_curves_are_read_on_the_epoch_axis(store):
    _log_run("d4_seed1", test_epoch=10)

    curves = load_run_curves("d4_seed1")

    np.testing.assert_array_equal(curves.train_loss.epochs, np.arange(1, 13))
    assert curves.validation_loss.values[0] == pytest.approx(0.069)
    assert curves.validation_metric.values[-1] == pytest.approx(0.86)
    np.testing.assert_array_equal(curves.test_metric.epochs, [10])
    assert curves.validation_held_out


def test_a_rescored_epoch_keeps_the_last_value(store):
    _log_run("d4_seed1", rescored=True)

    curves = load_run_curves("d4_seed1")

    assert curves.validation_metric.epochs.size == 12
    assert curves.validation_metric.values[4] == pytest.approx(0.99)


def test_run_trained_on_validation_is_recognised(store):
    _log_run("final", train_subset="train+val", test_epoch=12)

    curves = load_run_curves("final")

    assert not curves.validation_held_out
    assert curves.validation_metric.is_empty


def test_setting_without_curves_reads_empty(store):
    _log_run("d4_seed1", test_epoch=10)

    curves = load_run_curves("d4_seed1", postprocessing=CALIBRATED_2026_09_24)

    assert curves.validation_metric.is_empty
    assert curves.test_metric.is_empty
    assert not curves.train_loss.is_empty


def test_boundary_f1_can_be_read(store):
    _log_run("d4_seed1")
    mlflow.set_experiment(EXPERIMENT)
    run_id = mlflow.search_runs(output_format="list")[0].info.run_id
    MlflowClient().log_metric(run_id, f"val/{FROZEN_ID}/boundary_f1", 0.8,
                              step=1)

    curves = load_run_curves("d4_seed1", metric="boundary_f1")

    assert curves.validation_metric.values.tolist() == [0.8]


def test_missing_run_is_refused(store):
    with pytest.raises(ValueError, match="No run named"):
        load_run_curves("absent")


@pytest.mark.parametrize("field", ["loss", "metric", "postprocessing"])
def test_unknown_names_are_refused(store, field):
    _log_run("d4_seed1")

    with pytest.raises(ValueError, match="Unknown"):
        load_run_curves("d4_seed1", **{field: "nonsense"})


def test_figure_draws_both_panels(store):
    _log_run("d4_seed1", test_epoch=10)
    _log_run("final", train_subset="train+val", test_epoch=12)

    figure = training_curves_figure(
        [load_run_curves("d4_seed1"), load_run_curves("final")]
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    labels = [text.get_text() for legend in figure.legends
              for text in legend.get_texts()]
    assert "final (trained on validation images)" in labels
    assert "validation images also trained on" in labels


def test_runs_read_differently_are_refused(store):
    _log_run("d4_seed1")

    with pytest.raises(ValueError, match="one panel"):
        training_curves_figure([
            load_run_curves("d4_seed1"),
            load_run_curves("d4_seed1", metric="recall"),
        ])


def test_too_many_runs_are_refused(store):
    _log_run("d4_seed1")
    curves = load_run_curves("d4_seed1")

    with pytest.raises(ValueError):
        training_curves_figure([curves] * 4)
