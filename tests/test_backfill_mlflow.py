"""Tests for importing the runs trained before MLflow tracking existed.

The import reads two kinds of record written by other code: curve
reports keyed ``run/epoch-N@label`` and TensorBoard files. The tests pin
how both are read, and above all that a report's "frozen" label stays
bound to the setting it meant when it was written, whatever the code's
frozen setting becomes afterwards.
"""
import mlflow
import pytest
from mlflow import MlflowClient

import scripts.backfill_mlflow as backfill
from materials_vision import tracking
from materials_vision.evaluation.watershed import WatershedParams


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    monkeypatch.setattr(tracking, "DEFAULT_ARTIFACT_ROOT", tmp_path / "art")
    tracking.configure_tracking()
    yield tmp_path
    while mlflow.active_run() is not None:
        mlflow.end_run()


class TestCurveKeys:
    def test_labelled_key(self):
        assert backfill.parse_curve_key("d4_seed1/epoch-10@cdt=0.25") == (
            "d4_seed1", 10, "cdt=0.25",
        )

    def test_unlabelled_key(self):
        assert backfill.parse_curve_key("b0_seed1/epoch-3") == (
            "b0_seed1", 3, None,
        )

    def test_non_epoch_snapshot_is_refused(self):
        with pytest.raises(ValueError):
            backfill.parse_curve_key("b0_seed1/best")


class TestSettingForLabel:
    report = backfill.CurveReport(path=None)

    def test_frozen_is_the_calibration_of_its_day(self, monkeypatch):
        # Changing the code's frozen setting must not relabel history.
        import materials_vision.evaluation.watershed as watershed

        monkeypatch.setattr(
            watershed, "FROZEN_WATERSHED",
            WatershedParams(foreground_threshold=0.6),
        )
        setting = backfill.setting_for_label("frozen", self.report)
        assert setting == backfill.CALIBRATION_2026_09_14
        assert tracking.postprocessing_id(setting) == (
            "c0.3_b0.4_f0.5_fs1_d1.6_m0"
        )

    def test_centre_threshold_label_changes_only_that_value(self):
        setting = backfill.setting_for_label("cdt=0.25", self.report)
        assert setting.center_distance_threshold == 0.25
        assert setting.boundary_distance_threshold == 0.40

    def test_unknown_label_is_refused(self):
        with pytest.raises(ValueError):
            backfill.setting_for_label("fg=0.4", self.report)


def test_epoch_means_split_at_the_validation_steps():
    series = [(0, 1.0), (1, 3.0), (2, 10.0), (3, 20.0)]
    assert backfill.epoch_means(series, [2, 4]) == {1: 2.0, 2: 15.0}


def test_multi_run_provenance_is_reduced_to_one_run():
    record = {
        "provenance": {"git": {"commit": "abc"}},
        "runs": [
            {"name": "b0_seed1", "seed": 1},
            {"name": "b0_seed2", "seed": 2},
        ],
        "tonal": {"gamma_limit": [75, 125]},
    }
    params = backfill.flatten_provenance(record, "b0_seed2")
    assert params == {
        "run_name": "b0_seed2", "run_seed": "2",
        "tonal.gamma_limit": "75,125",
    }


def test_only_the_last_tensorboard_file_is_read(tmp_path):
    for stamp in ("1789892411", "1789892243"):
        (tmp_path / f"events.out.tfevents.{stamp}.host.1.0").touch()
    assert backfill.last_event_file(tmp_path).name.startswith(
        "events.out.tfevents.1789892411"
    )


def test_losses_land_on_both_axes(store, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    log_dir = tmp_path / "root" / "logs" / "run_x"
    writer = SummaryWriter(str(log_dir))
    for step, loss in enumerate((1.0, 3.0, 10.0, 20.0)):
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/learning_rate", 3e-4, step)
    writer.add_scalar("validation/loss", 0.5, 2)
    writer.add_scalar("validation/loss", 0.4, 4)
    writer.close()

    run = backfill.HistoricalRun(
        tmp_path / "root", "run_x", "E0", "B0", 1, provenance=None,
    )
    experiment = tracking.ensure_experiment("test_backfill")
    with mlflow.start_run(experiment_id=experiment) as active:
        run_id = active.info.run_id
    assert backfill.import_losses(run_id, run) == 2

    client = MlflowClient()
    epochs = sorted(
        (m.step, m.value)
        for m in client.get_metric_history(run_id, "epoch/train_loss")
    )
    assert epochs == [(1, 2.0), (2, 15.0)]
    validation = sorted(
        (m.step, round(m.value, 6))
        for m in client.get_metric_history(run_id, "epoch/val_loss")
    )
    assert validation == [(1, 0.5), (2, 0.4)]
    assert len(client.get_metric_history(run_id, "train/loss")) == 4
