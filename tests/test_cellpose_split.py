"""Tests for the Cellpose-SAM comparison on the frozen split.

Two seams matter here, and both are exercised end to end rather than
through the components in isolation. The first runs from a split file
on disk to what the Cellpose trainer is handed: it must be the rows the
PEFT-SAM runs trained on and nothing from TEST. The second runs from
predicted flows, through Cellpose's own post-processing, to the metrics
every PEFT-SAM figure is read with. The network itself needs weights
and a GPU and is replaced by a stand-in that returns the flows of a
known labelling, scaled as the network predicts them.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.evaluate_cellpose as evaluate_cli
import scripts.train_cellpose_split as train_cli
from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.evaluation import SizeBins, aggregate
from materials_vision.experiments.cellpose.split_evaluation import (
    CALIBRATION_NITERS, LIBRARY_DEFAULT, SELECTION_NITER, CellposeSettings,
    calibration_grid, masks_from_flows, score_settings)
from materials_vision.experiments.cellpose.split_training import (
    LEARNING_RATES, N_EPOCHS, REHEARSAL_EXPERIMENT, TRAINING_EXPERIMENT,
    SnapshotNameError, completed_epochs, loss_points, saved_epochs,
    snapshot_path)

BINS = SizeBins(edges_um2=(300.0, 600.0, 900.0),
                n_calibration_instances=100)

# Four square-micrometre pixels: the 5x5 instance below covers 100 um2
# and falls under the 200 um2 filter, the 12x12 ones cover 576 um2.
PIXEL_SIZE_UM = 2.0

# The network predicts flows five times the unit field Cellpose trains
# on, and Cellpose's post-processing divides by five before using them.
NETWORK_FLOW_SCALE = 5.0

CELLPROB_LOGIT = 6.0


# --------------------------------------------------------------------
# Snapshot layout
# --------------------------------------------------------------------
def _cellpose_saves_after(iepoch: int, n_epochs: int, save_every: int):
    """Cellpose 4.0.8's condition for saving, as written in train_seg."""
    return iepoch == n_epochs - 1 or (
        iepoch % save_every == 0 and iepoch != 0
    )


@pytest.mark.parametrize("n_epochs", [N_EPOCHS, 12, 11, 21])
def test_saved_epochs_are_the_ones_cellpose_saves(n_epochs):
    expected = tuple(
        iepoch + 1 for iepoch in range(n_epochs)
        if _cellpose_saves_after(iepoch, n_epochs, 10)
    )

    assert saved_epochs(n_epochs) == expected


def test_snapshot_names_round_trip_to_completed_epochs(tmp_path):
    for completed in saved_epochs():
        path = snapshot_path(tmp_path, "run", completed)
        assert completed_epochs(path) == completed
    assert snapshot_path(tmp_path, "run", 11).name == "run_epoch_0010"
    assert snapshot_path(tmp_path, "run", N_EPOCHS).name == "run"


@pytest.mark.parametrize("completed", [0, 1, 10, 12, 159, 161])
def test_an_epoch_without_a_snapshot_is_refused(tmp_path, completed):
    with pytest.raises(SnapshotNameError):
        snapshot_path(tmp_path, "run", completed)


def test_a_file_that_is_not_a_snapshot_is_refused(tmp_path):
    with pytest.raises(SnapshotNameError):
        completed_epochs(tmp_path / "run" / "models" / "notes.txt")


def test_zero_test_losses_are_not_logged():
    points = loss_points([3.0, 2.0, 1.0], [0.0, 2.5, 0.0])

    assert ("training/test_loss", 2.5, 2) in points
    assert [p for p in points if p[0] == "training/test_loss"] == [
        ("training/test_loss", 2.5, 2)
    ]
    assert len([p for p in points if p[0] == "training/train_loss"]) == 3


# --------------------------------------------------------------------
# Split to trainer
# --------------------------------------------------------------------
@pytest.fixture
def trainer_calls(monkeypatch):
    """Stand in for the trainer and record what it was handed."""
    calls = []

    def record(name, **kwargs):
        calls.append((name, kwargs))
        return Path("unused")

    monkeypatch.setattr("scripts.train_cellpose_split.train_cellpose", record)
    return calls


def _ids(source):
    return {record.image_id for record in source.records}


def _train_args(frozen_split, tmp_path, *extra):
    split_csv, manifest = frozen_split
    return train_cli.parse_args([
        "--arm", "lr1e-5", "--split", str(split_csv),
        "--manifest", str(manifest), "--cache-dir", str(tmp_path / "c"),
        *extra,
    ])


def test_an_arm_trains_on_train_and_monitors_on_val_only(
    tmp_path, frozen_split, trainer_calls
):
    args = _train_args(frozen_split, tmp_path)
    table = pd.read_csv(args.split)
    rows = {
        subset: set(table.loc[
            (table["split"] == subset) & table["used"], "image_id"
        ])
        for subset in ("train", "val", "test")
    }

    train_cli.execute(args, tmp_path / "provenance.json")

    [(name, kwargs)] = trainer_calls
    assert name == "cpsam_lr1e-5_seed20260907"
    assert _ids(kwargs["train_source"]) == rows["train"]
    assert _ids(kwargs["monitor_source"]) == rows["val"]
    assert not (
        _ids(kwargs["train_source"]) | _ids(kwargs["monitor_source"])
    ) & rows["test"]
    assert kwargs["learning_rate"] == LEARNING_RATES["lr1e-5"]
    assert kwargs["n_epochs"] == N_EPOCHS
    assert kwargs["train_positions"] is None
    assert kwargs["tracking"].experiment == TRAINING_EXPERIMENT
    assert kwargs["cache_dir"] == tmp_path / "c"


def test_a_rehearsal_is_short_and_kept_apart(
    tmp_path, frozen_split, trainer_calls
):
    args = _train_args(frozen_split, tmp_path, "--rehearsal")

    train_cli.execute(args, tmp_path / "provenance.json")

    [(_, kwargs)] = trainer_calls
    assert kwargs["n_epochs"] == train_cli.REHEARSAL_EPOCHS
    assert len(kwargs["train_positions"]) == train_cli.REHEARSAL_N_TRAIN
    assert kwargs["tracking"].experiment == REHEARSAL_EXPERIMENT
    assert kwargs["save_root"] == train_cli.REHEARSAL_SAVE_ROOT
    assert len(saved_epochs(train_cli.REHEARSAL_EPOCHS)) == 2


def test_provenance_records_the_arm_and_its_rate(tmp_path, frozen_split):
    args = _train_args(frozen_split, tmp_path)
    args.save_root = tmp_path / "ck"

    record = train_cli.provenance_record(args)

    assert record["learning_rate"] == LEARNING_RATES["lr1e-5"]
    assert record["augmentation"] == "cellpose_builtin"
    assert record["min_train_masks"] == 0
    assert record["n_epochs"] == N_EPOCHS


# --------------------------------------------------------------------
# Flows to metrics
# --------------------------------------------------------------------
def _labels() -> np.ndarray:
    labels = np.zeros((64, 64), dtype=np.uint16)
    labels[4:16, 4:16] = 1
    labels[4:16, 30:42] = 2
    labels[36:48, 10:22] = 3
    labels[40:45, 44:49] = 4
    return labels


def _network_output(labels: np.ndarray):
    """The flows and logits a perfect network would predict."""
    from cellpose import dynamics

    flows = dynamics.labels_to_flows(
        [labels.copy()], device=torch.device("cpu")
    )[0]
    flow_field = NETWORK_FLOW_SCALE * flows[2:4]
    cellprob = np.where(labels > 0, CELLPROB_LOGIT, -CELLPROB_LOGIT)
    return flow_field.astype(np.float32), cellprob.astype(np.float32)


class _PerfectNetwork:
    """Stands in for a snapshot, returning the flows of the annotation."""

    device = torch.device("cpu")

    def __init__(self, output):
        self._output = output
        self.calls = 0
        self.kwargs = []

    def eval(self, image, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        flow_field, cellprob = self._output
        return np.zeros(0), [None, flow_field, cellprob], None


class _Source:
    """Two identical samples, as the sample source would prepare them."""

    def __init__(self, labels):
        self._labels = labels

    def load(self, position):
        record = SampleRecord(
            index=position, image_id=f"AS1_{position}", formulation="AS1",
            material="AS", microscope="M1", scale_bin="coarse",
            pixel_size_um=PIXEL_SIZE_UM, q_max_i=1.3,
            source_path=Path("image.png"), mask_path=Path("mask.tif"),
            crop_bbox=(0, 0, 64, 64), n_instances_expected=4,
        )
        return PreparedSample(
            record=record,
            image=np.zeros(self._labels.shape, dtype=np.uint8),
            labels=self._labels,
            border_instance=np.zeros(4, dtype=bool),
            n_cut_by_crop=0, n_dropped_below_min_area=0,
            n_dropped_disconnected=0,
        )


def test_cellpose_post_processing_feeds_the_shared_metrics():
    labels = _labels()
    network = _PerfectNetwork(_network_output(labels))
    settings = (
        LIBRARY_DEFAULT,
        CellposeSettings(flow_threshold=0.8, cellprob_threshold=-1.0),
        LIBRARY_DEFAULT.with_area_filter(),
    )

    measured = score_settings(network, _Source(labels), [0, 1], BINS,
                              settings)

    assert network.calls == 2
    assert all(kwargs["compute_masks"] is False for kwargs in network.kwargs)
    assert all(kwargs["diameter"] is None for kwargs in network.kwargs)
    for setting in settings[:2]:
        result = aggregate(measured[setting])
        assert result.n_images == 2
        assert result.f1 == pytest.approx(1.0)
    filtered = aggregate(measured[settings[2]])
    assert filtered.precision == pytest.approx(1.0)
    assert filtered.recall == pytest.approx(0.75)


def test_scoring_under_no_setting_is_refused():
    with pytest.raises(ValueError, match="At least one setting"):
        score_settings(None, None, [0], BINS, ())


def test_the_calibration_grid_is_the_one_decided():
    grid = calibration_grid()

    assert len(grid) == 20
    assert {s.flow_threshold for s in grid} == {0.2, 0.4, 0.6, 0.8}
    assert {s.cellprob_threshold for s in grid} == {-2, -1, 0, 1, 2}
    assert {s.min_size_px for s in grid} == {15}
    assert len({s.label() for s in grid}) == 20
    assert LIBRARY_DEFAULT in grid


def test_the_second_grid_crosses_thresholds_with_steps():
    grid = calibration_grid(CALIBRATION_NITERS)

    assert CALIBRATION_NITERS == (1000, 2000)
    assert len(grid) == 40
    assert {s.niter for s in grid} == {1000, 2000}
    assert len({s.label() for s in grid}) == 40
    assert CellposeSettings(niter=SELECTION_NITER) in grid
    with pytest.raises(ValueError, match="positive"):
        calibration_grid(())


def test_labels_of_the_first_round_are_unchanged():
    """Figures recorded before the steps were a setting keep their names."""
    assert LIBRARY_DEFAULT.label() == "ft0.4_cp0_m15"
    assert LIBRARY_DEFAULT.with_area_filter().label() == "ft0.4_cp0_m15_a200"
    stepped = CellposeSettings(niter=1000)
    assert stepped.label() == "ft0.4_cp0_m15_n1000"
    assert stepped.with_area_filter().label() == "ft0.4_cp0_m15_n1000_a200"
    assert stepped.with_area_filter().niter == 1000


def test_the_steps_reach_cellposes_flow_following(monkeypatch):
    from cellpose import dynamics

    received = []
    monkeypatch.setattr(
        dynamics, "resize_and_compute_masks",
        lambda *args, **kwargs: received.append(kwargs) or np.zeros(
            (4, 4), dtype=np.uint16
        ),
    )

    masks_from_flows(
        np.zeros((2, 4, 4)), np.zeros((4, 4)),
        CellposeSettings(niter=2000), device=torch.device("cpu"),
    )

    assert received[0]["niter"] == 2000


# --------------------------------------------------------------------
# Evaluation entry point
# --------------------------------------------------------------------
def _snapshots(root: Path, run: str, names) -> list[Path]:
    directory = root / run / "models"
    directory.mkdir(parents=True)
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(b"weights")
        paths.append(path)
    return paths


def _eval_args(*extra) -> argparse.Namespace:
    return evaluate_cli.parse_args(list(extra))


def test_snapshots_are_ordered_as_trained_and_other_files_skipped(tmp_path):
    args = _eval_args()
    args.checkpoint = _snapshots(tmp_path, "run_a", [
        "run_a", "run_a_epoch_0100", "notes.txt", "run_a_epoch_0010",
    ])

    resolved = evaluate_cli.resolve_checkpoints(args)

    assert [completed_epochs(p) for p in resolved] == [11, 101, 160]


def test_test_is_locked_without_the_flag(tmp_path):
    args = _eval_args("--subset", "test")
    args.checkpoint = _snapshots(tmp_path, "run_a", ["run_a"])

    assert "locked" in evaluate_cli.locked_test_reason(args)


def test_test_accepts_one_snapshot_per_run(tmp_path):
    args = _eval_args("--subset", "test", "--unlock-test")
    args.checkpoint = (
        _snapshots(tmp_path, "run_a", ["run_a", "run_a_epoch_0010"])
        + _snapshots(tmp_path, "run_b", ["run_b"])
    )

    reason = evaluate_cli.locked_test_reason(args)
    assert "run_a" in reason and "run_b" not in reason

    args.checkpoint = [args.checkpoint[0], args.checkpoint[2]]
    assert evaluate_cli.locked_test_reason(args) is None


def test_settings_resolve_to_one_the_grid_or_both_filters():
    single = evaluate_cli.resolve_settings(_eval_args(
        "--flow-threshold", "0.6", "--cellprob-threshold", "-1",
        "--also-area-filter",
    ))
    grid = evaluate_cli.resolve_settings(_eval_args("--grid"))

    assert single == (
        CellposeSettings(flow_threshold=0.6, cellprob_threshold=-1.0),
        CellposeSettings(
            flow_threshold=0.6, cellprob_threshold=-1.0,
            min_instance_area_um2=200.0,
        ),
    )
    assert grid == calibration_grid()


def test_steps_resolve_for_one_setting_and_for_the_grid():
    single = evaluate_cli.resolve_settings(_eval_args(
        "--niter", "1000", "--also-area-filter",
    ))
    grid = evaluate_cli.resolve_settings(_eval_args(
        "--grid", "--grid-niter", "1000", "--grid-niter", "2000",
    ))

    assert [s.niter for s in single] == [1000, 1000]
    assert single[1].min_instance_area_um2 == 200.0
    assert grid == calibration_grid((1000, 2000))
