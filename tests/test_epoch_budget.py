"""Tests for the epoch-budget study: split, training, choice, comparison.

The trainer and the evaluation are replaced; what these pin is that the
split moves exactly the frozen VALIDATION out of fold four's training,
that each arm's loaders come from that split through the real reader
and sample source, that the epoch is chosen by the rule and refused on
an incomplete curve, and that the comparison is computed from summed
counts of the snapshots actually chosen.
"""
import json
import math

import mlflow
import pandas as pd
import pytest

import scripts.create_cv_folds as create_cli
import scripts.peft_sam.compare_epoch_budget as compare_cli
import scripts.peft_sam.create_epoch_budget_split as split_cli
import scripts.peft_sam.select_epoch_on_val as select_cli
import scripts.peft_sam.train_cv_fold as cv_cli
import scripts.peft_sam.train_epoch_budget as cli
from materials_vision import tracking
from materials_vision.data.cv_folds import FoldError, carve_validation
from materials_vision.data.split_io import load_split
from materials_vision.tracking import (late_epochs_of, log_metrics_at,
                                       summarize_curve)
from materials_vision.training import DECODER_LEARNING_RATE, LORA_LEARNING_RATE


@pytest.fixture
def folds(tmp_path, frozen_split):
    """Folds written from the miniature split, as the real ones were."""
    split_csv, manifest = frozen_split
    out_dir = tmp_path / "cv5"
    code = create_cli.main([
        "--split", str(split_csv), "--out-dir", str(out_dir),
        "--seed", "11",
    ])
    assert code == create_cli.EXIT_OK
    return out_dir, manifest


@pytest.fixture
def budget_split(tmp_path, folds):
    """The study's split, written from the miniature folds."""
    folds_dir, manifest = folds
    out_dir = tmp_path / "budget"
    code = split_cli.main([
        "--folds-dir", str(folds_dir), "--out-dir", str(out_dir),
    ])
    assert code == split_cli.EXIT_OK
    return out_dir / f"{split_cli.SPLIT_NAME}.csv", manifest


@pytest.fixture
def trainer_calls(monkeypatch):
    """Stand in for the trainer and record what it was handed."""
    calls = []
    monkeypatch.setattr(
        "scripts.peft_sam.train_final_model.train_run",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    return calls


def _used_ids(table, subset):
    rows = table[(table["split"] == subset) & table["used"].astype(bool)]
    return set(rows["image_id"])


def _image_ids(loader):
    return {record.image_id for record in loader.dataset.source.records}


class TestSplit:
    def test_validation_is_exactly_the_frozen_validation(
        self, folds, budget_split
    ):
        folds_dir, _ = folds
        split = pd.read_csv(budget_split[0])
        fold2 = pd.read_csv(folds_dir / "fold2.csv")

        assert _used_ids(split, "val") == _used_ids(fold2, "test")

    def test_held_out_set_is_fold_fours_unchanged(self, folds, budget_split):
        folds_dir, _ = folds
        split = pd.read_csv(budget_split[0])
        fold4 = pd.read_csv(folds_dir / "fold4.csv")

        assert _used_ids(split, "test") == _used_ids(fold4, "test")
        assert _used_ids(split, "train") == (
            _used_ids(fold4, "train") - _used_ids(split, "val")
        )

    def test_it_loads_and_verifies_like_any_split(self, budget_split):
        split_csv, manifest = budget_split
        val = load_split(split_csv, "val", verify_manifest=manifest)
        train = load_split(split_csv, "train", verify_manifest=manifest)

        assert not set(val.image_ids) & set(train.image_ids)

    def test_metadata_counts_the_validation_set(self, budget_split):
        split_csv, _ = budget_split
        meta = json.loads(
            split_csv.with_name(f"{split_csv.stem}_metadata.json").read_text()
        )
        split = pd.read_csv(split_csv)

        assert meta["counts"]["n_val_images"] == len(_used_ids(split, "val"))
        assert meta["fold"] == split_cli.FOLD

    def test_it_is_not_written_twice(self, folds, budget_split):
        code = split_cli.main([
            "--folds-dir", str(folds[0]),
            "--out-dir", str(budget_split[0].parent),
        ])

        assert code == split_cli.EXIT_FAILED

    def test_close_ups_of_a_moved_formulation_are_unused(self, frozen_table):
        fold = frozen_table.assign(split="train")

        carved = carve_validation(fold, frozenset({"AS12"}))

        rows = carved[carved["formulation"] == "AS12"]
        assert set(rows["split"]) == {"val"}
        assert list(rows.loc[rows["scale_bin"] == "outlier", "used"]) == [
            False
        ]
        assert rows.loc[rows["scale_bin"] != "outlier", "used"].all()

    def test_a_held_out_formulation_cannot_be_moved(self, frozen_table):
        fold = frozen_table.assign(
            split=frozen_table["split"].where(
                frozen_table["split"] == "test", "train"
            )
        )

        with pytest.raises(FoldError, match="not in the fold's training"):
            carve_validation(fold, frozenset({"AS3"}))


class TestTraining:
    def _args(self, budget_split, tmp_path, *extra):
        split_csv, manifest = budget_split
        return cli.parse_args([
            "--epochs", "50", "--split", str(split_csv),
            "--manifest", str(manifest), "--save-root", str(tmp_path / "ck"),
            *extra,
        ])

    def test_an_arm_trains_on_the_splits_training_rows_only(
        self, tmp_path, budget_split, trainer_calls
    ):
        args = self._args(budget_split, tmp_path)
        split = pd.read_csv(args.split)

        cli.execute(args)

        [(name, kwargs)] = trainer_calls
        assert name == "full_b_cv5_fold4_valv1_e50_seed20260907"
        assert _image_ids(kwargs["train_loader"]) == _used_ids(split, "train")
        monitor = _image_ids(kwargs["val_loader"])
        assert monitor <= _used_ids(split, "val")
        assert not monitor & _image_ids(kwargs["train_loader"])

    def test_an_arm_runs_the_recipe_for_its_budget_keeping_every_epoch(
        self, tmp_path, budget_split, trainer_calls
    ):
        cli.execute(self._args(budget_split, tmp_path))

        [(_, kwargs)] = trainer_calls
        assert kwargs["n_epochs"] == 50
        assert kwargs["save_every_kth_epoch"] == 1
        # The rate is annealed over the run's own length.
        assert kwargs.get("scheduler_horizon_epochs") is None
        assert kwargs["early_stopping"] is None
        tracking_spec = kwargs["tracking"]
        assert tracking_spec.experiment == cli.EPOCH_BUDGET_EXPERIMENT
        assert tracking_spec.tags["epochs"] == "50"
        assert tracking_spec.tags["fold"] == "4"

    def test_arms_never_share_a_directory_with_the_cv_fold(self):
        names = {cli.run_name(arm) for arm in cli.ARMS}

        assert len(names) == len(cli.ARMS)
        assert cv_cli.run_name(4) not in names
        assert cli.DEFAULT_SAVE_ROOT != cv_cli.DEFAULT_SAVE_ROOT

    def test_only_the_two_budgets_are_accepted(self):
        with pytest.raises(SystemExit):
            cli.parse_args(["--epochs", "24"])

    def test_the_recipes_rates_reach_the_trainer_by_default(
        self, tmp_path, budget_split, trainer_calls
    ):
        cli.execute(self._args(budget_split, tmp_path))

        [(name, kwargs)] = trainer_calls
        assert name == "full_b_cv5_fold4_valv1_e50_seed20260907"
        assert kwargs["lora_learning_rate"] == LORA_LEARNING_RATE
        assert kwargs["decoder_learning_rate"] == DECODER_LEARNING_RATE

    def test_a_rate_factor_halves_both_rates_under_its_own_name(
        self, tmp_path, budget_split, trainer_calls
    ):
        args = self._args(budget_split, tmp_path, "--lr-scale", "0.5")
        args.epochs = 12

        cli.execute(args)

        [(name, kwargs)] = trainer_calls
        assert name == "full_b_cv5_fold4_valv1_e12_lrx0.5_seed20260907"
        assert kwargs["lora_learning_rate"] == LORA_LEARNING_RATE / 2
        assert kwargs["decoder_learning_rate"] == DECODER_LEARNING_RATE / 2
        assert kwargs["tracking"].tags["lr_scale"] == "0.5"
        provenance = json.loads(
            (tmp_path / "ck" / f"{name}_provenance.json").read_text()
        )
        assert provenance["lora_learning_rate"] == LORA_LEARNING_RATE / 2
        assert provenance["decoder_learning_rate"] == (
            DECODER_LEARNING_RATE / 2
        )

    def test_only_the_declared_rate_factors_are_accepted(self):
        with pytest.raises(SystemExit):
            cli.parse_args(["--epochs", "12", "--lr-scale", "0.7"])

    def test_a_missing_split_is_refused(self, tmp_path):
        args = cli.parse_args([
            "--epochs", "12", "--split", str(tmp_path / "none.csv"),
        ])

        assert "create_epoch_budget_split" in cli.refusal(args)

    def test_a_rehearsal_is_short_and_kept_apart(
        self, tmp_path, budget_split, trainer_calls
    ):
        args = self._args(budget_split, tmp_path, "--rehearsal")

        cli.execute(args)

        [(_, kwargs)] = trainer_calls
        assert kwargs["n_epochs"] == cli.REHEARSAL_EPOCHS[50]
        assert len(kwargs["train_loader"]) == cli.REHEARSAL_N_TRAIN
        assert kwargs["tracking"].experiment == tracking.REHEARSAL_EXPERIMENT


def _section(label, tp, fp, fn, *, n_images=4, merges=0, splits=0):
    return {
        "label": label, "n_images": n_images, "n_gt": tp + fn,
        "n_pred": tp + fp, "n_true_positives": tp, "n_false_positives": fp,
        "n_false_negatives": fn, "n_merges": merges, "n_splits": splits,
        "f1": 2 * tp / (2 * tp + fp + fn), "precision": tp / (tp + fp),
        "recall": tp / (tp + fn), "merges_per_100_gt": 0.0,
        "splits_per_100_gt": 0.0, "pore_count_error": fp / (tp + fn),
    }


def _entry(checkpoint, f1_tp, *, setting="frozen", n_images=4):
    """One snapshot's report entry; ``f1_tp`` sets its F1."""
    overall = _section(f"{checkpoint}@{setting}", f1_tp, 10, 10,
                       n_images=n_images)
    return {
        "overall": overall,
        "per_material": [
            _section("material=AS", f1_tp, 6, 6),
            _section("material=K", f1_tp // 2, 4, 4),
        ],
        "per_formulation": [_section("formulation=AS16", f1_tp, 6, 6)],
    }


def _write_val_reports(directory, run, f1_tp_by_epoch, **kwargs):
    directory.mkdir(parents=True, exist_ok=True)
    for epoch, tp in f1_tp_by_epoch.items():
        checkpoint = f"ck/checkpoints/{run}/epoch-{epoch}.pt"
        (directory / f"epoch-{epoch}.json").write_text(json.dumps({
            f"{run}/epoch-{epoch}": _entry(checkpoint, tp, **kwargs)
        }))


class TestSelection:
    def test_the_highest_f1_wins_and_a_tie_goes_to_the_earlier(self):
        assert select_cli.select_epoch({1: 0.8, 2: 0.9, 3: 0.85}) == 2
        assert select_cli.select_epoch({1: 0.8, 4: 0.9, 2: 0.9}) == 2

    def test_an_undefined_f1_is_refused(self):
        with pytest.raises(select_cli.SelectionError, match="undefined"):
            select_cli.select_epoch({1: 0.8, 2: math.nan})

    def _choose(self, tmp_path, tp_by_epoch, n_epochs=3, **kwargs):
        run = cli.run_name(12)
        reports = tmp_path / "val"
        _write_val_reports(reports, run, tp_by_epoch, **kwargs)
        payloads = [json.loads(p.read_text()) for p in reports.glob("*")]
        entries = select_cli.snapshot_entries(payloads, "frozen")
        return select_cli.run_curve(entries, run, n_epochs)

    def test_a_missing_epoch_is_refused(self, tmp_path):
        with pytest.raises(
            select_cli.SelectionError, match=r"epoch\(s\) \[2\]"
        ):
            self._choose(tmp_path, {1: 50, 3: 60})

    def test_other_settings_are_not_read(self, tmp_path):
        with pytest.raises(select_cli.SelectionError, match="no score"):
            self._choose(tmp_path, {1: 50, 2: 55, 3: 60}, setting="deployed")

    def test_differing_populations_are_refused(self, tmp_path):
        run = cli.run_name(12)
        reports = tmp_path / "val"
        _write_val_reports(reports, run, {1: 50, 2: 55})
        _write_val_reports(reports / "x", run, {3: 60}, n_images=2)
        payloads = [
            json.loads(p.read_text()) for p in reports.rglob("*.json")
        ]

        with pytest.raises(select_cli.SelectionError, match="differing"):
            select_cli.run_curve(
                select_cli.snapshot_entries(payloads, "frozen"), run, 3
            )

    def test_scores_beyond_the_epochs_trained_are_refused(self, tmp_path):
        save_root = tmp_path / "ck"
        run = cli.run_name(12)
        _write_val_reports(
            save_root / "eval" / "val" / run, run, {1: 50, 2: 70},
        )
        # A rehearsal of the 12-epoch arm trains one epoch.
        argv = [
            "--epochs", "12", "--rehearsal", "--save-root", str(save_root),
            "--no-mlflow",
        ]

        with pytest.raises(select_cli.SelectionError, match="beyond"):
            select_cli.run(select_cli.parse_args(argv))

    def test_the_choice_is_recorded_once(self, tmp_path):
        save_root = tmp_path / "ck"
        run = cli.run_name(50)
        _write_val_reports(
            save_root / "eval" / "val" / run, run, {1: 50, 2: 70},
        )
        argv = [
            "--epochs", "50", "--rehearsal", "--save-root", str(save_root),
            "--no-mlflow",
        ]

        assert select_cli.main(argv) == select_cli.EXIT_OK
        record = json.loads(
            (save_root / "eval" / "selected_e50.json").read_text()
        )
        assert record["selected_epoch"] == 2
        assert record["checkpoint"].endswith(f"{run}/epoch-2.pt")
        assert set(record["curve"]) == {"1", "2"}
        assert select_cli.main(argv) == select_cli.EXIT_FAILED

    def test_a_rate_factor_is_chosen_into_its_own_record(self, tmp_path):
        save_root = tmp_path / "ck"
        run = cli.run_name(12, 0.5)
        _write_val_reports(save_root / "eval" / "val" / run, run, {1: 50})
        argv = [
            "--epochs", "12", "--lr-scale", "0.5", "--rehearsal",
            "--save-root", str(save_root), "--no-mlflow",
        ]

        assert select_cli.main(argv) == select_cli.EXIT_OK
        record = json.loads(
            (save_root / "eval" / "selected_e12_lrx0.5.json").read_text()
        )
        assert record["run"] == run
        assert record["lr_scale"] == 0.5
        assert not (save_root / "eval" / "selected_e12.json").exists()


def _choice(arm, epoch, curve_tp):
    run = cli.run_name(arm)
    return {
        "run": run, "selected_epoch": epoch, "val_f1": 0.9,
        "checkpoint": f"ck/checkpoints/{run}/epoch-{epoch}.pt",
        "curve": {
            str(k): {"f1": 2 * tp / (2 * tp + 20)}
            for k, tp in curve_tp.items()
        },
    }


class TestComparison:
    def _inputs(self, long_tp=80, short_tp=70, extra_snapshot=False):
        choices = {
            12: _choice(12, 11, {k: 60 + k for k in range(1, 13)}),
            50: _choice(50, 30, {k: 60 + min(k, 30) for k in range(1, 51)}),
        }
        payload = {
            "a": _entry(choices[12]["checkpoint"], short_tp),
            "b": _entry(choices[50]["checkpoint"], long_tp),
        }
        if extra_snapshot:
            payload["c"] = _entry(
                f"ck/checkpoints/{cli.run_name(50)}/epoch-50.pt", long_tp
            )
        return choices, payload

    def test_the_difference_is_from_summed_counts_of_the_chosen(self):
        choices, payload = self._inputs(long_tp=80, short_tp=70)

        result = compare_cli.compare(choices, payload, "frozen")

        assert result["delta"]["f1"] == pytest.approx(
            160 / 180 - 140 / 160
        )
        assert result["arms"]["50"]["selected_epoch"] == 30
        assert result["delta_by_material"]["AS"]["f1"] == pytest.approx(
            160 / 172 - 140 / 152
        )
        assert "AS16" in result["delta_by_formulation"]

    def test_the_verdict_reads_the_seed_noise_band(self):
        assert compare_cli.verdict(0.0060) == "long_arm_higher"
        assert compare_cli.verdict(-0.0060) == "long_arm_lower"
        assert compare_cli.verdict(0.0050) == "within_seed_noise"

    def test_another_snapshot_of_a_chosen_run_is_refused(self):
        choices, payload = self._inputs(extra_snapshot=True)

        with pytest.raises(compare_cli.ComparisonError, match="besides"):
            compare_cli.compare(choices, payload, "frozen")

    def test_the_curve_reading_splits_at_the_short_budget(self):
        choices, _ = self._inputs()

        reading = compare_cli.curve_reading(choices[50])

        assert reading["up_to_short"]["epoch"] == 12
        assert reading["after_short"]["epoch"] == 30
        assert reading["gain_after_short"] > 0

    def test_the_report_is_written(self, tmp_path):
        choices, payload = self._inputs()
        eval_dir = tmp_path / "ck" / "eval"
        eval_dir.mkdir(parents=True)
        for arm, choice in choices.items():
            (eval_dir / f"selected_e{arm}.json").write_text(
                json.dumps(choice)
            )
        (eval_dir / "test.json").write_text(json.dumps(payload))

        code = compare_cli.main([
            "--save-root", str(tmp_path / "ck"), "--rehearsal",
            "--no-mlflow",
        ])

        assert code == compare_cli.EXIT_OK
        stem = eval_dir / "epoch_budget_fold4_frozen"
        for suffix in (".json", ".md", ".png"):
            assert stem.with_suffix(suffix).exists()


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


class TestLateWindow:
    def test_is_the_last_three_epochs_of_the_run(self):
        assert late_epochs_of(12) == (10, 11, 12)
        assert late_epochs_of(50) == (48, 49, 50)
        assert late_epochs_of(2) == (1, 2)

    def _run(self, params):
        experiment = tracking.ensure_experiment("test_late")
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.log_params(params)
        for epoch in range(1, 51):
            log_metrics_at(
                run.info.run_id, {"val/pp/f1": epoch / 100}, step=epoch
            )
        return run.info.run_id

    def test_a_fifty_epoch_run_is_read_over_its_own_tail(self, store):
        run_id = self._run({"n_epochs": 50})

        summary = summarize_curve(run_id, "val", "pp")

        assert summary["val/pp/summary/f1_late_mean"] == pytest.approx(0.49)

    def test_a_run_without_its_length_falls_back(self, store):
        run_id = self._run({"seed": 1})

        summary = summarize_curve(run_id, "val", "pp")

        assert summary["val/pp/summary/f1_late_mean"] == pytest.approx(0.11)
