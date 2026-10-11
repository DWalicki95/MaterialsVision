"""Tests for training and scoring the Cellpose recipe on a CV fold.

The trainer and the evaluation are replaced; what these pin is the path
from a fold file and a recipe record to what the trainer and the
evaluation receive, through the real fold writer, split reader and
sample source.
"""
import json

import pandas as pd
import pytest

import scripts.cellpose.run_cellpose_fold as cli
import scripts.create_cv_folds as create_cli
from materials_vision.experiments.cellpose.split_training import (
    N_EPOCHS, snapshot_path)

CHOSEN_EPOCH = 101


@pytest.fixture
def folds(tmp_path, frozen_split):
    """Folds written from the miniature split, as the real ones were."""
    split_csv, manifest = frozen_split
    out_dir = tmp_path / "cv5"
    assert create_cli.main([
        "--split", str(split_csv), "--out-dir", str(out_dir),
        "--seed", "11",
    ]) == create_cli.EXIT_OK
    return out_dir, manifest


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({
        "snapshot": {
            "arm": "lr1e-5", "learning_rate": 1e-5,
            "completed_epochs": CHOSEN_EPOCH,
        },
        "postprocessing": {
            "flow_threshold": 0.6, "cellprob_threshold": -1.0,
            "min_size_px": 15, "niter": 1000,
        },
    }))
    return path


@pytest.fixture
def calls(monkeypatch):
    """Stand in for the trainer and the evaluation, recording both.

    The trainer writes the scored snapshot, as a real run would, so the
    script goes on to score it.
    """
    recorded = {"train": [], "evaluate": []}

    def train(name, **kwargs):
        recorded["train"].append((name, kwargs))
        snapshot = snapshot_path(
            kwargs["save_root"], name, CHOSEN_EPOCH, kwargs["n_epochs"]
        )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"weights")

    def evaluate(argv):
        recorded["evaluate"].append(argv)
        return cli.EXIT_OK

    monkeypatch.setattr(
        "scripts.cellpose.run_cellpose_fold.train_cellpose", train
    )
    monkeypatch.setattr("scripts.cellpose.evaluate_cellpose.main", evaluate)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    return recorded


def _argv(folds, recipe, tmp_path, fold=3):
    folds_dir, manifest = folds
    return [
        "--fold", str(fold), "--folds-dir", str(folds_dir),
        "--manifest", str(manifest), "--recipe", str(recipe),
        "--save-root", str(tmp_path / "ck"),
        "--cache-dir", str(tmp_path / "cache"),
    ]


def _rows(split_csv, subset):
    table = pd.read_csv(split_csv)
    return set(table.loc[
        (table["split"] == subset) & table["used"], "image_id"
    ])


def _ids(source):
    return {record.image_id for record in source.records}


def test_a_fold_trains_its_rows_and_scores_its_held_out_once(
    tmp_path, folds, recipe, calls
):
    folds_dir, _ = folds

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_OK

    [(name, kwargs)] = calls["train"]
    assert name == "cpsam_cv5_fold3_seed20260907"
    split = folds_dir / "fold3.csv"
    assert _ids(kwargs["train_source"]) == _rows(split, "train")
    assert not _ids(kwargs["train_source"]) & _rows(split, "test")
    assert kwargs["monitor_source"] is None
    assert kwargs["learning_rate"] == 1e-5
    assert kwargs["n_epochs"] == N_EPOCHS
    assert kwargs["train_positions"] is None

    [argv] = calls["evaluate"]
    snapshot = snapshot_path(tmp_path / "ck", name, CHOSEN_EPOCH)
    expected = {
        "--subset": "test", "--checkpoint": str(snapshot),
        "--flow-threshold": "0.6", "--cellprob-threshold": "-1.0",
        "--niter": "1000", "--split": str(split),
    }
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--unlock-test" in argv and "--also-area-filter" in argv
    assert "--n-images" not in argv


def test_fold_one_is_the_final_model_on_train_and_val(
    tmp_path, folds, recipe, calls, frozen_split
):
    split_csv, _ = frozen_split

    assert cli.main(_argv(folds, recipe, tmp_path, fold=1)) == cli.EXIT_OK

    [(_, kwargs)] = calls["train"]
    assert _ids(kwargs["train_source"]) == (
        _rows(split_csv, "train") | _rows(split_csv, "val")
    )


@pytest.mark.parametrize("missing", ["snapshot", "postprocessing"])
def test_no_fold_runs_before_the_recipe_is_complete(
    tmp_path, folds, recipe, calls, missing
):
    content = json.loads(recipe.read_text())
    del content[missing]
    recipe.write_text(json.dumps(content))

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_FAILED
    assert calls == {"train": [], "evaluate": []}


def test_a_first_round_recipe_without_steps_is_refused(
    tmp_path, folds, recipe, calls
):
    content = json.loads(recipe.read_text())
    del content["postprocessing"]["niter"]
    recipe.write_text(json.dumps(content))

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_FAILED
    assert calls == {"train": [], "evaluate": []}


def test_the_folds_read_the_second_round_by_default():
    assert cli.DEFAULT_RECIPE.name == "recipe_v2.json"


def test_a_rate_that_is_not_the_arms_is_refused(
    tmp_path, folds, recipe, calls
):
    content = json.loads(recipe.read_text())
    content["snapshot"]["learning_rate"] = 1e-3
    recipe.write_text(json.dumps(content))

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_FAILED
    assert calls == {"train": [], "evaluate": []}


def test_a_scored_fold_is_not_scored_again(tmp_path, folds, recipe, calls):
    report = tmp_path / "ck" / "eval" / "fold3.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_OK
    assert calls == {"train": [], "evaluate": []}


def test_a_trained_fold_is_scored_without_training_again(
    tmp_path, folds, recipe, calls
):
    snapshot = snapshot_path(
        tmp_path / "ck", cli.run_name(3), CHOSEN_EPOCH
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"weights")

    assert cli.main(_argv(folds, recipe, tmp_path)) == cli.EXIT_OK
    assert calls["train"] == []
    assert len(calls["evaluate"]) == 1


def test_a_rehearsal_scores_its_last_snapshot_on_a_few_images(
    tmp_path, folds, recipe, calls
):
    args = cli.parse_args(_argv(folds, recipe, tmp_path) + ["--rehearsal"])
    content = cli.read_recipe(recipe)

    argv = cli.evaluation_argv(args, content)

    assert cli.scored_epoch(args, content) == cli.REHEARSAL_EPOCHS
    assert argv[argv.index("--n-images") + 1] == str(
        cli.REHEARSAL_N_IMAGES
    )
