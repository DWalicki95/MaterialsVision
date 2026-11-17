"""Tests for training the adopted recipe on a cross-validation fold.

The trainer itself is replaced: what these pin is the path from a fold
file on disk to the loaders the trainer receives, through the real
split reader and the real sample source, and that the final model's
own path through the shared function did not change.
"""
import argparse

import pandas as pd
import pytest

import scripts.create_cv_folds as create_cli
import scripts.peft_sam.train_cv_fold as cli
import scripts.peft_sam.train_final_model as final
from materials_vision.tracking import REHEARSAL_EXPERIMENT, TRAINING_EXPERIMENT


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
def trainer_calls(monkeypatch):
    """Stand in for the trainer and record what it was handed."""
    calls = []
    monkeypatch.setattr(
        "scripts.peft_sam.train_final_model.train_run",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    return calls


def _image_ids(loader):
    return {record.image_id for record in loader.dataset.source.records}


def _fold_args(folds, tmp_path, *extra):
    folds_dir, manifest = folds
    return cli.parse_args([
        "--fold", "3", "--folds-dir", str(folds_dir),
        "--manifest", str(manifest), "--save-root", str(tmp_path / "ck"),
        *extra,
    ])


def test_a_fold_trains_on_its_training_rows_only(
    tmp_path, folds, trainer_calls
):
    args = _fold_args(folds, tmp_path)
    table = pd.read_csv(args.split)
    train_ids = set(table.loc[
        (table["split"] == "train") & table["used"], "image_id"
    ])
    held_ids = set(table.loc[table["split"] == "test", "image_id"])

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert name == "full_b_cv5_fold3_seed20260907"
    assert _image_ids(kwargs["train_loader"]) == train_ids
    assert not _image_ids(kwargs["train_loader"]) & held_ids
    assert _image_ids(kwargs["val_loader"]) == train_ids
    assert len(kwargs["val_loader"]) == final.MONITOR_IMAGES
    assert len(kwargs["train_loader"]) == len(train_ids)


def test_a_fold_runs_the_final_models_recipe(tmp_path, folds, trainer_calls):
    cli.execute(_fold_args(folds, tmp_path))

    [(_, kwargs)] = trainer_calls
    assert kwargs["n_epochs"] == final.N_EPOCHS
    assert kwargs["save_every_kth_epoch"] == final.N_EPOCHS
    assert kwargs["early_stopping"] is None
    assert kwargs["lora_learning_rate"] == final.LORA_LEARNING_RATE
    assert kwargs["decoder_learning_rate"] == final.DECODER_LEARNING_RATE
    assert kwargs["model_type"] == final.BASE_MODEL
    tracking = kwargs["tracking"]
    assert tracking.experiment == TRAINING_EXPERIMENT
    assert tracking.tags["fold"] == "3"
    assert tracking.tags["part"] == "cv"


def test_the_final_model_still_trains_on_train_and_val(
    tmp_path, frozen_table, frozen_split, trainer_calls
):
    split_csv, manifest = frozen_split
    args = argparse.Namespace(
        split=split_csv, manifest=manifest, save_root=tmp_path / "ck",
        epochs=final.N_EPOCHS, n_train=0,
    )

    final.execute(args)

    [(name, kwargs)] = trainer_calls
    subset = frozen_table["split"]
    assert name == final.run_name()
    assert _image_ids(kwargs["train_loader"]) == set(
        frozen_table.loc[subset.isin(["train", "val"]), "image_id"]
    )
    assert _image_ids(kwargs["val_loader"]) == set(
        frozen_table.loc[subset == "val", "image_id"]
    )


def test_the_final_model_is_recorded_with_its_provenance(
    tmp_path, frozen_split, trainer_calls
):
    split_csv, manifest = frozen_split
    args = argparse.Namespace(
        split=split_csv, manifest=manifest, save_root=tmp_path / "ck",
        epochs=final.N_EPOCHS, n_train=0,
    )

    final.execute(args)

    [(_, kwargs)] = trainer_calls
    tracking = kwargs["tracking"]
    assert tracking.experiment == TRAINING_EXPERIMENT
    assert tracking.tags["part"] == "final"
    [provenance] = tracking.artifacts
    assert provenance.exists()
    assert provenance.parent == tmp_path / "ck"


def test_a_final_model_rehearsal_is_kept_apart(
    tmp_path, frozen_split, trainer_calls
):
    split_csv, manifest = frozen_split
    args = argparse.Namespace(
        split=split_csv, manifest=manifest, save_root=tmp_path / "ck",
        epochs=1, n_train=2,
    )

    final.execute(args)

    [(_, kwargs)] = trainer_calls
    assert kwargs["tracking"].experiment == REHEARSAL_EXPERIMENT


def test_fold_one_is_the_final_model_and_is_refused(tmp_path, folds):
    args = cli.parse_args(["--fold", "1", "--folds-dir", str(folds[0])])

    assert "final model" in cli.refusal(args)


def test_an_unknown_fold_is_refused(folds):
    args = cli.parse_args(["--fold", "6", "--folds-dir", str(folds[0])])

    assert "no fold 6" in cli.refusal(args)


def test_a_fold_not_yet_written_is_refused(tmp_path):
    args = cli.parse_args(["--fold", "3", "--folds-dir", str(tmp_path)])

    assert "create_cv_folds" in cli.refusal(args)


def test_a_fold_from_another_manifest_is_refused_before_training(
    tmp_path, folds, trainer_calls
):
    other = tmp_path / "other_manifest.csv"
    other.write_text("image_id\n", encoding="utf-8")

    code = cli.main([
        "--fold", "3", "--folds-dir", str(folds[0]),
        "--manifest", str(other), "--save-root", str(tmp_path / "ck"),
    ])

    assert code == cli.EXIT_FAILED
    assert trainer_calls == []


def test_a_rehearsal_is_short_and_kept_apart(tmp_path, folds):
    args = cli.parse_args([
        "--fold", "3", "--folds-dir", str(folds[0]), "--rehearsal",
    ])

    assert args.epochs == cli.REHEARSAL_EPOCHS
    assert args.n_train == cli.REHEARSAL_N_TRAIN
    assert args.save_root == cli.REHEARSAL_SAVE_ROOT


def test_the_recipes_own_fold_keeps_its_original_name():
    assert cli.run_name(3) == "full_b_cv5_fold3_seed20260907"
    assert cli.run_name(
        3, final.BASE_MODEL, final.N_EPOCHS, final.DECODER_LEARNING_RATE
    ) == "full_b_cv5_fold3_seed20260907"


def test_every_setting_that_differs_enters_the_name():
    names = {
        cli.run_name(3, base_model, epochs, decoder_lr)
        for base_model in ("vit_l_lm", "vit_l_em_organelles", "vit_l")
        for epochs in (12, 24)
        for decoder_lr in (final.DECODER_LEARNING_RATE, 1e-4, 3e-4)
    }

    assert len(names) == 3 * 2 * 3


def test_another_base_model_reaches_the_trainer(
    tmp_path, folds, trainer_calls
):
    args = _fold_args(
        folds, tmp_path, "--base-model", "vit_l_em_organelles",
        "--epochs", "24",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert name == "full_b_vit_l_em_organelles_cv5_fold3_e24_seed20260907"
    assert kwargs["model_type"] == "vit_l_em_organelles"
    assert kwargs["n_epochs"] == 24
    assert kwargs["save_every_kth_epoch"] == 24
    assert kwargs["decoder_learning_rate"] == final.DECODER_LEARNING_RATE
    assert kwargs["tracking"].tags["base_model"] == "vit_l_em_organelles"
    assert kwargs["tracking"].tags["epochs"] == "24"


def test_the_provenance_states_the_base_model_and_decoder_rate(
    tmp_path, folds, trainer_calls
):
    import json

    args = _fold_args(
        folds, tmp_path, "--base-model", "vit_l", "--decoder-lr", "1e-4",
        "--epochs", "24",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert kwargs["decoder_learning_rate"] == 1e-4
    [provenance] = [
        path for path in kwargs["tracking"].artifacts
        if path.name.endswith("_provenance.json")
    ]
    record = json.loads(provenance.read_text())
    assert record["run"] == name
    assert record["base_model"] == "vit_l"
    assert record["decoder_learning_rate"] == 1e-4
    assert record["scored_snapshot"].endswith(f"{name}/epoch-24.pt")


def test_two_setups_of_one_fold_do_not_share_a_provenance_file(
    tmp_path, folds, trainer_calls
):
    cli.execute(_fold_args(folds, tmp_path))
    cli.execute(_fold_args(folds, tmp_path, "--epochs", "24"))

    first, second = (kwargs["tracking"].artifacts[0]
                     for _, kwargs in trainer_calls)
    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.parametrize("epochs", ["12", "24"])
def test_fold_one_of_an_existing_final_model_is_refused(folds, epochs):
    args = cli.parse_args([
        "--fold", "1", "--folds-dir", str(folds[0]), "--epochs", epochs,
    ])

    assert str(cli.FINAL_MODEL_SAVE_ROOTS[int(epochs)]) in cli.refusal(args)


@pytest.mark.parametrize("extra", [
    ["--base-model", "vit_l_em_organelles", "--epochs", "24"],
    ["--base-model", "vit_l", "--decoder-lr", "1e-4", "--epochs", "24"],
    ["--epochs", "18"],
])
def test_fold_one_without_a_final_model_is_trained(folds, extra):
    args = cli.parse_args(
        ["--fold", "1", "--folds-dir", str(folds[0]), *extra]
    )

    assert cli.refusal(args) is None


def test_a_base_model_without_a_decoder_needs_its_own_rate(folds):
    args = cli.parse_args([
        "--fold", "3", "--folds-dir", str(folds[0]), "--base-model", "vit_l",
    ])

    assert "--decoder-lr" in cli.refusal(args)


def test_a_rehearsal_keeps_the_base_model(folds):
    args = cli.parse_args([
        "--fold", "1", "--folds-dir", str(folds[0]), "--rehearsal",
        "--base-model", "vit_l_em_organelles", "--epochs", "24",
    ])

    assert args.epochs == cli.REHEARSAL_EPOCHS
    assert cli.fold_run_name(args) == (
        "full_b_vit_l_em_organelles_cv5_fold1_e1_seed20260907"
    )
    assert cli.refusal(args) is None
