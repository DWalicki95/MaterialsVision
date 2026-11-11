"""Tests for ``scripts/peft_sam/train_peft_sam.py``.

The trainer is replaced with a recorder, so the tests check which
settings reach it from the command line - defaults, learning rates,
trainable part, interactive loss, augmentation, data subsets, run
naming and input validation - on a small split that builds real data loaders.
"""
import json

import pandas as pd
import pytest

import scripts.peft_sam.train_peft_sam as cli
from materials_vision.tracking import REHEARSAL_EXPERIMENT
from materials_vision.training import (DECODER_LEARNING_RATE,
                                       ENCODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, LORA_RANK)
from scripts.peft_sam.run_e8_full import N_EPOCHS, RUN_SEED, build_full_policy


@pytest.fixture
def trainer_calls(monkeypatch):
    """Stand in for the trainer and record what it was handed."""
    calls = []
    monkeypatch.setattr(
        "scripts.peft_sam.train_peft_sam.train_run",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    return calls


def _args(frozen_split, tmp_path, *extra):
    split_csv, manifest = frozen_split
    return cli.parse_args([
        "--split", str(split_csv), "--manifest", str(manifest),
        "--save-root", str(tmp_path / "ck"), *extra,
    ])


def _image_ids(loader):
    return {record.image_id for record in loader.dataset.source.records}


def _used_ids(split_csv, subset):
    table = pd.read_csv(split_csv)
    rows = table[(table["split"] == subset) & table["used"].astype(bool)]
    return set(rows["image_id"])


def test_without_options_the_adopted_recipe_reaches_the_trainer(
    frozen_split, tmp_path, trainer_calls
):
    cli.execute(_args(frozen_split, tmp_path))

    [(name, kwargs)] = trainer_calls
    assert name == f"full_b_both_e{N_EPOCHS}_lora0.0003_dec1e-05_train" \
        f"_seed{RUN_SEED}"
    assert kwargs["n_epochs"] == N_EPOCHS
    assert kwargs["trainable"] == "both"
    assert kwargs["interactive_loss"] is True
    assert kwargs["lora_learning_rate"] == LORA_LEARNING_RATE
    assert kwargs["decoder_learning_rate"] == DECODER_LEARNING_RATE
    assert kwargs["lora_rank"] == LORA_RANK
    assert kwargs["scheduler_horizon_epochs"] is None
    assert kwargs["early_stopping"] is None
    assert kwargs["save_every_kth_epoch"] == 1
    assert kwargs["tracking"].experiment == cli.DEFAULT_EXPERIMENT


def test_it_trains_on_train_and_monitors_on_held_out_validation(
    frozen_split, tmp_path, trainer_calls
):
    split_csv, _ = frozen_split

    cli.execute(_args(frozen_split, tmp_path))

    [(_, kwargs)] = trainer_calls
    trained = _image_ids(kwargs["train_loader"])
    monitored = _image_ids(kwargs["val_loader"])
    assert trained == _used_ids(split_csv, "train")
    assert monitored == _used_ids(split_csv, "val")
    assert not trained & monitored


@pytest.mark.parametrize("trainable, fragment, absent", [
    ("lora", "_lora0.001_", "_dec"),
    ("instance_decoder", "_dec0.001_", "_lora0"),
])
def test_one_half_trains_under_a_name_without_the_frozen_rate(
    frozen_split, tmp_path, trainer_calls, trainable, fragment, absent
):
    args = _args(
        frozen_split, tmp_path, "--trainable", trainable, "--epochs", "20",
        "--lora-lr", "1e-3", "--decoder-lr", "1e-3",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert name.startswith(f"full_b_{trainable}_e20_")
    assert fragment in name
    assert absent not in name
    assert kwargs["trainable"] == trainable
    assert kwargs["n_epochs"] == 20
    assert kwargs["tracking"].tags["trainable"] == trainable


def test_the_provenance_records_no_rate_for_a_frozen_half(
    frozen_split, tmp_path, trainer_calls
):
    args = _args(frozen_split, tmp_path, "--trainable", "lora")

    cli.execute(args)

    record = json.loads(
        (args.save_root / f"{cli.run_name(args)}_provenance.json")
        .read_text()
    )
    assert record["trainable"] == "lora"
    assert record["lora_learning_rate"] == LORA_LEARNING_RATE
    assert record["decoder_learning_rate"] is None
    assert record["families"] == list(build_full_policy("full_b").families)


def test_a_rate_given_for_a_frozen_half_is_reported_as_ignored(
    frozen_split, tmp_path
):
    args = _args(
        frozen_split, tmp_path, "--trainable", "instance_decoder",
        "--lora-lr", "1e-3",
    )

    assert cli.ignored_rates(args) == ["--lora-lr"]


def test_the_adopted_rates_are_never_reported_as_ignored(
    frozen_split, tmp_path
):
    args = _args(
        frozen_split, tmp_path, "--trainable", "instance_decoder"
    )

    assert cli.ignored_rates(args) == []


def test_no_policy_trains_unaugmented(frozen_split, tmp_path, trainer_calls):
    cli.execute(_args(frozen_split, tmp_path, "--policy", "none"))

    [(name, kwargs)] = trainer_calls
    assert name.startswith("none_both_")
    assert kwargs["train_loader"].dataset._transform is None
    assert kwargs["tracking"].params["augmentation_families"] == ""


def test_the_adopted_policy_reaches_the_training_loader_only(
    frozen_split, tmp_path, trainer_calls
):
    cli.execute(_args(frozen_split, tmp_path))

    [(_, kwargs)] = trainer_calls
    policy = kwargs["train_loader"].dataset._transform
    assert list(policy.families) == list(
        build_full_policy("full_b").families
    )
    assert kwargs["val_loader"].dataset._transform is None


def test_a_rehearsal_goes_to_the_rehearsal_experiment(
    frozen_split, tmp_path, trainer_calls
):
    cli.execute(_args(
        frozen_split, tmp_path, "--n-train", "2", "--n-monitor", "1",
    ))

    [(name, kwargs)] = trainer_calls
    assert "_n2_" in name
    assert len(kwargs["train_loader"]) == 2
    assert len(kwargs["val_loader"]) == 1
    assert kwargs["tracking"].experiment == REHEARSAL_EXPERIMENT


def test_settings_that_differ_never_share_a_directory(
    frozen_split, tmp_path
):
    variants = [
        (), ("--epochs", "20"), ("--trainable", "lora"),
        ("--interactive-loss", "off"),
        ("--lora-lr", "1e-3"), ("--decoder-lr", "3e-5"),
        ("--scheduler-horizon", "24"), ("--lora-rank", "4"),
        ("--policy", "none"), ("--seed", "1"),
        ("--train-subset", "train+val"), ("--n-train", "4"),
        ("--peft-layers", "12-23"),
        ("--update-matrices", "q", "k", "v", "mlp"),
        ("--peft-layers", "12-23", "--update-matrices", "q", "k", "v",
         "mlp"),
        ("--peft-method", "late_ft", "--peft-layers", "12-23"),
        ("--peft-method", "late_ft", "--peft-layers", "18-23"),
        ("--peft-method", "late_ft", "--peft-layers", "12-23",
         "--encoder-lr", "3e-5"),
        ("--peft-method", "full_ft"),
        ("--base-model", "vit_l", "--decoder-lr", "1e-4"),
    ]
    names = {
        cli.run_name(_args(frozen_split, tmp_path, *extra))
        for extra in variants
    }

    assert len(names) == len(variants)


def test_snapshots_expected_follow_the_interval(frozen_split, tmp_path):
    args = _args(
        frozen_split, tmp_path, "--epochs", "10", "--save-every", "4",
    )

    assert [p.name for p in cli.snapshot_paths(args)] == [
        "epoch-4.pt", "epoch-8.pt",
    ]


@pytest.mark.parametrize("extra, fragment", [
    (("--epochs", "0"), "at least one epoch"),
    (("--epochs", "12", "--scheduler-horizon", "6"), "rise again"),
    (("--epochs", "4", "--save-every", "5"), "--save-every"),
    (("--lora-lr", "0"), "--lora-lr"),
    (("--trainable", "lora", "--decoder-lr", "-1"), None),
    (("--trainable", "instance_decoder", "--interactive-loss", "on"),
     "--interactive-loss on"),
    (("--trainable", "instance_decoder", "--interactive-loss", "off"),
     None),
    (("--lora-rank", "0"), "--lora-rank"),
    (("--n-train", "-1"), "negative"),
    (("--monitor-subset", "test"), "TEST"),
    (("--train-subset", "train+test"), "TEST"),
    (("--peft-method", "late_ft"), "needs the encoder blocks"),
    (("--peft-method", "full_ft", "--peft-layers", "1"), "whole encoder"),
    (("--peft-method", "full_ft", "--lora-rank", "32"), "no effect"),
    (("--peft-method", "late_ft", "--peft-layers", "1",
      "--update-matrices", "q"), "update_matrices"),
    (("--trainable", "instance_decoder", "--peft-method", "full_ft"),
     "encoder is frozen"),
    (("--peft-method", "full_ft", "--encoder-lr", "0"), "--encoder-lr"),
    (("--base-model", "vit_l", "--trainable", "lora"), "no instance decoder"),
    (("--base-model", "vit_l"), None),
    (("--peft-method", "late_ft", "--peft-layers", "12-23"), None),
])
def test_impossible_settings_are_refused(
    frozen_split, tmp_path, extra, fragment
):
    reason = cli.refusal(_args(frozen_split, tmp_path, *extra))

    if fragment is None:
        assert reason is None
    else:
        assert fragment in reason


def test_a_missing_split_is_refused(frozen_split, tmp_path):
    args = _args(frozen_split, tmp_path)
    args.split = tmp_path / "none.csv"

    assert "does not exist" in cli.refusal(args)


def test_an_unknown_choice_of_what_trains_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--trainable", "encoder"])


@pytest.mark.parametrize("trainable", ["both", "lora"])
def test_the_interactive_loss_can_be_turned_off(
    frozen_split, tmp_path, trainer_calls, trainable
):
    args = _args(
        frozen_split, tmp_path, "--trainable", trainable,
        "--interactive-loss", "off",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert name.startswith(f"full_b_{trainable}_nointeractive_e")
    assert kwargs["interactive_loss"] is False
    assert kwargs["tracking"].tags["interactive_loss"] == "off"
    record = json.loads(
        (args.save_root / f"{name}_provenance.json").read_text()
    )
    assert record["interactive_loss"] is False


def test_an_instance_decoder_run_has_no_interactive_loss_to_name(
    frozen_split, tmp_path, trainer_calls
):
    """It never had one, so its name stays what it always was."""
    cli.execute(_args(
        frozen_split, tmp_path, "--trainable", "instance_decoder",
    ))

    [(name, kwargs)] = trainer_calls
    assert "nointeractive" not in name
    assert kwargs["interactive_loss"] is False
    assert kwargs["tracking"].tags["interactive_loss"] == "off"


def test_turning_the_interactive_loss_on_explicitly_changes_nothing(
    frozen_split, tmp_path
):
    default = _args(frozen_split, tmp_path)
    explicit = _args(frozen_split, tmp_path, "--interactive-loss", "on")

    assert cli.run_name(explicit) == cli.run_name(default)


def test_late_lora_on_every_projection_reaches_the_trainer(
    frozen_split, tmp_path, trainer_calls
):
    args = _args(
        frozen_split, tmp_path, "--peft-layers", "12-23",
        "--update-matrices", "mlp", "q", "k", "v",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert "_L12-23_qkvmlp_lora0.0003_" in name
    assert kwargs["peft_method"] == "lora"
    assert kwargs["peft_layers"] == list(range(12, 24))
    assert kwargs["update_matrices"] == ["q", "k", "v", "mlp"]
    record = json.loads(
        (args.save_root / f"{name}_provenance.json").read_text()
    )
    assert record["lora_update_matrices"] == ["q", "k", "v", "mlp"]
    assert record["encoder_learning_rate"] is None


def test_late_fine_tuning_trains_under_the_encoder_rate(
    frozen_split, tmp_path, trainer_calls
):
    args = _args(
        frozen_split, tmp_path, "--peft-method", "late_ft",
        "--peft-layers", "12-23",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert "_lateft_L12-23_" in name
    assert "_lora0" not in name
    assert f"_enc{ENCODER_LEARNING_RATE:g}_" in name
    assert kwargs["peft_method"] == "late_ft"
    assert kwargs["encoder_learning_rate"] == ENCODER_LEARNING_RATE
    assert kwargs["tracking"].tags["peft_method"] == "late_ft"
    record = json.loads(
        (args.save_root / f"{name}_provenance.json").read_text()
    )
    assert record["lora_learning_rate"] is None
    assert record["lora_rank"] is None
    assert record["encoder_learning_rate"] == ENCODER_LEARNING_RATE


def test_the_base_model_reaches_the_trainer_and_the_name(
    frozen_split, tmp_path, trainer_calls
):
    args = _args(
        frozen_split, tmp_path, "--base-model", "vit_l",
        "--decoder-lr", "1e-4", "--epochs", "6",
    )

    cli.execute(args)

    [(name, kwargs)] = trainer_calls
    assert name.startswith("full_b_vit_l_both_e6_")
    assert kwargs["model_type"] == "vit_l"
    assert kwargs["tracking"].tags["base_model"] == "vit_l"


def test_the_default_name_does_not_change_with_the_new_options(
    frozen_split, tmp_path
):
    """Runs before the adaptation options existed keep their names."""
    explicit = _args(
        frozen_split, tmp_path, "--peft-method", "lora",
        "--update-matrices", "v", "q", "--base-model", "vit_l_lm",
    )

    assert cli.run_name(explicit) == cli.run_name(
        _args(frozen_split, tmp_path)
    )


@pytest.mark.parametrize("extra, ignored", [
    (("--peft-method", "late_ft", "--peft-layers", "1",
      "--lora-lr", "1e-3"), ["--lora-lr"]),
    (("--encoder-lr", "1e-5"), ["--encoder-lr"]),
    (("--peft-method", "full_ft", "--encoder-lr", "1e-5"), []),
])
def test_a_rate_for_something_that_does_not_train_is_reported(
    frozen_split, tmp_path, extra, ignored
):
    assert cli.ignored_rates(_args(frozen_split, tmp_path, *extra)) == ignored


def test_a_decoder_from_scratch_at_the_adopted_rate_is_warned_about(
    frozen_split, tmp_path
):
    assert cli.untrained_decoder_warning(
        _args(frozen_split, tmp_path, "--base-model", "vit_l")
    ) is not None
    assert cli.untrained_decoder_warning(_args(
        frozen_split, tmp_path, "--base-model", "vit_l",
        "--decoder-lr", "1e-4",
    )) is None
    assert cli.untrained_decoder_warning(
        _args(frozen_split, tmp_path)
    ) is None


@pytest.mark.parametrize("text, layers", [
    ("12-23", list(range(12, 24))), ("12,14,16", [12, 14, 16]),
    ("0-1,5", [0, 1, 5]), ("3,3", [3]),
])
def test_blocks_are_read_from_ranges_and_lists(text, layers):
    assert cli.parse_layers(text) == layers


@pytest.mark.parametrize("text", ["23-12", "a", "", "1-"])
def test_an_unreadable_choice_of_blocks_is_rejected(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_layers(text)
