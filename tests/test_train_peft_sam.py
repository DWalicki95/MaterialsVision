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
