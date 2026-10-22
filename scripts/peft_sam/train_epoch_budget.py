#!/usr/bin/env python3
"""
Train one arm of the epoch-budget study: the adopted recipe, 12 or 50 epochs.

**The question.** What happens when the number of epochs grows well
beyond the twelve the recipe was fixed at. Both arms train the adopted
recipe on the same images - fold four of the formulation folds, with
the formulations of the frozen VALIDATION moved out of training - and
differ in the number of epochs and in nothing else. The rate is
annealed over each run's own length, since a run of fifty epochs
annealed over twelve would be a different recipe, not a longer one.

**How each arm's model is taken.** As anyone with that many epochs
would take it: the epoch that does best on validation. A snapshot is
therefore kept after every epoch, all of them are scored on
validation, and the rule - highest instance F1 under the frozen
post-processing, the earlier epoch on a tie - is applied the same way
to both arms. The chosen snapshot of each is then scored once on the
formulations fold four holds out. Choosing among fifty candidates
favours the longer arm on validation, which is why the comparison is
read on the held-out formulations, where the choice is already made.

**Why a twelve-epoch arm rather than the fold's existing model.** Fold
four's model in the cross-validation trained on the validation
formulations as well. Against it, a difference would mix the number of
epochs with the amount of data.

The trainer's monitoring images are validation images, which are not
trained on, so the loss it reports on them is a genuine held-out loss.
It is recorded and decides nothing; the epoch is chosen by instance F1.

**A second question on the same path: half the learning rates.** A
twelve-epoch arm with both rates halved, the correction's and the
decoder's together, asks whether the adopted rates move the model too
fast. It differs from the twelve-epoch arm in the rates and in nothing
else - same images, seed, policy and schedule shape, annealed over the
same twelve epochs - and its epoch is chosen by the same rule. It is
compared with that arm alone, by ``compare_lr_scale.py``; it is not a
third budget.

Examples
--------
Rehearse the whole path in minutes:
    $ python scripts/peft_sam/train_epoch_budget.py --epochs 12 --rehearsal

Train an arm:
    $ python scripts/peft_sam/train_epoch_budget.py --epochs 50

Train the half-rate arm:
    $ python scripts/peft_sam/train_epoch_budget.py --epochs 12 \\
        --lr-scale 0.5

Score every snapshot of it on validation, one at a time so that an
interruption costs one snapshot (``<run>`` as ``run_name`` gives it):
    $ python scripts/peft_sam/evaluate_checkpoint.py \\
        --split /home/dwalicki/dane/splits/epoch_budget_v1/fold4_valv1.csv \\
        --checkpoint checkpoints/epoch_budget/checkpoints/<run>/epoch-1.pt \\
        --out checkpoints/epoch_budget/eval/val/<run>/epoch-1.json
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from materials_vision.data.split_io import SplitLoadError, load_split
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import REHEARSAL_EXPERIMENT, RunTracking
from materials_vision.training import DECODER_LEARNING_RATE, LORA_LEARNING_RATE
from scripts.peft_sam.create_epoch_budget_split import (BUDGET_ID,
                                                        DEFAULT_OUT_DIR, FOLD,
                                                        SPLIT_NAME)
from scripts.peft_sam.run_e8_full import RUN_SEED
from scripts.peft_sam.train_cv_fold import write_provenance
from scripts.peft_sam.train_final_model import (POLICY, recipe_record,
                                                train_recipe)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

EPOCH_BUDGET_EXPERIMENT = "peft_sam_epoch_budget"

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = DEFAULT_OUT_DIR / f"{SPLIT_NAME}.csv"

DEFAULT_SAVE_ROOT = Path("checkpoints/epoch_budget")

REHEARSAL_SAVE_ROOT = Path("checkpoints/rehearsal_epoch_budget")

# The two budgets compared: the one the recipe was fixed at and one
# far beyond it. Nothing else is accepted, so that no third arm can be
# added after the first two have been seen.
ARMS = (12, 50)

# Factors applied to both of the recipe's learning rates at once. One
# is the recipe itself and leaves every run name as it was before the
# factor existed, so the arms already trained keep their directories.
LR_SCALES = (1.0, 0.5)

TRAIN_SUBSET = "train"

MONITOR_SUBSET = "val"

# Every epoch is a candidate.
SAVE_EVERY_KTH_EPOCH = 1

REHEARSAL_N_TRAIN = 8

# Each arm rehearses on a run long enough to leave more than one
# snapshot to choose from in the longer arm.
REHEARSAL_EPOCHS = {12: 1, 50: 2}

SELECTION_RULE = (
    "highest instance F1 on validation under the frozen post-processing; "
    "the earlier epoch on a tie"
)


def lr_suffix(lr_scale: float) -> str:
    """What a rate factor adds to the names of an arm's files.

    Parameters
    ----------
    lr_scale : float
        One of ``LR_SCALES``.

    Returns
    -------
    str
        Empty for the recipe's own rates.
    """
    return "" if lr_scale == 1.0 else f"_lrx{lr_scale:g}"


def learning_rates(lr_scale: float) -> tuple[float, float]:
    """The correction's and the decoder's rate under a factor.

    Parameters
    ----------
    lr_scale : float

    Returns
    -------
    tuple of float
    """
    return LORA_LEARNING_RATE * lr_scale, DECODER_LEARNING_RATE * lr_scale


def run_name(arm: int, lr_scale: float = 1.0) -> str:
    """What one arm's checkpoints are filed under.

    Parameters
    ----------
    arm : int
        The budget in epochs, one of ``ARMS``.
    lr_scale : float, optional
        The factor on both rates, one of ``LR_SCALES``.

    Returns
    -------
    str
    """
    return (
        f"{POLICY}_cv5_fold{FOLD}_valv1_e{arm}{lr_suffix(lr_scale)}"
        f"_seed{RUN_SEED}"
    )


def snapshot_paths(
    save_root: Path, arm: int, epochs: int, lr_scale: float = 1.0
) -> list[Path]:
    """Every snapshot the run is expected to leave, in epoch order.

    Parameters
    ----------
    save_root : Path
    arm : int
    epochs : int
        Epochs actually trained; fewer than ``arm`` in a rehearsal.
    lr_scale : float, optional

    Returns
    -------
    list of Path
    """
    directory = save_root / "checkpoints" / run_name(arm, lr_scale)
    return [directory / f"epoch-{k}.pt" for k in range(1, epochs + 1)]


def provenance_record(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that produced the arm's run, for the file beside it.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
    """
    lora_rate, decoder_rate = learning_rates(args.lr_scale)
    return {
        "provenance": run_provenance(),
        "run": run_name(args.epochs, args.lr_scale),
        "budget_id": BUDGET_ID,
        "arm_epochs": args.epochs,
        "lr_scale": args.lr_scale,
        "epochs": args.n_epochs,
        "scheduler_horizon_epochs": args.n_epochs,
        "save_every_kth_epoch": SAVE_EVERY_KTH_EPOCH,
        "selection_rule": SELECTION_RULE,
        "train_subset": TRAIN_SUBSET,
        "monitor_subset": MONITOR_SUBSET,
        "n_train": args.n_train,
        "rehearsal": args.rehearsal,
        "split": str(args.split),
        "manifest": str(args.manifest),
        **recipe_record(),
        # After the recipe's record, which states the unscaled rates.
        "lora_learning_rate": lora_rate,
        "decoder_learning_rate": decoder_rate,
    }


def build_tracking(
    args: argparse.Namespace, provenance_path: Path, metadata_path: Path
) -> RunTracking:
    """Describe how the run is recorded in MLflow.

    Parameters
    ----------
    args : argparse.Namespace
    provenance_path, metadata_path : Path

    Returns
    -------
    RunTracking
    """
    families = ",".join(recipe_record()["families"])
    return RunTracking(
        experiment=REHEARSAL_EXPERIMENT if args.rehearsal
        else EPOCH_BUDGET_EXPERIMENT,
        tags={
            "study": "epoch_budget",
            "part": "epoch_budget",
            "budget_id": BUDGET_ID,
            "cv_id": "cv5_v1",
            "fold": str(FOLD),
            "epochs": str(args.epochs),
            "lr_scale": f"{args.lr_scale:g}",
            "val": "val_v1",
            "policy": POLICY.upper(),
            "seed": str(RUN_SEED),
            "train_subset": TRAIN_SUBSET,
            "source": "live",
        },
        params={
            "seed": RUN_SEED,
            "policy": POLICY.upper(),
            "augmentation_families": families,
            "budget_id": BUDGET_ID,
            "fold": FOLD,
            "arm_epochs": args.epochs,
            "lr_scale": args.lr_scale,
            "save_every_kth_epoch": SAVE_EVERY_KTH_EPOCH,
            "selection_rule": SELECTION_RULE,
        },
        artifacts=(provenance_path, metadata_path),
    )


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why the arm may not be trained as asked, or nothing.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    for path in (args.split, args.split_metadata):
        if not path.exists():
            return (
                f"{path} does not exist. The split is written by "
                f"scripts/peft_sam/create_epoch_budget_split.py, before "
                f"either arm is trained."
            )
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, required=True, choices=ARMS,
        help="The arm: its budget in epochs.",
    )
    parser.add_argument(
        "--lr-scale", type=float, default=1.0, choices=LR_SCALES,
        help="Factor on both learning rates; 1 is the recipe.",
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true",
        help=(
            f"{REHEARSAL_N_TRAIN} training images for "
            f"{REHEARSAL_EPOCHS} epoch(s) by arm, into "
            f"{REHEARSAL_SAVE_ROOT} and the {REHEARSAL_EXPERIMENT} "
            f"experiment."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if every snapshot already exists.",
    )
    args = parser.parse_args(argv)
    args.split_metadata = args.split.with_name(
        f"{args.split.stem}_metadata.json"
    )
    args.n_train = REHEARSAL_N_TRAIN if args.rehearsal else 0
    args.n_epochs = (
        REHEARSAL_EPOCHS[args.epochs] if args.rehearsal else args.epochs
    )
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    return args


def execute(args: argparse.Namespace) -> None:
    """Record the run and train it.

    Parameters
    ----------
    args : argparse.Namespace
    """
    record = provenance_record(args)
    provenance_path = write_provenance(record, args.save_root)
    lora_rate, decoder_rate = learning_rates(args.lr_scale)
    train_recipe(
        run_name(args.epochs, args.lr_scale),
        split=args.split,
        manifest=args.manifest,
        train_subset=TRAIN_SUBSET,
        monitor_subset=MONITOR_SUBSET,
        save_root=args.save_root,
        epochs=args.n_epochs,
        n_train=args.n_train,
        tracking=build_tracking(args, provenance_path, args.split_metadata),
        save_every_kth_epoch=SAVE_EVERY_KTH_EPOCH,
        lora_learning_rate=lora_rate,
        decoder_learning_rate=decoder_rate,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Train one arm.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    reason = refusal(args)
    if reason is not None:
        logger.error(reason)
        return EXIT_FAILED
    try:
        # Checked before the GPU is touched: a split drawn from another
        # manifest would train on images it does not describe.
        load_split(args.split, TRAIN_SUBSET, verify_manifest=args.manifest)
    except SplitLoadError as e:
        logger.error("%s", e)
        return EXIT_FAILED
    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    snapshots = snapshot_paths(
        args.save_root, args.epochs, args.n_epochs, args.lr_scale
    )
    if not args.force and all(path.exists() for path in snapshots):
        logger.info(
            "All %d snapshots of %s already exist; nothing to do.",
            len(snapshots), run_name(args.epochs, args.lr_scale),
        )
        return EXIT_OK
    execute(args)

    missing = [path for path in snapshots if not path.exists()]
    if missing:
        logger.error(
            "Training finished but %d snapshot(s) were not written, "
            "first %s; the epoch cannot be chosen among all of them.",
            len(missing), missing[0],
        )
        return EXIT_FAILED
    logger.info(
        "Run finished: %d snapshots in %s, to be scored on validation.",
        len(snapshots), snapshots[0].parent,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
