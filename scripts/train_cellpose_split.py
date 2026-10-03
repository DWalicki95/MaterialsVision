#!/usr/bin/env python3
"""
Train one arm of the Cellpose-SAM selection on TRAIN of the frozen split.

**What is being selected.** Two arms that differ in the learning rate
alone - the rate an earlier grid search found best, 1e-3, and the
library's own default for fine-tuning, 1e-5 - each trained for the full
budget of 160 epochs with a snapshot every ten. The arm and the epoch
are then chosen together, as the snapshot with the highest pooled
instance F1 on VALIDATION under Cellpose's default post-processing.
Everything else about the recipe is fixed in
``materials_vision.experiments.cellpose.split_training``.

**Why the earlier grid search is not simply adopted.** It split images
at random, so every formulation of AS - including those now held out
as TEST - sat on both sides of its split, and its configuration was
picked on its own test images. Its best learning rate is kept as a
candidate here, but the choice between candidates is made on
VALIDATION, which no formulation of TEST belongs to.

**VALIDATION is also handed to the trainer**, as the images Cellpose
computes a test loss on at epoch five and every tenth epoch. That loss
is of augmented crops and decides nothing; the snapshot is chosen by
instance F1 afterwards, with ``scripts/evaluate_cellpose.py``.

Examples
--------
Rehearse the whole path in minutes, into a throwaway directory:
    $ python scripts/train_cellpose_split.py --arm lr1e-3 --rehearsal

Train an arm:
    $ python scripts/train_cellpose_split.py --arm lr1e-3

Score every snapshot of both arms on VALIDATION:
    $ python scripts/evaluate_cellpose.py --out \\
        checkpoints/cellpose/selection/eval/selection.json \\
        --checkpoint-glob 'checkpoints/cellpose/selection/*/models/*'
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from materials_vision.experiments.cellpose.split_training import (
    DEFAULT_CACHE_DIR, LEARNING_RATES, N_EPOCHS, REHEARSAL_EXPERIMENT,
    RUN_SEED, TRAINING_EXPERIMENT, recipe_params, snapshot_path,
    train_cellpose)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import RunTracking
from materials_vision.training import build_source
from scripts.run_e8_full import spaced_indices

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/cellpose/selection")

REHEARSAL_SAVE_ROOT = Path("checkpoints/cellpose/rehearsal")

TRAIN_SUBSET = "train"

MONITOR_SUBSET = "val"

# A rehearsal trains on a few images for just long enough to leave one
# numbered snapshot and the last one, so both kinds of file name are
# exercised before a real run depends on them. Eleven epochs would not
# do: the eleventh is then also the last, and Cellpose saves it under
# the unnumbered name only.
REHEARSAL_N_TRAIN = 8

REHEARSAL_EPOCHS = 12


def run_name(arm: str) -> str:
    """What one arm's snapshots are filed under.

    Parameters
    ----------
    arm : str
        A key of ``LEARNING_RATES``.

    Returns
    -------
    str
    """
    return f"cpsam_{arm}_seed{RUN_SEED}"


def budget(args: argparse.Namespace) -> int:
    """Epochs this invocation trains.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
    """
    return REHEARSAL_EPOCHS if args.rehearsal else N_EPOCHS


def provenance_record(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that produced an arm's run, for the file beside it.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
    """
    return {
        "provenance": run_provenance(),
        "run": run_name(args.arm),
        "arm": args.arm,
        "rehearsal": args.rehearsal,
        "train_subset": TRAIN_SUBSET,
        "monitor_subset": MONITOR_SUBSET,
        "monitor_note": (
            "Cellpose's test loss on VALIDATION is of augmented crops "
            "and decides nothing; snapshots are chosen by instance F1."
        ),
        "split": str(args.split),
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        **recipe_params(LEARNING_RATES[args.arm]),
        "n_epochs": budget(args),
    }


def write_provenance(args: argparse.Namespace) -> Path:
    """Record what produced the run, before it starts.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / f"{run_name(args.arm)}_provenance.json"
    destination.write_text(
        json.dumps(provenance_record(args), indent=2, default=str)
    )
    logger.info("Wrote %s.", destination)
    return destination


def execute(args: argparse.Namespace, provenance: Path) -> Path:
    """Train the arm.

    Parameters
    ----------
    args : argparse.Namespace
    provenance : Path
        Attached to the MLflow run.

    Returns
    -------
    Path
        The directory holding the snapshots.
    """
    train_source = build_source(args.split, args.manifest, TRAIN_SUBSET)
    monitor_source = build_source(args.split, args.manifest, MONITOR_SUBSET)
    positions = (
        spaced_indices(len(train_source), REHEARSAL_N_TRAIN)
        if args.rehearsal else None
    )
    tracking = RunTracking(
        experiment=(
            REHEARSAL_EXPERIMENT if args.rehearsal else TRAINING_EXPERIMENT
        ),
        tags={"part": "selection", "arm": args.arm, "model": "cpsam"},
        params={"split": str(args.split), "manifest": str(args.manifest)},
        artifacts=(provenance,),
    )
    return train_cellpose(
        run_name(args.arm),
        train_source=train_source,
        monitor_source=monitor_source,
        learning_rate=LEARNING_RATES[args.arm],
        save_root=args.save_root,
        cache_dir=args.cache_dir,
        n_epochs=budget(args),
        train_positions=positions,
        tracking=tracking,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=sorted(LEARNING_RATES),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Where the cropped images, masks and flows are cached.",
    )
    parser.add_argument(
        "--rehearsal", action="store_true",
        help=f"{REHEARSAL_N_TRAIN} images, {REHEARSAL_EPOCHS} epochs, "
             f"into {REHEARSAL_SAVE_ROOT} and the rehearsal experiment.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the run's last snapshot already exists.",
    )
    args = parser.parse_args(argv)
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Train one arm of the selection.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    last = snapshot_path(
        args.save_root, run_name(args.arm), budget(args), budget(args)
    )
    if not args.force and last.exists():
        logger.info("%s already exists; nothing to do.", last)
        return EXIT_OK
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    execute(args, write_provenance(args))
    if not last.exists():
        logger.error(
            "Training finished but %s was not written.", last
        )
        return EXIT_FAILED
    logger.info("Run finished. Its last snapshot is %s.", last)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
