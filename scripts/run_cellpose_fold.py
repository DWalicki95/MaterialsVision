#!/usr/bin/env python3
"""
Train the chosen Cellpose recipe on one fold and score what it holds out.

**What this measures.** The same thing the PEFT-SAM cross-validation
measured, for the other recipe: how well it transfers to formulations
it has never seen, on five draws of held-out formulations instead of
one. The folds are the ones the PEFT-SAM runs were trained on, so each
fold's figure can be set beside the PEFT-SAM figure for the same
held-out formulations.

**Fold one is the final model.** It holds out exactly TEST of the
frozen split and trains on everything else, which is TRAIN and
VALIDATION together. PEFT-SAM trained its final model first and reused
it as fold one; here all five folds go through this one path, which
makes the final model a fold like any other rather than a separate
script that could drift from them.

**Nothing is chosen here.** The learning rate, the epoch scored and the
post-processing thresholds are read from the recipe record written by
``scripts/select_cellpose_recipe.py`` on VALIDATION, and this script
refuses to start until both choices are in it - so no fold can be
trained, let alone scored, before everything it depends on is fixed.
Every fold trains the full budget and is scored at the chosen epoch,
the condition that was selected rather than a shortened imitation of
it.

**Held-out formulations are scored once.** The evaluation goes through
``scripts/evaluate_cellpose.py`` with TEST unlocked for this fold's
single snapshot, and a fold whose report already exists is not scored
again.

Examples
--------
Rehearse the whole path in minutes, against a recipe written for it:
    $ python scripts/run_cellpose_fold.py --fold 3 --rehearsal \\
        --recipe <rehearsal recipe.json>

Train and score a fold:
    $ python scripts/run_cellpose_fold.py --fold 3
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

import scripts.evaluate_cellpose as evaluate_cli
from materials_vision.data.split_io import SplitLoadError, load_split
from materials_vision.experiments.cellpose.split_training import (
    DEFAULT_CACHE_DIR, LEARNING_RATES, N_EPOCHS, REHEARSAL_EXPERIMENT,
    RUN_SEED, TRAINING_EXPERIMENT, recipe_params, snapshot_path,
    train_cellpose)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import RunTracking
from materials_vision.training import build_source
from scripts.run_e8_full import spaced_indices
from scripts.select_cellpose_recipe import DEFAULT_RECIPE

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

CV_ID = "cv5_v1"

DEFAULT_FOLDS_DIR = Path("/home/dwalicki/dane/splits") / CV_ID

DEFAULT_SAVE_ROOT = Path("checkpoints/cellpose/cv")

REHEARSAL_SAVE_ROOT = Path("checkpoints/cellpose/rehearsal_cv")

N_FOLDS = 5

TRAIN_SUBSET = "train"

HELD_OUT_SUBSET = "test"

# Enough to leave a numbered snapshot and the last one; the rehearsal
# scores the last, since the chosen epoch lies beyond its budget.
REHEARSAL_N_TRAIN = 8

REHEARSAL_EPOCHS = 12

REHEARSAL_N_IMAGES = 4


class RecipeError(ValueError):
    """Raised when the recipe record is missing or incomplete."""


def run_name(fold: int) -> str:
    """What one fold's snapshots are filed under.

    Parameters
    ----------
    fold : int

    Returns
    -------
    str
    """
    return f"cpsam_cv5_fold{fold}_seed{RUN_SEED}"


def read_recipe(path: Path) -> dict[str, Any]:
    """Read the recipe and check that everything a fold needs is in it.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict

    Raises
    ------
    RecipeError
        If the file is missing, either choice has not been made, or
        the learning rate recorded disagrees with the arm it names.
    """
    if not path.exists():
        raise RecipeError(
            f"{path} does not exist. The recipe is chosen on VALIDATION "
            f"by scripts/select_cellpose_recipe.py before any fold runs."
        )
    recipe = json.loads(path.read_text())
    missing = [s for s in ("snapshot", "postprocessing") if s not in recipe]
    if missing:
        raise RecipeError(
            f"{path} does not yet record the {' and '.join(missing)} "
            f"choice; no fold runs before both are made."
        )
    if "niter" not in recipe["postprocessing"]:
        raise RecipeError(
            f"{path} records no flow-following steps. It predates their "
            f"calibration; the folds read the second round, "
            f"recipe_v2.json."
        )
    snapshot = recipe["snapshot"]
    if LEARNING_RATES.get(snapshot["arm"]) != snapshot["learning_rate"]:
        raise RecipeError(
            f"{path} names arm {snapshot['arm']!r} with learning rate "
            f"{snapshot['learning_rate']}, which is not that arm's."
        )
    return recipe


def budget(args: argparse.Namespace) -> int:
    """Epochs the fold trains.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
    """
    return REHEARSAL_EPOCHS if args.rehearsal else N_EPOCHS


def scored_epoch(args: argparse.Namespace, recipe: dict[str, Any]) -> int:
    """Completed epochs of the one snapshot that is scored.

    Parameters
    ----------
    args : argparse.Namespace
    recipe : dict

    Returns
    -------
    int
    """
    if args.rehearsal:
        return REHEARSAL_EPOCHS
    return int(recipe["snapshot"]["completed_epochs"])


def scored_snapshot(args: argparse.Namespace, recipe: dict[str, Any]) -> Path:
    """The fold's one scored snapshot, named before it exists.

    Parameters
    ----------
    args : argparse.Namespace
    recipe : dict

    Returns
    -------
    Path
    """
    return snapshot_path(
        args.save_root, run_name(args.fold), scored_epoch(args, recipe),
        budget(args),
    )


def report_path(args: argparse.Namespace) -> Path:
    """Where the fold's held-out figures are written.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
    """
    return args.save_root / "eval" / f"fold{args.fold}.json"


def provenance_record(
    args: argparse.Namespace, recipe: dict[str, Any]
) -> dict[str, Any]:
    """Everything that produced the fold's run, for the file beside it.

    Parameters
    ----------
    args : argparse.Namespace
    recipe : dict

    Returns
    -------
    dict
    """
    fold_metadata = json.loads(args.fold_metadata.read_text())
    return {
        "provenance": run_provenance(),
        "run": run_name(args.fold),
        "cv_id": CV_ID,
        "fold": args.fold,
        "held_out": fold_metadata.get("held_out"),
        "fold_split_table_sha256": fold_metadata.get("split_table_sha256"),
        "rehearsal": args.rehearsal,
        "recipe_path": str(args.recipe),
        "recipe_snapshot": {
            k: v for k, v in recipe["snapshot"].items()
            if k not in ("ranking", "provenance")
        },
        "recipe_postprocessing": {
            k: v for k, v in recipe["postprocessing"].items()
            if k not in ("ranking", "provenance")
        },
        "scored_snapshot": str(scored_snapshot(args, recipe)),
        "split": str(args.split),
        "manifest": str(args.manifest),
        **recipe_params(recipe["snapshot"]["learning_rate"]),
        "n_epochs": budget(args),
    }


def train_fold(args: argparse.Namespace, recipe: dict[str, Any]) -> None:
    """Train the recipe on the fold's training rows.

    Parameters
    ----------
    args : argparse.Namespace
    recipe : dict
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    provenance = args.save_root / f"{run_name(args.fold)}_provenance.json"
    provenance.write_text(json.dumps(
        provenance_record(args, recipe), indent=2, default=str
    ))
    train_source = build_source(args.split, args.manifest, TRAIN_SUBSET)
    positions = (
        spaced_indices(len(train_source), REHEARSAL_N_TRAIN)
        if args.rehearsal else None
    )
    tracking = RunTracking(
        experiment=(
            REHEARSAL_EXPERIMENT if args.rehearsal else TRAINING_EXPERIMENT
        ),
        tags={
            "part": "cv", "cv_id": CV_ID, "fold": str(args.fold),
            "arm": recipe["snapshot"]["arm"], "model": "cpsam",
        },
        params={"cv_id": CV_ID, "fold": args.fold},
        artifacts=(provenance, args.fold_metadata),
    )
    train_cellpose(
        run_name(args.fold),
        train_source=train_source,
        monitor_source=None,
        learning_rate=recipe["snapshot"]["learning_rate"],
        save_root=args.save_root,
        cache_dir=args.cache_dir,
        n_epochs=budget(args),
        train_positions=positions,
        tracking=tracking,
    )


def evaluation_argv(
    args: argparse.Namespace, recipe: dict[str, Any]
) -> list[str]:
    """The one evaluation of the fold's held-out formulations.

    Parameters
    ----------
    args : argparse.Namespace
    recipe : dict

    Returns
    -------
    list of str
    """
    chosen = recipe["postprocessing"]
    argv = [
        "--split", str(args.split), "--manifest", str(args.manifest),
        "--subset", HELD_OUT_SUBSET, "--unlock-test",
        "--checkpoint", str(scored_snapshot(args, recipe)),
        "--flow-threshold", str(chosen["flow_threshold"]),
        "--cellprob-threshold", str(chosen["cellprob_threshold"]),
        "--niter", str(chosen["niter"]),
        "--also-area-filter",
        "--out", str(report_path(args)),
    ]
    if args.rehearsal:
        argv += ["--n-images", str(REHEARSAL_N_IMAGES)]
    return argv


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
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true",
        help=f"{REHEARSAL_N_TRAIN} images, {REHEARSAL_EPOCHS} epochs, "
             f"{REHEARSAL_N_IMAGES} held-out images scored, into "
             f"{REHEARSAL_SAVE_ROOT}.",
    )
    args = parser.parse_args(argv)
    args.split = args.folds_dir / f"fold{args.fold}.csv"
    args.fold_metadata = args.folds_dir / f"fold{args.fold}_metadata.json"
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Train one fold, if not trained yet, and score it, if not scored.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if not 1 <= args.fold <= N_FOLDS:
        logger.error("There is no fold %d; folds run 1 to %d.",
                     args.fold, N_FOLDS)
        return EXIT_FAILED
    try:
        recipe = read_recipe(args.recipe)
        # Checked before the GPU is touched: a fold drawn from another
        # manifest would train on images the fold does not describe.
        load_split(args.split, TRAIN_SUBSET, verify_manifest=args.manifest)
    except (RecipeError, SplitLoadError, FileNotFoundError) as error:
        logger.error("%s", error)
        return EXIT_FAILED

    report = report_path(args)
    if report.exists():
        logger.info(
            "%s already exists; the held-out formulations of fold %d are "
            "scored once.", report, args.fold,
        )
        return EXIT_OK

    snapshot = scored_snapshot(args, recipe)
    if not snapshot.exists():
        if not torch.cuda.is_available():
            logger.error("No GPU is visible, so there is nothing to train on.")
            return EXIT_FAILED
        train_fold(args, recipe)
        if not snapshot.exists():
            logger.error(
                "Training finished but %s, the snapshot fixed in advance, "
                "was not written.", snapshot,
            )
            return EXIT_FAILED
    logger.info("Scoring %s on the formulations fold %d holds out.",
                snapshot, args.fold)
    return evaluate_cli.main(evaluation_argv(args, recipe))


if __name__ == "__main__":
    sys.exit(main())
