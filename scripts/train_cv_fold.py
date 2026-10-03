#!/usr/bin/env python3
"""
Train the adopted recipe on one cross-validation fold.

**What this measures.** The final model was trained on every
formulation except those of TEST and scored once on them. That is one
draw of unseen formulations, and for the materials of the second
microscope it is one formulation each. Each fold here holds out a
different set of formulations and trains the same recipe on the rest,
so that the transfer to new formulations can be read across all of
them rather than on one draw.

**Nothing is chosen here.** The recipe, the snapshot and the
post-processing were all fixed before the folds were written, and the
training itself is the final model's, called through the same
function: a fold differs from the final model in the images it
learns from and in nothing that was selected. Hence no validation set:
the trainer is handed a few of its own training images, as the final
model was, for a loss nothing reads.

**Fold one is not trained.** It holds out exactly TEST, so the final
model is its model and the final model's TEST score is its score.
Training it again would produce a second copy of one run.

The one departure from the final model that is not data: these runs
are recorded in MLflow, and the trainer's monitoring images are
training images rather than validation images. Neither changes what is
learnt from, but both change how the global random stream is drawn
between steps, so a fold differs from a hypothetical retraining of
fold one by about as much as a change of seed would.

Examples
--------
Rehearse the whole path in minutes:
    $ python scripts/train_cv_fold.py --fold 3 --rehearsal

Train a fold:
    $ python scripts/train_cv_fold.py --fold 3

Score its snapshot on the formulations it held out, once:
    $ python scripts/evaluate_checkpoint.py --subset test --unlock-test \\
        --split /home/dwalicki/dane/splits/cv5_v1/fold3.csv \\
        --postprocessing frozen --also-postprocessing deployed \\
        --checkpoint \\
        checkpoints/cv/checkpoints/full_b_cv5_fold3_seed20260907/epoch-12.pt \\
        --out checkpoints/cv/eval/fold3.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from materials_vision.data.split_io import SplitLoadError, load_split
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import (REHEARSAL_EXPERIMENT,
                                       TRAINING_EXPERIMENT, RunTracking)
from scripts.run_e8_full import RUN_SEED
from scripts.train_final_model import (N_EPOCHS, POLICY, recipe_record,
                                       scored_snapshot, train_recipe)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

CV_ID = "cv5_v1"

DEFAULT_FOLDS_DIR = Path("/home/dwalicki/dane/splits") / CV_ID

DEFAULT_SAVE_ROOT = Path("checkpoints/cv")

REHEARSAL_SAVE_ROOT = Path("checkpoints/rehearsal_cv")

N_FOLDS = 5

# Fold one holds out exactly TEST of the frozen split: its model is the
# final model, trained on TRAIN and VALIDATION together.
FINAL_MODEL_FOLD = 1

TRAIN_SUBSET = "train"

REHEARSAL_N_TRAIN = 8

REHEARSAL_EPOCHS = 1


def run_name(fold: int) -> str:
    """What one fold's checkpoints are filed under.

    Parameters
    ----------
    fold : int

    Returns
    -------
    str
    """
    return f"{POLICY}_cv5_fold{fold}_seed{RUN_SEED}"


def fold_paths(folds_dir: Path, fold: int) -> tuple[Path, Path]:
    """The fold's split table and its metadata sidecar.

    Parameters
    ----------
    folds_dir : Path
    fold : int

    Returns
    -------
    tuple of Path
    """
    return (
        folds_dir / f"fold{fold}.csv",
        folds_dir / f"fold{fold}_metadata.json",
    )


def provenance_record(
    args: argparse.Namespace, fold_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Everything that produced the fold's run, for the file beside it.

    Parameters
    ----------
    args : argparse.Namespace
    fold_metadata : dict

    Returns
    -------
    dict
    """
    return {
        "provenance": run_provenance(),
        "run": run_name(args.fold),
        "cv_id": CV_ID,
        "fold": args.fold,
        "held_out": fold_metadata.get("held_out"),
        "fold_split_table_sha256": fold_metadata.get("split_table_sha256"),
        "train_subset": TRAIN_SUBSET,
        "monitor_subset": TRAIN_SUBSET,
        "epochs": args.epochs,
        "n_train": args.n_train,
        "rehearsal": args.rehearsal,
        "scored_snapshot": str(scored_snapshot(
            args.save_root, args.epochs, run_name(args.fold)
        )),
        "split": str(args.split),
        "manifest": str(args.manifest),
        **recipe_record(),
    }


def write_provenance(record: dict[str, Any], save_root: Path) -> Path:
    """Write the provenance record next to the run's checkpoints.

    Parameters
    ----------
    record : dict
    save_root : Path

    Returns
    -------
    Path
    """
    save_root.mkdir(parents=True, exist_ok=True)
    destination = save_root / f"{record['run']}_provenance.json"
    destination.write_text(json.dumps(record, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return destination


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
        else TRAINING_EXPERIMENT,
        tags={
            "study": "cross_validation",
            "part": "cv",
            "cv_id": CV_ID,
            "fold": str(args.fold),
            "policy": POLICY.upper(),
            "seed": str(RUN_SEED),
            "train_subset": TRAIN_SUBSET,
            "source": "live",
        },
        params={
            "seed": RUN_SEED,
            "policy": POLICY.upper(),
            "augmentation_families": families,
            "cv_id": CV_ID,
            "fold": args.fold,
        },
        artifacts=(provenance_path, metadata_path),
    )


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why the fold may not be trained as asked, or nothing.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    if not 1 <= args.fold <= N_FOLDS:
        return f"There is no fold {args.fold}; folds run 1 to {N_FOLDS}."
    if args.fold == FINAL_MODEL_FOLD:
        return (
            f"Fold {FINAL_MODEL_FOLD} holds out exactly TEST, so its model "
            f"is the final model and its score the final model's TEST "
            f"score. Training it again would produce a second copy of one "
            f"run."
        )
    if not args.split.exists():
        return (
            f"{args.split} does not exist. The folds are written by "
            f"scripts/create_cv_folds.py, before any fold is trained."
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
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true",
        help=(
            f"{REHEARSAL_N_TRAIN} training images for {REHEARSAL_EPOCHS} "
            f"epoch, into {REHEARSAL_SAVE_ROOT} and the "
            f"{REHEARSAL_EXPERIMENT} experiment."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the scored snapshot already exists.",
    )
    args = parser.parse_args(argv)
    args.split, args.fold_metadata = fold_paths(args.folds_dir, args.fold)
    args.n_train = REHEARSAL_N_TRAIN if args.rehearsal else 0
    args.epochs = REHEARSAL_EPOCHS if args.rehearsal else N_EPOCHS
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
    fold_metadata = json.loads(
        args.fold_metadata.read_text(encoding="utf-8")
    )
    provenance_path = write_provenance(
        provenance_record(args, fold_metadata), args.save_root
    )
    train_recipe(
        run_name(args.fold),
        split=args.split,
        manifest=args.manifest,
        train_subset=TRAIN_SUBSET,
        monitor_subset=TRAIN_SUBSET,
        save_root=args.save_root,
        epochs=args.epochs,
        n_train=args.n_train,
        tracking=build_tracking(args, provenance_path, args.fold_metadata),
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Train one fold.

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
        # Checked before the GPU is touched: a fold drawn from another
        # manifest would train on images the fold does not describe.
        load_split(args.split, TRAIN_SUBSET, verify_manifest=args.manifest)
    except SplitLoadError as e:
        logger.error("%s", e)
        return EXIT_FAILED
    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    snapshot = scored_snapshot(
        args.save_root, args.epochs, run_name(args.fold)
    )
    if not args.force and snapshot.exists():
        logger.info("%s already exists; nothing to do.", snapshot)
        return EXIT_OK
    execute(args)

    if not snapshot.exists():
        logger.error(
            "Training finished but %s was not written; the snapshot "
            "that was fixed in advance does not exist.", snapshot,
        )
        return EXIT_FAILED
    logger.info("Run finished. The snapshot to score is %s.", snapshot)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
