#!/usr/bin/env python3
"""
Write the split of the epoch-budget study: fold four with a validation set.

**Why a fold needs one here.** The question is what happens when the
number of epochs grows from twelve to fifty, with the snapshot taken
where it does best on validation, as anyone with fifty epochs would
take it. A cross-validation fold has only a training set and the
formulations it holds out. Choosing the epoch on the held-out
formulations would let the same images choose the snapshot and then
measure it, so the choice is made on formulations taken out of
training instead.

**Which formulations.** Those of the frozen split's VALIDATION - the
set fold two holds out - because it is the set the twelve-epoch budget
was chosen on, so both budgets are judged where the shorter one was
fixed. They lie wholly in fold four's training set, since fold four
holds out other formulations. Everything else of fold four is left as
it was: its held-out set is unchanged image for image, and its training
set loses exactly these formulations.

The split is written once, before either run is trained, and refuses
to overwrite itself.

Examples
--------
Write the split and report it:
    $ python scripts/peft_sam/create_epoch_budget_split.py

Report without writing:
    $ python scripts/peft_sam/create_epoch_budget_split.py --dry-run
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from materials_vision.data.cv_folds import (HELD_OUT, TRAIN, VALIDATION,
                                            FoldError, carve_validation,
                                            file_sha256, fold_summary,
                                            write_fold)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

BUDGET_ID = "epoch_budget_v1"

FOLD = 4

# The fold whose held-out set is the frozen split's VALIDATION.
VALIDATION_FOLD = 2

DEFAULT_FOLDS_DIR = Path("/home/dwalicki/dane/splits/cv5_v1")

DEFAULT_OUT_DIR = Path("/home/dwalicki/dane/splits") / BUDGET_ID

SPLIT_NAME = f"fold{FOLD}_valv1"


def read_fold(folds_dir: Path, fold: int) -> tuple[pd.DataFrame, dict]:
    """A fold's table and its metadata.

    Parameters
    ----------
    folds_dir : Path
    fold : int

    Returns
    -------
    table : pandas.DataFrame
    metadata : dict

    Raises
    ------
    FoldError
        If either file is missing.
    """
    table_path = folds_dir / f"fold{fold}.csv"
    meta_path = folds_dir / f"fold{fold}_metadata.json"
    for path in (table_path, meta_path):
        if not path.exists():
            raise FoldError(f"Not found: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return pd.read_csv(table_path), metadata


def validation_formulations(metadata: Mapping[str, Any]) -> frozenset[str]:
    """The formulations of the frozen VALIDATION, from fold two's record.

    Parameters
    ----------
    metadata : mapping
        Fold two's metadata.

    Returns
    -------
    frozenset of str

    Raises
    ------
    FoldError
        If the fold given does not hold out the frozen VALIDATION.
    """
    if metadata.get("frozen_set_held_out") != VALIDATION:
        raise FoldError(
            f"Fold {metadata.get('fold')} does not hold out the frozen "
            f"VALIDATION (it records {metadata.get('frozen_set_held_out')!r})."
        )
    return frozenset(
        name for names in metadata["held_out"].values() for name in names
    )


def _ids(table: pd.DataFrame, subset: str) -> set[str]:
    """Image identifiers of one set, used rows only."""
    rows = table[(table["split"] == subset) & table["used"].astype(bool)]
    return set(rows["image_id"])


def check_split(
    carved: pd.DataFrame,
    fold: pd.DataFrame,
    validation_source: pd.DataFrame,
) -> list[str]:
    """Every property the split is meant to have, as violations.

    Parameters
    ----------
    carved : pandas.DataFrame
        The split about to be written.
    fold : pandas.DataFrame
        Fold four as it was written.
    validation_source : pandas.DataFrame
        Fold two, whose held-out images the validation set must equal.

    Returns
    -------
    list of str
        Empty when the split is sound.
    """
    violations = []
    train, val, test = (
        _ids(carved, TRAIN), _ids(carved, VALIDATION), _ids(carved, HELD_OUT)
    )
    if val != _ids(validation_source, HELD_OUT):
        violations.append(
            "validation is not exactly the images of the frozen VALIDATION"
        )
    if test != _ids(fold, HELD_OUT):
        violations.append("the held-out set differs from fold four's")
    if train != _ids(fold, TRAIN) - val:
        violations.append(
            "training is not fold four's training less the validation set"
        )
    for name, (a, b) in {
        "training and validation": (train, val),
        "training and held-out": (train, test),
        "validation and held-out": (val, test),
    }.items():
        if a & b:
            violations.append(f"{name} share image(s) {sorted(a & b)[:5]}")
    return violations


def split_metadata(
    carved: pd.DataFrame,
    formulations: frozenset[str],
    fold_meta: Mapping[str, Any],
    folds_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The sidecar of the split.

    Parameters
    ----------
    carved : pandas.DataFrame
    formulations : frozenset of str
        Moved to validation.
    fold_meta : mapping
        Fold four's metadata.
    folds_dir : Path
    provenance : mapping

    Returns
    -------
    dict
    """
    used = carved[carved["used"].astype(bool)]
    val = used[used["split"] == VALIDATION]
    return {
        "split_id": f"{BUDGET_ID}_{SPLIT_NAME}",
        "budget_id": BUDGET_ID,
        "cv_id": fold_meta.get("cv_id"),
        "fold": FOLD,
        "validation_formulations": {
            str(material): sorted(rows["formulation"].unique())
            for material, rows in val.groupby("material", sort=True)
        },
        "validation_source": str(
            folds_dir / f"fold{VALIDATION_FOLD}_metadata.json"
        ),
        "held_out_label": HELD_OUT,
        "held_out": fold_meta.get("held_out"),
        "source_fold": str(folds_dir / f"fold{FOLD}.csv"),
        "source_fold_sha256": file_sha256(folds_dir / f"fold{FOLD}.csv"),
        "manifest_path": fold_meta.get("manifest_path"),
        "manifest_sha256": fold_meta["manifest_sha256"],
        "counts": {
            **{k: v for k, v in fold_summary(carved).items()
               if k != "held_out"},
            "n_val_images": len(val),
            "n_val_pores": int(val["n_instances"].sum()),
            "n_val_images_by_material": {
                str(k): int(v) for k, v in
                val.groupby("material", sort=True).size().items()
            },
            "n_val_formulations": len(formulations),
        },
        "provenance": provenance,
    }


def run(args: argparse.Namespace) -> int:
    """Build, check and write the split.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
        Process exit code.
    """
    fold, fold_meta = read_fold(args.folds_dir, FOLD)
    validation_source, validation_meta = read_fold(
        args.folds_dir, VALIDATION_FOLD
    )
    formulations = validation_formulations(validation_meta)
    carved = carve_validation(fold, formulations)
    violations = check_split(carved, fold, validation_source)
    if violations:
        for violation in violations:
            logger.error("Split check failed: %s", violation)
        return EXIT_FAILED

    metadata = split_metadata(
        carved, formulations, fold_meta, args.folds_dir, run_provenance()
    )
    logger.info(
        "Fold %d with %s moved to validation:\n%s", FOLD,
        ", ".join(sorted(formulations)),
        json.dumps(metadata["counts"], indent=2),
    )
    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return EXIT_OK
    write_fold(
        carved, metadata, args.out_dir, SPLIT_NAME, overwrite=args.overwrite
    )
    return EXIT_OK


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
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build, check and report; write nothing.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing split; only for a draft nothing used.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Write the split of the epoch-budget study.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    try:
        return run(args)
    except FoldError as e:
        logger.error("%s", e)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
