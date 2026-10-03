#!/usr/bin/env python3
"""
Write the cross-validation folds over formulations.

Thin wrapper around ``materials_vision.data.cv_folds``: reads the
frozen split, decides which fold holds out each formulation, checks
the result and writes one split table per fold, with the metadata
sidecar ``load_split`` expects, plus the assignment and a short report.

The folds are written once, before any model is trained on them, and
the assignment is not redrawn afterwards: a fold chosen after seeing
how a model did on it would stop describing transfer to new
formulations. Writing over existing folds therefore needs
``--overwrite``.

Examples
--------
Draw and report without writing anything:
    $ python scripts/create_cv_folds.py --dry-run

Write the folds:
    $ python scripts/create_cv_folds.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from materials_vision.data.cv_folds import (FROZEN_SET_FOLDS, N_FOLDS,
                                            FoldError, assign_folds,
                                            check_folds, file_sha256,
                                            fold_summary, fold_table, held_out,
                                            write_fold)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

CV_ID = "cv5_v1"

DEFAULT_OUT_DIR = Path("/home/dwalicki/dane/splits") / CV_ID

CV_SEED = 20260926


def load_frozen_split(split_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Read the frozen split table and its metadata.

    Parameters
    ----------
    split_csv : Path

    Returns
    -------
    table : pandas.DataFrame
    metadata : dict

    Raises
    ------
    FoldError
        If either file is missing or the metadata records no manifest.
    """
    meta_path = split_csv.with_name(f"{split_csv.stem}_metadata.json")
    for path in (split_csv, meta_path):
        if not path.exists():
            raise FoldError(f"Not found: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if "manifest_sha256" not in metadata:
        raise FoldError(f"{meta_path} records no manifest_sha256")
    return pd.read_csv(split_csv), metadata


def fold_metadata(
    fold: int,
    summary: Mapping[str, Any],
    split_csv: Path,
    split_metadata: Mapping[str, Any],
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The sidecar of one fold.

    Parameters
    ----------
    fold : int
    summary : mapping
        As returned by ``fold_summary``.
    split_csv : Path
    split_metadata : mapping
    seed : int
    provenance : mapping

    Returns
    -------
    dict
    """
    return {
        "split_id": f"{CV_ID}_fold{fold}",
        "cv_id": CV_ID,
        "fold": fold,
        "n_folds": N_FOLDS,
        "grouping_unit": "formulation",
        "held_out_label": "test",
        "held_out": summary["held_out"],
        "frozen_set_held_out": FROZEN_SET_FOLDS.get(fold),
        "seed": seed,
        "forced_train": list(split_metadata.get("forced_train", [])),
        "source_split": str(split_csv),
        "source_split_id": split_metadata.get("split_id"),
        "source_split_sha256": file_sha256(split_csv),
        "manifest_path": split_metadata.get("manifest_path"),
        "manifest_sha256": split_metadata["manifest_sha256"],
        "counts": {k: v for k, v in summary.items() if k != "held_out"},
        "provenance": provenance,
    }


def format_report(
    summaries: Mapping[int, Mapping[str, Any]],
    n_tied: Mapping[str, int],
    seed: int,
    split_csv: Path,
) -> str:
    """A short Markdown report of the folds.

    Parameters
    ----------
    summaries : mapping of int to mapping
    n_tied : mapping of str to int
    seed : int
    split_csv : Path

    Returns
    -------
    str
    """
    lines = [
        f"# Foldy walidacji krzyzowej: `{CV_ID}`",
        "",
        f"Zrodlo: `{split_csv}` | seed: `{seed}` | foldow: {N_FOLDS}",
        "",
        "Remisy przy podziale (liczba rownie zbalansowanych podzialow, "
        "z ktorych losowano): "
        + ", ".join(f"{m} {n}" for m, n in sorted(n_tied.items())),
        "",
        "| fold | odlozone | obrazow testu | porow testu | AS / K / VAB "
        "obrazow | fine | obrazow treningu | nieuzyte |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for fold, s in sorted(summaries.items()):
        names = ", ".join(
            name for material in sorted(s["held_out"])
            for name in s["held_out"][material]
        )
        by_material = " / ".join(
            str(s["n_test_images_by_material"].get(m, 0))
            for m in ("AS", "K", "VAB")
        )
        lines.append(
            f"| {fold} | {names} | {s['n_test_images']} | "
            f"{s['n_test_pores']} | {by_material} | "
            f"{s['n_test_fine_images']} | {s['n_train_images']} | "
            f"{s['n_unused']} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    """Draw, check and write the folds.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
        Process exit code.
    """
    table, split_metadata = load_frozen_split(args.split)
    forced_train = list(split_metadata.get("forced_train", []))
    assignment, n_tied = assign_folds(
        table, forced_train=forced_train, seed=args.seed
    )
    violations = check_folds(table, assignment, forced_train=forced_train)
    if violations:
        for violation in violations:
            logger.error("Fold check failed: %s", violation)
        return EXIT_FAILED

    folds = {
        k: fold_table(table, held_out(assignment, k))
        for k in range(1, N_FOLDS + 1)
    }
    summaries = {k: fold_summary(fold) for k, fold in folds.items()}
    report = format_report(summaries, n_tied, args.seed, args.split)
    logger.info("Folds drawn:\n%s", report)
    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return EXIT_OK

    provenance = run_provenance()
    for k, fold in folds.items():
        metadata = fold_metadata(
            k, summaries[k], args.split, split_metadata, args.seed,
            provenance,
        )
        write_fold(
            fold, metadata, args.out_dir, f"fold{k}",
            overwrite=args.overwrite,
        )
    assignment_path = args.out_dir / f"{CV_ID}_assignment.json"
    assignment_path.write_text(json.dumps({
        "cv_id": CV_ID,
        "seed": args.seed,
        "forced_train": forced_train,
        "n_tied_divisions": n_tied,
        "assignment": dict(sorted(assignment.items())),
    }, indent=2), encoding="utf-8")
    report_path = args.out_dir / f"{CV_ID}_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s and %s.", assignment_path, report_path)
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
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--seed", type=int, default=CV_SEED,
        help="Breaks ties among equally balanced divisions.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Draw and report, write nothing.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing folds; only for a draft nothing used.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Write the cross-validation folds.

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
