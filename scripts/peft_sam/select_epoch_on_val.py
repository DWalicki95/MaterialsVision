#!/usr/bin/env python3
"""
Choose an arm's epoch on validation, by a rule fixed before training.

Every snapshot of the arm has been scored on the validation
formulations, one JSON report per snapshot or several together. This
reads them and applies the rule both arms share: the highest instance
F1 under the frozen post-processing, the earlier epoch on a tie. The
choice is written to a file before the held-out formulations are
opened, so the record shows which snapshot was chosen, and on what,
before its held-out score existed.

**It refuses rather than chooses among what happens to be there.** A
missing epoch, a snapshot scored on a different number of images than
the others, or two reports giving one snapshot different figures would
each make the maximum a maximum over something other than the run.

Examples
--------
Choose the epoch of the fifty-epoch arm:
    $ python scripts/peft_sam/select_epoch_on_val.py --epochs 50

Choose the epoch of the half-rate arm:
    $ python scripts/peft_sam/select_epoch_on_val.py --epochs 12 \\
        --lr-scale 0.5
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import (configure_tracking,
                                       find_run_for_checkpoint_dir,
                                       snapshot_epoch)
from scripts.peft_sam.train_epoch_budget import (ARMS, DEFAULT_SAVE_ROOT,
                                                 DEFAULT_SPLIT, LR_SCALES,
                                                 REHEARSAL_EPOCHS,
                                                 REHEARSAL_SAVE_ROOT,
                                                 SELECTION_RULE, lr_suffix,
                                                 run_name)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

# The post-processing the rule reads F1 under.
SELECTION_POSTPROCESSING = "frozen"

SELECTED_EPOCH_TAG = "mv.selected_epoch"

# Figures kept per epoch in the choice record, beside the F1 it is made
# on, so the curve can be read without the reports.
CURVE_FIELDS = (
    "f1", "precision", "recall", "merges_per_100_gt", "splits_per_100_gt",
    "pore_count_error",
)


class SelectionError(ValueError):
    """Raised when the reports cannot support a choice of epoch."""


def snapshot_entries(
    payloads: Iterable[Mapping[str, Any]], setting_label: str
) -> dict[Path, list[dict[str, Any]]]:
    """Every scored snapshot under one setting, by checkpoint path.

    Read from the label the evaluation gives each pooled result,
    ``<checkpoint>@<setting>``, rather than from the report's keys,
    whose form depends on how many settings were scored in that pass.

    Parameters
    ----------
    payloads : iterable of mapping
        Reports of ``evaluate_checkpoint.py``.
    setting_label : str

    Returns
    -------
    dict of Path to list of dict
        Each snapshot's entries, one per report that scored it.
    """
    found: dict[Path, list[dict[str, Any]]] = {}
    for payload in payloads:
        for entry in payload.values():
            checkpoint, _, label = entry["overall"]["label"].rpartition("@")
            if label == setting_label:
                found.setdefault(Path(checkpoint), []).append(entry)
    return found


def run_curve(
    entries: Mapping[Path, list[dict[str, Any]]], run: str, n_epochs: int
) -> dict[int, dict[str, Any]]:
    """One run's pooled figures per epoch, checked to be whole.

    Parameters
    ----------
    entries : mapping
        From :func:`snapshot_entries`.
    run : str
        The run directory's name.
    n_epochs : int
        Epochs the run trained; every one must be present.

    Returns
    -------
    dict of int to dict
        Epoch to the snapshot's ``overall`` figures.

    Raises
    ------
    SelectionError
        If an epoch is missing, if two reports disagree on a snapshot,
        or if the snapshots were scored on differing populations.
    """
    curve: dict[int, dict[str, Any]] = {}
    for checkpoint, found in entries.items():
        epoch = snapshot_epoch(checkpoint)
        if checkpoint.parent.name != run or epoch is None:
            continue
        overalls = [entry["overall"] for entry in found]
        if len({(o["f1"], o["n_images"]) for o in overalls}) > 1:
            raise SelectionError(
                f"Reports disagree on {checkpoint}: "
                f"{[(o['f1'], o['n_images']) for o in overalls]}."
            )
        curve[epoch] = overalls[0]
    missing = [k for k in range(1, n_epochs + 1) if k not in curve]
    if missing:
        raise SelectionError(
            f"{run}: no score for epoch(s) {missing}; the epoch is chosen "
            f"among all {n_epochs} or not at all."
        )
    extra = sorted(k for k in curve if k > n_epochs)
    if extra:
        raise SelectionError(
            f"{run}: scores for epoch(s) {extra} beyond the {n_epochs} "
            f"trained; these reports belong to another run."
        )
    populations = {curve[k]["n_images"] for k in curve}
    if len(populations) > 1:
        raise SelectionError(
            f"{run}: snapshots scored on differing numbers of images "
            f"{sorted(populations)}."
        )
    return curve


def select_epoch(f1_by_epoch: Mapping[int, float]) -> int:
    """The highest F1; the earlier epoch on an exact tie.

    Parameters
    ----------
    f1_by_epoch : mapping of int to float

    Returns
    -------
    int

    Raises
    ------
    SelectionError
        If the curve is empty or any value is not finite.
    """
    if not f1_by_epoch:
        raise SelectionError("Nothing to choose from.")
    bad = sorted(k for k, v in f1_by_epoch.items() if not v == v)
    if bad:
        raise SelectionError(f"F1 is undefined at epoch(s) {bad}.")
    best = max(f1_by_epoch.values())
    return min(k for k, v in f1_by_epoch.items() if v == best)


def expected_val_images(split: Path) -> int:
    """How many validation images the split's record says there are.

    Parameters
    ----------
    split : Path

    Returns
    -------
    int

    Raises
    ------
    SelectionError
        If the record is missing or carries no count.
    """
    meta_path = split.with_name(f"{split.stem}_metadata.json")
    if not meta_path.exists():
        raise SelectionError(f"No split metadata at {meta_path}")
    counts = json.loads(meta_path.read_text(encoding="utf-8"))["counts"]
    if "n_val_images" not in counts:
        raise SelectionError(f"{meta_path} records no n_val_images")
    return int(counts["n_val_images"])


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choice_record(
    args: argparse.Namespace,
    curve: Mapping[int, Mapping[str, Any]],
    epoch: int,
    reports: list[Path],
) -> dict[str, Any]:
    """What was chosen, how and on what.

    Parameters
    ----------
    args : argparse.Namespace
    curve : mapping
        From :func:`run_curve`.
    epoch : int
    reports : list of Path

    Returns
    -------
    dict
    """
    run = run_name(args.epochs, args.lr_scale)
    return {
        "run": run,
        "arm_epochs": args.epochs,
        "lr_scale": args.lr_scale,
        "epochs_trained": args.n_epochs,
        "selected_epoch": epoch,
        "checkpoint": str(
            args.save_root / "checkpoints" / run / f"epoch-{epoch}.pt"
        ),
        "rule": SELECTION_RULE,
        "postprocessing": SELECTION_POSTPROCESSING,
        "setting_label": args.setting_label,
        "val_f1": curve[epoch]["f1"],
        "n_val_images": curve[epoch]["n_images"],
        "curve": {
            str(k): {field: curve[k].get(field) for field in CURVE_FIELDS}
            for k in sorted(curve)
        },
        "reports": {str(path): _sha256(path) for path in reports},
        "provenance": run_provenance(),
    }


def tag_training_run(checkpoint: Path, epoch: int) -> None:
    """Put the choice on the arm's training run, for the interface.

    Parameters
    ----------
    checkpoint : Path
    epoch : int
    """
    import mlflow

    configure_tracking()
    run_id = find_run_for_checkpoint_dir(checkpoint.parent)
    if run_id is None:
        logger.warning(
            "No MLflow run is tagged with %s; the choice stays in its "
            "file only.", checkpoint.parent,
        )
        return
    mlflow.MlflowClient().set_tag(run_id, SELECTED_EPOCH_TAG, str(epoch))
    logger.info("Tagged run %s with epoch %d.", run_id, epoch)


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
    parser.add_argument("--epochs", type=int, required=True, choices=ARMS)
    parser.add_argument(
        "--lr-scale", type=float, default=1.0, choices=LR_SCALES,
        help="Factor on both learning rates the arm trained at.",
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--reports-dir", type=Path, default=None,
        help="Where the arm's validation reports are; every *.json in it "
             "is read. Defaults to <save-root>/eval/val/<run>.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true",
        help="Read a rehearsal: its shorter run, and validation scored "
             "on a sample of the images.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    args.n_epochs = (
        REHEARSAL_EPOCHS[args.epochs] if args.rehearsal else args.epochs
    )
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    run = run_name(args.epochs, args.lr_scale)
    if args.reports_dir is None:
        args.reports_dir = args.save_root / "eval" / "val" / run
    if args.out is None:
        args.out = args.save_root / "eval" / (
            f"selected_e{args.epochs}{lr_suffix(args.lr_scale)}.json"
        )
    args.setting_label = POSTPROCESSING_CONFIGS[
        SELECTION_POSTPROCESSING
    ].label()
    return args


def run(args: argparse.Namespace) -> int:
    """Read the reports, choose and record.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
        Process exit code.
    """
    if args.out.exists():
        logger.error(
            "%s already exists. The epoch is chosen once; remove the file "
            "only if the choice it records was never acted on.", args.out,
        )
        return EXIT_FAILED
    reports = sorted(args.reports_dir.glob("*.json"))
    if not reports:
        raise SelectionError(f"No report in {args.reports_dir}")
    payloads = [json.loads(path.read_text()) for path in reports]
    curve = run_curve(
        snapshot_entries(payloads, args.setting_label),
        run_name(args.epochs, args.lr_scale), args.n_epochs,
    )
    n_scored = next(iter(curve.values()))["n_images"]
    if not args.rehearsal:
        expected = expected_val_images(args.split)
        if n_scored != expected:
            raise SelectionError(
                f"Snapshots were scored on {n_scored} image(s); the "
                f"validation set has {expected}."
            )
    epoch = select_epoch({k: v["f1"] for k, v in curve.items()})
    record = choice_record(args, curve, epoch, reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    logger.info(
        "%s: epoch %d chosen, validation F1 %.4f over %d image(s); "
        "written to %s.", record["run"], epoch, record["val_f1"],
        n_scored, args.out,
    )
    if not args.no_mlflow:
        try:
            tag_training_run(Path(record["checkpoint"]), epoch)
        except Exception:  # noqa: BLE001
            logger.warning("Tagging the run in MLflow failed.", exc_info=True)
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    """Choose one arm's epoch.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    try:
        return run(args)
    except SelectionError as e:
        logger.error("%s", e)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
