#!/usr/bin/env python3
"""
Turn the VALIDATION readings into the Cellpose recipe, mechanically.

Two choices are made on VALIDATION, in this order, and both by the same
rule: the highest pooled instance F1.

1. ``snapshot``: the arm and the epoch together, among every snapshot
   of both selection arms scored under one post-processing setting.
2. ``postprocessing``: the thresholds and the flow-following steps,
   among every setting of the calibration grid scored on the snapshot
   chosen in step one.

Both are written into one record, which every later run reads instead
of being told the answer on its command line. What the final model and
the folds train and score is therefore whatever this record says, and
the record says what the rule picked.

**Two rounds, two records.** The first round chose the snapshot under
the library's post-processing, which follows the flows for 200 steps -
too few for the largest pores of this dataset, as a diagnosis on
VALIDATION showed afterwards. That round is kept as it was, in
``recipe.json``. The second round chooses the snapshot again under
1000 steps and calibrates the thresholds over 1000 and 2000, into
``recipe_v2.json``, which is the record the folds read.

**The rule is applied only to complete readings.** A snapshot missing
from the table would be a candidate that never competed, and snapshots
scored on differing numbers of images would be compared on different
populations; either is refused rather than read. So is a report scored
under a setting other than the one asked for, which the label of every
entry records. An exact tie is refused too: the rule has no answer for
it, and inventing one here would be a choice nobody made.

Examples
--------
    $ python scripts/select_cellpose_recipe.py snapshot --niter 1000 \\
        --readings checkpoints/cellpose/selection/eval/selection_v2.json
    $ python scripts/select_cellpose_recipe.py postprocessing \\
        --niters 1000 2000 \\
        --readings checkpoints/cellpose/selection/eval/calibration_v2.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from materials_vision.experiments.cellpose.split_evaluation import (
    CALIBRATION_NITERS, SELECTION_NITER, CellposeSettings, calibration_grid)
from materials_vision.experiments.cellpose.split_training import (
    LEARNING_RATES, N_EPOCHS, saved_epochs)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from scripts.train_cellpose_split import run_name

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_RECIPE = Path("checkpoints/cellpose/selection/recipe_v2.json")


class SelectionError(ValueError):
    """Raised when the readings cannot be selected from as they stand."""


def overall_f1(payload: dict[str, Any]) -> dict[str, tuple[float, int]]:
    """Pooled F1 and image count of every entry of a report.

    Parameters
    ----------
    payload : dict
        As written by ``scripts/evaluate_cellpose.py``.

    Returns
    -------
    dict of str to (float, int)
    """
    return {
        name: (float(entry["overall"]["f1"]),
               int(entry["overall"]["n_images"]))
        for name, entry in payload.items()
    }


def pick_highest(readings: dict[str, tuple[float, int]]) -> str:
    """The entry with the highest F1, on one population, without a tie.

    Parameters
    ----------
    readings : dict of str to (float, int)

    Returns
    -------
    str

    Raises
    ------
    SelectionError
        If the entries were scored on differing numbers of images, or
        the highest F1 is shared.
    """
    populations = {n_images for _, n_images in readings.values()}
    if len(populations) != 1:
        raise SelectionError(
            f"Entries were scored on differing numbers of images: "
            f"{sorted(populations)}."
        )
    best = max(f1 for f1, _ in readings.values())
    winners = sorted(n for n, (f1, _) in readings.items() if f1 == best)
    if len(winners) > 1:
        raise SelectionError(
            f"The highest F1, {best:.4f}, is shared by {winners}; the "
            f"rule does not decide between them."
        )
    return winners[0]


def expected_snapshots() -> set[str]:
    """Every snapshot of both arms, named as the report names them.

    Returns
    -------
    set of str
    """
    return {
        f"{run_name(arm)}/epoch-{epoch}"
        for arm in LEARNING_RATES
        for epoch in saved_epochs(N_EPOCHS)
    }


def check_scored_under(
    payload: dict[str, Any], setting: CellposeSettings
) -> None:
    """Verify every entry was scored under the setting asked for.

    Parameters
    ----------
    payload : dict
    setting : CellposeSettings

    Raises
    ------
    SelectionError
        If any entry's recorded label names another setting.
    """
    suffix = f"@{setting.label()}"
    wrong = sorted(
        name for name, entry in payload.items()
        if not str(entry["overall"].get("label", "")).endswith(suffix)
    )
    if wrong:
        raise SelectionError(
            f"{len(wrong)} entr(y/ies) were not scored under "
            f"{setting.label()}, e.g. {wrong[:3]}."
        )


def select_snapshot(
    payload: dict[str, Any], setting: CellposeSettings
) -> dict[str, Any]:
    """Choose the arm and the epoch.

    Parameters
    ----------
    payload : dict
        Every snapshot of both arms on VALIDATION, under one setting.
    setting : CellposeSettings
        The setting the snapshots were scored under.

    Returns
    -------
    dict
        The part of the recipe record this choice fills.

    Raises
    ------
    SelectionError
        If the report scored several settings or another one, or does
        not hold exactly the snapshots of both arms.
    """
    if any("@" in name for name in payload):
        raise SelectionError(
            "The snapshot is chosen under one setting; this report holds "
            "several."
        )
    expected = expected_snapshots()
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise SelectionError(
            f"The report must hold every snapshot of both arms and "
            f"nothing else. Missing: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}; unexpected: "
            f"{unexpected[:5]}."
        )
    check_scored_under(payload, setting)
    readings = overall_f1(payload)
    chosen = pick_highest(readings)
    run, epoch = chosen.split("/epoch-")
    arm = next(a for a in LEARNING_RATES if run_name(a) == run)
    return {
        "snapshot": {
            "arm": arm,
            "learning_rate": LEARNING_RATES[arm],
            "completed_epochs": int(epoch),
            "val_f1": readings[chosen][0],
            "n_val_images": readings[chosen][1],
            "selected_under": setting.label(),
            "selection_niter": setting.niter,
            "ranking": _ranking(readings),
        }
    }


def select_postprocessing(
    payload: dict[str, Any],
    recipe: dict[str, Any],
    niters: Sequence[int],
) -> dict[str, Any]:
    """Choose the thresholds and steps on the snapshot already chosen.

    Parameters
    ----------
    payload : dict
        The chosen snapshot on VALIDATION, under the whole grid.
    recipe : dict
        The record holding the snapshot choice.
    niters : sequence of int
        The steps the grid crossed the thresholds with.

    Returns
    -------
    dict
        The part of the recipe record this choice fills.

    Raises
    ------
    SelectionError
        If the snapshot has not been chosen yet, or the report is not
        the whole grid on exactly that snapshot.
    """
    if "snapshot" not in recipe:
        raise SelectionError(
            "Choose the snapshot first; the thresholds are calibrated "
            "on it."
        )
    chosen = recipe["snapshot"]
    stem = f"{run_name(chosen['arm'])}/epoch-{chosen['completed_epochs']}"
    grid = calibration_grid(niters)
    expected = {f"{stem}@{s.label()}" for s in grid}
    if set(payload) != expected:
        raise SelectionError(
            f"The report must hold the {len(expected)} grid settings on "
            f"{stem} and nothing else."
        )
    readings = overall_f1(payload)
    winner = pick_highest(readings)
    setting = next(s for s in grid if winner == f"{stem}@{s.label()}")
    selected_under = f"{stem}@{chosen['selected_under']}"
    return {
        "postprocessing": {
            "flow_threshold": setting.flow_threshold,
            "cellprob_threshold": setting.cellprob_threshold,
            "min_size_px": setting.min_size_px,
            "niter": setting.niter,
            "label": setting.label(),
            "grid_niters": list(niters),
            "val_f1": readings[winner][0],
            "val_f1_selected_under": (
                readings[selected_under][0]
                if selected_under in readings else None
            ),
            "ranking": _ranking(readings),
        }
    }


def _ranking(readings: dict[str, tuple[float, int]]) -> list[list[Any]]:
    """Every entry with its F1, best first, for the record."""
    return [
        [name, f1]
        for name, (f1, _) in sorted(
            readings.items(), key=lambda item: item[1][0], reverse=True
        )
    ]


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
    parser.add_argument("step", choices=["snapshot", "postprocessing"])
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--niter", type=int, default=SELECTION_NITER,
        help="Steps the snapshots were scored under (snapshot step).",
    )
    parser.add_argument(
        "--niters", type=int, nargs="+", default=list(CALIBRATION_NITERS),
        help="Steps the grid crossed (postprocessing step).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Apply one step of the selection and record it.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    payload = json.loads(args.readings.read_text())
    recipe = (
        json.loads(args.recipe.read_text()) if args.recipe.exists() else {}
    )
    if args.step in recipe:
        logger.error(
            "%s already records a %s choice; a choice is made once.",
            args.recipe, args.step,
        )
        return EXIT_FAILED
    try:
        if args.step == "snapshot":
            chosen = select_snapshot(
                payload, CellposeSettings(niter=args.niter)
            )
        else:
            chosen = select_postprocessing(payload, recipe, args.niters)
    except SelectionError as error:
        logger.error("%s", error)
        return EXIT_FAILED
    chosen[args.step]["readings"] = str(args.readings)
    chosen[args.step]["provenance"] = run_provenance()
    recipe.update(chosen)
    args.recipe.parent.mkdir(parents=True, exist_ok=True)
    args.recipe.write_text(json.dumps(recipe, indent=2, default=str))
    logger.info(
        "%s chosen: %s. Recorded in %s.", args.step,
        {k: v for k, v in chosen[args.step].items()
         if k not in ("ranking", "provenance")},
        args.recipe,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
