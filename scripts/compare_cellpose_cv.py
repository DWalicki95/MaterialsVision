#!/usr/bin/env python3
"""
Set the Cellpose folds beside the PEFT-SAM folds.

Both recipes were trained on the same five folds and scored on the same
held-out formulations with the same measuring code, so each fold gives
a paired reading: the same images, two recipes. This reads the fold
reports of both and writes the comparison, descriptively - nothing is
chosen from it.

**What is paired with what.** Cellpose under the thresholds calibrated
on VALIDATION against PEFT-SAM under its frozen post-processing, on
all five folds; fold one is TEST for both. The Cellpose variant with
the physical area filter of the PEFT-SAM deployment is reported
alongside for Cellpose alone, since PEFT-SAM has no score under its
deployed post-processing on fold one.

**How it is read.** Each recipe's own figures come first: its instance
F1 on every fold, their mean, spread and range across the folds, and
its F1 pooled over all of them and per material. The difference is
given beside them, never instead of them. Pooled figures are
recomputed from summed counts, as the PEFT-SAM cross-validation was,
never averaged from per-fold figures. Five folds are too few for a
test that would mean much, and the folds share most of their training
images, so their differences are not independent draws; the per-fold
figures describe how consistent the difference is rather than test it.

Examples
--------
    $ python scripts/compare_cellpose_cv.py
"""
import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from materials_vision.experiments.cellpose.split_evaluation import \
    CellposeSettings
from materials_vision.logging_config import setup_logging
from scripts.run_cellpose_fold import DEFAULT_SAVE_ROOT
from scripts.run_cellpose_fold import run_name as cellpose_run_name
from scripts.select_cellpose_recipe import DEFAULT_RECIPE
from scripts.summarize_cv import (DEFAULT_EVAL_DIR, DEFAULT_FOLDS_DIR,
                                  MATERIALS, N_FOLDS, FoldScores, SummaryError,
                                  fold_source, load_fold, pooled, summarize)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

PEFT_SAM_POSTPROCESSING = "frozen"


def cellpose_settings(recipe: dict[str, Any]) -> CellposeSettings:
    """The calibrated setting the recipe records.

    Parameters
    ----------
    recipe : dict

    Returns
    -------
    CellposeSettings
    """
    chosen = recipe["postprocessing"]
    return CellposeSettings(
        flow_threshold=float(chosen["flow_threshold"]),
        cellprob_threshold=float(chosen["cellprob_threshold"]),
        min_size_px=int(chosen["min_size_px"]),
        niter=int(chosen["niter"]),
    )


def load_cellpose_folds(
    recipe: dict[str, Any],
    setting: CellposeSettings,
    eval_dir: Path,
    folds_dir: Path,
) -> tuple[list[FoldScores], list[int]]:
    """Every Cellpose fold scored under one setting.

    Parameters
    ----------
    recipe : dict
    setting : CellposeSettings
    eval_dir, folds_dir : Path

    Returns
    -------
    tuple
        The folds found, and the numbers of those missing.
    """
    epoch = int(recipe["snapshot"]["completed_epochs"])
    found, missing = [], []
    for fold in range(1, N_FOLDS + 1):
        scores = load_fold(
            fold, eval_dir / f"fold{fold}.json",
            f"{cellpose_run_name(fold)}/epoch-{epoch}", setting.label(),
            None, folds_dir / f"fold{fold}_metadata.json",
        )
        (found if scores else missing).append(scores or fold)
    return found, missing


def load_peft_sam_folds(
    eval_dir: Path, folds_dir: Path
) -> tuple[list[FoldScores], list[int]]:
    """Every PEFT-SAM fold under its frozen post-processing.

    Parameters
    ----------
    eval_dir, folds_dir : Path

    Returns
    -------
    tuple
        The folds found, and the numbers of those missing.
    """
    label = POSTPROCESSING_CONFIGS[PEFT_SAM_POSTPROCESSING].label()
    found, missing = [], []
    for fold in range(1, N_FOLDS + 1):
        report, stem, bare = fold_source(fold, eval_dir)
        scores = load_fold(
            fold, report, stem, label, bare,
            folds_dir / f"fold{fold}_metadata.json",
        )
        (found if scores else missing).append(scores or fold)
    return found, missing


def spread(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation and range of per-fold figures.

    Parameters
    ----------
    values : list of float

    Returns
    -------
    dict
        ``sd`` is ``nan`` for a single fold, which has no spread.
    """
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else math.nan,
        "min": min(values),
        "max": max(values),
    }


def paired(
    cellpose: list[FoldScores], peft_sam: list[FoldScores]
) -> dict[str, Any]:
    """The paired reading over the folds both recipes have.

    Parameters
    ----------
    cellpose, peft_sam : list of FoldScores

    Returns
    -------
    dict

    Raises
    ------
    SummaryError
        If no fold is scored for both, or a fold's two reports do not
        cover the same number of annotated pores - which would mean
        they did not score the same images.
    """
    by_fold = {scores.fold: scores for scores in peft_sam}
    pairs = [
        (scores, by_fold[scores.fold])
        for scores in cellpose if scores.fold in by_fold
    ]
    if not pairs:
        raise SummaryError("No fold is scored for both recipes.")
    rows = []
    for ours, theirs in pairs:
        if ours.overall["n_gt"] != theirs.overall["n_gt"]:
            raise SummaryError(
                f"Fold {ours.fold}: Cellpose was scored on "
                f"{ours.overall['n_gt']} annotated pores and PEFT-SAM on "
                f"{theirs.overall['n_gt']}; they are not the same images."
            )
        f1_cellpose = pooled([ours.overall])["f1"]
        f1_peft_sam = pooled([theirs.overall])["f1"]
        rows.append({
            "fold": ours.fold,
            "f1_cellpose": f1_cellpose,
            "f1_peft_sam": f1_peft_sam,
            "delta_f1": f1_cellpose - f1_peft_sam,
        })
    deltas = [row["delta_f1"] for row in rows]
    included = [row["fold"] for row in rows]
    ours_sub = summarize([s for s, _ in pairs], "cellpose")
    theirs_sub = summarize([t for _, t in pairs], PEFT_SAM_POSTPROCESSING)
    by_material = {}
    for material in MATERIALS:
        a = ours_sub["pooled_by_material"].get(material)
        b = theirs_sub["pooled_by_material"].get(material)
        if a and b:
            by_material[material] = {
                "f1_cellpose": a["f1"], "f1_peft_sam": b["f1"],
                "delta_f1": a["f1"] - b["f1"], "n_gt": a["n_gt"],
            }
    return {
        "folds_included": included,
        "per_fold": rows,
        "f1_across_folds": {
            "cellpose": spread([row["f1_cellpose"] for row in rows]),
            "peft_sam": spread([row["f1_peft_sam"] for row in rows]),
        },
        "delta_across_folds": {
            **spread(deltas),
            "n_cellpose_ahead": sum(d > 0 for d in deltas),
        },
        "pooled": {
            "cellpose": ours_sub["pooled"],
            "peft_sam": theirs_sub["pooled"],
            "delta_f1": ours_sub["pooled"]["f1"] - theirs_sub["pooled"]["f1"],
        },
        "pooled_by_material": by_material,
    }


def format_markdown(comparison: dict[str, Any]) -> str:
    """The comparison as a short report.

    Parameters
    ----------
    comparison : dict

    Returns
    -------
    str
    """
    main = comparison["paired"]
    pooled_row = main["pooled"]
    lines = [
        "# Cellpose-SAM vs PEFT-SAM on cv5_v1",
        "",
        f"Cellpose `{comparison['cellpose_setting']}` vs PEFT-SAM "
        f"`{PEFT_SAM_POSTPROCESSING}`; folds {main['folds_included']}.",
        "",
        "## Instance F1 of each recipe",
        "",
        "| fold | Cellpose F1 | PEFT-SAM F1 | difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in main["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['f1_cellpose']:.4f} | "
            f"{row['f1_peft_sam']:.4f} | {row['delta_f1']:+.4f} |"
        )
    lines.append(
        f"| **pooled** | **{pooled_row['cellpose']['f1']:.4f}** | "
        f"**{pooled_row['peft_sam']['f1']:.4f}** | "
        f"{pooled_row['delta_f1']:+.4f} |"
    )
    across = main["f1_across_folds"]
    lines += [
        "",
        "| across folds | Cellpose | PEFT-SAM |",
        "|---|---:|---:|",
    ]
    for key in ("mean", "sd", "min", "max"):
        lines.append(
            f"| {key} | {across['cellpose'][key]:.4f} | "
            f"{across['peft_sam'][key]:.4f} |"
        )
    lines += [
        "",
        "| pooled | Cellpose | PEFT-SAM |",
        "|---|---:|---:|",
    ]
    for key in ("precision", "recall", "pore_count_error",
                "merges_per_100_gt", "splits_per_100_gt"):
        lines.append(
            f"| {key} | {pooled_row['cellpose'][key]:.4f} | "
            f"{pooled_row['peft_sam'][key]:.4f} |"
        )
    delta = main["delta_across_folds"]
    lines += [
        "",
        f"Per-fold difference: mean {delta['mean']:+.4f}, SD "
        f"{delta['sd']:.4f}, range {delta['min']:+.4f} to "
        f"{delta['max']:+.4f}; Cellpose ahead in "
        f"{delta['n_cellpose_ahead']} of {len(main['per_fold'])}.",
        "",
        "## Per material, pooled over the folds",
        "",
        "| material | Cellpose F1 | PEFT-SAM F1 | difference | pores |",
        "|---|---:|---:|---:|---:|",
    ]
    for material, row in main["pooled_by_material"].items():
        lines.append(
            f"| {material} | {row['f1_cellpose']:.4f} | "
            f"{row['f1_peft_sam']:.4f} | {row['delta_f1']:+.4f} | "
            f"{row['n_gt']} |"
        )
    filtered = comparison.get("cellpose_area_filter")
    if filtered:
        lines += [
            "",
            f"Cellpose alone with the area filter "
            f"(`{filtered['setting']}`): pooled F1 "
            f"{filtered['pooled']['f1']:.4f} over folds "
            f"{filtered['folds_included']}.",
        ]
    missing = comparison["missing"]
    if any(missing.values()):
        lines += ["", f"Missing folds: {missing}."]
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Read both sets of folds and pair them.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
    """
    recipe = json.loads(args.recipe.read_text())
    setting = cellpose_settings(recipe)
    ours, ours_missing = load_cellpose_folds(
        recipe, setting, args.cellpose_eval_dir, args.folds_dir
    )
    theirs, theirs_missing = load_peft_sam_folds(
        args.peft_sam_eval_dir, args.folds_dir
    )
    comparison: dict[str, Any] = {
        "cellpose_setting": setting.label(),
        "peft_sam_postprocessing": PEFT_SAM_POSTPROCESSING,
        "paired": paired(ours, theirs),
        "missing": {"cellpose": ours_missing, "peft_sam": theirs_missing},
    }
    filtered_setting = setting.with_area_filter()
    filtered, _ = load_cellpose_folds(
        recipe, filtered_setting, args.cellpose_eval_dir, args.folds_dir
    )
    if filtered:
        comparison["cellpose_area_filter"] = {
            "setting": filtered_setting.label(),
            "folds_included": [scores.fold for scores in filtered],
            "pooled": pooled([scores.overall for scores in filtered]),
        }
    return comparison


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
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--cellpose-eval-dir", type=Path, default=DEFAULT_SAVE_ROOT / "eval"
    )
    parser.add_argument(
        "--peft-sam-eval-dir", type=Path, default=DEFAULT_EVAL_DIR
    )
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.cellpose_eval_dir
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Write the comparison.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    try:
        comparison = compare(args)
    except (SummaryError, FileNotFoundError, KeyError) as error:
        logger.error("%s", error)
        return EXIT_FAILED
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "cellpose_vs_peft_sam.json"
    md_path = args.out_dir / "cellpose_vs_peft_sam.md"
    json_path.write_text(json.dumps(comparison, indent=2, default=str))
    md_path.write_text(format_markdown(comparison))
    logger.info("Wrote %s and %s.", json_path, md_path)
    logger.info("\n%s", format_markdown(comparison))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
