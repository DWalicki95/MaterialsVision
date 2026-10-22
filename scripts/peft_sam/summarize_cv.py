#!/usr/bin/env python3
"""
Read the cross-validation folds together.

Each fold's model was scored once, on the formulations it held out, and
the scores were written to a JSON report per fold. This joins them into
the three readings the folds exist for, all descriptive - nothing is
chosen from them:

1. **Instance F1 per fold**, with its mean, spread and range across the
   five, and where the final model's TEST score (fold one) falls in it.
2. **Each material pooled over every fold.** The materials of the
   second microscope are one formulation per fold, so this is the only
   figure for them that rests on more than one synthesis.
3. **Per formulation**: instance F1 and the signed error in pore count,
   to show whether an error belongs to the model or to a formulation.

**Pooled means pooled counts.** Figures across folds are recomputed
from the summed counts - true and false positives, false negatives,
merges, splits, instances - exactly as one evaluation pools images,
never averaged from per-fold figures: a fold with twice the pores
weighs twice as much, as it would had all the images been scored in
one pass.

**Fold one comes from the final model's TEST report.** TEST was scored
once, under the frozen post-processing, before the folds existed, and
is not opened again. So fold one has no per-formulation breakdown -
where a material has a single held-out formulation in it, its material
figure is that formulation's figure; the three AS formulations of TEST
stay unresolved - and it has no score under any other post-processing,
where it is left out and the report says so.

Examples
--------
Summarize under the frozen post-processing, the primary reading:
    $ python scripts/peft_sam/summarize_cv.py

The deployed post-processing alongside, folds two to five only:
    $ python scripts/peft_sam/summarize_cv.py --postprocessing deployed
"""
import argparse
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import configure_tracking, ensure_experiment
from scripts.peft_sam.train_cv_fold import (CV_ID, DEFAULT_FOLDS_DIR,
                                            FINAL_MODEL_FOLD, N_FOLDS,
                                            run_name)
from scripts.peft_sam.train_final_model import N_EPOCHS

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

ANALYSIS_EXPERIMENT = "peft_sam_analysis"

DEFAULT_EVAL_DIR = Path("checkpoints/cv/eval")

FINAL_MODEL_SCORES = Path("checkpoints/final/test_scores.json")

FINAL_MODEL_ENTRY = "full_b_trainval_seed20260907/epoch-12"

# The only post-processing TEST was scored under. Its entry carries no
# setting in its name because it was the only one scored in that pass.
FINAL_MODEL_POSTPROCESSING = "frozen"

MATERIALS = ("AS", "K", "VAB")

COUNT_FIELDS = (
    "n_images", "n_gt", "n_pred", "n_true_positives",
    "n_false_positives", "n_false_negatives", "n_merges", "n_splits",
)

# Colour follows the material everywhere in the figure; the pooled
# figure is drawn in ink, as it belongs to no material.
MATERIAL_COLORS = {"AS": "#2a78d6", "K": "#eb6834", "VAB": "#1baf7a"}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e4e3df"


class SummaryError(ValueError):
    """Raised when the fold reports cannot be read together."""


@dataclass(frozen=True)
class FoldScores:
    """One fold's figures, as its evaluation reported them.

    Parameters
    ----------
    fold : int
    overall : dict
    per_material : list of dict
    per_formulation : list of dict or None
        ``None`` where the report predates the breakdown.
    held_out : dict of str to list of str
        Material to the formulations the fold held out.
    """

    fold: int
    overall: dict[str, Any]
    per_material: list[dict[str, Any]]
    per_formulation: Optional[list[dict[str, Any]]]
    held_out: dict[str, list[str]]


def pooled(sections: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum the counts of several scored subsets and recompute from them.

    Parameters
    ----------
    sections : sequence of dict
        Each carries the fields of ``COUNT_FIELDS``.

    Returns
    -------
    dict
        The summed counts, and precision, recall, F1, merges and splits
        per hundred annotated pores and the signed pore count error,
        each ``nan`` where its denominator is zero.

    Raises
    ------
    SummaryError
        If no section is given or one lacks a count.
    """
    if not sections:
        raise SummaryError("Nothing to pool.")
    totals: dict[str, Any] = {}
    for field in COUNT_FIELDS:
        missing = [s.get("label", "?") for s in sections if field not in s]
        if missing:
            raise SummaryError(f"{missing} carry no {field!r}")
        totals[field] = sum(int(s[field]) for s in sections)
    tp = totals["n_true_positives"]
    fp = totals["n_false_positives"]
    fn = totals["n_false_negatives"]
    n_gt = totals["n_gt"]
    totals.update({
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "merges_per_100_gt": _ratio(100 * totals["n_merges"], n_gt),
        "splits_per_100_gt": _ratio(100 * totals["n_splits"], n_gt),
        "pore_count_error": _ratio(totals["n_pred"] - n_gt, n_gt),
    })
    return totals


def _ratio(numerator: float, denominator: float) -> float:
    """Guarded ratio; a zero denominator gives ``nan``."""
    return float("nan") if denominator == 0 else numerator / denominator


def find_entry(
    payload: dict[str, Any],
    stem: str,
    label: str,
    bare_setting: Optional[str],
) -> Optional[dict[str, Any]]:
    """Pick one snapshot's figures under one setting from a report.

    Parameters
    ----------
    payload : dict
        A JSON report of ``evaluate_checkpoint.py``.
    stem : str
        ``run/epoch-N``.
    label : str
        The setting's label, as the report appends it after ``@``.
    bare_setting : str, optional
        The label an entry without ``@`` was scored under, when that is
        known; a bare entry is otherwise never taken, since it does not
        say which setting it holds.

    Returns
    -------
    dict or None
    """
    entry = payload.get(f"{stem}@{label}")
    if entry is None and bare_setting == label:
        entry = payload.get(stem)
    return entry


def fold_source(
    fold: int, eval_dir: Path
) -> tuple[Path, str, Optional[str]]:
    """Where a fold's figures are, and what a bare entry there means.

    Parameters
    ----------
    fold : int
    eval_dir : Path

    Returns
    -------
    report : Path
    stem : str
    bare_setting : str or None
    """
    if fold == FINAL_MODEL_FOLD:
        return (
            FINAL_MODEL_SCORES, FINAL_MODEL_ENTRY,
            POSTPROCESSING_CONFIGS[FINAL_MODEL_POSTPROCESSING].label(),
        )
    return (
        eval_dir / f"fold{fold}.json",
        f"{run_name(fold)}/epoch-{N_EPOCHS}",
        None,
    )


def load_fold(
    fold: int,
    report: Path,
    stem: str,
    label: str,
    bare_setting: Optional[str],
    metadata_path: Path,
) -> Optional[FoldScores]:
    """Read one fold's figures and check they belong to that fold.

    Parameters
    ----------
    fold : int
    report, metadata_path : Path
    stem, label : str
    bare_setting : str, optional

    Returns
    -------
    FoldScores or None
        ``None`` when the report or the setting is absent.

    Raises
    ------
    SummaryError
        If the fold metadata is missing, or the report scored a
        different number of images than the fold holds out - which is
        what a report of another fold, or of a partial run, looks like.
    """
    if not metadata_path.exists():
        raise SummaryError(f"No fold metadata at {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not report.exists():
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    entry = find_entry(payload, stem, label, bare_setting)
    if entry is None:
        return None
    expected = metadata["counts"]["n_test_images"]
    scored = entry["overall"]["n_images"]
    if scored != expected:
        raise SummaryError(
            f"{report} scored {scored} image(s) for fold {fold}, which "
            f"holds out {expected}; it is not a full score of this fold."
        )
    return FoldScores(
        fold=fold,
        overall=entry["overall"],
        per_material=entry["per_material"],
        per_formulation=entry.get("per_formulation"),
        held_out=metadata["held_out"],
    )


def formulation_rows(scores: FoldScores) -> list[dict[str, Any]]:
    """The fold's figures per formulation, as far as they can be told.

    Parameters
    ----------
    scores : FoldScores

    Returns
    -------
    list of dict
        One per resolvable formulation, with ``formulation``,
        ``material``, ``fold`` and the pooled figures.
    """
    material_of = {
        name: material
        for material, names in scores.held_out.items() for name in names
    }
    if scores.per_formulation is not None:
        sections = {
            s["label"].split("=", 1)[1]: s for s in scores.per_formulation
        }
    else:
        sections = {}
        for section in scores.per_material:
            names = scores.held_out.get(section["label"].split("=", 1)[1])
            if names and len(names) == 1:
                sections[names[0]] = section
    return [
        {
            "formulation": name,
            "material": material_of[name],
            "fold": scores.fold,
            **pooled([section]),
        }
        for name, section in sorted(sections.items())
    ]


def material_sections(
    folds: Sequence[FoldScores], material: str
) -> list[dict[str, Any]]:
    """Every fold's section for one material.

    Parameters
    ----------
    folds : sequence of FoldScores
    material : str

    Returns
    -------
    list of dict
    """
    label = f"material={material}"
    return [
        section for scores in folds for section in scores.per_material
        if section["label"] == label
    ]


def summarize(
    folds: Sequence[FoldScores], postprocessing: str
) -> dict[str, Any]:
    """The three readings, from the folds that could be read.

    Parameters
    ----------
    folds : sequence of FoldScores
    postprocessing : str

    Returns
    -------
    dict
    """
    per_fold = []
    for scores in folds:
        row = {
            "fold": scores.fold,
            "held_out": scores.held_out,
            **pooled([scores.overall]),
        }
        for material in MATERIALS:
            sections = material_sections([scores], material)
            row[f"f1_{material}"] = (
                pooled(sections)["f1"] if sections else float("nan")
            )
        per_fold.append(row)
    f1s = [row["f1"] for row in per_fold]
    return {
        "cv_id": CV_ID,
        "postprocessing": postprocessing,
        "folds_included": [scores.fold for scores in folds],
        "per_fold": per_fold,
        "across_folds": {
            "f1_mean": statistics.fmean(f1s),
            "f1_sd": statistics.stdev(f1s) if len(f1s) > 1 else float("nan"),
            "f1_min": min(f1s),
            "f1_max": max(f1s),
        },
        "pooled": pooled([scores.overall for scores in folds]),
        "pooled_by_material": {
            material: pooled(material_sections(folds, material))
            for material in MATERIALS
            if material_sections(folds, material)
        },
        "per_formulation": [
            row for scores in folds for row in formulation_rows(scores)
        ],
    }


def _pct(value: float) -> str:
    """A signed fraction as a percentage, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{100 * value:+.1f}%"


def _num(value: float, digits: int = 4) -> str:
    """A figure, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def format_markdown(summary: dict[str, Any], missing: list[int]) -> str:
    """The readings as Markdown tables.

    Parameters
    ----------
    summary : dict
    missing : list of int
        Folds with no score under this post-processing.

    Returns
    -------
    str
    """
    lines = [
        f"# Walidacja krzyzowa `{summary['cv_id']}` - "
        f"postprocessing `{summary['postprocessing']}`",
        "",
        f"Foldy w odczycie: {summary['folds_included']}"
        + (f"; bez wyniku: {missing}" if missing else ""),
        "",
        "## F1 per fold",
        "",
        "| fold | obrazow | porow GT | F1 | P | R | skl./100 | rozc./100 "
        "| blad liczby porow | F1 AS | F1 K | F1 VAB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['n_images']} | {row['n_gt']} | "
            f"{_num(row['f1'])} | {_num(row['precision'])} | "
            f"{_num(row['recall'])} | {_num(row['merges_per_100_gt'], 2)} | "
            f"{_num(row['splits_per_100_gt'], 2)} | "
            f"{_pct(row['pore_count_error'])} | "
            + " | ".join(_num(row[f"f1_{m}"]) for m in MATERIALS) + " |"
        )
    across = summary["across_folds"]
    lines += [
        "",
        f"Miedzy foldami: srednia {_num(across['f1_mean'])}, SD "
        f"{_num(across['f1_sd'])}, zakres {_num(across['f1_min'])}-"
        f"{_num(across['f1_max'])}.",
        "",
        "## Zbiorczo (sumy licznikow po foldach)",
        "",
        "| | porow GT | F1 | P | R | skl./100 | rozc./100 "
        "| blad liczby porow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("wszystko", summary["pooled"])] + list(
        summary["pooled_by_material"].items()
    )
    for name, figures in rows:
        lines.append(
            f"| {name} | {figures['n_gt']} | {_num(figures['f1'])} | "
            f"{_num(figures['precision'])} | {_num(figures['recall'])} | "
            f"{_num(figures['merges_per_100_gt'], 2)} | "
            f"{_num(figures['splits_per_100_gt'], 2)} | "
            f"{_pct(figures['pore_count_error'])} |"
        )
    lines += [
        "",
        "## Per formulacja",
        "",
        "| formulacja | material | fold | obrazow | porow GT | F1 "
        "| blad liczby porow |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _formulation_order(summary["per_formulation"]):
        lines.append(
            f"| {row['formulation']} | {row['material']} | {row['fold']} | "
            f"{row['n_images']} | {row['n_gt']} | {_num(row['f1'])} | "
            f"{_pct(row['pore_count_error'])} |"
        )
    return "\n".join(lines) + "\n"


def _formulation_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Grouped by material, best F1 first within a material."""
    rank = {material: i for i, material in enumerate(MATERIALS)}
    return sorted(
        rows, key=lambda r: (rank.get(r["material"], 99), -r["f1"])
    )


def plot(summary: dict[str, Any], path: Path) -> Path:
    """F1 per fold above; F1 and pore count error per formulation below.

    Parameters
    ----------
    summary : dict
    path : Path

    Returns
    -------
    Path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _formulation_order(summary["per_formulation"])[::-1]
    height_in = 2.8 + 0.22 * max(len(rows), 8)
    fig = plt.figure(figsize=(10, height_in))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(2.2, height_in - 2.8), hspace=0.25,
        wspace=0.08,
    )
    by_fold = fig.add_subplot(grid[0, :])
    f1_ax = fig.add_subplot(grid[1, 0])
    count_ax = fig.add_subplot(grid[1, 1], sharey=f1_ax)

    folds = [row["fold"] for row in summary["per_fold"]]
    offsets = {"AS": -0.15, "K": 0.0, "VAB": 0.15}
    by_fold.scatter(
        folds, [row["f1"] for row in summary["per_fold"]], s=90,
        marker="_", color=INK, linewidths=2, label="wszystko", zorder=3,
    )
    for material in MATERIALS:
        by_fold.scatter(
            [f + offsets[material] for f in folds],
            [row[f"f1_{material}"] for row in summary["per_fold"]],
            s=40, color=MATERIAL_COLORS[material], label=material,
            edgecolors="white", linewidths=1, zorder=4,
        )
    by_fold.axhline(
        summary["pooled"]["f1"], color=MUTED, linewidth=1, linestyle="--",
    )
    by_fold.text(
        max(folds) + 1.05, summary["pooled"]["f1"],
        f"zbiorczo {summary['pooled']['f1']:.3f}", color=MUTED,
        fontsize=8, va="bottom", ha="right",
    )
    by_fold.set_xticks(folds)
    by_fold.set_xticklabels(
        [f"{f}\n(TEST v1)" if f == FINAL_MODEL_FOLD else str(f)
         for f in folds]
    )
    by_fold.set_xlim(min(folds) - 0.5, max(folds) + 1.1)
    by_fold.set_ylabel("F1 instancji")
    by_fold.legend(
        frameon=False, fontsize=8, ncol=4, loc="lower left",
        bbox_to_anchor=(0, 1.0),
    )

    positions = range(len(rows))
    colors = [MATERIAL_COLORS.get(r["material"], MUTED) for r in rows]
    f1_ax.scatter(
        [r["f1"] for r in rows], positions, s=40, color=colors,
        edgecolors="white", linewidths=1, zorder=3,
    )
    count_ax.scatter(
        [100 * r["pore_count_error"] for r in rows], positions, s=40,
        color=colors, edgecolors="white", linewidths=1, zorder=3,
    )
    count_ax.axvline(0, color=MUTED, linewidth=1)
    for material, figures in summary["pooled_by_material"].items():
        span = [i for i, r in enumerate(rows) if r["material"] == material]
        if span:
            f1_ax.vlines(
                figures["f1"], min(span) - 0.4, max(span) + 0.4,
                color=MUTED, linewidth=1, linestyle="--",
            )
    f1_ax.set_yticks(list(positions))
    f1_ax.set_yticklabels(
        [f"{r['formulation']} (f{r['fold']})" for r in rows], fontsize=8
    )
    count_ax.tick_params(axis="y", labelleft=False)
    f1_ax.set_xlabel("F1 instancji (kreska: material zbiorczo)")
    count_ax.set_xlabel("blad liczby porow [%]")

    for axis in (by_fold, f1_ax, count_ax):
        axis.grid(axis="x" if axis is not by_fold else "y", color=GRID,
                  linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=INK, labelcolor=INK)
    fig.suptitle(
        f"Walidacja krzyzowa {summary['cv_id']}, postprocessing "
        f"{summary['postprocessing']}", x=0.01, ha="left", fontsize=10,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def record(summary: dict[str, Any], files: list[Path]) -> str:
    """One MLflow run holding the readings and the files.

    Parameters
    ----------
    summary : dict
    files : list of Path

    Returns
    -------
    str
        The run's identifier.
    """
    import mlflow

    configure_tracking()
    metrics = {
        f"fold{row['fold']}/f1": row["f1"] for row in summary["per_fold"]
    }
    metrics.update({
        f"across_folds/{k}": v for k, v in summary["across_folds"].items()
    })
    for key in ("f1", "precision", "recall", "pore_count_error",
                "merges_per_100_gt", "splits_per_100_gt"):
        metrics[f"pooled/{key}"] = summary["pooled"][key]
        for material, figures in summary["pooled_by_material"].items():
            metrics[f"pooled/{material}/{key}"] = figures[key]
    with mlflow.start_run(
        experiment_id=ensure_experiment(ANALYSIS_EXPERIMENT),
        run_name=f"{CV_ID}_{summary['postprocessing']}",
        tags={"part": "cv", "cv_id": CV_ID,
              "postprocessing": summary["postprocessing"],
              "folds": ",".join(map(str, summary["folds_included"]))},
    ) as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
        for path in files:
            mlflow.log_artifact(str(path))
        return run.info.run_id


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
        "--postprocessing", default="frozen",
        choices=sorted(POSTPROCESSING_CONFIGS),
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Summarize the folds scored so far instead of refusing.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Summarize the folds.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    label = POSTPROCESSING_CONFIGS[args.postprocessing].label()

    folds: list[FoldScores] = []
    missing: list[int] = []
    try:
        for fold in range(1, N_FOLDS + 1):
            report, stem, bare_setting = fold_source(fold, args.eval_dir)
            scores = load_fold(
                fold, report, stem, label, bare_setting,
                args.folds_dir / f"fold{fold}_metadata.json",
            )
            if scores is None:
                missing.append(fold)
            else:
                folds.append(scores)
    except SummaryError as e:
        logger.error("%s", e)
        return EXIT_FAILED

    expected_missing = (
        [] if args.postprocessing == FINAL_MODEL_POSTPROCESSING
        else [FINAL_MODEL_FOLD]
    )
    unexpected = sorted(set(missing) - set(expected_missing))
    if expected_missing and FINAL_MODEL_FOLD in missing:
        logger.info(
            "Fold %d has no score under %s: TEST was scored once, under "
            "%s, and is not opened again.", FINAL_MODEL_FOLD,
            args.postprocessing, FINAL_MODEL_POSTPROCESSING,
        )
    if unexpected and not args.allow_missing:
        logger.error(
            "No score under %s for fold(s) %s. Score them first, or pass "
            "--allow-missing to summarize what is there.",
            args.postprocessing, unexpected,
        )
        return EXIT_FAILED
    if not folds:
        logger.error("No fold has a score under %s.", args.postprocessing)
        return EXIT_FAILED

    summary = summarize(folds, args.postprocessing)
    stem = args.eval_dir / f"{CV_ID}_summary_{args.postprocessing}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(
        {**summary, "folds_missing": missing}, indent=2
    ))
    markdown = format_markdown(summary, missing)
    md_path = stem.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    png_path = plot(summary, stem.with_suffix(".png"))
    logger.info("Wrote %s, %s and %s.\n%s", json_path, md_path.name,
                png_path.name, markdown)
    if not args.no_mlflow:
        run_id = record(summary, [json_path, md_path, png_path])
        logger.info("Recorded in MLflow run %s.", run_id)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
