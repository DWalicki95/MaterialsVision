#!/usr/bin/env python3
"""
Set the classical watershed segmentations beside the two models.

The models are read from their cross-validation: PEFT-SAM under its
frozen post-processing and Cellpose-SAM under the setting its recipe
records, each pooled over all five folds of ``cv5_v1`` - every image
of the dataset scored once, by a model that never saw its
formulation. The classical methods are read from
``evaluate_morphological_operations``, scored with the same measuring
code on the few images they were run on.

**The classical row is an estimate, and the table says so.** It rests
on ten images of one material and one microscope, against roughly
seven hundred images of three materials for the models. Read across,
the rows say how far apart the approaches are, not by how much to
three decimals; and the classical pipelines had their parameters set
by hand for each image, which favours them.

**Only figures that pool across folds are set side by side.** Counts
are summed and divided once, as for every pooled figure in this
study: precision, recall, F1, pore count error, merges and splits per
hundred annotated pores, recall per size class, and the mean overlap
of matched pairs - the last weighted by the number of pairs, which
makes it the mean over every pair - and the porosity error, a mean
over images, weighted by the number of images.

**The rest is a mean of folds, and labelled so.** Distribution
distances, diameter shifts, per-pair median shape errors and the
orientation statistics need every pore, and a fold report keeps one
value per fold. For the models these are the mean of the five fold
values; for the classical methods, scored in one pass, they are exact.
Boundary F1 cannot be recovered at all from the fold reports, so it is
given for the classical methods alone.

Examples
--------
    $ PYTHONPATH=. python scripts/compare_classical_baseline.py
"""
import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from materials_vision.evaluation import DECISION_SCALE, SIZE_BIN_LABELS
from materials_vision.logging_config import setup_logging
from scripts.compare_cellpose_cv import (cellpose_settings,
                                         load_cellpose_folds,
                                         load_peft_sam_folds)
from scripts.run_cellpose_fold import DEFAULT_SAVE_ROOT
from scripts.select_cellpose_recipe import DEFAULT_RECIPE
from scripts.summarize_cv import (DEFAULT_EVAL_DIR, DEFAULT_FOLDS_DIR, N_FOLDS,
                                  SummaryError, pooled)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_CLASSICAL_SCORES = Path(
    "outputs/morphological_evaluation/classical_scores.json"
)

# Figures given for the classical methods only, because the fold
# reports of the models keep no counts to pool them from.
CLASSICAL_ONLY_FIELDS = (
    "macro_f1", "mean_porosity_error_pp", "wasserstein_um",
    "median_diameter_log_ratio",
)

# Errors in the descriptors a material report is built from, which the
# fold reports keep as one value per fold. Name in the row to the keys
# leading to the figure in a report. The two orientation statistics go
# together: the first detects one preferred axis, the second a pair of
# perpendicular ones, and neither reads correctly without the other.
FOLD_MEAN_FIELDS = {
    "wasserstein_um": ("wasserstein_um",),
    "median_diameter_drift_um": ("median_diameter_drift_um",),
    "iqr_drift_um": ("iqr_drift_um",),
    "median_diameter_error": ("median_diameter_error",),
    "median_elongation_error": ("median_elongation_error",),
    "median_angle_error_deg": ("median_angle_error_deg",),
    "orientation_resultant_difference": (
        "orientation", "resultant_difference",
    ),
    "orientation_grid_resultant_difference": (
        "orientation", "grid_resultant_difference",
    ),
}


def pooled_pair_iou(sections: Sequence[dict[str, Any]]) -> float:
    """Mean overlap of matched pairs over several scored subsets.

    Each subset's mean is weighted by its number of pairs, so the
    result is the mean over every pair, as if scored in one pass.

    Parameters
    ----------
    sections : sequence of dict

    Returns
    -------
    float
        ``nan`` when no subset has a pair.
    """
    total_pairs = 0
    weighted = 0.0
    for section in sections:
        n_pairs = section["n_true_positives"]
        if n_pairs == 0:
            continue
        total_pairs += n_pairs
        weighted += section["mean_pair_iou"] * n_pairs
    return weighted / total_pairs if total_pairs else math.nan


def pooled_size_bins(
    sections: Sequence[dict[str, Any]]
) -> dict[str, float]:
    """Recall per size class over several scored subsets.

    Parameters
    ----------
    sections : sequence of dict
        Each with a ``size_bin_recall`` list of ``label``, ``n_gt`` and
        ``n_matched``.

    Returns
    -------
    dict
        Class label to pooled recall, ``nan`` for an empty class.

    Raises
    ------
    SummaryError
        If a subset lacks its size-class counts, which would pool the
        classes over fewer images than the other figures.
    """
    n_gt = dict.fromkeys(SIZE_BIN_LABELS, 0)
    n_matched = dict.fromkeys(SIZE_BIN_LABELS, 0)
    for section in sections:
        bins = section.get("size_bin_recall")
        if not bins:
            raise SummaryError(
                f"{section.get('label', '?')} has no size-class counts."
            )
        for item in bins:
            n_gt[item["label"]] += item["n_gt"]
            n_matched[item["label"]] += item["n_matched"]
    return {
        label: n_matched[label] / n_gt[label] if n_gt[label] else math.nan
        for label in SIZE_BIN_LABELS
    }


def pooled_porosity_error(sections: Sequence[dict[str, Any]]) -> float:
    """Mean signed porosity error over every image of several subsets.

    Each subset reports the mean over its images, so weighting it by
    its number of images gives the mean over all of them. Porosity is
    defined on every non-empty frame, so no image drops out of a
    subset's mean and the weights are exact.

    Parameters
    ----------
    sections : sequence of dict

    Returns
    -------
    float
        ``nan`` when no subset has a defined value.
    """
    n_images = 0
    weighted = 0.0
    for section in sections:
        value = section["mean_porosity_error_pp"]
        if not math.isfinite(value):
            continue
        n_images += section["n_images"]
        weighted += value * section["n_images"]
    return weighted / n_images if n_images else math.nan


def fold_mean(
    sections: Sequence[dict[str, Any]], path: tuple[str, ...]
) -> float:
    """Mean over subsets of a figure that cannot be pooled from them.

    Medians and distribution distances need every pore, and a fold
    report keeps one value per fold, so a model's figure is the mean
    of its folds' figures - an approximation, labelled as such where
    it is shown. A single subset, as for the classical methods, gives
    its own figure unchanged.

    Parameters
    ----------
    sections : sequence of dict
    path : tuple of str
        Keys leading to the figure, e.g. ``("orientation",
        "resultant_difference")``.

    Returns
    -------
    float
        ``nan`` when no subset has a finite value.

    Raises
    ------
    SummaryError
        If a subset lacks the figure, which would average it over
        fewer folds than the row claims.
    """
    values = []
    for section in sections:
        value: Any = section
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise SummaryError(
                    f"{section.get('label', '?')} has no "
                    f"{'.'.join(path)}."
                )
            value = value[key]
        if value is not None and math.isfinite(value):
            values.append(value)
    return statistics.fmean(values) if values else math.nan


def method_row(
    method: str, population: str, sections: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """One row of the table, pooled from its scored subsets.

    Parameters
    ----------
    method, population : str
    sections : sequence of dict

    Returns
    -------
    dict
        Pooled counts and recall per size class; under
        ``descriptor_errors`` the porosity error pooled over images and
        the figures of ``FOLD_MEAN_FIELDS`` averaged over subsets;
        ``n_sections`` says how many subsets were averaged and
        ``f1_per_section`` gives each one's F1, to show the spread
        behind the pooled figure.
    """
    descriptor_errors = {
        "mean_porosity_error_pp": pooled_porosity_error(sections),
    }
    for name, path in FOLD_MEAN_FIELDS.items():
        descriptor_errors[name] = fold_mean(sections, path)
    return {
        "method": method,
        "population": population,
        **pooled(sections),
        "mean_pair_iou": pooled_pair_iou(sections),
        "size_bin_recall": pooled_size_bins(sections),
        "descriptor_errors": descriptor_errors,
        "n_sections": len(sections),
        "f1_per_section": [pooled([section])["f1"] for section in sections],
    }


def model_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The two models, each pooled over all five folds.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    list of dict

    Raises
    ------
    SummaryError
        If either model is missing a fold: the reference is the whole
        cross-validation, and four folds of it are a different
        population.
    """
    recipe = json.loads(args.recipe.read_text())
    setting = cellpose_settings(recipe)
    cellpose, cellpose_missing = load_cellpose_folds(
        recipe, setting, args.cellpose_eval_dir, args.folds_dir
    )
    peft_sam, peft_sam_missing = load_peft_sam_folds(
        args.peft_sam_eval_dir, args.folds_dir
    )
    if cellpose_missing or peft_sam_missing:
        raise SummaryError(
            f"Folds missing - Cellpose: {cellpose_missing}, PEFT-SAM: "
            f"{peft_sam_missing}."
        )
    population = f"cv5_v1, {N_FOLDS} folds, held-out formulations"
    return [
        method_row("PEFT-SAM (frozen)", population,
                   [scores.overall for scores in peft_sam]),
        method_row(f"Cellpose-SAM ({setting.label()})", population,
                   [scores.overall for scores in cellpose]),
    ]


def classical_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The classical variants, each pooled over its images.

    Parameters
    ----------
    payload : dict
        The report of ``evaluate_morphological_operations``.

    Returns
    -------
    list of dict
        Each row also carries ``classical_only`` figures and the
        ``per_image`` F1.

    Raises
    ------
    SummaryError
        If the report is empty or its variants were scored on
        different annotated pores.
    """
    if not payload:
        raise SummaryError("The classical report holds no variant.")
    populations = {
        variant: (report["overall"]["n_images"], report["overall"]["n_gt"])
        for variant, report in payload.items()
    }
    if len(set(populations.values())) != 1:
        raise SummaryError(
            f"Classical variants were scored on different populations "
            f"(images, pores): {populations}."
        )
    rows = []
    for variant, report in payload.items():
        overall = report["overall"]
        n_images = overall["n_images"]
        row = method_row(
            f"classical {variant}", f"{n_images} images, estimate",
            [overall],
        )
        row["classical_only"] = {
            field: overall[field] for field in CLASSICAL_ONLY_FIELDS
        }
        row["classical_only"]["boundary_f1"] = _decision_boundary_f1(overall)
        row["per_image"] = {
            item["label"]: item["f1"] for item in report["per_image"]
        }
        rows.append(row)
    return rows


def _decision_boundary_f1(overall: dict[str, Any]) -> float:
    """Boundary F1 at the decision tolerance, whichever way it is keyed.

    A report read back from JSON keys its tolerances as strings.
    """
    scores = overall["boundary_f1"]
    for key in (DECISION_SCALE, str(DECISION_SCALE)):
        if key in scores:
            return scores[key]
    return math.nan


def _fmt(value: float, signed: bool = False) -> str:
    """A figure for the table, ``n/a`` where undefined."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def format_markdown(rows: list[dict[str, Any]]) -> str:
    """The comparison as a short report.

    Parameters
    ----------
    rows : list of dict
        Model rows first, then the classical ones.

    Returns
    -------
    str
    """
    bins = " | ".join(f"R {label}" for label in SIZE_BIN_LABELS)
    lines = [
        "# Classical watershed vs PEFT-SAM and Cellpose-SAM",
        "",
        "Models: pooled over all five folds of cv5_v1. Classical: an "
        "estimate on the images it was run on, with parameters set by "
        "hand per image - a different and much smaller population.",
        "",
        f"| method | population | images | pores | F1 | P | R | "
        f"count error | merges/100 | splits/100 | pair IoU | {bins} |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        + "---:|" * len(SIZE_BIN_LABELS),
    ]
    for row in rows:
        recall_bins = " | ".join(
            _fmt(row["size_bin_recall"][label]) for label in SIZE_BIN_LABELS
        )
        lines.append(
            f"| {row['method']} | {row['population']} | "
            f"{row['n_images']} | {row['n_gt']} | {_fmt(row['f1'])} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['pore_count_error'], signed=True)} | "
            f"{row['merges_per_100_gt']:.2f} | "
            f"{row['splits_per_100_gt']:.2f} | "
            f"{_fmt(row['mean_pair_iou'])} | {recall_bins} |"
        )

    classical = [row for row in rows if "classical_only" in row]
    if not classical:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "## Classical only",
        "",
        "| variant | boundary F1 | macro F1 | porosity error (pp) | "
        "Wasserstein (um) | median log(d_pred/d_gt) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in classical:
        extra = row["classical_only"]
        lines.append(
            f"| {row['method']} | {_fmt(extra['boundary_f1'])} | "
            f"{_fmt(extra['macro_f1'])} | "
            f"{_fmt(extra['mean_porosity_error_pp'], signed=True)} | "
            f"{_fmt(extra['wasserstein_um'])} | "
            f"{_fmt(extra['median_diameter_log_ratio'], signed=True)} |"
        )
    image_ids = list(classical[0]["per_image"])
    lines += [
        "",
        "## Classical F1 per image",
        "",
        "| image | " + " | ".join(row["method"] for row in classical)
        + " |",
        "|---|" + "---:|" * len(classical),
    ]
    for image_id in image_ids:
        lines.append(
            f"| {image_id} | "
            + " | ".join(_fmt(row["per_image"][image_id])
                         for row in classical)
            + " |"
        )
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Read the models and the classical report into one table.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
    """
    payload = json.loads(args.classical_scores.read_text())
    return {
        "rows": model_rows(args) + classical_rows(payload),
        "classical_scores": str(args.classical_scores),
    }


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
        "--classical-scores", type=Path, default=DEFAULT_CLASSICAL_SCORES
    )
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
        args.out_dir = args.classical_scores.parent
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
    json_path = args.out_dir / "classical_vs_models.json"
    md_path = args.out_dir / "classical_vs_models.md"
    report = format_markdown(comparison["rows"])
    json_path.write_text(json.dumps(comparison, indent=2, default=str))
    md_path.write_text(report)
    logger.info("Wrote %s and %s.", json_path, md_path)
    logger.info("\n%s", report)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
