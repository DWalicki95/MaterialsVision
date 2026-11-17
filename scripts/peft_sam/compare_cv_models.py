#!/usr/bin/env python3
"""
Set base models beside each other on the same cross-validation folds.

Every model compared was trained by ``train_cv_fold.py`` on the same
five folds, with the same recipe and budget, and scored on the same
held-out formulations by the same measuring code. Each fold therefore
gives a paired reading - the same images, two starting points - and so
does each held-out formulation. This reads every model's folds exactly
as ``summarize_cv.py`` reads them and writes the comparison against a
reference model.

**What decides.** Instance F1 at a mask overlap of one half, pooled
over pores, and nothing else. A model beats the reference when all
three hold:

1. its F1, averaged over the five folds, is at least half a point
   above the reference's;
2. it is ahead in at least four of the five folds;
3. a Wilcoxon signed-rank test over the paired per-formulation F1
   differences gives p below 0.05, in the model's favour.

Anything else confirms the reference or is a tie. A comparison with
fewer than five paired folds gets no verdict.

**What is only reported.** Recall of the smallest pores (Q1), the
signed error in pore count and F1 for each material are given for
every model as diagnostics. They do not change the verdict. Neither do
the mean without fold two, which is VALIDATION and on which the recipe
was selected, or the zero-shot generalists, which are read alongside
when their fold reports are given.

**How the figures are formed.** Pooled figures are recomputed from
summed counts, never averaged from per-fold figures. The folds share
most of their training images, so their differences are not
independent draws: the per-fold figures say how consistent a
difference is, and the per-formulation test is the closest to a test
the design allows.

Examples
--------
The three base models at 24 epochs, under the setting calibrated in
each fold, with the zero-shot generalists alongside:
    $ python scripts/peft_sam/compare_cv_models.py \\
        --reference vit_l_lm --model vit_l_em_organelles --model vit_l \\
        --decoder-lr vit_l=1e-4 --epochs 24 \\
        --zero-shot vit_l_em_organelles --zero-shot vit_l_lm \\
        --out checkpoints/cv_e24/base_models_paired.json
"""
import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from scipy.stats import wilcoxon

from materials_vision.logging_config import setup_logging
from materials_vision.training import (BASE_MODELS,
                                       BASE_MODELS_WITHOUT_DECODER,
                                       DECODER_LEARNING_RATE)
from scripts.peft_sam.summarize_cv import (CALIBRATED, DEFAULT_FOLDS_DIR,
                                           MATERIALS, N_FOLDS,
                                           POSTPROCESSING_CHOICES, FoldScores,
                                           SummaryError, formulation_rows,
                                           load_folds, pooled, reading,
                                           summarize)
from scripts.peft_sam.train_cv_fold import VALIDATION_FOLD
from scripts.peft_sam.train_final_model import N_EPOCHS

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_EVAL_ROOT = Path("checkpoints/cv_e24/eval")

DEFAULT_CALIBRATION_ROOT = Path("checkpoints/cv_e24/calibration")

# Where the zero-shot reports sit under the evaluation root.
ZERO_SHOT_DIR_NAME = "zero_shot"

# The two settings a zero-shot generalist is read under: the one the
# study froze, and the library's own, tuned for the untrained decoder.
ZERO_SHOT_SETTINGS = ("frozen", "library_default")

MIN_MEAN_GAIN_F1 = 0.005

MIN_FOLDS_AHEAD = 4

ALPHA = 0.05

SMALLEST_BIN = "Q1"

VERDICT_MODEL_WINS = "model_wins"
VERDICT_REFERENCE_HOLDS = "reference_holds"
VERDICT_INCOMPLETE = "incomplete"


def parse_decoder_rate(text: str) -> tuple[str, float]:
    """Read one ``model=rate`` pair from the command line.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple of str and float

    Raises
    ------
    argparse.ArgumentTypeError
        For anything but a known base model and a positive rate.
    """
    model, _, rate = text.partition("=")
    try:
        value = float(rate)
    except ValueError:
        value = math.nan
    if model not in BASE_MODELS or not value > 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a base model and a positive rate, like "
            f"vit_l=1e-4."
        )
    return model, value


def model_reading(
    base_model: str, args: argparse.Namespace
) -> argparse.Namespace:
    """How one trained model's folds are found.

    Parameters
    ----------
    base_model : str
    args : argparse.Namespace

    Returns
    -------
    argparse.Namespace
        For :func:`scripts.peft_sam.summarize_cv.load_folds`.
    """
    return reading(
        args.eval_root / base_model, args.folds_dir, args.postprocessing,
        base_model=base_model, epochs=args.epochs,
        decoder_lr=args.decoder_lr.get(base_model, DECODER_LEARNING_RATE),
        calibration_dir=args.calibration_root / base_model,
    )


def smallest_bin_recall(folds: Sequence[FoldScores]) -> float:
    """Recall of the smallest pores, pooled over the folds.

    Parameters
    ----------
    folds : sequence of FoldScores

    Returns
    -------
    float
        ``nan`` when no fold reports the bin.
    """
    n_gt = n_matched = 0
    for scores in folds:
        for section in scores.overall.get("size_bin_recall") or []:
            if section["label"] == SMALLEST_BIN:
                n_gt += section["n_gt"]
                n_matched += section["n_matched"]
    return n_matched / n_gt if n_gt else math.nan


def model_figures(
    folds: Sequence[FoldScores], postprocessing: str
) -> dict[str, Any]:
    """One model's own figures, before any difference is taken.

    Parameters
    ----------
    folds : sequence of FoldScores
    postprocessing : str

    Returns
    -------
    dict
    """
    summary = summarize(folds, postprocessing)
    return {
        "folds_included": summary["folds_included"],
        "f1_per_fold": {
            row["fold"]: row["f1"] for row in summary["per_fold"]
        },
        "across_folds": summary["across_folds"],
        "pooled_f1": summary["pooled"]["f1"],
        "pooled_f1_by_material": {
            material: figures["f1"]
            for material, figures in summary["pooled_by_material"].items()
        },
        "pore_count_error": summary["pooled"]["pore_count_error"],
        "q1_recall": smallest_bin_recall(folds),
    }


def paired_folds(
    model: Sequence[FoldScores], reference: Sequence[FoldScores]
) -> list[tuple[FoldScores, FoldScores]]:
    """The folds both models have, checked to be the same images.

    Parameters
    ----------
    model, reference : sequence of FoldScores

    Returns
    -------
    list of tuple
        In fold order.

    Raises
    ------
    SummaryError
        If a fold's two reports cover different numbers of annotated
        pores, which means they did not score the same images.
    """
    by_fold = {scores.fold: scores for scores in reference}
    pairs = []
    for scores in sorted(model, key=lambda s: s.fold):
        other = by_fold.get(scores.fold)
        if other is None:
            continue
        if scores.overall["n_gt"] != other.overall["n_gt"]:
            raise SummaryError(
                f"Fold {scores.fold}: {scores.overall['n_gt']} annotated "
                f"pores against {other.overall['n_gt']} for the "
                f"reference; they are not the same images."
            )
        pairs.append((scores, other))
    return pairs


def formulation_test(
    pairs: Sequence[tuple[FoldScores, FoldScores]]
) -> dict[str, Any]:
    """Paired per-formulation F1 differences and their signed-rank test.

    Parameters
    ----------
    pairs : sequence of tuple
        From :func:`paired_folds`.

    Returns
    -------
    dict
        ``p_value`` is ``nan`` when there is nothing to test: no
        formulation resolved for both, or every difference zero.
    """
    deltas = {}
    for model, reference in pairs:
        theirs = {
            row["formulation"]: row["f1"]
            for row in formulation_rows(reference)
        }
        for row in formulation_rows(model):
            if row["formulation"] in theirs:
                deltas[row["formulation"]] = (
                    row["f1"] - theirs[row["formulation"]]
                )
    values = list(deltas.values())
    p_value = math.nan
    if any(value != 0 for value in values):
        p_value = float(wilcoxon(values).pvalue)
    return {
        "n_formulations": len(values),
        "median_delta_f1": statistics.median(values) if values else math.nan,
        "n_model_ahead": sum(value > 0 for value in values),
        "p_value": p_value,
        "delta_f1": dict(sorted(deltas.items())),
    }


def verdict(
    n_pairs: int, mean_delta: float, n_ahead: int, test: dict[str, Any]
) -> str:
    """Whether the model beats the reference, by the stated criterion.

    Parameters
    ----------
    n_pairs : int
        Folds paired.
    mean_delta : float
        Model minus reference, F1 averaged over the folds.
    n_ahead : int
        Folds in which the model is ahead.
    test : dict
        From :func:`formulation_test`.

    Returns
    -------
    str
    """
    if n_pairs < N_FOLDS:
        return VERDICT_INCOMPLETE
    if (mean_delta >= MIN_MEAN_GAIN_F1
            and n_ahead >= MIN_FOLDS_AHEAD
            and test["p_value"] < ALPHA
            and test["median_delta_f1"] > 0):
        return VERDICT_MODEL_WINS
    return VERDICT_REFERENCE_HOLDS


def compare_model(
    model: Sequence[FoldScores], reference: Sequence[FoldScores]
) -> dict[str, Any]:
    """The paired reading of one model against the reference.

    Parameters
    ----------
    model, reference : sequence of FoldScores

    Returns
    -------
    dict
    """
    pairs = paired_folds(model, reference)
    rows = []
    for ours, theirs in pairs:
        f1_model = pooled([ours.overall])["f1"]
        f1_reference = pooled([theirs.overall])["f1"]
        rows.append({
            "fold": ours.fold,
            "f1_model": f1_model,
            "f1_reference": f1_reference,
            "delta_f1": f1_model - f1_reference,
        })
    deltas = [row["delta_f1"] for row in rows]
    without_validation = [
        row["delta_f1"] for row in rows if row["fold"] != VALIDATION_FOLD
    ]
    mean_delta = statistics.fmean(deltas) if deltas else math.nan
    n_ahead = sum(delta > 0 for delta in deltas)
    test = formulation_test(pairs)
    return {
        "folds_paired": [row["fold"] for row in rows],
        "per_fold": rows,
        "mean_delta_f1": mean_delta,
        "n_folds_model_ahead": n_ahead,
        "mean_delta_f1_without_validation_fold": (
            statistics.fmean(without_validation)
            if without_validation else math.nan
        ),
        "per_formulation": test,
        "verdict": verdict(len(rows), mean_delta, n_ahead, test),
    }


def load_model(
    reading_args: argparse.Namespace, allow_missing: bool, name: str
) -> list[FoldScores]:
    """One model's folds, refusing gaps unless they are allowed.

    Parameters
    ----------
    reading_args : argparse.Namespace
    allow_missing : bool
    name : str
        For the messages.

    Returns
    -------
    list of FoldScores

    Raises
    ------
    SummaryError
        If folds are missing and that is not allowed, or none is there.
    """
    folds, missing, _ = load_folds(reading_args)
    if missing and not allow_missing:
        raise SummaryError(
            f"{name}: no score under {reading_args.postprocessing} for "
            f"fold(s) {missing} in {reading_args.eval_dir}. Score them "
            f"first, or pass --allow-missing."
        )
    if not folds:
        raise SummaryError(
            f"{name}: no fold has a score in {reading_args.eval_dir}."
        )
    return folds


def zero_shot_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Every zero-shot generalist under each of its two settings.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
        Generalist to setting to its figures; a setting without every
        fold scored is left out with a warning.
    """
    found: dict[str, Any] = {}
    directory = args.eval_root / ZERO_SHOT_DIR_NAME
    for name in args.zero_shot:
        for setting in ZERO_SHOT_SETTINGS:
            folds, missing, _ = load_folds(reading(
                directory, args.folds_dir, setting, pretrained=name,
            ))
            if missing:
                logger.warning(
                    "Zero-shot %s under %s: fold(s) %s not scored in %s; "
                    "left out.", name, setting, missing, directory,
                )
                continue
            found.setdefault(name, {})[setting] = model_figures(
                folds, setting
            )
    return found


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Read every model and set each beside the reference.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict

    Raises
    ------
    SummaryError
        If a model's folds cannot be read or paired.
    """
    reference = load_model(
        model_reading(args.reference, args), args.allow_missing,
        args.reference,
    )
    models = {
        name: load_model(model_reading(name, args), args.allow_missing, name)
        for name in args.model
    }
    return {
        "postprocessing": args.postprocessing,
        "epochs": args.epochs,
        "reference": args.reference,
        "decoder_learning_rates": {
            name: args.decoder_lr.get(name, DECODER_LEARNING_RATE)
            for name in (args.reference, *args.model)
        },
        "criterion": {
            "min_mean_gain_f1": MIN_MEAN_GAIN_F1,
            "min_folds_ahead": MIN_FOLDS_AHEAD,
            "of_folds": N_FOLDS,
            "alpha": ALPHA,
        },
        "validation_fold": VALIDATION_FOLD,
        "figures": {
            name: model_figures(folds, args.postprocessing)
            for name, folds in {args.reference: reference, **models}.items()
        },
        "against_reference": {
            name: compare_model(folds, reference)
            for name, folds in models.items()
        },
        "zero_shot": zero_shot_figures(args),
    }


def _num(value: float, digits: int = 4) -> str:
    """A figure, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def _pp(value: float) -> str:
    """A difference of fractions in percentage points."""
    return "-" if math.isnan(value) else f"{100 * value:+.2f} pp"


def _pct(value: float) -> str:
    """A signed fraction as a percentage."""
    return "-" if math.isnan(value) else f"{100 * value:+.1f}%"


VERDICT_TEXT = {
    VERDICT_MODEL_WINS: "model lepszy od referencji",
    VERDICT_REFERENCE_HOLDS: "referencja potwierdzona albo remis",
    VERDICT_INCOMPLETE: "brak werdyktu (niepelna CV)",
}


def _figures_rows(figures: dict[str, Any]) -> list[str]:
    """One table row per model, its own figures."""
    rows = []
    for name, own in figures.items():
        per_fold = " | ".join(
            _num(own["f1_per_fold"].get(fold, math.nan))
            for fold in range(1, N_FOLDS + 1)
        )
        rows.append(
            f"| {name} | {per_fold} | {_num(own['across_folds']['f1_mean'])} "
            f"| {_num(own['across_folds']['f1_mean_without_validation_fold'])}"
            f" | {_num(own['pooled_f1'])} |"
        )
    return rows


def _diagnostic_rows(figures: dict[str, Any]) -> list[str]:
    """One table row per model, its diagnostics."""
    rows = []
    for name, own in figures.items():
        by_material = " | ".join(
            _num(own["pooled_f1_by_material"].get(m, math.nan))
            for m in MATERIALS
        )
        rows.append(
            f"| {name} | {_num(own['q1_recall'])} | "
            f"{_pct(own['pore_count_error'])} | {by_material} |"
        )
    return rows


def format_markdown(comparison: dict[str, Any]) -> str:
    """The comparison as Markdown tables.

    Parameters
    ----------
    comparison : dict

    Returns
    -------
    str
    """
    folds = " | ".join(f"fold {k}" for k in range(1, N_FOLDS + 1))
    lines = [
        f"# Modele bazowe w CV - {comparison['epochs']} epok, "
        f"postprocessing `{comparison['postprocessing']}`",
        "",
        f"Referencja: `{comparison['reference']}`. Metryka decyzyjna: F1 "
        f"instancji (IoU >= 0.5, sumowane po porach).",
        "",
        "## F1 kazdego modelu",
        "",
        f"| model | {folds} | srednia | srednia bez foldu "
        f"{comparison['validation_fold']} (VAL) | zbiorczo |",
        "|---|" + "---:|" * (N_FOLDS + 3),
        *_figures_rows(comparison["figures"]),
        "",
        "## Wzgledem referencji",
        "",
        "| model | sr. delta F1 | foldy na plus | delta bez foldu 2 "
        "| mediana delta (formulacje) | formulacje na plus | p Wilcoxona "
        "| werdykt |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, against in comparison["against_reference"].items():
        test = against["per_formulation"]
        lines.append(
            f"| {name} | {_pp(against['mean_delta_f1'])} | "
            f"{against['n_folds_model_ahead']}/"
            f"{len(against['folds_paired'])} | "
            f"{_pp(against['mean_delta_f1_without_validation_fold'])} | "
            f"{_pp(test['median_delta_f1'])} | {test['n_model_ahead']}/"
            f"{test['n_formulations']} | {_num(test['p_value'], 3)} | "
            f"{VERDICT_TEXT[against['verdict']]} |"
        )
    criterion = comparison["criterion"]
    lines += [
        "",
        f"Model wygrywa, gdy srednia delta >= "
        f"{_pp(criterion['min_mean_gain_f1'])}, jest na plus w >= "
        f"{criterion['min_folds_ahead']}/{criterion['of_folds']} foldach "
        f"i Wilcoxon po formulacjach daje p < {criterion['alpha']} na "
        f"jego korzysc.",
        "",
        "## Diagnostyka (bez prawa weta)",
        "",
        "| model | recall Q1 | blad liczby porow | "
        + " | ".join(f"F1 {m}" for m in MATERIALS) + " |",
        "|---|---:|---:|" + "---:|" * len(MATERIALS),
        *_diagnostic_rows(comparison["figures"]),
    ]
    if comparison["zero_shot"]:
        lines += [
            "",
            "## Zero-shot",
            "",
            f"| model | ustawienie | {folds} | srednia | zbiorczo |",
            "|---|---|" + "---:|" * (N_FOLDS + 2),
        ]
        for name, settings in comparison["zero_shot"].items():
            for setting, own in settings.items():
                per_fold = " | ".join(
                    _num(own["f1_per_fold"].get(fold, math.nan))
                    for fold in range(1, N_FOLDS + 1)
                )
                lines.append(
                    f"| {name} | {setting} | {per_fold} | "
                    f"{_num(own['across_folds']['f1_mean'])} | "
                    f"{_num(own['pooled_f1'])} |"
                )
    return "\n".join(lines) + "\n"


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why the comparison may not be made as asked, or nothing.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    if not args.model:
        return "Name at least one --model to set beside the reference."
    if args.reference in args.model:
        return f"{args.reference} is the reference; it is not a --model."
    for name in (args.reference, *args.model):
        if (name in BASE_MODELS_WITHOUT_DECODER
                and name not in args.decoder_lr):
            return (
                f"{name} trains its decoder from scratch at its own rate; "
                f"name it with --decoder-lr {name}=<rate>, as its folds "
                f"were named."
            )
    for name in args.zero_shot:
        if name in BASE_MODELS_WITHOUT_DECODER:
            return f"{name} ships no instance decoder to score zero-shot."
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
    parser.add_argument(
        "--reference", choices=BASE_MODELS, default="vit_l_lm",
    )
    parser.add_argument(
        "--model", choices=BASE_MODELS, action="append", default=[],
        help="A base model to set beside the reference; repeatable.",
    )
    parser.add_argument(
        "--decoder-lr", type=parse_decoder_rate, action="append",
        default=[],
        help="model=rate for a model whose folds were trained at a "
             "decoder rate other than the recipe's; repeatable.",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument(
        "--postprocessing", default=CALIBRATED,
        choices=POSTPROCESSING_CHOICES,
        help=f"Default {CALIBRATED}: each fold under the setting "
             f"calibrated on its own training images.",
    )
    parser.add_argument(
        "--zero-shot", choices=BASE_MODELS, action="append", default=[],
        help="A generalist whose zero-shot fold reports to read "
             "alongside; repeatable.",
    )
    parser.add_argument(
        "--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
        help="Holds one directory of fold reports per model, and "
             f"{ZERO_SHOT_DIR_NAME}/ for the zero-shot ones.",
    )
    parser.add_argument(
        "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT,
        help="Holds one directory of calibration reports per model.",
    )
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Compare the folds scored so far; no verdict is given.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSON report; a Markdown copy is written beside it.",
    )
    args = parser.parse_args(argv)
    args.decoder_lr = dict(args.decoder_lr)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Compare the models.

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
        comparison = compare(args)
    except SummaryError as e:
        logger.error("%s", e)
        return EXIT_FAILED
    markdown = format_markdown(comparison)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(comparison, indent=2, default=str))
        args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")
        logger.info("Wrote %s and %s.", args.out, args.out.with_suffix(".md"))
    logger.info("\n%s", markdown)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
