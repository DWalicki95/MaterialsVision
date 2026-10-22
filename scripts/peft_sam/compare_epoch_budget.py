#!/usr/bin/env python3
"""
Set the two arms of the epoch-budget study side by side.

Each arm's epoch was chosen on validation and its chosen snapshot
scored once on the formulations fold four holds out. This reads both
and reports three things, all descriptive:

1. **The held-out difference.** Instance F1 of the fifty-epoch arm's
   chosen snapshot minus that of the twelve-epoch arm's, read against
   the seed noise of this recipe. That noise was measured as the range
   of instance F1 across three seeds of the same recipe on VALIDATION -
   the mean of epochs ten to twelve, 0.8605 to 0.8658 - so a
   difference smaller than 0.0053 is one a change of seed alone could
   produce. Each arm here has one seed, and a single snapshot varies
   somewhat more than a mean of three, so the band is if anything
   generous to a difference.
2. **The validation curves.** Whether instance F1 on validation still
   rises after the twelfth epoch, and where each arm's best epoch fell.
3. **Where any difference sits**: per material and per formulation,
   with the signed error in pore count and the merges and splits per
   hundred annotated pores beside instance F1.

Figures are recomputed from summed counts, as every pooled figure in
this study is.

Examples
--------
The primary reading, under the frozen post-processing:
    $ python scripts/peft_sam/compare_epoch_budget.py

The deployed post-processing alongside:
    $ python scripts/peft_sam/compare_epoch_budget.py \\
        --postprocessing deployed
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (REHEARSAL_EXPERIMENT,
                                       configure_tracking, ensure_experiment,
                                       log_metric_series)
from scripts.peft_sam.create_epoch_budget_split import BUDGET_ID, FOLD
from scripts.peft_sam.select_epoch_on_val import snapshot_entries
from scripts.peft_sam.summarize_cv import (ANALYSIS_EXPERIMENT, GRID, INK,
                                           MATERIALS, MUTED, pooled)
from scripts.peft_sam.train_epoch_budget import (ARMS, DEFAULT_SAVE_ROOT,
                                                 DEFAULT_SPLIT,
                                                 REHEARSAL_SAVE_ROOT, run_name)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

# The range of instance F1 across three seeds of the recipe; see the
# module docstring.
DELTA_SIG = 0.0053

SHORT_ARM, LONG_ARM = ARMS

# Two categorical hues kept apart from the material colours of the
# cross-validation figures, so an arm is never read as a material.
ARM_COLORS = {SHORT_ARM: "#6250d6", LONG_ARM: "#e34948"}

DELTA_FIELDS = (
    "f1", "precision", "recall", "pore_count_error", "merges_per_100_gt",
    "splits_per_100_gt",
)


class ComparisonError(ValueError):
    """Raised when the two arms cannot be read together."""


def load_choice(path: Path, arm: int) -> dict[str, Any]:
    """One arm's choice record, checked to belong to that arm.

    Parameters
    ----------
    path : Path
    arm : int

    Returns
    -------
    dict

    Raises
    ------
    ComparisonError
    """
    if not path.exists():
        raise ComparisonError(
            f"No choice for the {arm}-epoch arm at {path}; it is made by "
            f"select_epoch_on_val.py before the held-out set is scored."
        )
    choice = json.loads(path.read_text(encoding="utf-8"))
    if choice.get("run") != run_name(arm):
        raise ComparisonError(
            f"{path} records run {choice.get('run')!r}, not "
            f"{run_name(arm)!r}."
        )
    return choice


def held_out_entry(
    payload: Mapping[str, Any], choice: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """The chosen snapshot's held-out figures under one setting.

    Parameters
    ----------
    payload : mapping
        The held-out report.
    choice : mapping
        From :func:`load_choice`.
    label : str
        The setting's label.

    Returns
    -------
    dict

    Raises
    ------
    ComparisonError
        If the report holds no figures for the chosen snapshot under
        this setting, or holds another snapshot of the same run - which
        would mean the held-out set saw more than the chosen one.
    """
    entries = snapshot_entries([payload], label)
    chosen = Path(choice["checkpoint"])
    same_run = [p for p in entries if p.parent.name == chosen.parent.name]
    others = [p for p in same_run if p.name != chosen.name]
    if others:
        raise ComparisonError(
            f"The held-out report scores {[p.name for p in others]} of "
            f"{chosen.parent.name} besides the chosen {chosen.name}."
        )
    if not same_run:
        raise ComparisonError(
            f"The held-out report has no figures for {chosen} under "
            f"{label}."
        )
    return entries[same_run[0]][0]


def section_figures(
    sections: list[Mapping[str, Any]], prefix: str
) -> dict[str, dict[str, Any]]:
    """Pooled figures of each labelled section, by name.

    Parameters
    ----------
    sections : list of mapping
        Labelled like ``"material=K"``.
    prefix : str
        ``"material"`` or ``"formulation"``.

    Returns
    -------
    dict of str to dict
    """
    return {
        s["label"].split("=", 1)[1]: pooled([s])
        for s in sections if s["label"].startswith(f"{prefix}=")
    }


def arm_reading(
    choice: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """One arm's figures: its choice and its held-out score.

    Parameters
    ----------
    choice, entry : mapping

    Returns
    -------
    dict
    """
    return {
        "run": choice["run"],
        "selected_epoch": choice["selected_epoch"],
        "val_f1": choice["val_f1"],
        "held_out": pooled([entry["overall"]]),
        "by_material": section_figures(entry["per_material"], "material"),
        "by_formulation": section_figures(
            entry.get("per_formulation", []), "formulation"
        ),
    }


def deltas(
    long: Mapping[str, Any], short: Mapping[str, Any]
) -> dict[str, float]:
    """Long arm minus short arm, field by field.

    Parameters
    ----------
    long, short : mapping
        Pooled figures.

    Returns
    -------
    dict of str to float
    """
    return {field: long[field] - short[field] for field in DELTA_FIELDS}


def verdict(delta_f1: float) -> str:
    """Read a difference in F1 against the seed noise.

    Parameters
    ----------
    delta_f1 : float

    Returns
    -------
    str
    """
    if math.isnan(delta_f1):
        return "undefined"
    if delta_f1 > DELTA_SIG:
        return "long_arm_higher"
    if delta_f1 < -DELTA_SIG:
        return "long_arm_lower"
    return "within_seed_noise"


def curve_reading(choice: Mapping[str, Any]) -> dict[str, Any]:
    """Where the validation curve of an arm peaked, before and after 12.

    Parameters
    ----------
    choice : mapping

    Returns
    -------
    dict
        The best epoch and F1 up to the short budget and, for a longer
        run, after it.
    """
    curve = {int(k): v["f1"] for k, v in choice["curve"].items()}
    reading: dict[str, Any] = {}
    for name, epochs in (
        ("up_to_short", [k for k in curve if k <= SHORT_ARM]),
        ("after_short", [k for k in curve if k > SHORT_ARM]),
    ):
        if epochs:
            best = min(epochs, key=lambda k: (-curve[k], k))
            reading[name] = {"epoch": best, "f1": curve[best]}
    if "up_to_short" in reading and "after_short" in reading:
        reading["gain_after_short"] = (
            reading["after_short"]["f1"] - reading["up_to_short"]["f1"]
        )
    return reading


def compare(
    choices: Mapping[int, Mapping[str, Any]],
    payload: Mapping[str, Any],
    postprocessing: str,
) -> dict[str, Any]:
    """The three readings.

    Parameters
    ----------
    choices : mapping of int to mapping
        Arm to its choice record.
    payload : mapping
        The held-out report.
    postprocessing : str

    Returns
    -------
    dict
    """
    label = POSTPROCESSING_CONFIGS[postprocessing].label()
    arms = {
        arm: arm_reading(choice, held_out_entry(payload, choice, label))
        for arm, choice in choices.items()
    }
    populations = {a["held_out"]["n_images"] for a in arms.values()}
    if len(populations) > 1:
        raise ComparisonError(
            f"The arms were scored on differing numbers of held-out "
            f"images {sorted(populations)}."
        )
    long, short = arms[LONG_ARM], arms[SHORT_ARM]
    overall = deltas(long["held_out"], short["held_out"])
    return {
        "budget_id": BUDGET_ID,
        "fold": FOLD,
        "postprocessing": postprocessing,
        "delta_sig": DELTA_SIG,
        "arms": {str(arm): reading for arm, reading in arms.items()},
        "delta": overall,
        "verdict": verdict(overall["f1"]),
        "delta_by_material": {
            m: deltas(long["by_material"][m], short["by_material"][m])
            for m in MATERIALS
            if m in long["by_material"] and m in short["by_material"]
        },
        "delta_by_formulation": {
            name: deltas(long["by_formulation"][name], figures)
            for name, figures in sorted(short["by_formulation"].items())
            if name in long["by_formulation"]
        },
        "validation_curves": {
            str(arm): curve_reading(choice)
            for arm, choice in choices.items()
        },
    }


def _num(value: float, digits: int = 4) -> str:
    """A figure, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def _signed(value: float, digits: int = 4) -> str:
    """A signed figure, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{value:+.{digits}f}"


def _pct(value: float) -> str:
    """A signed fraction as a percentage, or a dash for ``nan``."""
    return "-" if math.isnan(value) else f"{100 * value:+.1f}%"


VERDICT_TEXT = {
    "long_arm_higher": "50 epok wyzej, ponad prog szumu seeda",
    "long_arm_lower": "50 epok nizej, ponad prog szumu seeda",
    "within_seed_noise": "roznica w granicach szumu seeda",
    "undefined": "nieokreslone",
}


def format_markdown(result: Mapping[str, Any]) -> str:
    """The readings as Markdown tables.

    Parameters
    ----------
    result : mapping
        From :func:`compare`.

    Returns
    -------
    str
    """
    arms = result["arms"]
    short, long = arms[str(SHORT_ARM)], arms[str(LONG_ARM)]
    lines = [
        f"# Budzet epok {SHORT_ARM} vs {LONG_ARM} - fold {result['fold']}, "
        f"postprocessing `{result['postprocessing']}`",
        "",
        f"Odczyt glowny: dF1 = {_signed(result['delta']['f1'])} "
        f"(prog {result['delta_sig']}) - {VERDICT_TEXT[result['verdict']]}.",
        "",
        "## Zbior odlozony",
        "",
        "| ramie | wybrana epoka | F1 VAL | F1 | P | R | skl./100 "
        "| rozc./100 | blad liczby porow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, reading in ((SHORT_ARM, short), (LONG_ARM, long)):
        h = reading["held_out"]
        lines.append(
            f"| {arm} ep. | {reading['selected_epoch']} | "
            f"{_num(reading['val_f1'])} | {_num(h['f1'])} | "
            f"{_num(h['precision'])} | {_num(h['recall'])} | "
            f"{_num(h['merges_per_100_gt'], 2)} | "
            f"{_num(h['splits_per_100_gt'], 2)} | "
            f"{_pct(h['pore_count_error'])} |"
        )
    d = result["delta"]
    lines += [
        f"| roznica | | | {_signed(d['f1'])} | {_signed(d['precision'])} | "
        f"{_signed(d['recall'])} | {_signed(d['merges_per_100_gt'], 2)} | "
        f"{_signed(d['splits_per_100_gt'], 2)} | "
        f"{_pct(d['pore_count_error'])} |",
        "",
        "## Krzywe walidacyjne",
        "",
        f"| ramie | najlepsza do ep. {SHORT_ARM} | F1 | najlepsza po ep. "
        f"{SHORT_ARM} | F1 | przyrost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        c = result["validation_curves"][str(arm)]
        up, after = c.get("up_to_short"), c.get("after_short")
        lines.append(
            f"| {arm} ep. | {up['epoch'] if up else '-'} | "
            f"{_num(up['f1']) if up else '-'} | "
            f"{after['epoch'] if after else '-'} | "
            f"{_num(after['f1']) if after else '-'} | "
            f"{_signed(c['gain_after_short']) if after else '-'} |"
        )
    lines += [
        "",
        "## Per material",
        "",
        f"| material | F1 {SHORT_ARM} ep. | F1 {LONG_ARM} ep. | dF1 "
        "| d skl./100 | d rozc./100 | d blad liczby porow |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for material, d in result["delta_by_material"].items():
        lines.append(
            f"| {material} | {_num(short['by_material'][material]['f1'])} | "
            f"{_num(long['by_material'][material]['f1'])} | "
            f"{_signed(d['f1'])} | {_signed(d['merges_per_100_gt'], 2)} | "
            f"{_signed(d['splits_per_100_gt'], 2)} | "
            f"{_pct(d['pore_count_error'])} |"
        )
    lines += [
        "",
        "## Per formulacja",
        "",
        f"| formulacja | F1 {SHORT_ARM} ep. | F1 {LONG_ARM} ep. | dF1 "
        f"| blad liczby porow {SHORT_ARM} ep. | blad liczby porow "
        f"{LONG_ARM} ep. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in result["delta_by_formulation"].items():
        s, g = short["by_formulation"][name], long["by_formulation"][name]
        lines.append(
            f"| {name} | {_num(s['f1'])} | {_num(g['f1'])} | "
            f"{_signed(d['f1'])} | {_pct(s['pore_count_error'])} | "
            f"{_pct(g['pore_count_error'])} |"
        )
    return "\n".join(lines) + "\n"


def _style(axis: Any, grid_axis: str) -> None:
    """Recessive grid and axes, as in the cross-validation figures."""
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=INK, labelcolor=INK)


def plot(
    result: Mapping[str, Any],
    choices: Mapping[int, Mapping[str, Any]],
    path: Path,
) -> Path:
    """Validation curves on the left; held-out F1 per material right.

    Parameters
    ----------
    result : mapping
    choices : mapping of int to mapping
    path : Path

    Returns
    -------
    Path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (curves, held) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": (1.6, 1)}
    )
    for arm in ARMS:
        curve = {int(k): v["f1"] for k, v in choices[arm]["curve"].items()}
        epochs = sorted(curve)
        color = ARM_COLORS[arm]
        curves.plot(
            epochs, [curve[k] for k in epochs], color=color, linewidth=2,
            label=f"{arm} epok",
        )
        chosen = choices[arm]["selected_epoch"]
        curves.scatter(
            [chosen], [curve[chosen]], s=64, color=color,
            edgecolors="white", linewidths=2, zorder=4,
        )
        curves.annotate(
            f"wybrana: ep. {chosen}", (chosen, curve[chosen]),
            xytext=(0, 8), textcoords="offset points", ha="center",
            fontsize=8, color=INK,
        )
    curves.axvline(SHORT_ARM, color=MUTED, linewidth=1, linestyle="--")
    curves.set_xlabel("epoka")
    curves.set_ylabel("F1 instancji na walidacji (frozen)")
    curves.legend(frameon=False, fontsize=8, loc="lower right")
    _style(curves, "y")

    arms = result["arms"]
    rows = ["wszystko"] + [
        m for m in MATERIALS if m in result["delta_by_material"]
    ]
    positions = list(range(len(rows)))[::-1]
    for arm in ARMS:
        reading = arms[str(arm)]
        values = [reading["held_out"]["f1"]] + [
            reading["by_material"][m]["f1"] for m in rows[1:]
        ]
        held.scatter(
            values, positions, s=64, color=ARM_COLORS[arm],
            edgecolors="white", linewidths=2, zorder=4, label=f"{arm} epok",
        )
    for y, row in zip(positions, rows):
        pair = [
            arms[str(arm)]["held_out"]["f1"] if row == "wszystko"
            else arms[str(arm)]["by_material"][row]["f1"] for arm in ARMS
        ]
        held.plot(pair, [y, y], color=MUTED, linewidth=1, zorder=3)
    held.set_yticks(positions)
    held.set_yticklabels(rows)
    held.set_xlabel(
        f"F1 instancji, odlozone formulacje ({result['postprocessing']})"
    )
    held.legend(frameon=False, fontsize=8, loc="lower right")
    _style(held, "x")

    fig.suptitle(
        f"Budzet epok, fold {result['fold']}: dF1 = "
        f"{_signed(result['delta']['f1'])} "
        f"({VERDICT_TEXT[result['verdict']]})",
        x=0.01, ha="left", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def record(
    result: Mapping[str, Any],
    choices: Mapping[int, Mapping[str, Any]],
    files: list[Path],
    experiment: str = ANALYSIS_EXPERIMENT,
) -> str:
    """One MLflow run holding the readings, both curves and the files.

    Parameters
    ----------
    result : mapping
    choices : mapping of int to mapping
    files : list of Path
    experiment : str, optional
        A rehearsal goes to the rehearsal experiment, so it can never
        be mistaken for the comparison.

    Returns
    -------
    str
        The run's identifier.
    """
    import mlflow

    configure_tracking()
    metrics: dict[str, float] = {}
    for arm in ARMS:
        reading = result["arms"][str(arm)]
        metrics[f"e{arm}/selected_epoch"] = reading["selected_epoch"]
        metrics[f"e{arm}/val_f1"] = reading["val_f1"]
        for field in DELTA_FIELDS:
            metrics[f"e{arm}/held_out/{field}"] = reading["held_out"][field]
        for material, figures in reading["by_material"].items():
            metrics[f"e{arm}/held_out/{material}/f1"] = figures["f1"]
    for field, value in result["delta"].items():
        metrics[f"delta/{field}"] = value
    for material, figures in result["delta_by_material"].items():
        metrics[f"delta/{material}/f1"] = figures["f1"]
    with mlflow.start_run(
        experiment_id=ensure_experiment(experiment),
        run_name=f"epoch_budget_fold{FOLD}_{result['postprocessing']}",
        tags={"part": "epoch_budget", "budget_id": BUDGET_ID,
              "fold": str(FOLD), "postprocessing": result["postprocessing"],
              "verdict": result["verdict"]},
    ) as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
        for path in files:
            mlflow.log_artifact(str(path))
        run_id = run.info.run_id
    log_metric_series(run_id, [
        (f"e{arm}/val_curve_f1", float(v["f1"]), int(k))
        for arm in ARMS for k, v in choices[arm]["curve"].items()
    ])
    return run_id


def expected_held_out_images(split: Path) -> int:
    """How many held-out images the split's record says there are.

    Parameters
    ----------
    split : Path

    Returns
    -------
    int
    """
    meta_path = split.with_name(f"{split.stem}_metadata.json")
    if not meta_path.exists():
        raise ComparisonError(f"No split metadata at {meta_path}")
    counts = json.loads(meta_path.read_text(encoding="utf-8"))["counts"]
    return int(counts["n_test_images"])


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
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--held-out-report", type=Path, default=None,
        help="Defaults to <save-root>/eval/test.json.",
    )
    parser.add_argument(
        "--rehearsal", action="store_true",
        help="Read a rehearsal, scored on a sample of the held-out "
             "images.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    args.eval_dir = args.save_root / "eval"
    if args.held_out_report is None:
        args.held_out_report = args.eval_dir / "test.json"
    return args


def run(args: argparse.Namespace) -> int:
    """Read both arms, compare and record.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
        Process exit code.
    """
    choices = {
        arm: load_choice(args.eval_dir / f"selected_e{arm}.json", arm)
        for arm in ARMS
    }
    if not args.held_out_report.exists():
        raise ComparisonError(f"No held-out report at {args.held_out_report}")
    payload = json.loads(args.held_out_report.read_text(encoding="utf-8"))
    result = compare(choices, payload, args.postprocessing)
    n_scored = result["arms"][str(SHORT_ARM)]["held_out"]["n_images"]
    if not args.rehearsal:
        expected = expected_held_out_images(args.split)
        if n_scored != expected:
            raise ComparisonError(
                f"The held-out report scored {n_scored} image(s); fold "
                f"{FOLD} holds out {expected}."
            )

    stem = args.eval_dir / f"epoch_budget_fold{FOLD}_{args.postprocessing}"
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    markdown = format_markdown(result)
    md_path = stem.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    png_path = plot(result, choices, stem.with_suffix(".png"))
    logger.info("Wrote %s, %s and %s.\n%s", json_path, md_path.name,
                png_path.name, markdown)
    if not args.no_mlflow:
        run_id = record(
            result, choices, [json_path, md_path, png_path],
            REHEARSAL_EXPERIMENT if args.rehearsal else ANALYSIS_EXPERIMENT,
        )
        logger.info("Recorded in MLflow run %s.", run_id)
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    """Compare the two arms.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    try:
        return run(args)
    except ComparisonError as e:
        logger.error("%s", e)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
