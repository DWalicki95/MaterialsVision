#!/usr/bin/env python3
"""
Step 7 of the optimization plan: read the scored snapshots and screen.

Reads the VALIDATION reports of the three FULL_A reference seeds and of
arms A1 and A2, all scored under the post-processing part I froze, and
applies the registered rule: the noise band from the reference seeds,
then each arm against the reference, then the branch table. Everything
is read as the mean of epochs 10 to 12.

Beside the decision, two readings for the report and not for the rule:
the robustness cross-section - the same gains at centre thresholds 0.20
and 0.30 - and what part I changed on VALIDATION, from the attribution
study's frozen configuration scored in the same pass.

Runs whose report does not exist yet are skipped, so the band can be
fixed from the reference alone before any arm has been read.

Examples
--------
    $ python scripts/screen_hparam_arms.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.screening import (branch, late_mean,
                                                   noise_band, screen_arm)
from materials_vision.evaluation.watershed import (CALIBRATED_2026_09_24,
                                                   FROZEN_WATERSHED,
                                                   robustness_series)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import configure_tracking, ensure_experiment

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_EVAL = Path("checkpoints/hparams/eval")

ANALYSIS_EXPERIMENT = "peft_sam_analysis"

REFERENCE_RUNS = (
    "d4_seed20260907", "d4_seed20260908", "d4_seed20260909",
)

ARM_RUNS = {
    "a1_r32": "d4_a1_r32_seed20260907",
    "a2_lr1e-3": "d4_a2_lr1e-3_seed20260907",
}

CROSS_SECTION = robustness_series(CALIBRATED_2026_09_24, (0.20, 0.25, 0.30))

SETTINGS = {
    "calibrated": CALIBRATED_2026_09_24,
    "cdt0.20": CROSS_SECTION[0],
    "cdt0.30": CROSS_SECTION[2],
    "frozen": FROZEN_WATERSHED,
}

MATERIALS = ("AS", "K", "VAB")


def load_report(path: Path) -> Optional[dict]:
    """A run's VALIDATION report, or None while it is still being made."""
    return json.loads(path.read_text()) if path.exists() else None


def curves(report: dict, run: str, label: str) -> dict[str, dict[int, float]]:
    """Per-epoch figures of one run under one setting.

    Returns
    -------
    dict of str to dict of int to float
        Figure name to epoch to value.
    """
    out: dict[str, dict[int, float]] = {}
    for epoch in range(1, 13):
        record = report.get(f"{run}/epoch-{epoch}@{label}")
        if record is None:
            continue
        overall = record["overall"]
        values = {
            "f1": overall["f1"],
            "merges_per_100_gt": overall["merges_per_100_gt"],
            "splits_per_100_gt": overall["splits_per_100_gt"],
            "abs_pore_count_error": abs(overall["pore_count_error"]),
            "pore_count_error": overall["pore_count_error"],
            "boundary_f1": overall["boundary_f1"][str(DECISION_SCALE)],
            "median_diameter_log_ratio": overall.get(
                "median_diameter_log_ratio", float("nan")
            ),
        }
        for section in record["per_material"]:
            material = section["label"].split("=")[1]
            values[f"{material}/f1"] = section["f1"]
            values[f"{material}/pore_count_error"] = (
                section["pore_count_error"]
            )
        for name, value in values.items():
            out.setdefault(name, {})[epoch] = value
    return out


def late(curve_set: dict[str, dict[int, float]]) -> dict[str, float]:
    """Late-epoch mean of every figure that has all three late epochs."""
    means = {}
    for name, by_epoch in curve_set.items():
        try:
            means[name] = late_mean(by_epoch)
        except KeyError:
            continue
    return means


def read_all(eval_dir: Path) -> dict:
    """Late-epoch figures and curves of every run and setting available."""
    runs = {}
    for run in (*REFERENCE_RUNS, *ARM_RUNS.values()):
        report = load_report(eval_dir / f"{run}_val.json")
        if report is None:
            logger.info("%s: no report yet; skipped.", run)
            continue
        runs[run] = {
            key: {
                "curves": curves(report, run, setting.label()),
            }
            for key, setting in SETTINGS.items()
        }
        for key in runs[run]:
            runs[run][key]["late"] = late(runs[run][key]["curves"])
    return runs


def decide(runs: dict) -> dict:
    """The registered rule, as far as the available runs allow."""
    references = [
        runs[r]["calibrated"]["late"] for r in REFERENCE_RUNS if r in runs
    ]
    if len(references) < len(REFERENCE_RUNS):
        return {"status": "reference incomplete",
                "n_reference": len(references)}
    band, how = noise_band([r["f1"] for r in references])
    decision: dict = {
        "band": band, "band_from": how,
        "reference_late_f1": [r["f1"] for r in references],
        "reference_mean_f1": sum(r["f1"] for r in references) / 3,
        "arms": {},
    }
    verdicts = {}
    for arm, run in ARM_RUNS.items():
        if run not in runs:
            continue
        verdict = screen_arm(runs[run]["calibrated"]["late"], references,
                             band)
        verdicts[arm] = verdict
        decision["arms"][arm] = verdict.__dict__ | {
            "robustness_gain": {
                key: runs[run][key]["late"]["f1"] - sum(
                    runs[r][key]["late"]["f1"] for r in REFERENCE_RUNS
                ) / 3
                for key in ("cdt0.20", "calibrated", "cdt0.30")
            },
        }
    if len(verdicts) == len(ARM_RUNS):
        reading, chosen = branch(
            verdicts["a1_r32"], verdicts["a2_lr1e-3"], band
        )
        decision["branch_reading"] = reading
        decision["branch"] = chosen
    decision["part_one_gain_on_val"] = {
        run: runs[run]["calibrated"]["late"]["f1"]
        - runs[run]["frozen"]["late"]["f1"]
        for run in runs
    }
    return decision


def figure(runs: dict, decision: dict, path: Path) -> Path:
    """F1 curves under both configurations, late window shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "d4_seed20260907": "#2a78d6", "d4_seed20260908": "#6da7ec",
        "d4_seed20260909": "#184f95",
        "d4_a1_r32_seed20260907": "#eb6834",
        "d4_a2_lr1e-3_seed20260907": "#1baf7a",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = (("f1", "instance F1 (pooled)"), ("K/f1", "instance F1, K"),
              ("pore_count_error", "pore count error"))
    for axis, (name, title) in zip(axes, panels):
        for run, settings in runs.items():
            for key, style in (("calibrated", "-"), ("frozen", ":")):
                curve = settings[key]["curves"].get(name, {})
                if not curve:
                    continue
                epochs = sorted(curve)
                axis.plot(
                    epochs, [curve[e] for e in epochs], style,
                    color=colours.get(run, "#52514e"), linewidth=2,
                    label=f"{run}" if key == "calibrated" else None,
                )
        axis.axvspan(9.5, 12.5, color="#52514e", alpha=0.08)
        axis.set_title(title, loc="left")
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    band = decision.get("band")
    fig.suptitle(
        "VALIDATION, solid: calibrated_2026_09_24, dotted: frozen; "
        "shaded: epochs 10-12"
        + (f"; noise band {band:.4f}" if band else ""),
        x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def record(decision: dict, runs: dict, files: list[Path]) -> str:
    """One MLflow run holding the decision, the late means and files."""
    import mlflow

    configure_tracking()
    with mlflow.start_run(
        experiment_id=ensure_experiment(ANALYSIS_EXPERIMENT),
        run_name="step7_screening",
        tags={"part": "II", "plan_step": "7",
              "status": decision.get("status", "decided")},
    ) as run:
        metrics = {}
        for name, settings in runs.items():
            for key, value in settings["calibrated"]["late"].items():
                metrics[f"{name}/late/{key}"] = value
            metrics[f"{name}/late_frozen/f1"] = (
                settings["frozen"]["late"].get("f1", float("nan"))
            )
        if "band" in decision:
            metrics["band"] = decision["band"]
        for arm, verdict in decision.get("arms", {}).items():
            metrics[f"{arm}/gain"] = verdict["gain"]
            metrics[f"{arm}/passes"] = float(verdict["passes"])
        mlflow.log_metrics({
            k: v for k, v in metrics.items() if v == v
        })
        for path in files:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Read, decide, write and record.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    runs = read_all(args.eval_dir)
    if not runs:
        logger.error("No report in %s yet.", args.eval_dir)
        return EXIT_FAILED
    decision = decide(runs)
    report = args.eval_dir / "screening.json"
    report.write_text(json.dumps({
        "decision": decision,
        "late": {run: {k: s["late"] for k, s in settings.items()}
                 for run, settings in runs.items()},
    }, indent=2, default=str))
    plot = figure(runs, decision, args.eval_dir / "screening.png")
    run_id = record(decision, runs, [report, plot])
    logger.info("Decision: %s", json.dumps(decision, default=str))
    logger.info("Wrote %s; MLflow run %s.", report, run_id)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
