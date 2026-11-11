#!/usr/bin/env python3
"""
Compare runs with a reference epoch by epoch, the way the seed pairs them.

Every run trained with one seed sees the same images in the same order,
and the epoch-to-epoch wobble of its validation figures follows that
order: the runs of seed 20260907 move together whatever their rank,
learning rate, trainable part or augmentation (correlation of
epoch-to-epoch changes 0.87-0.97, against 0.15 and 0.53 for two other
seeds). Most of the noise in one run's curve is therefore shared with
the reference, and it cancels in their difference. Comparing the
difference epoch by epoch can show an effect a few tenths of a point
wide, which two curves read separately would bury.

**What decides.** For each run, against the reference, over a window
of late epochs:

1. the mean difference in instance F1 is at least ``--min-gain-pp``;
2. the difference is positive in at least ``--min-positive`` epochs of
   the window;
3. no guard falls by more than ``--max-guard-loss-pp`` on average: F1
   on the weakest materials, and recall on the smallest pores where an
   evaluation report holding it is given.

A run lacking an epoch of the window is reported as incomplete rather
than judged on fewer points.

**Where the figures come from.** F1 and the per-material F1 are read
from MLflow, where ``evaluate_checkpoint.py`` writes them per epoch.
Recall on the smallest size bin is only in that script's JSON reports,
read when given with ``--curves``.

Examples
--------
Judge every run of the method study against the adopted recipe:
    $ PYTHONPATH=. python scripts/peft_sam/compare_paired_runs.py \\
        --reference full_b_seed20260907 \\
        --curves checkpoints/e8/e8_full_b_curves.json \\
        --curves checkpoints/peft_methods/stage1_curves.json \\
        --out checkpoints/peft_methods/stage1_paired.json

The same at another centre threshold, to see that a result does not
depend on where the calibration landed:
    $ PYTHONPATH=. python scripts/peft_sam/compare_paired_runs.py \\
        --reference full_b_seed20260907 --center-threshold 0.25
"""
import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from materials_vision.evaluation.watershed import (POSTPROCESSING_CONFIGS,
                                                   WatershedParams)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (TRAINING_EXPERIMENT, configure_tracking,
                                       postprocessing_id)
from scripts.peft_sam.average_lora_snapshots import parse_epochs

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_EXPERIMENT = "peft_sam_methods"

DEFAULT_WINDOW = "7-12"

MIN_GAIN_PP = 0.5

MIN_POSITIVE_EPOCHS = 5

MAX_GUARD_LOSS_PP = 1.0

# The two materials the models segment worst, where a gain on the pooled
# figure could hide a loss.
GUARD_MATERIALS = ("K", "VAB")

SMALLEST_SIZE_BIN = "Q1"

N_LAST_EPOCHS = 3

PERCENT = 100.0

VERDICT_WINS = "wins"
VERDICT_NO_GAIN = "no gain"
VERDICT_GUARD = "guard failed"
VERDICT_INCOMPLETE = "incomplete"

_SNAPSHOT_NAME = re.compile(
    r"^(?P<run>.+)/epoch-(?P<epoch>\d+)(?:@(?P<label>.+))?$"
)


@dataclass
class PairedComparison:
    """One run against the reference, over the window."""

    run: str
    epochs: list[int]
    deltas_pp: list[float] = field(default_factory=list)
    mean_delta_pp: Optional[float] = None
    n_positive: int = 0
    last_epochs_mean_run: Optional[float] = None
    last_epochs_mean_reference: Optional[float] = None
    guard_deltas_pp: dict[str, Optional[float]] = field(default_factory=dict)
    missing_epochs: list[int] = field(default_factory=list)
    verdict: str = VERDICT_INCOMPLETE


def paired_deltas_pp(
    reference: Mapping[int, float],
    candidate: Mapping[int, float],
    epochs: Sequence[int],
) -> list[float]:
    """Candidate minus reference at each epoch, in percentage points.

    Parameters
    ----------
    reference, candidate : mapping of int to float
        Figure per epoch, as fractions.
    epochs : sequence of int

    Returns
    -------
    list of float

    Raises
    ------
    KeyError
        If either side lacks an epoch.
    """
    return [
        (candidate[epoch] - reference[epoch]) * PERCENT for epoch in epochs
    ]


def mean_guard_delta_pp(
    reference: Mapping[int, float],
    candidate: Mapping[int, float],
    epochs: Sequence[int],
) -> Optional[float]:
    """Mean difference of a guard figure, or nothing if it is incomplete.

    Parameters
    ----------
    reference, candidate : mapping of int to float
    epochs : sequence of int

    Returns
    -------
    float or None
    """
    if not all(e in reference and e in candidate for e in epochs):
        return None
    deltas = paired_deltas_pp(reference, candidate, epochs)
    return sum(deltas) / len(deltas)


def judge(
    comparison: PairedComparison,
    min_gain_pp: float = MIN_GAIN_PP,
    min_positive: int = MIN_POSITIVE_EPOCHS,
    max_guard_loss_pp: float = MAX_GUARD_LOSS_PP,
) -> str:
    """Decide a comparison by the study's rule.

    Parameters
    ----------
    comparison : PairedComparison
    min_gain_pp, min_positive, max_guard_loss_pp : optional

    Returns
    -------
    str
        One of the ``VERDICT_*`` values. A guard that could not be
        computed does not fail the run; it is reported as missing.
    """
    if comparison.missing_epochs or comparison.mean_delta_pp is None:
        return VERDICT_INCOMPLETE
    if (comparison.mean_delta_pp < min_gain_pp
            or comparison.n_positive < min_positive):
        return VERDICT_NO_GAIN
    if any(
        delta is not None and delta < -max_guard_loss_pp
        for delta in comparison.guard_deltas_pp.values()
    ):
        return VERDICT_GUARD
    return VERDICT_WINS


def compare(
    run: str,
    reference: Mapping[str, Mapping[int, float]],
    candidate: Mapping[str, Mapping[int, float]],
    epochs: Sequence[int],
) -> PairedComparison:
    """Assemble one run's comparison from its figures and the reference's.

    Parameters
    ----------
    run : str
    reference, candidate : mapping of str to mapping of int to float
        Per figure name, the value per epoch. ``"f1"`` must be present;
        every other name is a guard.
    epochs : sequence of int

    Returns
    -------
    PairedComparison
        Without a verdict; see :func:`judge`.
    """
    comparison = PairedComparison(run=run, epochs=list(epochs))
    comparison.missing_epochs = [
        epoch for epoch in epochs
        if epoch not in candidate.get("f1", {})
        or epoch not in reference.get("f1", {})
    ]
    if comparison.missing_epochs:
        return comparison
    deltas = paired_deltas_pp(reference["f1"], candidate["f1"], epochs)
    comparison.deltas_pp = deltas
    comparison.mean_delta_pp = sum(deltas) / len(deltas)
    comparison.n_positive = sum(delta > 0 for delta in deltas)
    last = list(epochs)[-N_LAST_EPOCHS:]
    comparison.last_epochs_mean_run = (
        sum(candidate["f1"][e] for e in last) / len(last)
    )
    comparison.last_epochs_mean_reference = (
        sum(reference["f1"][e] for e in last) / len(last)
    )
    for name in sorted(set(reference) | set(candidate)):
        if name == "f1":
            continue
        comparison.guard_deltas_pp[name] = mean_guard_delta_pp(
            reference.get(name, {}), candidate.get(name, {}), epochs,
        )
    return comparison


def size_bin_recall_by_epoch(
    reports: Sequence[Mapping[str, Any]],
    run: str,
    label: str,
    size_bin: str = SMALLEST_SIZE_BIN,
) -> dict[int, float]:
    """Recall on one size bin per epoch, out of evaluation reports.

    Parameters
    ----------
    reports : sequence of mapping
        JSON reports of ``evaluate_checkpoint.py``.
    run : str
    label : str
        The watershed setting's label. A report scored under a single
        setting names its entries without one, and those are taken as
        that setting.
    size_bin : str, optional

    Returns
    -------
    dict of int to float
    """
    recall: dict[int, float] = {}
    for report in reports:
        for name, entry in report.items():
            match = _SNAPSHOT_NAME.match(name)
            if match is None or match.group("run") != run:
                continue
            if match.group("label") not in (None, label):
                continue
            for size in entry["overall"].get("size_bin_recall", []):
                if size["label"] == size_bin:
                    recall[int(match.group("epoch"))] = size["recall"]
    return recall


def _metric_by_epoch(client: Any, run_id: str, key: str) -> dict[int, float]:
    """A logged figure, keyed by the epoch it was logged at."""
    return {
        int(point.step): float(point.value)
        for point in client.get_metric_history(run_id, key)
    }


def _runs_in(client: Any, experiment: str) -> dict[str, str]:
    """Finished runs of an experiment, newest of each name."""
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"No experiment named {experiment!r} in the store.")
    runs: dict[str, str] = {}
    for run in client.search_runs(
        [found.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    ):
        runs[run.info.run_name] = run.info.run_id
    return runs


def figures_of(
    client: Any,
    run_id: str,
    run: str,
    setting: WatershedParams,
    reports: Sequence[Mapping[str, Any]],
    materials: Sequence[str] = GUARD_MATERIALS,
) -> dict[str, dict[int, float]]:
    """Every figure a comparison reads, for one run.

    Parameters
    ----------
    client : mlflow.tracking.MlflowClient
    run_id, run : str
    setting : WatershedParams
    reports : sequence of mapping
    materials : sequence of str, optional

    Returns
    -------
    dict of str to dict of int to float
    """
    setting_id = postprocessing_id(setting)
    figures = {"f1": _metric_by_epoch(client, run_id, f"val/{setting_id}/f1")}
    for material in materials:
        figures[f"F1 {material}"] = _metric_by_epoch(
            client, run_id, f"val/{setting_id}/material_{material}/f1"
        )
    if reports:
        figures[f"recall {SMALLEST_SIZE_BIN}"] = size_bin_recall_by_epoch(
            reports, run, setting.label()
        )
    return figures


def resolve_setting(args: argparse.Namespace) -> WatershedParams:
    """The watershed setting the figures are read under.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    WatershedParams
    """
    fields = POSTPROCESSING_CONFIGS[args.postprocessing].to_kwargs()
    if args.center_threshold is not None:
        fields["center_distance_threshold"] = args.center_threshold
    return WatershedParams(**fields)


def report(comparisons: Sequence[PairedComparison]) -> None:
    """Write one line per run, then its guards."""
    for c in comparisons:
        if c.verdict == VERDICT_INCOMPLETE:
            logger.info(
                "%-60s %s: epochs %s missing.", c.run, c.verdict,
                c.missing_epochs,
            )
            continue
        logger.info(
            "%-60s %-12s mean dF1 %+.2f pp, positive in %d/%d, last-%d "
            "F1 %.4f vs %.4f.", c.run, c.verdict, c.mean_delta_pp,
            c.n_positive, len(c.epochs), N_LAST_EPOCHS,
            c.last_epochs_mean_run, c.last_epochs_mean_reference,
        )
        guards = ", ".join(
            f"{name} {'n/a' if delta is None else f'{delta:+.2f} pp'}"
            for name, delta in c.guard_deltas_pp.items()
        )
        logger.info("%-60s guards: %s.", "", guards)


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
        "--reference", required=True,
        help="Name of the reference run, trained with the same seed.",
    )
    parser.add_argument(
        "--reference-experiment", default=TRAINING_EXPERIMENT,
        help=f"Experiment holding it. Default {TRAINING_EXPERIMENT}.",
    )
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help=f"Experiment holding the runs. Default {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--run", action="append", default=[],
        help="Run to judge; repeatable. Default: every finished run of "
             "--experiment other than the reference.",
    )
    parser.add_argument(
        "--epochs", type=parse_epochs, default=parse_epochs(DEFAULT_WINDOW),
        help=f"Window of epochs compared. Default {DEFAULT_WINDOW}.",
    )
    parser.add_argument(
        "--postprocessing", default="frozen",
        choices=sorted(POSTPROCESSING_CONFIGS),
        help="Named watershed setting the figures were scored under.",
    )
    parser.add_argument(
        "--center-threshold", type=float, default=None,
        help="Override its centre threshold, e.g. 0.25 or 0.35 from the "
             "robustness sweep.",
    )
    parser.add_argument(
        "--curves", type=Path, action="append", default=[],
        help="JSON report of evaluate_checkpoint.py to read recall on the "
             "smallest pores from; repeatable. Give the reference's too.",
    )
    parser.add_argument("--min-gain-pp", type=float, default=MIN_GAIN_PP)
    parser.add_argument(
        "--min-positive", type=int, default=MIN_POSITIVE_EPOCHS,
    )
    parser.add_argument(
        "--max-guard-loss-pp", type=float, default=MAX_GUARD_LOSS_PP,
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Judge every run against the reference.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    if args.min_positive > len(args.epochs):
        logger.error(
            "--min-positive %d cannot be met in a window of %d epoch(s).",
            args.min_positive, len(args.epochs),
        )
        return EXIT_FAILED
    missing = [path for path in args.curves if not path.exists()]
    if missing:
        logger.error("No such report: %s.", missing[0])
        return EXIT_FAILED

    from mlflow.tracking import MlflowClient

    configure_tracking()
    client = MlflowClient()
    setting = resolve_setting(args)
    reports = [json.loads(path.read_text()) for path in args.curves]
    try:
        reference_runs = _runs_in(client, args.reference_experiment)
        candidate_runs = _runs_in(client, args.experiment)
    except ValueError as error:
        logger.error("%s", error)
        return EXIT_FAILED
    if args.reference not in reference_runs:
        logger.error(
            "No finished run named %s in %s.", args.reference,
            args.reference_experiment,
        )
        return EXIT_FAILED
    names = args.run or [n for n in candidate_runs if n != args.reference]
    unknown = [n for n in names if n not in candidate_runs]
    if unknown or not names:
        logger.error(
            "No finished run named %s in %s.",
            unknown[0] if unknown else "anything", args.experiment,
        )
        return EXIT_FAILED

    logger.info(
        "Comparing %d run(s) with %s over epochs %s, under %s.",
        len(names), args.reference, args.epochs, postprocessing_id(setting),
    )
    reference = figures_of(
        client, reference_runs[args.reference], args.reference, setting,
        reports,
    )
    comparisons = []
    for name in names:
        comparison = compare(
            name, reference,
            figures_of(client, candidate_runs[name], name, setting, reports),
            args.epochs,
        )
        comparison.verdict = judge(
            comparison, args.min_gain_pp, args.min_positive,
            args.max_guard_loss_pp,
        )
        comparisons.append(comparison)
    report(comparisons)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "reference": args.reference,
            "postprocessing": postprocessing_id(setting),
            "rule": {
                "min_gain_pp": args.min_gain_pp,
                "min_positive": args.min_positive,
                "max_guard_loss_pp": args.max_guard_loss_pp,
            },
            "comparisons": [asdict(c) for c in comparisons],
        }, indent=2))
        logger.info("Wrote %s.", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
