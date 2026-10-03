#!/usr/bin/env python3
"""
How well a Cellpose-SAM snapshot segments the images it was held out from.

The counterpart of ``scripts/evaluate_checkpoint.py`` for the Cellpose
comparison. The snapshot is run differently - Cellpose follows
predicted flows where PEFT-SAM grows a watershed - but everything that
measures is shared: the same split reader and sample source, the same
``evaluate_image``, the same pooling, the same JSON report. A figure
from here and one from there are read with one measuring stick.

**Three uses, one script.**

* *Selecting the snapshot:* every snapshot of both arms on VALIDATION,
  under Cellpose's default post-processing. The snapshot with the
  highest pooled instance F1 is the one the recipe keeps.
* *Calibrating the post-processing:* the chosen snapshot on VALIDATION
  under every setting of the grid (``--grid``), twenty settings in one
  pass over the images.
* *Scoring held-out formulations:* one snapshot per run on TEST, or on
  the formulations a fold holds out, under the calibrated setting and,
  with ``--also-area-filter``, the same setting with the physical area
  filter the PEFT-SAM deployment applies.

**TEST is scored once, and only snapshots chosen beforehand**, under
the same rules as for PEFT-SAM: it stays locked without
``--unlock-test``, and even then accepts one snapshot per run, so that
TEST cannot pick among the epochs of a run.

Examples
--------
Score every snapshot of both selection arms on VALIDATION:
    $ python scripts/evaluate_cellpose.py \\
        --checkpoint-glob 'checkpoints/cellpose/selection/*/models/*' \\
        --out checkpoints/cellpose/selection/eval/selection.json

Calibrate the post-processing on the chosen snapshot:
    $ python scripts/evaluate_cellpose.py --grid --checkpoint <snapshot> \\
        --out checkpoints/cellpose/selection/eval/calibration.json

Score a fold's snapshot on the formulations it holds out, once:
    $ python scripts/evaluate_cellpose.py --subset test --unlock-test \\
        --split /home/dwalicki/dane/splits/cv5_v1/fold3.csv \\
        --flow-threshold <ft> --cellprob-threshold <cp> \\
        --also-area-filter --checkpoint <snapshot> --out <json>
"""
import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from materials_vision.evaluation import (DECISION_SCALE, AggregateResult,
                                         aggregate, cross_sections,
                                         load_size_bins)
from materials_vision.experiments.cellpose.split_evaluation import (
    CellposeSettings, calibration_grid, load_snapshot, score_settings)
from materials_vision.experiments.cellpose.split_training import (
    SnapshotNameError, completed_epochs, run_of)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (configure_tracking,
                                       find_run_for_checkpoint_dir,
                                       log_evaluation_to_run)
from materials_vision.training import build_source
from scripts.evaluate_checkpoint import mlflow_refusal, report, write_json

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")


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
        "--checkpoint", type=Path, action="append", default=[],
        help="Snapshot to score; repeat to compare several.",
    )
    parser.add_argument(
        "--checkpoint-glob", action="append", default=[],
        help="Pattern selecting snapshots; repeatable.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument(
        "--subset", default="val", choices=["train", "val", "test"],
        help="Which split subset to score on. TEST needs --unlock-test.",
    )
    parser.add_argument(
        "--unlock-test", action="store_true",
        help="Open TEST, for the single evaluation of snapshots chosen "
             "on VALIDATION beforehand.",
    )
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="Score this many images; 0 means all of them.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--flow-threshold", type=float,
        default=CellposeSettings.flow_threshold,
    )
    parser.add_argument(
        "--cellprob-threshold", type=float,
        default=CellposeSettings.cellprob_threshold,
    )
    parser.add_argument(
        "--niter", type=int, default=CellposeSettings.niter,
        help="Steps the flows are followed for.",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Score under every setting of the calibration grid instead "
             "of the one given by the thresholds and steps.",
    )
    parser.add_argument(
        "--grid-niter", type=int, action="append", default=None,
        help="Steps the grid crosses the thresholds with; repeatable. "
             "The library's steps alone when not given.",
    )
    parser.add_argument(
        "--also-area-filter", action="store_true",
        help="Score each setting a second time with the physical area "
             "filter of the PEFT-SAM deployment.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Do not add the figures to the runs the snapshots were "
             "trained in.",
    )
    return parser.parse_args(argv)


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Expand the patterns, drop duplicates, and order as trained.

    Files that are not snapshots - a provenance record that a broad
    pattern happened to catch, say - are dropped with a note rather
    than scored.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    list of Path
    """
    found = list(args.checkpoint)
    for pattern in args.checkpoint_glob:
        found.extend(Path().glob(pattern))
    snapshots = []
    for path in dict.fromkeys(found):
        try:
            completed_epochs(path)
        except SnapshotNameError:
            logger.info("%s is not a snapshot; skipped.", path)
            continue
        snapshots.append(path)
    return sorted(
        snapshots, key=lambda path: (run_of(path), completed_epochs(path))
    )


def locked_test_reason(args: argparse.Namespace) -> Optional[str]:
    """Say why TEST may not be scored as asked, or nothing if it may.

    Parameters
    ----------
    args : argparse.Namespace
        With ``checkpoint`` already resolved.

    Returns
    -------
    str or None
    """
    if args.subset != "test":
        return None
    if not args.unlock_test:
        return (
            "TEST is locked. Pass --unlock-test, and only for the single "
            "evaluation of snapshots already chosen on VALIDATION."
        )
    per_run = Counter(run_of(path) for path in args.checkpoint)
    repeated = sorted(run for run, count in per_run.items() if count > 1)
    if repeated:
        return (
            f"TEST scores one snapshot per run, chosen on VALIDATION "
            f"beforehand. Several were given for {', '.join(repeated)}, "
            f"which would let TEST choose among them."
        )
    return None


def resolve_settings(
    args: argparse.Namespace
) -> tuple[CellposeSettings, ...]:
    """Work out which settings to score under.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    tuple of CellposeSettings
    """
    if args.grid:
        niters = args.grid_niter or [CellposeSettings.niter]
        settings = list(calibration_grid(niters))
    else:
        settings = [CellposeSettings(
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            niter=args.niter,
        )]
    if args.also_area_filter:
        settings += [setting.with_area_filter() for setting in settings]
    return tuple(settings)


def snapshot_name(snapshot: Path) -> str:
    """How a snapshot is named in the report.

    Parameters
    ----------
    snapshot : Path

    Returns
    -------
    str
        ``<run>/epoch-<completed>``, counting completed epochs as the
        PEFT-SAM reports do.
    """
    return f"{run_of(snapshot)}/epoch-{completed_epochs(snapshot)}"


def score_snapshot(
    snapshot: Path,
    args: argparse.Namespace,
    settings: tuple[CellposeSettings, ...],
) -> list[tuple[CellposeSettings, AggregateResult, list]]:
    """Segment every image of a subset and measure the result.

    Parameters
    ----------
    snapshot : Path
    args : argparse.Namespace
    settings : tuple of CellposeSettings

    Returns
    -------
    list of tuple
        Per setting: the setting, the pooled figures, and the per-image
        evaluations behind them.
    """
    source = build_source(
        args.split, args.manifest, args.subset,
        allow_test=args.unlock_test,
    )
    size_bins = load_size_bins(args.size_bins)
    model = load_snapshot(snapshot)

    positions = list(range(len(source)))
    if 0 < args.n_images < len(source):
        stride = len(source) / args.n_images
        positions = [int(step * stride) for step in range(args.n_images)]

    started_s = time.perf_counter()
    evaluations = score_settings(
        model, source, positions, size_bins, settings,
        boundary_scales=(DECISION_SCALE,),
    )
    elapsed_s = time.perf_counter() - started_s
    logger.info(
        "%s: %d image(s) x %d setting(s) in %.1f s (%.2f s each).",
        snapshot, len(positions), len(settings), elapsed_s,
        elapsed_s / max(1, len(positions)),
    )
    return [
        (
            setting,
            aggregate(
                evaluations[setting],
                label=f"{snapshot_name(snapshot)}@{setting.label()}",
            ),
            evaluations[setting],
        )
        for setting in settings
    ]


def record_in_mlflow(
    snapshot: Path,
    subset: str,
    scored: list[tuple[CellposeSettings, AggregateResult, list]],
) -> Optional[str]:
    """Add one snapshot's figures to the run it was trained in.

    Parameters
    ----------
    snapshot : Path
    subset : str
    scored : list of tuple

    Returns
    -------
    str or None
        The run written to, or ``None`` when its run is not in the
        store.
    """
    run_id = find_run_for_checkpoint_dir(snapshot.parent)
    if run_id is None:
        logger.warning(
            "No MLflow run is tagged with %s; its figures stay in the "
            "JSON report only.", snapshot.parent,
        )
        return None
    for setting, result, per_image in scored:
        sections = (
            cross_sections(per_image, "material")
            + cross_sections(per_image, "scale_bin")
        )
        log_evaluation_to_run(
            run_id, completed_epochs(snapshot), subset,
            f"cellpose_{setting.label()}", result, sections,
        )
    return run_id


def main(argv: Optional[list[str]] = None) -> int:
    """Score every snapshot given.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    args.checkpoint = resolve_checkpoints(args)
    if not args.checkpoint:
        logger.error(
            "No snapshot selected. Pass --checkpoint or --checkpoint-glob."
        )
        return EXIT_FAILED
    missing = [path for path in args.checkpoint if not path.exists()]
    if missing:
        logger.error(
            "No such snapshot: %s.", ", ".join(str(p) for p in missing)
        )
        return EXIT_FAILED
    refusal = locked_test_reason(args)
    if refusal is not None:
        logger.error(refusal)
        return EXIT_FAILED

    settings = resolve_settings(args)
    logger.info(
        "Scoring %d snapshot(s) on %s under %s.",
        len(args.checkpoint), args.subset.upper(),
        ", ".join(setting.label() for setting in settings),
    )
    refusal = mlflow_refusal(args)
    if refusal is None:
        configure_tracking()
    else:
        logger.info("Figures are not added to MLflow: %s.", refusal)

    results: dict[str, AggregateResult] = {}
    evaluations: dict[str, list] = {}
    for snapshot in args.checkpoint:
        scored = score_snapshot(snapshot, args, settings)
        stem = snapshot_name(snapshot)
        for setting, result, per_image in scored:
            name = stem if len(scored) == 1 else f"{stem}@{setting.label()}"
            results[name], evaluations[name] = result, per_image
        if refusal is None:
            try:
                record_in_mlflow(snapshot, args.subset, scored)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s: writing to MLflow failed; the JSON report is "
                    "unaffected.", snapshot, exc_info=True,
                )
    report(results)
    if args.out is not None:
        write_json(args.out, results, evaluations)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
