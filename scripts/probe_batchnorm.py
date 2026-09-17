#!/usr/bin/env python3
"""
Whether the decoder's batch statistics are what make snapshots jump.

Scoring every epoch of the unaugmented baseline produced a spread that
nothing in the training curve explained. Neighbouring epochs of one run
differed by as much as 0.045 in instance F1, while three independent
seeds differed by 0.014 at their peaks and the sampling uncertainty of
the score is about 0.006. The models really were that different from
one epoch to the next, and the loss curve was smooth throughout.

**One candidate explanation is not about learning at all.** The
instance decoder normalizes over the batch, and the batch is a single
image. Its running statistics are an exponential average with most of
their weight on the last few dozen images, so a snapshot saved at the
end of an epoch carries whichever images happened to end that epoch.
Two snapshots an epoch apart would then differ partly for that reason,
independently of anything the optimizer did.

**How this separates the two.** Each snapshot is scored twice: once as
saved, and once after its batch statistics have been recomputed over
one fixed set of training images, the same set for every snapshot. If
the statistics were the source of the jumping, the spread between
neighbouring snapshots shrinks once they all share them. If the spread
survives, the models genuinely differ and the cause is elsewhere.

**What a confirmation would be worth.** The spread between snapshots
sets the threshold below which no augmentation can be credited with
anything. Removing a source of it that has nothing to do with the
models makes every later comparison in this study more sensitive.

Examples
--------
Compare the first three epochs of every baseline run:
    $ PYTHONPATH=. python scripts/probe_batchnorm.py \\
        --checkpoint-glob "checkpoints/e0/checkpoints/b0_*/epoch-[123].pt"

Rehearse the path on a handful of images:
    $ PYTHONPATH=. python scripts/probe_batchnorm.py \\
        --checkpoint-glob "checkpoints/e0/checkpoints/b0_*/epoch-[12].pt" \\
        --n-images 12 --n-calibration-images 8
"""
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

from materials_vision.evaluation import (FROZEN_WATERSHED, AggregateResult,
                                         aggregate, load_size_bins)
from materials_vision.evaluation.inference import (
    build_segmenter, prepared_images, recalibrate_batch_statistics,
    score_settings)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.training import (LORA_RANK, PEFT_KWARGS, build_source,
                                       prepare_geometry)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

MODEL_TYPE = "vit_l"

# Training images the statistics are recomputed over. Enough that the
# average is stable, few enough that the probe stays cheap; they are
# spread through the ordering so that both microscopes and both scale
# bins are represented, since the statistics are meant to describe the
# training distribution rather than one corner of it.
N_CALIBRATION_IMAGES = 100

# The spread this is trying to explain, measured over the unaugmented
# baseline runs, and the seed-to-seed spread it should be compared
# against.
OBSERVED_NEIGHBOUR_SPREAD = 0.045

SEED_SPREAD = 0.0136


def spaced_positions(n_available: int, n_wanted: int) -> list[int]:
    """Evenly spaced positions, or every one of them.

    Parameters
    ----------
    n_available : int
    n_wanted : int
        Zero or less asks for all of them.

    Returns
    -------
    list of int
    """
    if n_wanted <= 0 or n_wanted >= n_available:
        return list(range(n_available))
    stride = n_available / n_wanted
    return [int(step * stride) for step in range(n_wanted)]


def epoch_of(path: Path) -> int:
    """Epoch a snapshot's filename names, or zero.

    Parameters
    ----------
    path : Path

    Returns
    -------
    int
    """
    match = re.search(r"epoch-(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Expand the patterns, in the order the snapshots were trained.

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
    unique = list(dict.fromkeys(found))
    return sorted(unique, key=lambda path: (path.parent.name, epoch_of(path)))


def score(segmenter, source, positions, size_bins, label) -> AggregateResult:
    """Segment the chosen images and pool the result.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
    source : SampleSource
    positions : list of int
    size_bins : SizeBins
    label : str

    Returns
    -------
    AggregateResult
    """
    measured = score_settings(
        segmenter, source, positions, size_bins, (FROZEN_WATERSHED,)
    )
    return aggregate(measured[FROZEN_WATERSHED], label=label)


def probe_one(
    checkpoint: Path,
    val_source,
    val_positions: list[int],
    size_bins,
    calibration_images: list,
) -> tuple[AggregateResult, AggregateResult]:
    """Score one snapshot as saved and after recalibration.

    The order matters: recalibration replaces the statistics in place,
    so the snapshot has to be scored as saved first.

    Parameters
    ----------
    checkpoint : Path
    val_source : SampleSource
    val_positions : list of int
    size_bins : SizeBins
    calibration_images : list of np.ndarray

    Returns
    -------
    tuple of AggregateResult
        As saved, then recalibrated.
    """
    segmenter = build_segmenter(
        checkpoint, MODEL_TYPE, PEFT_KWARGS, LORA_RANK
    )
    as_saved = score(
        segmenter, val_source, val_positions, size_bins,
        f"{checkpoint}@as-saved",
    )
    layers = recalibrate_batch_statistics(segmenter, calibration_images)
    if layers == 0:
        logger.warning(
            "%s: the decoder holds no batch-normalization layers, so "
            "this probe cannot say anything about it.", checkpoint,
        )
    recalibrated = score(
        segmenter, val_source, val_positions, size_bins,
        f"{checkpoint}@recalibrated",
    )
    return as_saved, recalibrated


def neighbour_spread(rows: list[dict], field: str) -> float:
    """Largest gap between snapshots one epoch apart within a run.

    This, not the spread over a whole run, is what the probe is about:
    a run that improves steadily has a large total spread and small
    neighbouring ones, and only the second kind is noise. Gaps are
    taken within a run and never across two of them, since snapshots
    from different seeds are not neighbours in any useful sense.

    Parameters
    ----------
    rows : list of dict
        In the order the snapshots were trained.
    field : str
        Which score to read from each row.

    Returns
    -------
    float
        Zero when no two snapshots of one run were given.
    """
    by_run: dict[str, list[float]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row[field])
    gaps = [
        abs(later - earlier)
        for scores in by_run.values()
        for earlier, later in zip(scores, scores[1:])
    ]
    return max(gaps) if gaps else 0.0


def report(rows: list[dict]) -> None:
    """Write the comparison and say what it supports.

    Parameters
    ----------
    rows : list of dict
    """
    logger.info(
        "%-34s %10s %10s %10s", "snapshot", "as-saved", "recalib.", "delta",
    )
    for row in rows:
        logger.info(
            "%-34s %10.4f %10.4f %+10.4f",
            row["name"], row["as_saved_f1"], row["recalibrated_f1"],
            row["recalibrated_f1"] - row["as_saved_f1"],
        )

    before = neighbour_spread(rows, "as_saved_f1")
    after = neighbour_spread(rows, "recalibrated_f1")
    logger.info(
        "Largest gap between neighbouring snapshots: %.4f as saved, "
        "%.4f recalibrated.", before, after,
    )
    if before == 0.0:
        logger.info(
            "No two snapshots of one run were given, so there is no "
            "gap to shrink; pass at least two neighbouring epochs."
        )
        return
    logger.info(
        "For reference, the spread between independent seeds is %.4f "
        "and the baseline runs showed neighbouring gaps up to %.4f.",
        SEED_SPREAD, OBSERVED_NEIGHBOUR_SPREAD,
    )
    if after < before / 2:
        logger.info(
            "The gap more than halves once the snapshots share their "
            "batch statistics, so much of it was the statistics rather "
            "than the models."
        )
    elif after < before:
        logger.info(
            "The gap narrows but survives, so the statistics explain "
            "part of it and something else explains the rest."
        )
    else:
        logger.info(
            "The gap does not narrow, so the snapshots differ as "
            "models and the batch statistics are not the cause."
        )


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
        help="Snapshot to probe; repeat for several.",
    )
    parser.add_argument(
        "--checkpoint-glob", action="append", default=[],
        help="Pattern selecting snapshots; repeatable.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="Score this many validation images; 0 means all of them.",
    )
    parser.add_argument(
        "--n-calibration-images", type=int,
        default=N_CALIBRATION_IMAGES,
        help="Training images the statistics are recomputed over.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the full figures as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Score every snapshot twice and compare the spreads.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    checkpoints = resolve_checkpoints(args)
    if not checkpoints:
        logger.error(
            "No snapshot selected. Pass --checkpoint or --checkpoint-glob."
        )
        return EXIT_FAILED
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        logger.error(
            "No such snapshot: %s.",
            ", ".join(str(path) for path in missing),
        )
        return EXIT_FAILED

    prepare_geometry()
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    size_bins = load_size_bins(args.size_bins)

    calibration_positions = spaced_positions(
        len(train_source), args.n_calibration_images
    )
    calibration_images = prepared_images(
        train_source, calibration_positions
    )
    val_positions = spaced_positions(len(val_source), args.n_images)
    logger.info(
        "Probing %d snapshot(s) on %d validation image(s), statistics "
        "recomputed over %d training image(s).",
        len(checkpoints), len(val_positions), len(calibration_images),
    )

    rows = []
    started_s = time.perf_counter()
    for checkpoint in checkpoints:
        as_saved, recalibrated = probe_one(
            checkpoint, val_source, val_positions, size_bins,
            calibration_images,
        )
        rows.append({
            "name": f"{checkpoint.parent.name}/{checkpoint.stem}",
            "run": checkpoint.parent.name,
            "epoch": epoch_of(checkpoint),
            "checkpoint": str(checkpoint),
            "as_saved_f1": as_saved.f1,
            "recalibrated_f1": recalibrated.f1,
            "as_saved_splits_per_100_gt": as_saved.splits_per_100_gt,
            "recalibrated_splits_per_100_gt":
                recalibrated.splits_per_100_gt,
        })
        logger.info(
            "%s: %.4f as saved, %.4f recalibrated.",
            rows[-1]["name"], as_saved.f1, recalibrated.f1,
        )
    logger.info(
        "Probed %d snapshot(s) in %.1f s.",
        len(rows), time.perf_counter() - started_s,
    )

    report(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "provenance": run_provenance(),
            "n_calibration_images": len(calibration_images),
            "calibration_positions": calibration_positions,
            "n_validation_images": len(val_positions),
            "watershed": FROZEN_WATERSHED.to_kwargs(),
            "rows": rows,
        }, indent=2, default=str))
        logger.info("Wrote %s.", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
