#!/usr/bin/env python3
"""
Where the thresholds that separate touching pores belong.

The decoder does not predict instances. It predicts, per pixel, a
foreground probability and two distances - to the pore's centre and to
its boundary. Instances appear afterwards: pixels where both distances
are small become seeds, and a watershed grows them until they meet. The
thresholds deciding what counts as small are read against the scale of
those distance maps, and fine-tuning moves that scale.

**Why that matters enough to calibrate.** Left at the value that suited
the decoder before any training, the thresholds make the reported
metric measure the drift of that scale as well as the quality of the
model. Measured on one baseline run, a snapshot scores 0.807 at a
centre threshold of one half and 0.833 at three tenths - the same
predictions underneath, only the seeding changed. Roughly two thirds of
what looked like a model getting worse over its last eighteen epochs
was the threshold falling out of step with it.

**Calibrated on training images, not on validation ones.** The
validation split decides which checkpoint the study reports, so a
threshold chosen by looking at validation scores would be chosen to
flatter the thing it is later used to measure. Training images have
already been learned from, so reading one more number off them costs
nothing that is not already spent.

**Calibrated once, then frozen.** Whatever value this produces applies
to every run afterwards - baseline, candidates, ablations and the final
test - because a comparison that attributes a difference to
augmentation needs everything else held still. It is therefore worth
running on a checkpoint from the configuration that will actually be
used, not on an older one.

**The result is reported alongside a robustness check.** A single
calibrated value invites the objection that it was chosen on runs
without augmentation and so favours them. The answer is evidence rather
than argument: the main metric is reported at several thresholds, and a
policy that wins at one and loses at the others has not won.

Examples
--------
Calibrate on the best snapshot of a run:
    $ PYTHONPATH=. python scripts/calibrate_watershed.py \\
        --checkpoint checkpoints/e0/checkpoints/b0_seed20260907/epoch-2.pt \\
        --out checkpoints/watershed_calibration.json

Rehearse the path in a couple of minutes:
    $ PYTHONPATH=. python scripts/calibrate_watershed.py \\
        --checkpoint <path> --n-images 6 \\
        --center-thresholds 0.4 0.5 --boundary-thresholds 0.5
"""
import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from materials_vision.evaluation import (AggregateResult, WatershedParams,
                                         aggregate)
from materials_vision.evaluation.inference import (build_segmenter,
                                                   score_settings)
from materials_vision.evaluation.postprocessing_calibration import (
    boundary_at_decision_scale, f1_noise, rank_key)
from materials_vision.evaluation.size_bins import load_size_bins
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

# Calibrated on TRAIN, never on VALIDATION; see the module docstring.
SUBSET = "train"

# Enough images for the pooled score to be stable - they carry some
# forty pores each, so this is on the order of two thousand instances -
# and few enough that the whole grid fits in one sitting. Spread
# through the ordering so that both microscopes and both scale bins are
# represented.
N_IMAGES = 48

# The seeding thresholds. Below the library's default in both cases,
# because that is the direction fine-tuning moves the distance maps:
# the trained decoder predicts smaller distances, so the default seeds
# too freely and cuts single pores in two.
CENTER_THRESHOLDS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)

BOUNDARY_THRESHOLDS = (0.4, 0.5, 0.6)


def grid(
    center_thresholds: list[float], boundary_thresholds: list[float]
) -> list[WatershedParams]:
    """Every combination of the two seeding thresholds.

    Everything else is left at its frozen value: these two decide where
    seeds appear, which is what the drift affects, and varying more
    would turn a calibration into a search.

    Parameters
    ----------
    center_thresholds, boundary_thresholds : list of float

    Returns
    -------
    list of WatershedParams
    """
    return [
        WatershedParams(
            center_distance_threshold=center,
            boundary_distance_threshold=boundary,
        )
        for center, boundary in itertools.product(
            center_thresholds, boundary_thresholds
        )
    ]


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


def report(
    ranked: list[tuple[WatershedParams, AggregateResult]]
) -> None:
    """Write the grid, best first, and name the winner.

    Parameters
    ----------
    ranked : list of tuple
    """
    logger.info(
        "%8s %8s %8s %8s %8s %8s %8s",
        "centre", "boundary", "f1", "boundary", "splits", "merges", "count",
    )
    for setting, result in ranked:
        logger.info(
            "%8.2f %8.2f %8.4f %8.4f %8.2f %8.2f %+8.4f",
            setting.center_distance_threshold,
            setting.boundary_distance_threshold,
            result.f1, boundary_at_decision_scale(result),
            result.splits_per_100_gt, result.merges_per_100_gt,
            result.pore_count_error,
        )
    best, best_result = ranked[0]
    worst_result = ranked[-1][1]
    logger.info(
        "Best: centre %.2f, boundary %.2f, at %.4f - %.4f above the "
        "worst setting of the grid.",
        best.center_distance_threshold,
        best.boundary_distance_threshold,
        best_result.f1, best_result.f1 - worst_result.f1,
    )
    if best.center_distance_threshold in (
        min(s.center_distance_threshold for s, _ in ranked),
        max(s.center_distance_threshold for s, _ in ranked),
    ):
        logger.warning(
            "The winner sits at the edge of the grid, so the optimum "
            "may lie outside it. Widen --center-thresholds and repeat "
            "before freezing this value."
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
        "--checkpoint", type=Path, required=True,
        help="Snapshot to calibrate against; use the best one of the "
             "configuration that will actually run.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument(
        "--n-images", type=int, default=N_IMAGES,
        help="Training images to calibrate on; 0 means all of them.",
    )
    parser.add_argument(
        "--center-thresholds", type=float, nargs="+",
        default=list(CENTER_THRESHOLDS),
    )
    parser.add_argument(
        "--boundary-thresholds", type=float, nargs="+",
        default=list(BOUNDARY_THRESHOLDS),
    )
    parser.add_argument(
        "--tie-tolerance", type=float, default=None,
        help="F1 differences below this count as ties and are settled "
             "on the other criteria. Defaults to the sampling "
             "uncertainty of the subsample being scored.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the chosen setting and the whole grid.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Score the grid on training images and name the best setting.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if not args.checkpoint.exists():
        logger.error("No such snapshot: %s.", args.checkpoint)
        return EXIT_FAILED

    prepare_geometry()
    source = build_source(args.split, args.manifest, SUBSET)
    size_bins = load_size_bins(args.size_bins)
    positions = spaced_positions(len(source), args.n_images)
    settings = grid(args.center_thresholds, args.boundary_thresholds)
    logger.info(
        "Calibrating %s over %d setting(s) on %d %s image(s).",
        args.checkpoint, len(settings), len(positions), SUBSET.upper(),
    )

    segmenter = build_segmenter(
        args.checkpoint, MODEL_TYPE, PEFT_KWARGS, LORA_RANK
    )
    started_s = time.perf_counter()
    measured = score_settings(
        segmenter, source, positions, size_bins, settings
    )
    logger.info(
        "Scored the grid in %.1f s.", time.perf_counter() - started_s
    )

    scored = [
        (setting, aggregate(evaluations, label=setting.label()))
        for setting, evaluations in measured.items()
    ]
    tolerance = (
        args.tie_tolerance if args.tie_tolerance is not None
        else max(f1_noise(result) for _, result in scored)
    )
    logger.info(
        "Treating F1 differences below %.4f as ties, and resolving "
        "them on wall agreement, pore count and merge or split count.",
        tolerance,
    )
    ranked = sorted(
        scored, key=lambda item: rank_key(item[1], tolerance), reverse=True
    )
    report(ranked)

    if args.out is not None:
        best, best_result = ranked[0]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "provenance": run_provenance(),
            "checkpoint": str(args.checkpoint),
            "subset": SUBSET,
            "positions": positions,
            "tie_tolerance": tolerance,
            "chosen": best.to_kwargs(),
            "grid": [
                {
                    "setting": setting.to_kwargs(),
                    "f1": result.f1,
                    "boundary_f1": boundary_at_decision_scale(result),
                    "splits_per_100_gt": result.splits_per_100_gt,
                    "merges_per_100_gt": result.merges_per_100_gt,
                    "pore_count_error": result.pore_count_error,
                }
                for setting, result in ranked
            ],
        }, indent=2, default=str))
        logger.info("Wrote %s.", args.out)
        logger.info(
            "Freeze the chosen setting in "
            "materials_vision/evaluation/watershed.py before starting "
            "the runs it will apply to; changing it afterwards "
            "invalidates every comparison made under the old one."
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
