#!/usr/bin/env python3
"""
Health check for the evaluation layer, on real masks.

The unit tests pin each metric on small synthetic frames. This script
answers a different question: run the whole layer over real annotated
images, deform them in ways whose consequence is known in advance, and
verify that every metric moves the way it must. A metric that is
correct on a 20x30 test frame and silently wrong on a 1280x960 foam
with five hundred pores would pass the tests and fail here.

Each deformation is built from the annotation itself, so the expected
outcome is not a judgement call:

- **identity** - the prediction is the annotation. Everything must be
  perfect; anything that is not is a defect in the metric, since there
  is no error to measure.
- **shrink** / **grow** - pores eroded, or the erosion treated as the
  annotation so the prediction is the larger of the two. Diameters,
  porosity and the size distribution must move, and in opposite
  directions between the two.
- **missing** / **spurious** - instances removed from the prediction,
  or from the annotation so the prediction carries extra ones. Recall
  and precision must fall respectively, and the signed pore count
  error must change sign between them.
- **merged** / **split** - neighbouring pores joined, or single pores
  cut in two. The dedicated counters must fire, and only the matching
  one.
- **rotated** - both sides turned by a quarter turn. Every figure must
  be identical to the identity case: the metrics are compared across
  the eight orientations of the D4 augmentation family, so an
  orientation-dependent metric would make that comparison meaningless.

The script reports one line per expectation and exits non-zero if any
of them fails, so it can be run as a regression gate after a change to
the layer.

Examples
--------
Run on a dozen images at the decision tolerance only:
    $ python scripts/check_evaluation_layer.py

Run over the whole validation set at every tolerance:
    $ python scripts/check_evaluation_layer.py --n-images 0 --all-scales
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, grey_erosion

from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.evaluation import (DECISION_SCALE, DIAGNOSTIC_SCALES,
                                         AggregateResult, aggregate,
                                         evaluate_image, load_size_bins)
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED_CHECKS = 1
EXIT_FATAL = 2

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v2/manifest_v2.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

A_MIN_FRAGMENT_PX2 = 432.0

DEFAULT_N_IMAGES = 12

SAMPLE_SEED = 20260902

EROSION_FOOTPRINT = np.ones((5, 5), dtype=bool)

DEFORMED_FRACTION = 0.1


@dataclass(frozen=True)
class Check:
    """One expectation about how a deformation must show up.

    Parameters
    ----------
    deformation : str
        Which deformation the expectation applies to.
    description : str
        What is being asserted, in words.
    passed : bool
    observed : str
        The value that decided it, for the report.
    """

    deformation: str
    description: str
    passed: bool
    observed: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument("--subset", default="val", choices=["train", "val"])
    parser.add_argument(
        "--n-images", type=int, default=DEFAULT_N_IMAGES,
        help="images to sample; 0 uses the whole subset",
    )
    parser.add_argument(
        "--all-scales", action="store_true",
        help=(
            "score every boundary tolerance instead of the decision "
            "one alone; several times slower"
        ),
    )
    return parser.parse_args(argv)


def shrink(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erode every pore, leaving the walls thicker than annotated."""
    eroded = grey_erosion(labels, footprint=EROSION_FOOTPRINT)
    eroded[labels == 0] = 0
    return eroded


def drop_instances(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Remove a tenth of the instances outright."""
    present = np.unique(labels[labels > 0])
    n_dropped = max(1, int(round(present.size * DEFORMED_FRACTION)))
    doomed = rng.choice(present, size=n_dropped, replace=False)
    out = labels.copy()
    out[np.isin(out, doomed)] = 0
    return out


def merge_neighbours(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Give a tenth of the pores the label of a neighbour."""
    present = np.unique(labels[labels > 0])
    n_merged = max(1, int(round(present.size * DEFORMED_FRACTION)))
    out = labels.copy()
    for label in rng.choice(present, size=n_merged, replace=False):
        touching = out[binary_dilation(out == label) & (out != label)]
        touching = touching[touching > 0]
        if touching.size:
            out[out == np.bincount(touching).argmax()] = label
    return out


def split_instances(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cut a tenth of the pores in half, giving each half an id."""
    present = np.unique(labels[labels > 0])
    n_split = max(1, int(round(present.size * DEFORMED_FRACTION)))
    out = labels.copy()
    next_label = int(out.max()) + 1
    for label in rng.choice(present, size=n_split, replace=False):
        rows = np.nonzero((out == label).any(axis=1))[0]
        if rows.size < 4:
            continue
        middle = rows[rows.size // 2]
        out[:middle][out[:middle] == label] = next_label
        next_label += 1
    return out


def _rotate(labels: np.ndarray) -> np.ndarray:
    """Turn the frame by a quarter turn."""
    return np.rot90(labels).copy()


def evaluate_deformation(
    source: SampleSource,
    indices: np.ndarray,
    size_bins,
    scales: tuple[float, ...],
    build: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    *,
    swap_sides: bool = False,
    rotate: bool = False,
) -> AggregateResult:
    """Evaluate one deformation over the sampled images.

    Parameters
    ----------
    source : SampleSource
    indices : np.ndarray
        Positions in the source to evaluate.
    size_bins : SizeBins
    scales : tuple of float
    build : callable
        Turns an annotation into the deformed label image.
    swap_sides : bool, optional
        Treat the deformation as the annotation and the original as
        the prediction, which inverts the direction of every error
        without needing a second deformation.
    rotate : bool, optional
        Turn both sides by a quarter turn before evaluating.

    Returns
    -------
    AggregateResult
    """
    evaluations = []
    for index in indices:
        sample = source.load(int(index))
        rng = np.random.default_rng(SAMPLE_SEED + int(index))
        deformed = build(sample.labels, rng)
        gt, pred = ((deformed, sample.labels) if swap_sides
                    else (sample.labels, deformed))
        if rotate:
            gt, pred = _rotate(gt), _rotate(pred)
        evaluations.append(evaluate_image(
            sample.record, gt, pred,
            size_bins=size_bins, boundary_scales=scales,
        ))
    return aggregate(evaluations)


def _identity(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The annotation itself; nothing to measure but perfection."""
    return labels.copy()


def build_checks(
    results: dict[str, AggregateResult], scale: float
) -> list[Check]:
    """Turn the aggregated results into pass or fail expectations."""
    checks: list[Check] = []

    def expect(name: str, description: str, passed: bool, observed):
        checks.append(Check(name, description, bool(passed), str(observed)))

    perfect = results["identity"]
    expect("identity", "instance F1 is exactly 1",
           perfect.f1 == 1.0, f"{perfect.f1:.6f}")
    expect("identity", "boundary F1 is exactly 1",
           perfect.boundary_f1[scale] == 1.0,
           f"{perfect.boundary_f1[scale]:.6f}")
    expect("identity", "no merges and no splits",
           perfect.n_merges == 0 and perfect.n_splits == 0,
           f"{perfect.n_merges}/{perfect.n_splits}")
    expect("identity", "no shape error",
           np.isclose(perfect.median_diameter_error, 0.0),
           f"{perfect.median_diameter_error:.2e}")
    expect("identity", "no distribution drift",
           np.isclose(perfect.wasserstein_um, 0.0),
           f"{perfect.wasserstein_um:.2e}")
    expect("identity", "no porosity error",
           np.isclose(perfect.mean_porosity_error_pp, 0.0),
           f"{perfect.mean_porosity_error_pp:.2e}")

    shrunk, grown = results["shrink"], results["grow"]
    expect("shrink", "predicted pores are smaller",
           shrunk.median_diameter_drift_um < 0,
           f"{shrunk.median_diameter_drift_um:+.2f} um")
    expect("shrink", "porosity falls",
           shrunk.mean_porosity_error_pp < 0,
           f"{shrunk.mean_porosity_error_pp:+.2f} pp")
    expect("grow", "predicted pores are larger",
           grown.median_diameter_drift_um > 0,
           f"{grown.median_diameter_drift_um:+.2f} um")
    expect("grow", "porosity rises",
           grown.mean_porosity_error_pp > 0,
           f"{grown.mean_porosity_error_pp:+.2f} pp")
    expect("shrink/grow", "the size drift reverses between them",
           np.sign(shrunk.median_diameter_drift_um)
           != np.sign(grown.median_diameter_drift_um),
           f"{shrunk.median_diameter_drift_um:+.2f} vs "
           f"{grown.median_diameter_drift_um:+.2f}")
    expect("shrink", "boundary agreement falls further than instance F1",
           (1 - shrunk.boundary_f1[scale]) > (1 - shrunk.f1),
           f"bF1 {shrunk.boundary_f1[scale]:.4f} vs F1 {shrunk.f1:.4f}")

    missing, spurious = results["missing"], results["spurious"]
    expect("missing", "recall falls and precision does not",
           missing.recall < 1.0 and missing.precision == 1.0,
           f"P {missing.precision:.4f} R {missing.recall:.4f}")
    expect("missing", "too few pores are predicted",
           missing.pore_count_error < 0,
           f"{missing.pore_count_error:+.4f}")
    expect("spurious", "precision falls and recall does not",
           spurious.precision < 1.0 and spurious.recall == 1.0,
           f"P {spurious.precision:.4f} R {spurious.recall:.4f}")
    expect("spurious", "too many pores are predicted",
           spurious.pore_count_error > 0,
           f"{spurious.pore_count_error:+.4f}")

    merged, split = results["merged"], results["split"]
    expect("merged", "merges are counted and splits are not",
           merged.n_merges > 0 and merged.n_splits == 0,
           f"{merged.n_merges} merges, {merged.n_splits} splits")
    expect("split", "splits are counted and merges are not",
           split.n_splits > 0 and split.n_merges == 0,
           f"{split.n_splits} splits, {split.n_merges} merges")

    rotated = results["rotated"]
    expect("rotated", "instance F1 survives a quarter turn",
           rotated.f1 == perfect.f1,
           f"{rotated.f1:.6f} vs {perfect.f1:.6f}")
    expect("rotated", "boundary F1 survives a quarter turn",
           np.isclose(rotated.boundary_f1[scale],
                      perfect.boundary_f1[scale]),
           f"{rotated.boundary_f1[scale]:.6f} vs "
           f"{perfect.boundary_f1[scale]:.6f}")
    expect("rotated", "the instance population is unchanged",
           rotated.n_gt == perfect.n_gt,
           f"{rotated.n_gt} vs {perfect.n_gt}")
    expect("rotated", "orientation concentration is unchanged",
           np.isclose(rotated.orientation.gt_grid_resultant,
                      perfect.orientation.gt_grid_resultant),
           f"{rotated.orientation.gt_grid_resultant:.6f} vs "
           f"{perfect.orientation.gt_grid_resultant:.6f}")

    return checks


def report(checks: list[Check]) -> int:
    """Log every expectation and return the number that failed."""
    n_failed = 0
    for check in checks:
        if check.passed:
            logger.info("  PASS  %-12s %-46s %s",
                        check.deformation, check.description, check.observed)
        else:
            n_failed += 1
            logger.error("  FAIL  %-12s %-46s %s",
                         check.deformation, check.description,
                         check.observed)
    return n_failed


def main(argv: list[str] | None = None) -> int:
    """Run the health check and report.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    try:
        size_bins = load_size_bins(args.size_bins)
        subset = load_split(args.split, subset=args.subset)
        manifest = read_manifest(args.manifest)
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("Cannot start the check: %s", error)
        return EXIT_FATAL

    source = SampleSource(subset, manifest,
                          min_fragment_area_px2=A_MIN_FRAGMENT_PX2)
    indices = _sample_indices(len(source), args.n_images)
    scales = DIAGNOSTIC_SCALES if args.all_scales else (DECISION_SCALE,)
    scale = DECISION_SCALE if DECISION_SCALE in scales else scales[0]
    logger.info(
        "Checking the evaluation layer on %d of %d %s image(s), "
        "boundary tolerance(s) %s.",
        indices.size, len(source), args.subset, scales,
    )

    deformations = {
        "identity": (_identity, False),
        "shrink": (shrink, False),
        "grow": (shrink, True),
        "missing": (drop_instances, False),
        "spurious": (drop_instances, True),
        "merged": (merge_neighbours, False),
        "split": (split_instances, False),
    }
    results = {
        name: evaluate_deformation(source, indices, size_bins, scales,
                                   build, swap_sides=swap)
        for name, (build, swap) in deformations.items()
    }
    results["rotated"] = evaluate_deformation(
        source, indices, size_bins, scales, _identity, rotate=True
    )

    _log_summary(results, scale)
    n_failed = report(build_checks(results, scale))
    if n_failed:
        logger.error("%d expectation(s) failed.", n_failed)
        return EXIT_FAILED_CHECKS
    logger.info("All expectations hold.")
    return EXIT_OK


def _sample_indices(n_available: int, n_requested: int) -> np.ndarray:
    """Choose which images to run on, reproducibly."""
    if n_requested <= 0 or n_requested >= n_available:
        return np.arange(n_available)
    rng = np.random.default_rng(SAMPLE_SEED)
    return np.sort(rng.choice(n_available, size=n_requested,
                              replace=False))


def _log_summary(
    results: dict[str, AggregateResult], scale: float
) -> None:
    """Log the figures the expectations are drawn from."""
    logger.info(
        "%-10s %7s %7s %7s %8s %6s %6s %9s %9s",
        "deformacja", "F1", "bF1", "meanIoU", "W[um]", "merge",
        "split", "dPore", "poroz[pp]",
    )
    for name, result in results.items():
        logger.info(
            "%-10s %7.4f %7.4f %7.4f %8.2f %6d %6d %+9.4f %+9.2f",
            name, result.f1, result.boundary_f1[scale],
            result.mean_pair_iou, result.wasserstein_um,
            result.n_merges, result.n_splits, result.pore_count_error,
            result.mean_porosity_error_pp,
        )


if __name__ == "__main__":
    sys.exit(main())
