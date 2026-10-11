"""Score the classical watershed segmentations as the models were scored.

The classical predictions - marker-controlled watershed pipelines run
in Fiji, with parameters set by hand for each image - are measured
with the evaluation the segmentation models were measured with: the
same annotation, read through the same sample source and content
crop, the same pore matching at mask IoU 0.5, the same pooled figures
and cross-sections, and a JSON report of the same layout.

Each variant's mask is scored as the method returned it. These
pipelines partition the whole frame, so the walls between pores are
assigned to the pores beside them; that shows up in the porosity
error, the diameters and the overlap of matched pairs, and it is a
property of the method rather than something to correct after the
results are known.

Run from the repository root::

    PYTHONPATH=. python scripts/evaluate_morphological_operations.py \\
        --unlock-test
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

from materials_vision.data.samples import SampleSourceError
from materials_vision.data.split_io import LockedTestSetError, SplitLoadError
from materials_vision.evaluation import (DECISION_SCALE, ImageEvaluation,
                                         SizeBinsLoadError, aggregate,
                                         evaluate_image, load_size_bins)
from materials_vision.logging_config import setup_logging

from .alignment import prediction_on_content_frame
from .data_loading import (ClassicalSample, PredictionLoadError, find_samples,
                           load_prediction_labels)
from .ground_truth import sources_for_images
from .reporting import log_headline, write_json
from .visualization import render_sample_figure, save_sample_figure

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_PREDICTIONS_ROOT = Path(
    "/home/dwalicki/results/operacje_morfologiczne/predykcje"
)

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

DEFAULT_OUT_DIR = Path("outputs/morphological_evaluation")

DEFAULT_VARIANTS = ("interactive_watershed", "marker_watershed")

SCORES_FILE = "classical_scores.json"

GT_TITLE = "Reczna adnotacja (GT)"


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
        "--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS),
        help="Variant subdirectory names to score.",
    )
    parser.add_argument(
        "--unlock-test", action="store_true",
        help="Read the annotation of images that belong to TEST. No "
             "model is scored on it.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip the overlay figures.",
    )
    return parser.parse_args(argv)


def score_samples(
    samples: tuple[ClassicalSample, ...],
    args: argparse.Namespace,
) -> dict[str, list[ImageEvaluation]]:
    """Evaluate every variant on every sample.

    Parameters
    ----------
    samples : tuple of ClassicalSample
    args : argparse.Namespace

    Returns
    -------
    dict
        Variant name to its per-image evaluations, in the same image
        order for every variant.
    """
    by_id = {sample.image_id: sample for sample in samples}
    sources = sources_for_images(
        args.split, args.manifest, by_id, allow_test=args.unlock_test,
    )
    size_bins = load_size_bins(args.size_bins)
    evaluations: dict[str, list[ImageEvaluation]] = {
        variant: [] for variant in args.variants
    }
    for source in sources:
        for index in range(len(source)):
            prepared = source.load(index)
            record = prepared.record
            predictions = {
                variant: prediction_on_content_frame(
                    load_prediction_labels(path), record
                )
                for variant, path in by_id[record.image_id].mask_paths.items()
            }
            for variant, pred_labels in predictions.items():
                evaluations[variant].append(evaluate_image(
                    record, prepared.labels, pred_labels,
                    size_bins=size_bins, boundary_scales=(DECISION_SCALE,),
                ))
            if not args.no_figures:
                figure = render_sample_figure(
                    prepared.image, prepared.labels, predictions,
                    [GT_TITLE, *predictions],
                )
                save_sample_figure(figure, args.out_dir, record.image_id)
    return evaluations


def run(args: argparse.Namespace) -> int:
    """Score the classical predictions and write the report.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
        Process exit code.
    """
    try:
        samples = find_samples(args.predictions_root, tuple(args.variants))
        evaluations = score_samples(samples, args)
    except (PredictionLoadError, SampleSourceError, SplitLoadError,
            LockedTestSetError, SizeBinsLoadError) as error:
        logger.error("%s", error)
        return EXIT_FAILED
    write_json(args.out_dir / SCORES_FILE, evaluations)
    log_headline({
        variant: aggregate(items, label=variant)
        for variant, items in evaluations.items()
    })
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    """Module entry point.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    return run(args)
