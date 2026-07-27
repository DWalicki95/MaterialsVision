"""CLI entry point for morphological watershed evaluation.

Run from the repository root::

    python scripts/evaluate_morphological_operations.py \\
        --json-path project-26-....json \\
        --predictions-root /path/to/results_root \\
        --output-dir outputs/morphological_evaluation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np # type: ignore

from .alignment import align_mask_to_shape
from .data_loading import (
    find_original_image,
    find_prediction_masks,
    iter_sample_dirs,
    load_image,
    load_tif,
    match_sample_to_gt,
)
from .ground_truth import (
    build_ground_truth_index,
    load_label_studio_tasks,
    percent_points_to_pixels,
    rasterize_instances,
)
from .reporting import (
    evaluate_variant,
    evaluate_variant_per_sample,
    write_xlsx_report,
)
from .visualization import render_sample_figure, save_sample_figure

logger = logging.getLogger(__name__)

DEFAULT_VARIANTS = ["interactive_watershed", "marker_watershed"]
GT_TITLE = "Reczna adnotacja (GT)"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate watershed segmentation against Label Studio "
                    "annotations."
    )
    parser.add_argument("--json-path", type=Path, required=True,
                        help="Label Studio JSON export.")
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="Root directory with prediction outputs.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for the report and visualizations.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU greedy-match threshold.")
    parser.add_argument("--boundary-scales", type=float, nargs="+",
                        default=[0.2], help="Boundary tolerance scales.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="Watershed variant subdirectory names.")
    parser.add_argument("--label-name", default="Por",
                        help="Polygon label to evaluate.")
    return parser.parse_args(argv)


def configure_logging() -> None:
    """Configure root logging with timestamps at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _count_instances(mask: np.ndarray) -> int:
    """Return the number of non-background instances in a mask."""
    labels = np.unique(mask)
    return int(np.count_nonzero(labels != 0))


def _build_gt_mask(gt_entry: dict, shape: tuple) -> np.ndarray:
    """Rasterize ground-truth polygons at the original image shape."""
    height, width = shape
    polygons_px = [
        percent_points_to_pixels(points, width, height)
        for points in gt_entry["polygons_pct"]
    ]
    return rasterize_instances(polygons_px, (height, width))


def _collect_sample(
    sample_dir: Path, gt_index: Dict[str, dict], variants: List[str]
):
    """
    Build the data needed for one sample.

    Returns a dict with ``stem``, ``image``, ``gt_mask`` and
    ``pred_masks`` (variant to aligned mask), or None when the sample
    cannot be evaluated.
    """
    gt_entry = match_sample_to_gt(sample_dir.name, gt_index)
    if gt_entry is None:
        return None
    image_path = find_original_image(sample_dir)
    if image_path is None:
        return None
    image = load_image(image_path)
    shape = image.shape[:2]
    gt_mask = _build_gt_mask(gt_entry, shape)
    if _count_instances(gt_mask) == 0:
        logger.warning("Sample %s has empty ground-truth, skipping",
                       sample_dir.name)
        return None
    mask_paths = find_prediction_masks(sample_dir, variants)
    if not mask_paths:
        return None
    pred_masks = {
        variant: align_mask_to_shape(load_tif(path), shape)
        for variant, path in mask_paths.items()
    }
    return {
        "stem": sample_dir.name,
        "image": image,
        "gt_mask": gt_mask,
        "pred_masks": pred_masks,
    }


def _accumulate(variant_batches: Dict[str, dict], sample: dict) -> None:
    """Append a sample's masks to the per-variant evaluation batches."""
    for variant, pred_mask in sample["pred_masks"].items():
        batch = variant_batches.setdefault(
            variant, {"stems": [], "true": [], "pred": []}
        )
        batch["stems"].append(sample["stem"])
        batch["true"].append(sample["gt_mask"])
        batch["pred"].append(pred_mask)


def _visualize(sample: dict, output_dir: Path) -> None:
    """Render and save the overlay figure for one sample."""
    titles = [GT_TITLE] + list(sample["pred_masks"].keys())
    figure = render_sample_figure(
        sample["image"], sample["gt_mask"], sample["pred_masks"], titles
    )
    save_sample_figure(figure, output_dir, sample["stem"])


def _report(
    variant_batches: Dict[str, dict],
    output_dir: Path,
    threshold: float,
    scales: List[float],
) -> None:
    """Compute metrics for all variants and write the xlsx report."""
    aggregate_by_variant = {}
    per_sample_by_variant = {}
    for variant, batch in variant_batches.items():
        aggregate_by_variant[variant] = evaluate_variant(
            batch["true"], batch["pred"], threshold, scales
        )
        per_sample_by_variant[variant] = evaluate_variant_per_sample(
            batch["stems"], batch["true"], batch["pred"], threshold, scales
        )
    write_xlsx_report(
        output_dir, aggregate_by_variant, per_sample_by_variant
    )


def run(args: argparse.Namespace) -> int:
    """Run the full evaluation pipeline. Returns a process exit code."""
    if not args.json_path.is_file():
        logger.error("JSON file not found: %s", args.json_path)
        return 1
    if not args.predictions_root.is_dir():
        logger.error("Predictions root not found: %s",
                     args.predictions_root)
        return 1

    tasks = load_label_studio_tasks(args.json_path)
    gt_index = build_ground_truth_index(tasks, args.label_name)

    variant_batches: Dict[str, dict] = {}
    processed, skipped = 0, []
    matched_stems = set()
    for sample_dir in iter_sample_dirs(args.predictions_root):
        sample = _collect_sample(sample_dir, gt_index, args.variants)
        if sample is None:
            skipped.append(sample_dir.name)
            continue
        matched_stems.add(sample["stem"])
        _accumulate(variant_batches, sample)
        _visualize(sample, args.output_dir)
        processed += 1

    _log_missing_on_disk(gt_index, matched_stems)

    if processed == 0:
        logger.error("No samples were processed")
        return 1

    _report(variant_batches, args.output_dir, args.iou_threshold,
            args.boundary_scales)
    logger.info("Processed %d samples, skipped %d", processed, len(skipped))
    if skipped:
        logger.info("Skipped samples: %s", ", ".join(skipped))
    return 0


def _log_missing_on_disk(
    gt_index: Dict[str, dict], matched_stems: set
) -> None:
    """Warn about ground-truth stems that had no sample on disk."""
    for stem in gt_index:
        if not any(stem.startswith(m) or m.startswith(stem)
                   for m in matched_stems):
            logger.warning("GT sample %s has no prediction on disk", stem)


def main(argv: List[str] = None) -> int:
    """Module entry point."""
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args)
