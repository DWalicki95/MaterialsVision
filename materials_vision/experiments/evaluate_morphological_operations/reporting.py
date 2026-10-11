"""Folding the classical evaluations into the report the models have.

No metric is computed here. Every figure comes from
``materials_vision.evaluation`` - ``aggregate`` and ``cross_sections``
over the per-image evaluations - so the JSON written below has the
layout of the segmentation models' reports, entry for entry, and a
script reading one can read the other.

One addition: a breakdown **per image**. The classical predictions
cover a handful of images, and on so few a pooled figure can rest
largely on one of them; the per-image rows show whether it does.
"""
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from materials_vision.evaluation import (AggregateResult, ImageEvaluation,
                                         aggregate, cross_sections)

logger = logging.getLogger(__name__)

# The same cross-sections, under the same keys, as the model reports.
JSON_CROSS_SECTIONS = {
    "per_material": "material",
    "per_scale_bin": "scale_bin",
    "per_formulation": "formulation",
}

HEADLINE_FIELDS = (
    "f1", "precision", "recall", "mean_pair_iou",
    "merges_per_100_gt", "splits_per_100_gt", "pore_count_error",
)


def variant_report(evaluations: Sequence[ImageEvaluation]) -> dict:
    """Every figure for one variant: pooled, per cross-section, per image.

    Parameters
    ----------
    evaluations : sequence of ImageEvaluation

    Returns
    -------
    dict
        ``overall``, the keys of ``JSON_CROSS_SECTIONS``, and
        ``per_image``, each holding ``AggregateResult`` fields.
    """
    report = {"overall": asdict(aggregate(evaluations))}
    for field, key in JSON_CROSS_SECTIONS.items():
        report[field] = [
            asdict(section) for section in cross_sections(evaluations, key)
        ]
    report["per_image"] = [
        asdict(aggregate([evaluation], label=evaluation.record.image_id))
        for evaluation in evaluations
    ]
    return report


def write_json(
    destination: Path,
    evaluations_by_variant: dict[str, Sequence[ImageEvaluation]],
) -> dict:
    """Save the report of every variant.

    Parameters
    ----------
    destination : Path
    evaluations_by_variant : dict
        Variant name to its per-image evaluations.

    Returns
    -------
    dict
        The payload written.
    """
    payload = {
        variant: variant_report(evaluations)
        for variant, evaluations in evaluations_by_variant.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return payload


def log_headline(results: dict[str, AggregateResult]) -> None:
    """Log the headline figures of every variant, one row each.

    Parameters
    ----------
    results : dict
        Variant name to its pooled figures.
    """
    header = "  ".join(f"{field:>10.10}" for field in HEADLINE_FIELDS)
    logger.info("%-24s %s", "variant", header)
    for name, result in results.items():
        values = "  ".join(
            f"{getattr(result, field):10.4f}" for field in HEADLINE_FIELDS
        )
        logger.info("%-24s %s", name, values)
    first = next(iter(results.values()))
    logger.info(
        "Scored on %d image(s), %d annotated instance(s).",
        first.n_images, first.n_gt,
    )
