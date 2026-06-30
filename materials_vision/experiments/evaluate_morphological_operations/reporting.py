"""Quantitative metric computation and xlsx reporting.

All metrics come from ``materials_vision.metrics``; this module only
orchestrates those functions and serializes the results. No metric is
defined or modified here.
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np # type: ignore
import pandas as pd # type: ignore

from materials_vision.metrics import (
    boundary_scores_batched,
    iou_scores_batch,
    summarize_evaluation_boundary_score,
    summarize_evaluation_iou,
)

logger = logging.getLogger(__name__)


def _evaluate(
    true_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    threshold: float,
    scales: Sequence[float],
) -> pd.Series:
    """
    Compute the combined IoU and boundary metric series for a batch.

    Parameters
    ----------
    true_masks : List[np.ndarray]
        Ground-truth instance masks.
    pred_masks : List[np.ndarray]
        Prediction instance masks (same order as ``true_masks``).
    threshold : float
        IoU greedy-match threshold.
    scales : Sequence[float]
        Boundary tolerance scales.

    Returns
    -------
    pd.Series
        Metric name to value mapping.
    """
    iou_results = iou_scores_batch(true_masks, pred_masks, threshold)
    precision, recall, fscore = boundary_scores_batched(
        true_masks, pred_masks, list(scales)
    )
    iou_report = summarize_evaluation_iou(iou_results)
    boundary_report = summarize_evaluation_boundary_score(
        precision, recall, fscore
    )
    combined = pd.concat([iou_report, boundary_report])
    return combined["value"]


def evaluate_variant(
    true_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    threshold: float,
    scales: Sequence[float],
) -> pd.Series:
    """
    Aggregate metrics over all samples of a single variant.

    Parameters
    ----------
    true_masks : List[np.ndarray]
        Ground-truth masks.
    pred_masks : List[np.ndarray]
        Prediction masks.
    threshold : float
        IoU greedy-match threshold.
    scales : Sequence[float]
        Boundary tolerance scales.

    Returns
    -------
    pd.Series
        Aggregated metric values.
    """
    return _evaluate(true_masks, pred_masks, threshold, scales)


def evaluate_variant_per_sample(
    stems: List[str],
    true_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    threshold: float,
    scales: Sequence[float],
) -> pd.DataFrame:
    """
    Compute metrics independently for each sample of a variant.

    Parameters
    ----------
    stems : List[str]
        Sample identifiers, used as the row index.
    true_masks : List[np.ndarray]
        Ground-truth masks.
    pred_masks : List[np.ndarray]
        Prediction masks.
    threshold : float
        IoU greedy-match threshold.
    scales : Sequence[float]
        Boundary tolerance scales.

    Returns
    -------
    pd.DataFrame
        One row per sample, columns are metric names.
    """
    rows = {}
    for stem, true_mask, pred_mask in zip(stems, true_masks, pred_masks):
        rows[stem] = _evaluate(
            [true_mask], [pred_mask], threshold, scales
        )
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "sample"
    return report


def _build_summary(
    aggregate_by_variant: Dict[str, pd.Series],
    per_sample_by_variant: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the summary frame with a mean and a std column per variant.

    The mean column holds the aggregated metric value; the std column
    holds the per-sample standard deviation (ddof=1) of that metric.

    Parameters
    ----------
    aggregate_by_variant : Dict[str, pd.Series]
        Mapping ``variant -> aggregated metric series``.
    per_sample_by_variant : Dict[str, pd.DataFrame]
        Mapping ``variant -> per-sample metric frame``.

    Returns
    -------
    pd.DataFrame
        Summary indexed by metric, with ``<variant>`` and
        ``<variant>_std`` columns.
    """
    summary = pd.DataFrame(aggregate_by_variant)
    summary.index.name = "metric"
    columns = []
    for variant in aggregate_by_variant:
        std = per_sample_by_variant[variant].std(ddof=1)
        summary[f"{variant}_std"] = std
        columns.extend([variant, f"{variant}_std"])
    return summary[columns]


def write_xlsx_report(
    output_dir: Path,
    aggregate_by_variant: Dict[str, pd.Series],
    per_sample_by_variant: Dict[str, pd.DataFrame],
) -> Path:
    """
    Write aggregate and per-sample metrics to an xlsx report.

    Parameters
    ----------
    output_dir : Path
        Directory where ``report.xlsx`` is written.
    aggregate_by_variant : Dict[str, pd.Series]
        Mapping ``variant -> aggregated metric series``.
    per_sample_by_variant : Dict[str, pd.DataFrame]
        Mapping ``variant -> per-sample metric frame``.

    Returns
    -------
    Path
        Path to the written report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.xlsx"
    summary = _build_summary(aggregate_by_variant, per_sample_by_variant)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary")
        for variant, frame in per_sample_by_variant.items():
            sheet = f"per_sample_{variant}"[:31]
            frame.to_excel(writer, sheet_name=sheet)
    logger.info("Report written to %s", report_path)
    return report_path
