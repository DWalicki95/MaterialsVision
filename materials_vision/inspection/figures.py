"""
Figures of one image and what the model made of it.

Built without pyplot: a figure made through it is registered as open,
shown a second time by a notebook at the end of the cell, and kept in
memory until closed by hand. A plain ``Figure`` is displayed where it
is displayed and nowhere else, and is freed with its last reference.
"""
import logging

import numpy as np
from matplotlib.figure import Figure

from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.evaluation.matching import match_instances
from materials_vision.inspection.overlay import (MatchSummary, outcome_overlay,
                                                 paired_colours,
                                                 summarise_match)

logger = logging.getLogger(__name__)

FIGURE_SIZE_IN = (16.0, 8.0)


def describe(record: SampleRecord, summary: MatchSummary) -> str:
    """One line naming an image and what the model made of it.

    Parameters
    ----------
    record : SampleRecord
    summary : MatchSummary

    Returns
    -------
    str
    """
    return (
        f"{record.image_id} | {record.material}, {record.formulation}, "
        f"{record.scale_bin} | F1 {summary.f1:.3f} | "
        f"TP {summary.n_true_positives}, FP {summary.n_false_positives}, "
        f"FN {summary.n_false_negatives} | merges {summary.n_merges}, "
        f"splits {summary.n_splits}"
    )


def comparison_figure(
    sample: PreparedSample,
    pred_labels: np.ndarray,
    seed: int = 0,
    figsize_in: tuple[float, float] = FIGURE_SIZE_IN,
) -> Figure:
    """Prediction on the left, annotation on the right, colours paired.

    Parameters
    ----------
    sample : PreparedSample
    pred_labels : np.ndarray
        ``(H, W)`` prediction on the sample's frame. Not modified.
    seed : int, optional
        Fixes the pair colours.
    figsize_in : tuple of float, optional
        Width and height in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    match = match_instances(sample.labels, pred_labels)
    gt_rgb, pred_rgb = paired_colours(
        sample.labels, pred_labels, match, seed=seed
    )
    summary = summarise_match(match)

    figure = Figure(figsize=figsize_in)
    ax_pred, ax_gt = figure.subplots(1, 2, sharex=True, sharey=True)
    ax_pred.imshow(outcome_overlay(sample.image, pred_labels, pred_rgb))
    ax_pred.set_title(f"prediction: {summary.n_pred} pores")
    ax_gt.imshow(outcome_overlay(sample.image, sample.labels, gt_rgb))
    ax_gt.set_title(f"annotation: {summary.n_gt} pores")
    for axis in (ax_pred, ax_gt):
        axis.set_axis_off()
    figure.suptitle(
        f"{describe(sample.record, summary)}\n"
        "shared colour = matched pair, red = no counterpart"
    )
    figure.tight_layout()
    return figure
