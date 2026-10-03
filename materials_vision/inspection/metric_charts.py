"""
Every evaluation metric as grouped bars: one group per subset of
images, one bar per model.

**One list of metric definitions feeds both the charts and the table.**
Each metric is declared once - where its value comes from, its unit,
which direction is better - so a figure and the table beside it cannot
disagree about what a number means.

**Metrics are grouped by the question they answer**, one figure per
group: detection and overlap, recall by pore size, size bias of matched
pores by pore size, separation, shape of matched pores, material
quantities relative to the annotation, and orientation. Within a figure
every metric has its own panel and its own axis; two quantities of
different scale never share one.

**A value resting on few pores says so.** Where a metric is a median
over matched pairs split by size, the number of pairs is printed under
each value and kept in the table: a class of small pores the model
rarely finds can hold few enough pairs for its median to be noise.

**Signed metrics are drawn from a zero line.** For a drift, the length
of a bar is its size and the side of zero its direction, and the best
value is zero rather than the top of the axis.

**The images behind each group must be the same for every model.** A
comparison of two models is only a comparison if both were scored on
the same pores, so the annotated pore counts are checked to agree
subset by subset before anything is drawn.

Colours are the first categorical slots of the reference palette of
the data-visualisation guidelines this project follows, in fixed order:
the first model given is always the first colour.
"""
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.size_bins import SIZE_BIN_LABELS
from materials_vision.inspection.scoring import PERCENT, SubsetScores

logger = logging.getLogger(__name__)

# Categorical slots 1-3; the three validate as distinguishable pairwise,
# including under colour-vision deficiency, which a fourth would not.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE = "#fcfcfb"

INK_PRIMARY = "#0b0b0b"

INK_SECONDARY = "#52514e"

INK_MUTED = "#898781"

GRIDLINE = "#e1e0d9"

BASELINE = "#c3c2b7"

PANEL_COLUMNS = 4

PANEL_SIZE_IN = (4.2, 3.4)

GROUP_WIDTH = 0.8

# Fraction of a panel's value range added beyond its extremes, to hold
# the value printed at the end of each bar.
LABEL_HEADROOM = 0.18

HIGHER = "higher is better"

LOWER = "lower is better"

ZERO = "closer to zero is better"

SECTION_TITLES = {
    "detection": "Detection and overlap",
    "size": "Recall by pore size (area quartiles of the annotation)",
    "size_bias": "Size bias of matched pores by pore size "
                 "(area quartiles of the annotation)",
    "separation": "Separation of neighbouring pores",
    "shape": "Shape of matched pores",
    "material": "Material quantities, prediction relative to annotation",
    "orientation": "Orientation distribution",
}


@dataclass(frozen=True)
class MetricSpec:
    """How one metric is read out of a subset's scores and drawn.

    Parameters
    ----------
    key : str
        Stable identifier, used as the table's row name.
    title : str
    unit : str
        Empty for a plain fraction.
    reading : str
        Which direction is better.
    section : str
        A key of ``SECTION_TITLES``.
    extract : callable
        ``SubsetScores -> float``.
    signed : bool, optional
        Drawn from a zero line.
    unit_interval : bool, optional
        Bounded by 0 and 1, and drawn on that full axis.
    count : callable, optional
        ``SubsetScores -> int``: how many matched pairs the value rests
        on, printed under it and added to the table.
    """

    key: str
    title: str
    unit: str
    reading: str
    section: str
    extract: Callable[[SubsetScores], float]
    signed: bool = False
    unit_interval: bool = False
    count: Optional[Callable[[SubsetScores], int]] = None


def _size_bin_recall(index: int) -> Callable[[SubsetScores], float]:
    return lambda scores: scores.result.size_bin_recall[index].recall


def _size_bin_bias(index: int) -> Callable[[SubsetScores], float]:
    return lambda scores: scores.size_bias[index].median_pct


def _size_bin_pairs(index: int) -> Callable[[SubsetScores], int]:
    return lambda scores: scores.size_bias[index].n_pairs


def _size_bin_title(prefix: str, index: int) -> str:
    label = SIZE_BIN_LABELS[index]
    if index == 0:
        return f"{prefix}, {label} (smallest pores)"
    if index == len(SIZE_BIN_LABELS) - 1:
        return f"{prefix}, {label} (largest pores)"
    return f"{prefix}, {label}"


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("f1", "F1 (pooled)", "", HIGHER, "detection",
               lambda s: s.result.f1, unit_interval=True),
    MetricSpec("precision", "Precision", "", HIGHER, "detection",
               lambda s: s.result.precision, unit_interval=True),
    MetricSpec("recall", "Recall", "", HIGHER, "detection",
               lambda s: s.result.recall, unit_interval=True),
    MetricSpec("macro_f1", "F1 (mean over images)", "", HIGHER,
               "detection", lambda s: s.result.macro_f1,
               unit_interval=True),
    MetricSpec("mean_pair_iou", "IoU of matched pairs", "", HIGHER,
               "detection", lambda s: s.result.mean_pair_iou,
               unit_interval=True),
    MetricSpec("boundary_f1", f"Boundary F1 (tolerance {DECISION_SCALE:g})",
               "", HIGHER, "detection",
               lambda s: s.result.boundary_f1.get(DECISION_SCALE,
                                                  float("nan")),
               unit_interval=True),
    *(
        MetricSpec(f"recall_{label}", _size_bin_title("Recall", index), "",
                   HIGHER, "size", _size_bin_recall(index),
                   unit_interval=True)
        for index, label in enumerate(SIZE_BIN_LABELS)
    ),
    MetricSpec("size_bias", "Size bias, all sizes",
               "median, % of diameter", ZERO, "size_bias",
               lambda s: PERCENT * math.expm1(
                   s.result.median_diameter_log_ratio),
               signed=True, count=lambda s: s.result.n_shape_pairs),
    *(
        MetricSpec(f"size_bias_{label}", _size_bin_title("Size bias", index),
                   "median, % of diameter", ZERO, "size_bias",
                   _size_bin_bias(index), signed=True,
                   count=_size_bin_pairs(index))
        for index, label in enumerate(SIZE_BIN_LABELS)
    ),
    MetricSpec("merges", "Merges", "per 100 annotated pores", LOWER,
               "separation", lambda s: s.result.merges_per_100_gt),
    MetricSpec("splits", "Splits", "per 100 annotated pores", LOWER,
               "separation", lambda s: s.result.splits_per_100_gt),
    MetricSpec("diameter_error", "Diameter error of matched pores",
               "median, % of diameter", LOWER, "shape",
               lambda s: PERCENT * s.result.median_diameter_error),
    MetricSpec("elongation_error", "Elongation error of matched pores",
               "median, % of elongation", LOWER, "shape",
               lambda s: PERCENT * s.result.median_elongation_error),
    MetricSpec("angle_error", "Angle error of matched pores",
               "median, degrees", LOWER, "shape",
               lambda s: s.result.median_angle_error_deg),
    MetricSpec("pore_count", "Pore count", "% of annotated", ZERO,
               "material", lambda s: s.drift.pore_count_pct, signed=True),
    MetricSpec("porosity", "Porosity", "% of annotated", ZERO, "material",
               lambda s: s.drift.porosity_pct, signed=True),
    MetricSpec("median_diameter", "Median pore diameter", "% of annotated",
               ZERO, "material", lambda s: s.drift.median_diameter_pct,
               signed=True),
    MetricSpec("iqr_diameter", "Spread of pore diameter (IQR)",
               "% of annotated", ZERO, "material",
               lambda s: s.drift.iqr_diameter_pct, signed=True),
    MetricSpec("wasserstein", "Diameter distribution distance",
               "Wasserstein, % of annotated median", LOWER, "material",
               lambda s: s.drift.wasserstein_pct),
    MetricSpec("mean_angle_difference", "Mean orientation difference",
               "degrees; read only where alignment is high", LOWER,
               "orientation",
               lambda s: s.result.orientation.mean_angle_difference_deg),
    MetricSpec("alignment_difference", "Alignment strength difference",
               "axial resultant, prediction minus annotation", ZERO,
               "orientation",
               lambda s: s.result.orientation.resultant_difference,
               signed=True),
)


def metrics_table(
    scores_by_model: Mapping[str, Mapping[str, SubsetScores]],
    metrics: Sequence[MetricSpec] = METRICS,
) -> pd.DataFrame:
    """Every metric of every model on every subset, as one table.

    Parameters
    ----------
    scores_by_model : mapping
        Model name to its subset scores, as ``scores_by_material``
        returns them.
    metrics : sequence of MetricSpec, optional

    Returns
    -------
    pandas.DataFrame
        Rows indexed by section, metric and unit; columns by subset and
        then model. A metric resting on a count of matched pairs is
        followed by a row holding that count.

    Raises
    ------
    ValueError
        If the models were not scored on the same subsets and pores.
    """
    subsets = _common_subsets(scores_by_model)
    rows = {}
    for spec in metrics:
        rows[(spec.section, spec.title, spec.unit)] = {
            (subset, model): spec.extract(scores[subset])
            for subset in subsets
            for model, scores in scores_by_model.items()
        }
        if spec.count is not None:
            rows[(spec.section, f"{spec.title}: matched pairs", "count")] = {
                (subset, model): spec.count(scores[subset])
                for subset in subsets
                for model, scores in scores_by_model.items()
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["section", "metric", "unit"]
    table.columns.names = ["subset", "model"]
    return table


def metric_figures(
    scores_by_model: Mapping[str, Mapping[str, SubsetScores]],
    metrics: Sequence[MetricSpec] = METRICS,
) -> dict[str, Figure]:
    """One figure per group of metrics, bars grouped by subset.

    Parameters
    ----------
    scores_by_model : mapping
        Model name to its subset scores, as ``scores_by_material``
        returns them. The order given is the order - and the colour -
        of the bars.
    metrics : sequence of MetricSpec, optional

    Returns
    -------
    dict of str to matplotlib.figure.Figure
        Keyed by section, in the order the sections first appear.

    Raises
    ------
    ValueError
        If there are more models than colours, or the models were not
        scored on the same subsets and pores.
    """
    if not 0 < len(scores_by_model) <= len(SERIES_COLOURS):
        raise ValueError(
            f"Between 1 and {len(SERIES_COLOURS)} models can be drawn, "
            f"got {len(scores_by_model)}."
        )
    subsets = _common_subsets(scores_by_model)
    sections: dict[str, list[MetricSpec]] = {}
    for spec in metrics:
        sections.setdefault(spec.section, []).append(spec)
    return {
        section: _section_figure(section, specs, scores_by_model, subsets)
        for section, specs in sections.items()
    }


def _section_figure(
    section: str,
    specs: Sequence[MetricSpec],
    scores_by_model: Mapping[str, Mapping[str, SubsetScores]],
    subsets: Sequence[str],
) -> Figure:
    """Lay out one panel per metric of a section."""
    n_columns = min(len(specs), PANEL_COLUMNS)
    n_rows = math.ceil(len(specs) / n_columns)
    figure = Figure(
        figsize=(PANEL_SIZE_IN[0] * n_columns, PANEL_SIZE_IN[1] * n_rows),
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(figure.subplots(n_rows, n_columns, squeeze=False))
    first_scores = next(iter(scores_by_model.values()))
    tick_labels = [
        f"{subset}\n{first_scores[subset].result.n_images} img\n"
        f"{first_scores[subset].result.n_gt} pores"
        for subset in subsets
    ]
    for axis, spec in zip(axes.flat, specs):
        _draw_panel(axis, spec, scores_by_model, subsets, tick_labels)
    for axis in list(axes.flat)[len(specs):]:
        axis.set_visible(False)
    figure.legend(
        handles=[
            Patch(facecolor=colour, label=model)
            for model, colour in zip(scores_by_model, SERIES_COLOURS)
        ],
        loc="upper right", ncol=len(scores_by_model), frameon=False,
        labelcolor=INK_PRIMARY,
    )
    figure.suptitle(
        SECTION_TITLES.get(section, section), x=0.01, ha="left",
        color=INK_PRIMARY, fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _draw_panel(
    axis,
    spec: MetricSpec,
    scores_by_model: Mapping[str, Mapping[str, SubsetScores]],
    subsets: Sequence[str],
    tick_labels: Sequence[str],
) -> None:
    """Grouped bars of one metric, with its value above each bar."""
    n_models = len(scores_by_model)
    bar_width = GROUP_WIDTH / n_models
    centres = np.arange(len(subsets))
    values_by_model = {
        model: np.array([
            spec.extract(scores_by_model[model][subset])
            for subset in subsets
        ], dtype=float)
        for model in scores_by_model
    }
    for position, (model, colour) in enumerate(
        zip(scores_by_model, SERIES_COLOURS)
    ):
        values = values_by_model[model]
        counts: list[Optional[int]] = [
            None if spec.count is None
            else spec.count(scores_by_model[model][subset])
            for subset in subsets
        ]
        offsets = centres - GROUP_WIDTH / 2 + bar_width * (position + 0.5)
        drawn = np.where(np.isfinite(values), values, 0.0)
        # A surface-coloured edge keeps two adjacent bars apart.
        axis.bar(offsets, drawn, width=bar_width, color=colour,
                 edgecolor=SURFACE, linewidth=1.5, zorder=2)
        for x, value, count in zip(offsets, values, counts):
            _label_bar(axis, x, value, spec, count)

    axis.set_facecolor(SURFACE)
    axis.set_xticks(centres, tick_labels, fontsize=8, color=INK_SECONDARY)
    axis.tick_params(axis="y", labelsize=8, colors=INK_MUTED, length=0)
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE)
    if spec.signed:
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=3)
        axis.spines["bottom"].set_visible(False)
    axis.set_ylim(*_value_limits(spec, np.concatenate(
        list(values_by_model.values())
    )))
    subtitle = ", ".join(part for part in (spec.unit, spec.reading) if part)
    axis.set_title(f"{spec.title}\n{subtitle}", loc="left", fontsize=9,
                   color=INK_PRIMARY)


def _value_limits(
    spec: MetricSpec, values: np.ndarray
) -> tuple[float, float]:
    """Axis range holding every bar, zero, and room for the labels.

    Set explicitly because a bar's base is a hard limit to matplotlib's
    automatic margins: a panel whose values are all negative would
    otherwise end exactly at zero, with the labels of zero-length bars
    printed into the title above it.
    """
    if spec.unit_interval:
        return 0.0, 1.08
    finite = values[np.isfinite(values)]
    low = min(0.0, float(finite.min())) if finite.size else 0.0
    high = max(0.0, float(finite.max())) if finite.size else 0.0
    if high == low:
        return (-1.0, 1.0) if spec.signed else (0.0, 1.0)
    # A label carrying a pair count takes two lines.
    headroom = LABEL_HEADROOM * (2 if spec.count is not None else 1)
    padding = headroom * (high - low)
    return (low - padding if low < 0 else 0.0), high + padding


def _label_bar(
    axis,
    x: float,
    value: float,
    spec: MetricSpec,
    count: Optional[int] = None,
) -> None:
    """Print a bar's value at its end, or mark it as undefined.

    A count, when given, is printed as a second line: ``n=`` pairs.
    """
    count_line = "" if count is None else f"\nn={count}"
    if not np.isfinite(value):
        axis.text(x, 0.0, f"n/a{count_line}", ha="center", va="bottom",
                  fontsize=7, color=INK_MUTED)
        return
    # Adding zero turns a negative zero into a positive one, so a value
    # that rounds to nothing is not printed as "-0.0".
    if spec.unit_interval:
        text = f"{round(value, 3) + 0.0:.3f}"
    elif spec.signed:
        text = f"{round(value, 1) + 0.0:+.1f}"
    else:
        text = f"{round(value, 1) + 0.0:.1f}"
    axis.text(x, value, text + count_line, ha="center",
              va="bottom" if value >= 0 else "top", fontsize=7,
              color=INK_SECONDARY, zorder=4)


def _common_subsets(
    scores_by_model: Mapping[str, Mapping[str, SubsetScores]],
) -> list[str]:
    """The subsets every model was scored on, checked to be the same.

    Raises
    ------
    ValueError
        If there is no model, the models were scored on different
        subsets, or on a different number of annotated pores in any.
    """
    if not scores_by_model:
        raise ValueError("No model to compare.")
    models = list(scores_by_model)
    subsets = list(scores_by_model[models[0]])
    for model in models[1:]:
        if list(scores_by_model[model]) != subsets:
            raise ValueError(
                f"{model!r} was scored on subsets "
                f"{list(scores_by_model[model])}, {models[0]!r} on "
                f"{subsets}."
            )
        for subset in subsets:
            n_first = scores_by_model[models[0]][subset].result.n_gt
            n_this = scores_by_model[model][subset].result.n_gt
            if n_this != n_first:
                raise ValueError(
                    f"Subset {subset!r} holds {n_first} annotated pores "
                    f"for {models[0]!r} and {n_this} for {model!r}; the "
                    f"models were not scored on the same images."
                )
    return subsets
