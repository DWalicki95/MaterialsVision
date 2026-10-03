"""
Training curves of a run, with training, validation and test figures on
one epoch axis.

Read straight from the MLflow store the training and scoring scripts
write to. What that store holds decides what can be drawn:

- **losses per epoch**, on the training images and on the validation
  images, for every run;
- **instance metrics per epoch on VALIDATION** (F1 and the rest), for
  runs trained on TRAIN alone, under each post-processing setting the
  run was scored with;
- **at most one TEST point per run**, at the snapshot fixed before TEST
  was opened. There is no TEST curve and there must not be one: scoring
  every epoch of a run on TEST would let TEST choose among them, and
  the figure reported would be the best of several draws on the set
  kept to measure generalisation;
- **no instance metric on the training images**. Only the losses exist
  on that side.

Two panels therefore, one quantity each, never sharing an axis: the
loss, training against validation, and the instance metric, the
validation curve with the TEST point placed at its epoch.

**The loss drawn by default is the instance decoder's**, the branch the
segmentation is made from. It is computed the same way on both sides.
The loss of the prompted branch is not: training feeds it several
rounds of corrective prompts per step, validation a single pass, so the
gap between its two curves is not a generalisation gap. It can still be
asked for by name.

**A run that trained on the validation images says so.** The trainer
insists on a validation set, and the run that trained on TRAIN and
VALIDATION together was handed images it also trained on. Its
validation curve is kept, but labelled as such, recognised from the
``train_subset`` tag every run carries.
"""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   POSTPROCESSING_CONFIGS,
                                                   WatershedParams)
from materials_vision.inspection.metric_charts import (BASELINE, GRIDLINE,
                                                       INK_MUTED, INK_PRIMARY,
                                                       INK_SECONDARY,
                                                       SERIES_COLOURS, SURFACE)
from materials_vision.tracking import (EVALUATION_FIELDS, TRAINING_EXPERIMENT,
                                       configure_tracking,
                                       evaluation_metric_name,
                                       postprocessing_id)

logger = logging.getLogger(__name__)

LOSSES = ("instance_loss", "loss", "mask_loss", "iou_loss")

# Every evaluation field the scoring script logs per epoch. Boundary F1
# is logged beside the listed fields, at the decision tolerance, rather
# than among them.
METRIC_FIELDS = EVALUATION_FIELDS + ("boundary_f1",)

TRAIN_SUBSET_TAG = "train_subset"

VALIDATION_SUBSET = "val"

TEST_SUBSET = "test"

FIGURE_SIZE_IN = (13.0, 4.8)

TEST_MARKER_SIZE = 9

TRAIN_STYLE = "-"

VALIDATION_STYLE = "--"

# Dotted rather than dashed, so that a validation curve on images the
# model trained on cannot be read as a held-out one.
SEEN_VALIDATION_STYLE = ":"

TEST_MARKER = "D"

# Height of one legend row, as a fraction of the figure.
LEGEND_ROW_HEIGHT = 0.06


@dataclass(frozen=True)
class Series:
    """One quantity per epoch.

    Parameters
    ----------
    epochs : np.ndarray
        Ascending, one entry per epoch that was recorded.
    values : np.ndarray
    """

    epochs: np.ndarray
    values: np.ndarray

    @property
    def is_empty(self) -> bool:
        """Whether nothing was recorded.

        Returns
        -------
        bool
        """
        return self.epochs.size == 0


@dataclass(frozen=True)
class RunCurves:
    """Everything the store holds for one run, on the epoch axis.

    Parameters
    ----------
    run_name : str
    loss : str
        Which loss the two loss series are.
    metric : str
        Which evaluation field the two metric series are.
    postprocessing : str
        The setting the metric series were scored under, as spelled in
        the metric names.
    train_loss, validation_loss : Series
    validation_metric : Series
        Empty for a run never scored on VALIDATION under the setting.
    test_metric : Series
        Empty, or a single point at the snapshot scored on TEST.
    validation_held_out : bool
        False when the run also trained on the validation images.
    """

    run_name: str
    loss: str
    metric: str
    postprocessing: str
    train_loss: Series
    validation_loss: Series
    validation_metric: Series
    test_metric: Series
    validation_held_out: bool


def load_run_curves(
    run_name: str,
    *,
    experiment: str = TRAINING_EXPERIMENT,
    loss: str = "instance_loss",
    metric: str = "f1",
    postprocessing: Union[str, WatershedParams] = FROZEN_WATERSHED,
    tracking_uri: Optional[str] = None,
) -> RunCurves:
    """Read one run's curves from the store.

    Parameters
    ----------
    run_name : str
        As shown in the MLflow interface, e.g. ``d4_seed20260907``.
    experiment : str, optional
    loss : str, optional
        One of ``LOSSES``.
    metric : str, optional
        One of ``METRIC_FIELDS``, e.g. ``"f1"``, ``"merges_per_100_gt"``.
        ``median_diameter_log_ratio`` exists only in runs scored after
        it was added to the evaluation; older runs read it as empty.
    postprocessing : str or WatershedParams, optional
        A setting, or its name in ``POSTPROCESSING_CONFIGS``. Defaults
        to the frozen setting, the only one under which the runs carry
        both a validation curve and a TEST point.
    tracking_uri : str, optional
        Overrides the project store.

    Returns
    -------
    RunCurves

    Raises
    ------
    ValueError
        For an unknown loss, metric or setting name, or a run that is
        not in the experiment.
    """
    from mlflow.tracking import MlflowClient

    if loss not in LOSSES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSSES}.")
    if metric not in METRIC_FIELDS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {METRIC_FIELDS}."
        )
    setting_id = postprocessing_id(_setting(postprocessing))

    configure_tracking(tracking_uri)
    client = MlflowClient()
    run = _find_run(client, experiment, run_name)
    run_id = run.info.run_id

    curves = RunCurves(
        run_name=run_name,
        loss=loss,
        metric=metric,
        postprocessing=setting_id,
        train_loss=_series(client, run_id, f"epoch/train_{loss}"),
        validation_loss=_series(client, run_id, f"epoch/val_{loss}"),
        validation_metric=_series(client, run_id, evaluation_metric_name(
            VALIDATION_SUBSET, setting_id, metric
        )),
        test_metric=_series(client, run_id, evaluation_metric_name(
            TEST_SUBSET, setting_id, metric
        )),
        validation_held_out=VALIDATION_SUBSET not in run.data.tags.get(
            TRAIN_SUBSET_TAG, ""
        ).split("+"),
    )
    if curves.validation_metric.is_empty and curves.test_metric.is_empty:
        logger.warning(
            "%s holds no %s under %s on VALIDATION or TEST; only the "
            "losses can be drawn.", run_name, metric, setting_id,
        )
    return curves


def training_curves_figure(curves: Sequence[RunCurves]) -> Figure:
    """Loss and instance metric against the epoch, for up to three runs.

    Colour tells the runs apart and line style the image sets: solid
    for training, dashed for validation (dotted where the run trained
    on those images too), a diamond for the single TEST point.

    Parameters
    ----------
    curves : sequence of RunCurves
        All read with the same loss, metric and setting.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no run, more runs than colours, or runs read with different
        losses, metrics or settings are given.
    """
    _check_comparable(curves)
    figure = Figure(figsize=FIGURE_SIZE_IN, facecolor=SURFACE)
    ax_loss, ax_metric = figure.subplots(1, 2)

    for run, colour in zip(curves, SERIES_COLOURS):
        _plot_series(ax_loss, run.train_loss, colour, TRAIN_STYLE)
        _plot_series(ax_loss, run.validation_loss, colour,
                     _validation_style(run))
        _plot_series(ax_metric, run.validation_metric, colour,
                     _validation_style(run))
        _plot_test_point(ax_metric, run.test_metric, colour)

    first = curves[0]
    _style_axis(ax_loss, f"Loss: {first.loss}",
                "training and validation images")
    _style_axis(ax_metric, f"Instance {first.metric}",
                f"post-processing {first.postprocessing}")
    _mark_empty(ax_loss, [c.train_loss for c in curves]
                + [c.validation_loss for c in curves])
    _mark_empty(ax_metric, [c.validation_metric for c in curves]
                + [c.test_metric for c in curves])

    run_handles, set_handles = _legend_handles(curves)
    rows = ((run_handles, LEGEND_ROW_HEIGHT), (set_handles, 0.0))
    for handles, height in rows:
        figure.legend(
            handles=handles, loc="lower center", ncol=len(handles),
            bbox_to_anchor=(0.5, height), frameon=False,
            labelcolor=INK_PRIMARY, fontsize=9,
        )
    figure.tight_layout(rect=(0, 2.2 * LEGEND_ROW_HEIGHT, 1, 1))
    return figure


def _setting(postprocessing: Union[str, WatershedParams]) -> WatershedParams:
    """Resolve a setting given by name."""
    if isinstance(postprocessing, WatershedParams):
        return postprocessing
    if postprocessing not in POSTPROCESSING_CONFIGS:
        raise ValueError(
            f"Unknown post-processing {postprocessing!r}; expected one of "
            f"{sorted(POSTPROCESSING_CONFIGS)} or a WatershedParams."
        )
    return POSTPROCESSING_CONFIGS[postprocessing]


def _find_run(client: Any, experiment: str, run_name: str) -> Any:
    """The newest run of that name in the experiment."""
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"No experiment named {experiment!r} in the store.")
    runs = client.search_runs(
        [found.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise ValueError(f"No run named {run_name!r} in {experiment!r}.")
    if len(runs) > 1:
        logger.warning(
            "%d runs named %s in %s; using the newest, %s.",
            len(runs), run_name, experiment, runs[0].info.run_id,
        )
    return runs[0]


def _series(client: Any, run_id: str, key: str) -> Series:
    """A metric's history, one value per step.

    A step logged more than once - a snapshot scored again - keeps the
    value logged last, which is the one the interface shows.
    """
    latest: dict[int, tuple[int, float]] = {}
    for point in client.get_metric_history(run_id, key):
        kept = latest.get(point.step)
        if kept is None or point.timestamp >= kept[0]:
            latest[point.step] = (point.timestamp, point.value)
    epochs = np.array(sorted(latest), dtype=int)
    return Series(
        epochs=epochs,
        values=np.array([latest[e][1] for e in epochs], dtype=float),
    )


def _check_comparable(curves: Sequence[RunCurves]) -> None:
    """Refuse a set of runs that cannot share the two panels."""
    if not 0 < len(curves) <= len(SERIES_COLOURS):
        raise ValueError(
            f"Between 1 and {len(SERIES_COLOURS)} runs can be drawn, got "
            f"{len(curves)}."
        )
    first = curves[0]
    for run in curves[1:]:
        if (run.loss, run.metric, run.postprocessing) != (
            first.loss, first.metric, first.postprocessing
        ):
            raise ValueError(
                f"{run.run_name} was read as {run.loss} / {run.metric} / "
                f"{run.postprocessing} and {first.run_name} as "
                f"{first.loss} / {first.metric} / {first.postprocessing}; "
                "one panel cannot hold both."
            )


def _validation_style(run: RunCurves) -> str:
    if run.validation_held_out:
        return VALIDATION_STYLE
    return SEEN_VALIDATION_STYLE


def _plot_series(axis: Any, series: Series, colour: str, style: str) -> None:
    if series.is_empty:
        return
    axis.plot(series.epochs, series.values, style, color=colour,
              linewidth=2, marker="o", markersize=3, zorder=3)


def _plot_test_point(axis: Any, series: Series, colour: str) -> None:
    """The TEST point, with its value printed beside it."""
    if series.is_empty:
        return
    axis.plot(series.epochs, series.values, TEST_MARKER, color=colour,
              markersize=TEST_MARKER_SIZE, markeredgecolor=SURFACE,
              markeredgewidth=1.5, zorder=4)
    for epoch, value in zip(series.epochs, series.values):
        axis.annotate(f"{value:.3f}", (epoch, value), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=8,
                      color=INK_SECONDARY)


def _style_axis(axis: Any, title: str, subtitle: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=10,
                   color=INK_PRIMARY)
    axis.set_xlabel("epoch", fontsize=9, color=INK_SECONDARY)
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    axis.grid(color=GRIDLINE, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE)
    axis.margins(x=0.08, y=0.12)


def _mark_empty(axis: Any, series: Sequence[Series]) -> None:
    """Say so in the panel when the store holds nothing to draw there."""
    if all(item.is_empty for item in series):
        axis.text(0.5, 0.5, "nothing recorded in the store",
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=9, color=INK_MUTED)


def _legend_handles(curves: Sequence[RunCurves]) -> tuple[list, list]:
    """Runs by colour, and the image sets by line style, as two rows."""
    run_handles = [
        Line2D([], [], color=colour, linewidth=2, label=(
            run.run_name if run.validation_held_out
            else f"{run.run_name} (trained on validation images)"
        ))
        for run, colour in zip(curves, SERIES_COLOURS)
    ]
    handles = [
        Line2D([], [], color=INK_SECONDARY, linestyle=TRAIN_STYLE,
               label="training"),
        Line2D([], [], color=INK_SECONDARY, linestyle=VALIDATION_STYLE,
               label="validation"),
        Line2D([], [], color=INK_SECONDARY, marker=TEST_MARKER,
               linestyle="none", label="test (one snapshot)"),
    ]
    if not all(run.validation_held_out for run in curves):
        handles.append(Line2D(
            [], [], color=INK_SECONDARY, linestyle=SEEN_VALIDATION_STYLE,
            label="validation images also trained on",
        ))
    return run_handles, handles
