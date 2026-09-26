"""
Where training runs and their evaluations are recorded for inspection.

Every run is one MLflow run. Training writes its losses and learning
rates into it while it happens; scoring the snapshots afterwards writes
the instance metrics into the same run, one point per epoch, so a run's
losses and its F1 curve sit side by side however far apart in time the
two were produced. The link between them is the checkpoint directory,
stored as a tag, rather than a run identifier carried around by hand.

**Every evaluation metric names the post-processing it was read under.**
The watershed settings that turn the decoder's output into instances
are being recalibrated, and the same snapshot scores differently under
different settings - by as much as 0.03 in instance F1, more than any
training change this study has measured. A metric called ``f1`` would
silently put numbers read under two measuring sticks on one chart. The
name therefore carries the settings in full, spelled out as values
rather than as a label like "frozen", because what "frozen" refers to
changes the moment the calibration does.

**Nothing here decides anything.** This module records and displays.
Readings that decide - the late-epoch mean, the noise band, the
screening rule - are computed by the scripts that apply them, and only
copied here so they can be seen next to the curves.
"""
import logging
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from materials_vision.evaluation.watershed import WatershedParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunTracking:
    """How a training run is to be recorded.

    Parameters
    ----------
    experiment : str
        MLflow experiment the run goes into.
    tags : mapping of str to str, optional
        What the run is within the study - part, arm, policy, seed -
        so runs can be filtered and grouped in the interface.
    params : mapping of str to Any, optional
        Settings the training module cannot see, such as the
        augmentation policy, logged alongside the ones it can.
    artifacts : tuple of Path, optional
        Files attached to the run, typically its provenance record.
    """

    experiment: str
    tags: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()


REPO_ROOT = Path(__file__).resolve().parents[1]

# The store the project's MLflow UI already reads, so that the command
# documented for it shows these runs without further configuration.
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "mlruns"

TRAINING_EXPERIMENT = "peft_sam_training"

POSTPROCESSING_EXPERIMENT = "peft_sam_postprocessing"

# Kept apart so a shortened dry run can never be mistaken for a run of
# the study when runs are compared in the interface.
REHEARSAL_EXPERIMENT = "peft_sam_rehearsal"

CHECKPOINT_DIR_TAG = "mv.checkpoint_dir"

# Scalar fields of an aggregate result shown as curves. The remaining
# fields are either counts that only make sense next to their totals, or
# nested structures written to the JSON reports instead.
EVALUATION_FIELDS = (
    "f1", "precision", "recall", "macro_f1", "mean_pair_iou",
    "merges_per_100_gt", "splits_per_100_gt", "pore_count_error",
    "macro_abs_pore_count_error", "mean_porosity_error_pp",
    "wasserstein_um", "median_diameter_drift_um",
    "median_diameter_error", "median_diameter_log_ratio",
    "median_elongation_error", "n_gt", "n_pred",
)

# Epochs the headline of part II is read over: the mean of the last
# three snapshots of a twelve-epoch run, because the deployed model is
# the last snapshot and no validation set is left to pick a peak with.
LATE_EPOCHS = (10, 11, 12)


def configure_tracking(tracking_uri: Optional[str] = None) -> str:
    """Point MLflow at the project store.

    Parameters
    ----------
    tracking_uri : str, optional
        Overrides the store. When omitted, ``MLFLOW_TRACKING_URI`` wins
        if set, and the project database otherwise.

    Returns
    -------
    str
        The URI in use.
    """
    import mlflow

    uri = (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or DEFAULT_TRACKING_URI
    )
    mlflow.set_tracking_uri(uri)
    return uri


def ensure_experiment(
    name: str, artifact_root: Optional[Path] = None
) -> str:
    """Return the experiment's identifier, creating it if needed.

    The artifact location is given explicitly on creation. Left to
    MLflow it would resolve against the working directory of whichever
    process happened to create the experiment, and images from a run
    started elsewhere would be written where the interface never looks.

    Parameters
    ----------
    name : str
    artifact_root : Path, optional
        Defaults to the repository's ``mlruns``.

    Returns
    -------
    str
    """
    import mlflow

    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    root = DEFAULT_ARTIFACT_ROOT if artifact_root is None else artifact_root
    location = (Path(root) / name).resolve().as_uri()
    return mlflow.create_experiment(name, artifact_location=location)


def checkpoint_dir_tag(checkpoint_dir: Path) -> str:
    """Name a checkpoint directory the same way from every entry point.

    Relative to the repository when it lies inside it, so the tag
    written by a training launched from the repository root matches the
    one looked up by an evaluation launched from anywhere.

    Parameters
    ----------
    checkpoint_dir : Path

    Returns
    -------
    str
        A POSIX path.
    """
    resolved = Path(checkpoint_dir).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def find_run_for_checkpoint_dir(checkpoint_dir: Path) -> Optional[str]:
    """Find the run a checkpoint directory was trained in.

    Parameters
    ----------
    checkpoint_dir : Path

    Returns
    -------
    str or None
        The most recent matching run, or ``None`` when there is none.
        More than one match means the run was trained again over the
        same directory; the snapshots on disk are the newest run's, so
        that is the one returned, and the ambiguity is logged.
    """
    import mlflow

    tag = checkpoint_dir_tag(checkpoint_dir)
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{CHECKPOINT_DIR_TAG}` = '{tag}'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    if not runs:
        return None
    if len(runs) > 1:
        logger.warning(
            "%d runs are tagged with %s; using the newest, %s.",
            len(runs), tag, runs[0].info.run_id,
        )
    return runs[0].info.run_id


def postprocessing_id(settings: WatershedParams) -> str:
    """Spell out a watershed setting as a metric-name component.

    Every value appears, including the ones at their defaults, so that
    two settings can never share a name and no name depends on what the
    frozen setting happened to be when it was written.

    Parameters
    ----------
    settings : WatershedParams

    Returns
    -------
    str
        For example ``c0.3_b0.4_f0.5_fs1_d1.6_m0``, with ``_a<um2>``
        appended when the setting drops instances by physical area.
        The suffix is left out at zero so that every figure recorded
        before the filter existed keeps the name it was logged under.
    """
    name = (
        f"c{settings.center_distance_threshold:g}"
        f"_b{settings.boundary_distance_threshold:g}"
        f"_f{settings.foreground_threshold:g}"
        f"_fs{settings.foreground_smoothing:g}"
        f"_d{settings.distance_smoothing:g}"
        f"_m{settings.min_size:g}"
    )
    if settings.min_instance_area_um2 > 0:
        name += f"_a{settings.min_instance_area_um2:g}"
    return name


def evaluation_metric_name(
    subset: str,
    postprocessing: str,
    field: str,
    section: Optional[str] = None,
) -> str:
    """Compose the name one evaluation figure is logged under.

    Parameters
    ----------
    subset : str
        ``"train"``, ``"val"`` or ``"test"``.
    postprocessing : str
        From :func:`postprocessing_id`.
    field : str
    section : str, optional
        A cross-section label such as ``"material=K"``, written as
        ``material_K`` because MLflow does not accept ``=`` in names.

    Returns
    -------
    str
    """
    parts = [subset, postprocessing]
    if section is not None:
        parts.append(section.replace("=", "_"))
    parts.append(field)
    return "/".join(parts)


def _as_mapping(result: Any) -> Mapping[str, Any]:
    """Accept an aggregate result or its JSON form alike."""
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    return result


def evaluation_metrics(
    result: Any,
    subset: str,
    postprocessing: str,
    section: Optional[str] = None,
    boundary_scale: float = 0.1,
) -> dict[str, float]:
    """Flatten one aggregate result into named figures.

    Parameters
    ----------
    result : AggregateResult or mapping
        Either the dataclass or the dictionary it is saved as, so that
        live evaluations and results already on disk go through the
        same path.
    subset, postprocessing : str
    section : str, optional
    boundary_scale : float, optional
        The boundary tolerance the study decides on; boundary F1 is
        stored per tolerance and only this one is shown.

    Returns
    -------
    dict of str to float
        Fields that are missing or not finite are left out rather than
        logged as zero, which a chart would draw as a real value.
    """
    data = _as_mapping(result)
    figures: dict[str, Any] = {
        name: data.get(name) for name in EVALUATION_FIELDS
    }
    boundary = data.get("boundary_f1") or {}
    figures["boundary_f1"] = _lookup_scale(boundary, boundary_scale)

    named = {}
    for figure, value in figures.items():
        if not _is_finite_number(value):
            continue
        name = evaluation_metric_name(
            subset, postprocessing, figure, section
        )
        named[name] = float(value)
    return named


def _lookup_scale(per_scale: Mapping[Any, Any], scale: float) -> Any:
    """Find a tolerance whether it is keyed as a float or as a string."""
    for key, value in per_scale.items():
        try:
            if abs(float(key) - scale) < 1e-9:
                return value
        except (TypeError, ValueError):
            continue
    return None


def _is_finite_number(value: Any) -> bool:
    """Whether a value can be drawn as a point on a curve."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and abs(value) != float("inf")


def section_metrics(
    sections: Iterable[Any],
    subset: str,
    postprocessing: str,
) -> dict[str, float]:
    """Flatten a set of cross-sections, such as the three materials.

    Parameters
    ----------
    sections : iterable of AggregateResult or mapping
        Each labelled like ``"material=K"``.
    subset, postprocessing : str

    Returns
    -------
    dict of str to float
    """
    named: dict[str, float] = {}
    for section in sections:
        data = _as_mapping(section)
        named.update(
            evaluation_metrics(
                data, subset, postprocessing, section=str(data["label"])
            )
        )
    return named


def late_epoch_summary(
    f1_by_epoch: Mapping[int, float],
    late_epochs: Sequence[int] = LATE_EPOCHS,
) -> dict[str, float]:
    """Summarize a curve the way part II reads it, and its peak beside.

    Parameters
    ----------
    f1_by_epoch : mapping of int to float
        One-based epoch to instance F1.
    late_epochs : sequence of int, optional

    Returns
    -------
    dict
        ``late_mean`` only when every late epoch is present, since a
        mean over whichever of them exist would compare unequal things;
        ``peak`` and ``peak_epoch`` whenever the curve is not empty.
    """
    summary: dict[str, float] = {}
    if not f1_by_epoch:
        return summary
    peak_epoch = max(f1_by_epoch, key=lambda epoch: f1_by_epoch[epoch])
    summary["peak"] = float(f1_by_epoch[peak_epoch])
    summary["peak_epoch"] = float(peak_epoch)
    if all(epoch in f1_by_epoch for epoch in late_epochs):
        summary["late_mean"] = float(
            sum(f1_by_epoch[epoch] for epoch in late_epochs)
            / len(late_epochs)
        )
    return summary


def snapshot_epoch(checkpoint: Path) -> Optional[int]:
    """Read the epoch a snapshot was saved after from its file name.

    Parameters
    ----------
    checkpoint : Path

    Returns
    -------
    int or None
        ``None`` for ``best.pt`` and ``latest.pt``, which carry no epoch
        in their name and so have no place on a curve.
    """
    match = re.fullmatch(r"epoch-(\d+)", Path(checkpoint).stem)
    return int(match.group(1)) if match else None


def log_evaluation_to_run(
    run_id: str,
    epoch: int,
    subset: str,
    postprocessing: str,
    overall: Any,
    sections: Iterable[Any] = (),
) -> int:
    """Put one snapshot's figures on its run's curves.

    Parameters
    ----------
    run_id : str
    epoch : int
        One-based; the step the figures are logged at.
    subset, postprocessing : str
    overall : AggregateResult or mapping
    sections : iterable of AggregateResult or mapping, optional
        Cross-sections such as the materials and the scale bins.

    Returns
    -------
    int
        How many figures were written.
    """
    metrics = evaluation_metrics(overall, subset, postprocessing)
    metrics.update(section_metrics(sections, subset, postprocessing))
    log_metrics_at(run_id, metrics, step=epoch)
    return len(metrics)


def summarize_curve(
    run_id: str, subset: str, postprocessing: str
) -> dict[str, float]:
    """Log the late-epoch mean and the peak of a run's F1 curve.

    Read back from the store rather than from the evaluation in hand,
    so that a curve scored in several sittings is summarized whole.

    Parameters
    ----------
    run_id : str
    subset, postprocessing : str

    Returns
    -------
    dict of str to float
        The figures logged, by full metric name.
    """
    from mlflow import MlflowClient

    key = evaluation_metric_name(subset, postprocessing, "f1")
    history = MlflowClient().get_metric_history(run_id, key)
    latest: dict[int, tuple[int, float]] = {}
    for entry in history:
        known = latest.get(entry.step)
        if known is None or entry.timestamp >= known[0]:
            latest[entry.step] = (entry.timestamp, entry.value)
    curve = {step: value for step, (_, value) in latest.items()}
    summary = {
        evaluation_metric_name(
            subset, postprocessing, f"f1_{name}", section="summary"
        ): value
        for name, value in late_epoch_summary(curve).items()
    }
    log_metrics_at(run_id, summary)
    return summary


def log_metrics_at(
    run_id: str, metrics: Mapping[str, float], step: Optional[int] = None
) -> None:
    """Write figures to a run that is not necessarily the active one.

    Parameters
    ----------
    run_id : str
    metrics : mapping of str to float
    step : int, optional
    """
    log_metric_series(
        run_id,
        [(key, value, step or 0) for key, value in metrics.items()],
    )


def log_metric_series(
    run_id: str, points: Sequence[tuple[str, float, int]]
) -> None:
    """Write many points, across names and steps, in as few requests.

    Parameters
    ----------
    run_id : str
    points : sequence of (name, value, step)
    """
    import time

    from mlflow import MlflowClient
    from mlflow.entities import Metric

    if not points:
        return
    timestamp_ms = int(time.time() * 1000)
    entries = [
        Metric(key, float(value), timestamp_ms, int(step))
        for key, value, step in points
    ]
    client = MlflowClient()
    # The store accepts at most a thousand metrics per request.
    for start in range(0, len(entries), 1000):
        client.log_batch(run_id, metrics=entries[start:start + 1000])
