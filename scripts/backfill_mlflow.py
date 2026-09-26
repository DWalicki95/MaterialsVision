#!/usr/bin/env python3
"""
Put the runs trained before MLflow tracking existed into MLflow.

Part II of the optimization study compares new runs against a reference
of three seeds, one of which - the orientation run - was trained during
the augmentation study and recorded only in TensorBoard and in JSON
reports. Seeing the reference and the arms on one chart needs that run
in the same store, and the rest of the augmentation study comes along
at no extra cost, so its curves can be browsed in the same place.

**What is imported.** For every run listed in ``HISTORICAL_RUNS``: its
provenance record as parameters and as an attached file; its training
and validation losses from TensorBoard, per step and averaged per epoch,
under the names a live run uses; its instance metrics from the curve
reports, one point per epoch snapshot; and, for the snapshots that were
scored on TEST, those figures under ``test/``.

**What is not.** The runs trained before the geometry correction and
the watershed calibration - the first baseline, the learning-rate and
rank probes, the backbone pilots. Their figures were read with settings
this study has since replaced, and one of them, the rank probe, was
declared invalid in the plan. Importing them would put numbers on the
charts that the study itself no longer stands behind.

**Which watershed setting a figure was read under is stated here, not
inferred.** Every curve report imported was scored after the
calibration of 2026-09-14, under centre and boundary thresholds of 0.30
and 0.40. The reports call that setting "frozen", and the code's frozen
setting is about to change; resolving "frozen" through the code would
relabel all of these figures the day it does. The setting is therefore
written out below as a constant of its own.

**Only the last TensorBoard file of a run is read.** A run that was
restarted wrote a new file and began again from step zero, so its last
file is the complete record and the earlier ones are the abandoned
attempts.

Examples
--------
Import everything not yet in the store:
    $ python scripts/backfill_mlflow.py

Replace runs imported earlier, after changing this script:
    $ python scripts/backfill_mlflow.py --replace
"""
import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from materials_vision.evaluation.watershed import WatershedParams
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (CHECKPOINT_DIR_TAG, TRAINING_EXPERIMENT,
                                       checkpoint_dir_tag, configure_tracking,
                                       ensure_experiment,
                                       find_run_for_checkpoint_dir,
                                       log_evaluation_to_run,
                                       log_metric_series, postprocessing_id,
                                       summarize_curve)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

CHECKPOINTS = Path("checkpoints")

# The setting every imported curve report calls "frozen". See the module
# docstring for why it is not read from the code.
CALIBRATION_2026_09_14 = WatershedParams(
    center_distance_threshold=0.30, boundary_distance_threshold=0.40,
)

TEST_SCORES = CHECKPOINTS / "final" / "test_scores.json"

# TensorBoard tags and the names a live run logs the same figure under.
# The library records the rate of the first parameter group only, which
# is the low-rank correction's.
TRAIN_TAGS = {
    "train/loss": "loss",
    "train/mask_loss": "mask_loss",
    "train/iou_loss": "iou_loss",
    "train/model_iou": "model_iou",
    "train/instance_loss": "instance_loss",
}

LEARNING_RATE_TAG = "train/learning_rate"

VALIDATION_TAGS = {
    "validation/loss": "loss",
    "validation/metric": "metric",
    "validation/mask_loss": "mask_loss",
    "validation/iou_loss": "iou_loss",
    "validation/model_iou": "model_iou",
    "validation/instance_loss": "instance_loss",
}


@dataclass(frozen=True)
class CurveReport:
    """One file of per-epoch evaluations.

    Parameters
    ----------
    path : Path
    unlabelled_setting : WatershedParams
        What a key without an ``@label`` suffix was scored under.
    """

    path: Path
    unlabelled_setting: WatershedParams = CALIBRATION_2026_09_14


@dataclass(frozen=True)
class HistoricalRun:
    """A run trained before tracking, and where its records are.

    Parameters
    ----------
    save_root : Path
    name : str
    step : str
        The step of the augmentation study it belongs to.
    policy : str
    seed : int
    provenance : Path
    curves : tuple of CurveReport
    extra_tags : tuple of (str, str)
        For the orientation run, its role as the first seed of the
        part II reference.
    """

    save_root: Path
    name: str
    step: str
    policy: str
    seed: int
    provenance: Path
    curves: tuple[CurveReport, ...] = ()
    extra_tags: tuple[tuple[str, str], ...] = ()

    @property
    def checkpoint_dir(self) -> Path:
        return self.save_root / "checkpoints" / self.name

    @property
    def log_dir(self) -> Path:
        return self.save_root / "logs" / self.name


def _e0(name: str, seed: int, policy: str) -> HistoricalRun:
    root = CHECKPOINTS / "e0_v2"
    return HistoricalRun(
        root, name, "E0", policy, seed, root / "e0_provenance.json",
        curves=(
            CurveReport(root / "e0_v2_curves.json"),
            CurveReport(
                root / "e0_v2_curves_cdt025.json",
                WatershedParams(
                    center_distance_threshold=0.25,
                    boundary_distance_threshold=0.40,
                ),
            ),
        ),
    )


def _single(
    directory: str, name: str, step: str, policy: str, curves: str,
    provenance: str, extra_tags: tuple[tuple[str, str], ...] = (),
) -> HistoricalRun:
    root = CHECKPOINTS / directory
    return HistoricalRun(
        root, name, step, policy, 20260907, root / provenance,
        curves=(CurveReport(root / curves),), extra_tags=extra_tags,
    )


HISTORICAL_RUNS = (
    _e0("b0_seed20260907", 20260907, "B0"),
    _e0("b0_seed20260908", 20260908, "B0"),
    _e0("b0_seed20260909", 20260909, "B0"),
    _e0("full_seed20260907", 20260907, "FULL_calibration"),
    _single(
        "e1", "d4_seed20260907", "E1", "FULL_A", "e1_curves.json",
        "e1_provenance.json",
        extra_tags=(("arm", "reference"), ("part", "II")),
    ),
    _single("e2", "scale_seed20260907", "E2", "D4+scale",
            "e2_curves.json", "e2_provenance.json"),
    _single("e3", "tonal_seed20260907", "E3", "D4+tonal",
            "e3_curves.json", "tonal_seed20260907_provenance.json"),
    _single("e3_bc", "tonal_bc_seed20260907", "E3", "D4+tonal_bc",
            "curves.json", "tonal_bc_seed20260907_provenance.json"),
    _single("e3_gamma", "tonal_gamma_seed20260907", "E3", "D4+tonal_gamma",
            "curves.json", "tonal_gamma_seed20260907_provenance.json"),
    _single("e4", "blur_seed20260907", "E4", "D4+blur",
            "e4_curves.json", "blur_seed20260907_provenance.json"),
    _single("e5", "mask_aware_seed20260907", "E5", "D4+mask_aware",
            "e5_curves.json", "mask_aware_seed20260907_provenance.json"),
    _single("e6", "septum_seed20260907", "E6", "D4+septum",
            "e6_curves.json", "septum_seed20260907_provenance.json"),
    _single("e8", "full_b_seed20260907", "E8", "FULL_B",
            "e8_full_b_curves.json", "full_b_seed20260907_provenance.json"),
    _single("e8", "full_b_no_d4_seed20260907", "E8", "FULL_B-D4",
            "e8_full_b_no_d4_curves.json",
            "full_b_no_d4_seed20260907_provenance.json"),
    HistoricalRun(
        CHECKPOINTS / "final", "full_b_trainval_seed20260907", "final",
        "FULL_B", 20260907,
        CHECKPOINTS / "final" / "full_b_trainval_seed20260907_provenance.json",
        extra_tags=(("train_subset", "train+val"),),
    ),
)


def setting_for_label(
    label: Optional[str], report: CurveReport
) -> WatershedParams:
    """Resolve the ``@label`` of a curve key to the setting it names.

    Parameters
    ----------
    label : str or None
        ``None`` or ``"frozen"`` for the report's own setting, or
        ``"cdt=<value>"`` for that setting at another centre threshold.
    report : CurveReport

    Returns
    -------
    WatershedParams

    Raises
    ------
    ValueError
        For a label this study never produced, rather than guessing.
    """
    if label is None or label == "frozen":
        return report.unlabelled_setting
    match = re.fullmatch(r"cdt=([0-9.]+)", label)
    if match is None:
        raise ValueError(f"Unrecognized setting label {label!r}.")
    fields = dict(report.unlabelled_setting.to_kwargs())
    fields["center_distance_threshold"] = float(match.group(1))
    return WatershedParams(**fields)


def parse_curve_key(key: str) -> tuple[str, int, Optional[str]]:
    """Split ``run/epoch-N@label`` into its three parts.

    Parameters
    ----------
    key : str

    Returns
    -------
    tuple
        Run name, epoch, and the label or ``None``.

    Raises
    ------
    ValueError
        If the key names no epoch snapshot.
    """
    match = re.fullmatch(r"([^/]+)/epoch-(\d+)(?:@(.+))?", key)
    if match is None:
        raise ValueError(f"Not an epoch snapshot: {key!r}.")
    return match.group(1), int(match.group(2)), match.group(3)


def flatten_provenance(record: dict[str, Any], run: str) -> dict[str, str]:
    """Turn a provenance record into flat parameters.

    The environment block is left out - it goes in as the attached file
    and two tags - and a record describing several runs is reduced to
    the entry for this one.

    Parameters
    ----------
    record : dict
    run : str

    Returns
    -------
    dict of str to str
    """
    params: dict[str, str] = {}
    for key, value in record.items():
        if key == "provenance":
            continue
        if key == "runs":
            for entry in value:
                if entry.get("name") == run:
                    params.update(
                        {f"run_{k}": str(v) for k, v in entry.items()}
                    )
            continue
        params.update(_flatten(key, value))
    return params


def _flatten(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        flat: dict[str, str] = {}
        for key, inner in value.items():
            flat.update(_flatten(f"{prefix}.{key}", inner))
        return flat
    if isinstance(value, (list, tuple)):
        return {prefix: ",".join(map(str, value))}
    return {prefix: str(value)}


def last_event_file(log_dir: Path) -> Optional[Path]:
    """The newest TensorBoard file of a run, or ``None`` if there is none.

    Parameters
    ----------
    log_dir : Path

    Returns
    -------
    Path or None
    """
    files = sorted(
        log_dir.glob("events.out.tfevents.*"),
        key=lambda path: int(path.name.split(".")[3]),
    )
    if len(files) > 1:
        logger.info(
            "%s: %d TensorBoard files; reading only the last, %s.",
            log_dir, len(files), files[-1].name,
        )
    return files[-1] if files else None


def read_scalars(event_file: Path) -> dict[str, list[tuple[int, float]]]:
    """Every scalar series of one TensorBoard file.

    Parameters
    ----------
    event_file : Path

    Returns
    -------
    dict of str to list of (step, value)
    """
    from tensorboard.backend.event_processing.event_accumulator import \
        EventAccumulator

    accumulator = EventAccumulator(
        str(event_file), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def epoch_means(
    series: list[tuple[int, float]], epoch_end_steps: list[int]
) -> dict[int, float]:
    """Average a per-step series over each epoch.

    Parameters
    ----------
    series : list of (step, value)
    epoch_end_steps : list of int
        The step validation was logged at after each epoch, which is
        the first step of the next.

    Returns
    -------
    dict of int to float
        One-based epoch to mean; an epoch with no step is left out.
    """
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for step, value in series:
        epoch = next(
            (index + 1 for index, end in enumerate(epoch_end_steps)
             if step < end),
            None,
        )
        if epoch is None:
            continue
        sums[epoch] += value
        counts[epoch] += 1
    return {epoch: sums[epoch] / counts[epoch] for epoch in sums}


def import_losses(run_id: str, run: HistoricalRun) -> int:
    """Copy a run's TensorBoard scalars into its MLflow run.

    Parameters
    ----------
    run_id : str
    run : HistoricalRun

    Returns
    -------
    int
        Validation points found, one per epoch trained.
    """
    event_file = last_event_file(run.log_dir)
    if event_file is None:
        logger.warning("%s: no TensorBoard record.", run.name)
        return 0
    scalars = read_scalars(event_file)
    validation = scalars.get("validation/loss", [])
    epoch_end_steps = [step for step, _ in validation]

    per_step: dict[int, dict[str, float]] = defaultdict(dict)
    per_epoch: dict[int, dict[str, float]] = defaultdict(dict)
    for tag, name in TRAIN_TAGS.items():
        series = scalars.get(tag, [])
        for step, value in series:
            per_step[step][f"train/{name}"] = value
        for epoch, mean in epoch_means(series, epoch_end_steps).items():
            per_epoch[epoch][f"epoch/train_{name}"] = mean
    for step, value in scalars.get(LEARNING_RATE_TAG, []):
        per_step[step]["train/lr_lora"] = value
    for epoch, mean in epoch_means(
        scalars.get(LEARNING_RATE_TAG, []), epoch_end_steps
    ).items():
        per_epoch[epoch]["epoch/lr_lora"] = mean
    for tag, name in VALIDATION_TAGS.items():
        for index, (_, value) in enumerate(scalars.get(tag, [])):
            per_epoch[index + 1][f"epoch/val_{name}"] = value

    log_metric_series(run_id, [
        (name, value, step)
        for table in (per_step, per_epoch)
        for step, metrics in table.items()
        for name, value in metrics.items()
    ])
    return len(epoch_end_steps)


def import_curves(run_id: str, run: HistoricalRun) -> set[str]:
    """Copy a run's per-epoch evaluations into its MLflow run.

    Parameters
    ----------
    run_id : str
    run : HistoricalRun

    Returns
    -------
    set of str
        The post-processing settings imported, by identifier.
    """
    settings: set[str] = set()
    for report in run.curves:
        if not report.path.exists():
            logger.warning("%s: %s is missing.", run.name, report.path)
            continue
        payload = json.loads(report.path.read_text())
        for key, record in payload.items():
            name, epoch, label = parse_curve_key(key)
            if name != run.name:
                continue
            setting = postprocessing_id(setting_for_label(label, report))
            log_evaluation_to_run(
                run_id, epoch, "val", setting, record["overall"],
                list(record.get("per_material", []))
                + list(record.get("per_scale_bin", [])),
            )
            settings.add(setting)
    return settings


def import_test_scores(run_id: str, run: HistoricalRun) -> int:
    """Copy the TEST figures of a run's scored snapshot, if it has one.

    Parameters
    ----------
    run_id : str
    run : HistoricalRun

    Returns
    -------
    int
        Snapshots imported: zero or one.
    """
    if not TEST_SCORES.exists():
        return 0
    payload = json.loads(TEST_SCORES.read_text())
    imported = 0
    for key, record in payload.items():
        name, epoch, label = parse_curve_key(key)
        if name != run.name:
            continue
        setting = postprocessing_id(
            setting_for_label(label, CurveReport(TEST_SCORES))
        )
        log_evaluation_to_run(
            run_id, epoch, "test", setting, record["overall"],
            list(record.get("per_material", []))
            + list(record.get("per_scale_bin", [])),
        )
        imported += 1
    return imported


def run_tags(run: HistoricalRun, record: dict[str, Any]) -> dict[str, str]:
    """Tags for an imported run, matching those of a live one.

    Parameters
    ----------
    run : HistoricalRun
    record : dict
        Its provenance record.

    Returns
    -------
    dict of str to str
    """
    environment = record.get("provenance", {})
    git = environment.get("git", {})
    tags = {
        "study": "augmentation",
        "experiment_step": run.step,
        "policy": run.policy,
        "seed": str(run.seed),
        "train_subset": "train",
        "source": "backfill",
        "git_commit": str(git.get("commit", "")),
        "git_dirty": str(git.get("dirty", "")),
        CHECKPOINT_DIR_TAG: checkpoint_dir_tag(run.checkpoint_dir),
    }
    tags.update(dict(run.extra_tags))
    return tags


def import_run(run: HistoricalRun, experiment_id: str) -> str:
    """Create one run in the store and fill it from the records on disk.

    Parameters
    ----------
    run : HistoricalRun
    experiment_id : str

    Returns
    -------
    str
        The new run's identifier.
    """
    import mlflow

    record = json.loads(run.provenance.read_text())
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run.name,
        tags=run_tags(run, record),
    ) as active:
        run_id = active.info.run_id
        mlflow.log_params(flatten_provenance(record, run.name))
        mlflow.log_artifact(str(run.provenance), artifact_path="provenance")
    n_epochs = import_losses(run_id, run)
    settings = import_curves(run_id, run)
    for setting in sorted(settings):
        summarize_curve(run_id, "val", setting)
    n_test = import_test_scores(run_id, run)
    logger.info(
        "%s: %d epoch(s) of losses, curves under %s, %d TEST snapshot(s).",
        run.name, n_epochs, ", ".join(sorted(settings)) or "nothing",
        n_test,
    )
    return run_id


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
        "--replace", action="store_true",
        help="Delete runs imported earlier and import them again.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Import every listed run not yet in the store.

    Returns
    -------
    int
        Process exit code.
    """
    import mlflow

    args = parse_args(argv)
    setup_logging()
    configure_tracking()
    experiment_id = ensure_experiment(TRAINING_EXPERIMENT)

    failed = []
    for run in HISTORICAL_RUNS:
        existing = find_run_for_checkpoint_dir(run.checkpoint_dir)
        if existing is not None and not args.replace:
            logger.info("%s is already in the store; skipped.", run.name)
            continue
        if existing is not None:
            mlflow.delete_run(existing)
            logger.info("%s: deleted earlier import %s.", run.name, existing)
        try:
            import_run(run, experiment_id)
        except (OSError, ValueError, KeyError):
            logger.exception("%s could not be imported.", run.name)
            failed.append(run.name)
    if failed:
        logger.error("Not imported: %s.", ", ".join(failed))
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
