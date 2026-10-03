"""
Fine-tuning Cellpose-SAM on the frozen split of the PEFT-SAM study.

This is the one place the Cellpose recipe of the comparison is defined,
for the same reason ``materials_vision.training`` is for PEFT-SAM: a
difference between two runs can only be attributed to what differed
between them if everything else has one definition.

**What is shared with PEFT-SAM, and what is not.** The images, the
instances and the splits are shared: every sample comes from the same
``SampleSource``, with the information panel cropped and the cut
instances rebuilt, and is written to disk exactly as prepared because
Cellpose trains from files. The training itself is Cellpose's own, and
deliberately so - this comparison is between two recipes, each run the
way its library runs it. In particular the augmentation is the one
built into Cellpose's trainer, applied to every step and not optional:
a rotation by an arbitrary angle, a flip, a rescaling between 0.75 and
1.25, and a random 256-pixel crop. The PEFT-SAM study found rescaling
harmful for its own model and rejected it; here it is part of the
recipe under comparison, not a choice made for it.

**Every library default the recipe depends on is named.** The rescaling
range, the crop size and the minimum mask count are passed explicitly
rather than inherited, so that a Cellpose release that changed its mind
could not change the recipe without the change showing up here.

**Snapshots are kept every ten epochs, and all runs train the full
budget.** The epoch scored is chosen on VALIDATION among these
snapshots. Cellpose's schedule ends by halving the rate every five
epochs over the last fifty, so a run shortened to the chosen epoch
would put that decay somewhere the chosen snapshot never saw. The
final model and the cross-validation folds therefore train all 160
epochs and are scored at the chosen snapshot, which reproduces the
condition that was selected rather than an approximation of it.

**Where snapshots land, and what their numbers mean.** Cellpose writes
``<run>/models/<run>_epoch_NNNN`` after the zero-based epoch ``NNNN``
whenever it divides by ten, and ``<run>/models/<run>`` after the last.
The number in the name is therefore one less than the epochs completed:
``_epoch_0010`` has seen eleven. The helpers below convert, so that
every report counts completed epochs as the PEFT-SAM reports do.
"""
import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from materials_vision.data.samples import SampleSource
from materials_vision.experiments.cellpose.split_cache import (CachedSubset,
                                                               export_source)
from materials_vision.tracking import RunTracking, log_metric_series

logger = logging.getLogger(__name__)

BASE_MODEL = "cpsam"

# The seed of every PEFT-SAM run, so that the two studies share it.
RUN_SEED = 20260907

N_EPOCHS = 160

BATCH_SIZE = 8

WEIGHT_DECAY = 1e-4

# Per-image percentile normalization, Cellpose's default and the
# setting the earlier grid search found best.
NORMALIZE = True

# Images are not rescaled to a nominal object diameter before
# augmentation; Cellpose-SAM is trained to work at native resolution.
RESCALE = False

# Cellpose's default range, stated rather than inherited: images are
# rescaled by a factor drawn uniformly from [1 - r/2, 1 + r/2].
SCALE_RANGE = 0.5

CROP_PX = 256

# Zero, against the library's default of five: an image with few pores
# is still one the PEFT-SAM runs trained on, and dropping it would make
# the two training sets differ.
MIN_TRAIN_MASKS = 0

SAVE_EVERY = 10

# The two arms of the selection on VALIDATION. They differ in the
# learning rate alone: the rate the earlier grid search found best, and
# the library's own default for fine-tuning.
LEARNING_RATES = {"lr1e-3": 1e-3, "lr1e-5": 1e-5}

TRAINING_EXPERIMENT = "cellpose_training"

REHEARSAL_EXPERIMENT = "cellpose_rehearsal"

DEFAULT_CACHE_DIR = Path("/home/dwalicki/dane/cellpose_cache/manifest_v3")

_EPOCH_SUFFIX = re.compile(r".+_epoch_(\d+)")


class SnapshotNameError(ValueError):
    """Raised when a file is not a snapshot of a run laid out here."""


def models_dir(save_root: Path, name: str) -> Path:
    """Directory Cellpose writes a run's snapshots into.

    Parameters
    ----------
    save_root : Path
    name : str

    Returns
    -------
    Path
    """
    return save_root / name / "models"


def snapshot_path(
    save_root: Path, name: str, completed: int, n_epochs: int = N_EPOCHS
) -> Path:
    """The snapshot a run leaves after a given number of epochs.

    Parameters
    ----------
    save_root : Path
    name : str
    completed : int
        Epochs completed, one-based.
    n_epochs : int, optional
        The run's budget, whose last snapshot carries no epoch suffix.

    Returns
    -------
    Path

    Raises
    ------
    SnapshotNameError
        If no snapshot is saved after that many epochs.
    """
    saved = saved_epochs(n_epochs)
    if completed not in saved:
        raise SnapshotNameError(
            f"No snapshot is saved after {completed} epoch(s) of a "
            f"{n_epochs}-epoch run; they are saved after "
            f"{', '.join(map(str, saved))}."
        )
    directory = models_dir(save_root, name)
    if completed == n_epochs:
        return directory / name
    return directory / f"{name}_epoch_{completed - 1:04d}"


def saved_epochs(n_epochs: int = N_EPOCHS) -> tuple[int, ...]:
    """Completed epochs after which a run leaves a snapshot.

    Parameters
    ----------
    n_epochs : int, optional

    Returns
    -------
    tuple of int
    """
    intermediate = [
        index + 1 for index in range(SAVE_EVERY, n_epochs - 1, SAVE_EVERY)
    ]
    return (*intermediate, n_epochs)


def run_of(snapshot: Path) -> str:
    """The run a snapshot belongs to, read from where it lies.

    Parameters
    ----------
    snapshot : Path

    Returns
    -------
    str
    """
    return snapshot.parent.parent.name


def completed_epochs(snapshot: Path, n_epochs: int = N_EPOCHS) -> int:
    """Epochs a snapshot had completed when it was saved.

    Parameters
    ----------
    snapshot : Path
    n_epochs : int, optional
        The run's budget, which the unsuffixed last snapshot stands for.

    Returns
    -------
    int

    Raises
    ------
    SnapshotNameError
        If the name is neither a numbered snapshot nor the last one of
        the run it lies in.
    """
    match = _EPOCH_SUFFIX.fullmatch(snapshot.name)
    if match is not None:
        return int(match.group(1)) + 1
    if snapshot.name == run_of(snapshot):
        return n_epochs
    raise SnapshotNameError(
        f"{snapshot} is not a snapshot of a run laid out as "
        f"<run>/models/<run>[_epoch_NNNN]."
    )


def recipe_params(learning_rate: float) -> dict[str, Any]:
    """Everything that defines a run of the recipe, for MLflow.

    Parameters
    ----------
    learning_rate : float

    Returns
    -------
    dict
    """
    return {
        "base_model": BASE_MODEL,
        "seed": RUN_SEED,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "normalize": NORMALIZE,
        "rescale": RESCALE,
        "scale_range": SCALE_RANGE,
        "crop_px": CROP_PX,
        "min_train_masks": MIN_TRAIN_MASKS,
        "save_every": SAVE_EVERY,
        "augmentation": "cellpose_builtin",
    }


def select_positions(
    cached: CachedSubset, positions: Optional[Sequence[int]]
) -> tuple[list[str], list[str]]:
    """The image and mask files of a run, optionally restricted.

    Parameters
    ----------
    cached : CachedSubset
    positions : sequence of int, optional
        ``None`` keeps every sample.

    Returns
    -------
    tuple of list of str
        Image files, then mask files, aligned.
    """
    chosen = range(len(cached)) if positions is None else positions
    return (
        [str(cached.image_paths[i]) for i in chosen],
        [str(cached.mask_paths[i]) for i in chosen],
    )


def train_cellpose(
    name: str,
    *,
    train_source: SampleSource,
    monitor_source: Optional[SampleSource],
    learning_rate: float,
    save_root: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    n_epochs: int = N_EPOCHS,
    train_positions: Optional[Sequence[int]] = None,
    tracking: Optional[RunTracking] = None,
) -> Path:
    """Run one fine-tuning of the recipe to completion.

    Parameters
    ----------
    name : str
        Names the run directory, the snapshots and the MLflow run.
    train_source : SampleSource
    monitor_source : SampleSource, optional
        Images Cellpose computes a test loss on at epoch five and every
        tenth epoch. The loss is of augmented crops and decides
        nothing; it is kept for the curves. ``None`` computes none.
    learning_rate : float
    save_root : Path
    cache_dir : Path, optional
    n_epochs : int, optional
        Differs from the recipe's budget only in a rehearsal.
    train_positions : sequence of int, optional
        Train on these samples only, for a rehearsal.
    tracking : RunTracking, optional
        Record the run in MLflow. ``None`` records nothing.

    Returns
    -------
    Path
        The directory holding the run's snapshots.
    """
    from cellpose import models, train

    from materials_vision.experiments.cellpose.utils import (
        patch_cellpose_get_batch, precompute_flows_batched)
    from materials_vision.training import tracked_run

    torch.manual_seed(RUN_SEED)
    train_images, train_masks = select_positions(
        export_source(train_source, cache_dir), train_positions
    )
    monitor_images: Optional[list[str]] = None
    monitor_masks: Optional[list[str]] = None
    if monitor_source is not None:
        monitor_images, monitor_masks = select_positions(
            export_source(monitor_source, cache_dir), None
        )

    model = models.CellposeModel(gpu=True, pretrained_model=BASE_MODEL)
    device = model.net.device
    train_flows = precompute_flows_batched(
        train_masks, train_images, device=device
    )
    monitor_flows = None
    if monitor_images is not None and monitor_masks is not None:
        monitor_flows = precompute_flows_batched(
            monitor_masks, monitor_images, device=device
        )

    directory = models_dir(save_root, name)
    run_dir = directory.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Training %r from %s: %d image(s), %d monitoring, %d epoch(s), "
        "lr %.0e, into %s.", name, BASE_MODEL, len(train_images),
        0 if monitor_images is None else len(monitor_images), n_epochs,
        learning_rate, directory,
    )

    def fit():
        patch_cellpose_get_batch()
        return train.train_seg(
            model.net,
            train_files=train_images,
            train_labels_files=train_flows,
            test_files=monitor_images,
            test_labels_files=monitor_flows,
            load_files=False,
            batch_size=BATCH_SIZE,
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            n_epochs=n_epochs,
            normalize=NORMALIZE,
            rescale=RESCALE,
            scale_range=SCALE_RANGE,
            bsize=CROP_PX,
            min_train_masks=MIN_TRAIN_MASKS,
            save_path=str(run_dir),
            save_every=SAVE_EVERY,
            save_each=True,
            model_name=name,
        )

    if tracking is None:
        fit()
        return directory

    params = {
        **recipe_params(learning_rate),
        "n_epochs": n_epochs,
        "n_train_images": len(train_images),
        "n_monitor_images": (
            0 if monitor_images is None else len(monitor_images)
        ),
        "cache_dir": str(cache_dir),
    }
    with tracked_run(name, directory, tracking, params) as run_id:
        _, train_losses, test_losses = fit()
        log_metric_series(run_id, loss_points(train_losses, test_losses))
    return directory


def loss_points(
    train_losses: Sequence[float], test_losses: Sequence[float]
) -> list[tuple[str, float, int]]:
    """The loss curves as MLflow points, stepped by completed epochs.

    Cellpose returns one entry per epoch in both arrays but computes the
    test loss only on some of them, leaving zeros elsewhere. Those zeros
    are not losses and are left out rather than drawn as a value.

    Parameters
    ----------
    train_losses, test_losses : sequence of float

    Returns
    -------
    list of (name, value, step)
    """
    points = [
        ("training/train_loss", float(loss), index + 1)
        for index, loss in enumerate(train_losses)
    ]
    points.extend(
        ("training/test_loss", float(loss), index + 1)
        for index, loss in enumerate(test_losses)
        if loss > 0
    )
    return points
