"""
Shared training setup for PEFT-SAM (LoRA) fine-tuning.

Every training script builds its model, data loaders and optimizer
through this module, so runs differ only in the arguments they pass.

Contents
--------
- Model defaults: LoRA rank and target layers, frozen components,
  trainable part (``both``, ``lora`` or ``decoder``).
- Learning rates: separate rates for the LoRA layers and the instance
  decoder, with a cosine schedule annealed over the run length.
- Data: ``build_source`` reads a subset of the split, ``build_loader``
  wraps it in a DataLoader (batch size 1, seeded sampler, optional
  augmentation, images at native resolution).
- ``prepare_geometry``: fixes the library's encoder input resize, which
  otherwise distorts the image aspect ratio during training. Must be
  called before a model is built.
- ``train_run``: runs one fine-tuning to completion.
- ``training_params`` / ``tracked_run``: MLflow parameters and run
  logging.
"""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from micro_sam.models.peft_sam import LoRASurgery
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from materials_vision.data import (ProportionalImageSampler, SampleSource,
                                   load_split, merge_subsets, read_manifest)
from materials_vision.data.dataset import (InstanceSegmentationDataset,
                                           build_label_transform)
from materials_vision.sam_geometry import (patch_resize_longest_side,
                                           verify_preprocess_geometry)
from materials_vision.tracking import (CHECKPOINT_DIR_TAG, RunTracking,
                                       checkpoint_dir_tag, configure_tracking,
                                       ensure_experiment)
from materials_vision.training_logger import (PARAMETER_GROUP_NAMES,
                                              MlflowJointSamLogger)

logger = logging.getLogger(__name__)

# Smallest surviving piece of an instance the panel crop may cut, in
# square pixels of the source grid. Measured once over the training
# annotations as the first percentile of instance area.
A_MIN_FRAGMENT_PX2 = 388.43

# Rank of the low-rank correction. Chosen as the middle of the range
# usual in the literature rather than optimized: it affects every run
# identically, so it cannot bias a comparison between policies, and
# the reference work for this domain reports no value at all.
LORA_RANK = 8

PEFT_KWARGS: dict[str, Any] = {
    "rank": LORA_RANK,
    # Named rather than left to the library's default, which it
    # currently also is. A value this study treats as frozen should not
    # be able to change underneath it because a dependency changed its
    # mind, and the same reliance on an unstated default is what made
    # the encoder input geometry wrong for weeks.
    "peft_module": LoRASurgery,
    "update_matrices": ["q", "v"],
    # Empty means every attention block of the backbone.
    "attention_layers_to_update": [],
    "quantize": False,
}

# Frozen outright, against the library default. See the module
# docstring for why leaving the mask decoder trainable would change
# what the run is.
FREEZE_PARTS = ["prompt_encoder", "mask_decoder"]

# What a run adapts: the low-rank correction and the instance decoder
# together, the correction alone, or the decoder alone.
TRAINABLE_CHOICES = ("both", "lora", "decoder")

_BATCHNORM_KINDS = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
)

# Thirty times the library's default, and measured rather than argued.
# A low-rank correction starts at exactly zero and holds under a
# million parameters, so a rate tuned for whole decoders barely moves
# it. A four-point grid over both rates, ten epochs each, put the two
# arms at this rate a full point of instance F1 above the two at a
# third of it - 0.858 and 0.860 against 0.847 and 0.844 - and that gap
# held whichever rate the decoder ran at.
LORA_LEARNING_RATE = 3e-4

# The library's own default, which is where a pretrained decoder
# trained in full belongs. The same grid could not separate this from
# three times it: the two differed by 0.0026 where the spread between
# seeds is 0.0136. Where a probe cannot decide, this holds the flatter
# curve and the better average across epochs two to ten, it is the
# value a reader would expect to see defended, and it is not the arm
# the watershed calibration was performed on - which removes the one
# way that calibration could have favoured a winner.
DECODER_LEARNING_RATE = 1e-5

# Instances sampled per step to prompt the interactive branch. With
# images carrying hundreds of pores this, not the batch size, is what
# sets how wide a step is.
N_OBJECTS_PER_BATCH = 25

BATCH_SIZE = 1

# Iterative prompting rounds per step, the probability of feeding the
# previous round's mask back in, how much the boxes derived from the
# annotation are distorted, steps between logged images, and batches of
# each loader checked before training starts. All five are the
# library's own values, restated because the optimizer below is built
# here rather than there: a fork that quietly changed one of them would
# make every number this study reports incomparable with the runs that
# came before it.
N_SUB_ITERATION = 8

MASK_PROB = 0.5

BOX_DISTORTION_FACTOR = 0.025

LOG_IMAGE_INTERVAL = 100

VERIFY_N_LABELS_IN_LOADER = 50

# Worker processes preparing samples ahead of the GPU. Measured to hide
# sample preparation entirely behind the optimizer step, with a margin
# of four; it cannot change what a run produces, because a sample's
# augmentation is seeded from its position in the epoch rather than
# from the order in which workers finish.
NUM_WORKERS = 4

TRAIN_SUBSET = "train"


def prepare_geometry() -> None:
    """Install and verify the encoder input geometry.

    Must be called before a model is built, by every entry point that
    trains or evaluates one. Verification is separate from installation
    on purpose: it calls the library's own function rather than the
    replacement, so it catches both an installation that did not take
    and a library update that changed the behaviour underneath it.

    Raises
    ------
    SamGeometryError
        If any input geometry would reach the encoder at a size nobody
        chose.
    """
    patch_resize_longest_side()
    for geometry_px, content_px in verify_preprocess_geometry().items():
        logger.info(
            "Content %dx%d reaches the encoder as %dx%d.",
            geometry_px[0], geometry_px[1], content_px[0], content_px[1],
        )


def build_source(
    split_csv: Path,
    manifest_csv: Path,
    subset: str,
    *,
    allow_test: bool = False,
) -> SampleSource:
    """Open one subset of the frozen split, or several joined, for reading.

    Parameters
    ----------
    split_csv, manifest_csv : Path
    subset : str
        ``"train"`` or ``"val"``, or several joined by ``"+"``:
        ``"train+val"`` is what the model trained after every choice
        has been made on VALIDATION learns from. ``"test"`` is reachable
        only with ``allow_test`` and is never joined to anything.
    allow_test : bool, optional
        Handed to the lock on TEST, which lives with the split reader
        rather than here so that no loader can route around it. Only
        the single, final evaluation sets it.

    Returns
    -------
    SampleSource
    """
    parts = [
        load_split(split_csv, subset=name, allow_test=allow_test)
        for name in subset.split("+")
    ]
    split = parts[0] if len(parts) == 1 else merge_subsets(parts)
    manifest = read_manifest(manifest_csv)
    return SampleSource(
        split, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )


def build_loader(
    source: SampleSource,
    *,
    policy: Optional[Callable] = None,
    run_seed: int = 0,
    shuffle: bool = False,
    indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """Wrap a sample source as the loader the trainer consumes.

    Parameters
    ----------
    source : SampleSource
    policy : Callable, optional
        Augmentation policy. ``None`` is both the baseline condition
        and the only correct setting for validation, where a
        transformed image would be measuring the transformation rather
        than the model.
    run_seed : int, optional
        Seeds both the image order and the augmentation, through
        separate namespaces, so a run reproduces regardless of how many
        workers load it.
    shuffle : bool, optional
        Permute the order each epoch. On for training, off elsewhere.
    indices : sequence of int, optional
        Restrict to these positions, for pilots that do not need the
        whole subset.

    Returns
    -------
    DataLoader

    Notes
    -----
    The sampler is always supplied, never the loader's own ``shuffle``:
    it owns a random stream that no augmentation policy can disturb,
    and it is what puts the epoch into the index the dataset receives.
    Restricting to a subset is expressed through the sampler too rather
    than by wrapping the dataset, because a wrapper would renumber the
    indices and the epoch packed into them would be read back as a
    position.
    """
    dataset = InstanceSegmentationDataset(
        source,
        label_transform=build_label_transform(),
        transform=policy,
        run_seed=run_seed,
    )
    sampler = ProportionalImageSampler(
        len(dataset), run_seed, positions=indices, shuffle=shuffle,
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS,
    )


class CosineAnnealingIgnoringMetric(CosineAnnealingLR):
    """Cosine annealing that tolerates being handed a metric.

    The trainer advances its scheduler as ``step(current_metric)``,
    written for the plateau scheduler it defaults to. Every other
    scheduler reads that positional argument as an epoch number, so a
    cosine schedule handed a validation loss of 0.077 would look up the
    rate for epoch zero on every epoch and hold its initial rate for
    the whole run, while appearing to work. Swallowing the argument is
    the smallest correct answer, and subclassing rather than wrapping
    keeps the state dictionary and the checkpoint round-trip the
    trainer expects.
    """

    def step(  # type: ignore[override]
        self,
        metric: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Advance the schedule by one epoch.

        Parameters
        ----------
        metric : float, optional
            Accepted and ignored; see the class docstring.
        epoch : int, optional
            Accepted and ignored, so that neither calling convention
            can silently re-target the schedule.
        """
        super().step()


def _trainable_groups(
    model: Any, unetr: Any
) -> tuple[list[Any], list[Any]]:
    """Split what trains into its two learning regimes.

    The instance decoder holds the image encoder as a submodule - the
    same object, not a copy - so its parameters have to be selected by
    excluding that submodule rather than by taking everything. Without
    the exclusion the correction would appear in both groups and be
    stepped twice per iteration, at two different rates.

    Parameters
    ----------
    model : TrainableSAM
    unetr : torch.nn.Module

    Returns
    -------
    tuple of list
        The low-rank correction's parameters, then the decoder's.
    """
    correction = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    decoder = [
        parameter for name, parameter in unetr.named_parameters()
        if not name.startswith("encoder")
    ]
    return correction, decoder


def _check_trainable(
    trainable: str, decoder_batchnorm_momentum: Optional[float]
) -> None:
    """Reject a choice of what trains before any model is built.

    Parameters
    ----------
    trainable : str
    decoder_batchnorm_momentum : float, optional

    Raises
    ------
    ValueError
        If ``trainable`` is not one of :data:`TRAINABLE_CHOICES`, or if
        a batch-norm momentum is given for a run whose decoder is
        frozen - its batch statistics are held fixed there, so the
        momentum would be silently ignored.
    """
    if trainable not in TRAINABLE_CHOICES:
        raise ValueError(
            f"trainable must be one of {TRAINABLE_CHOICES}, "
            f"got {trainable!r}."
        )
    if trainable == "lora" and decoder_batchnorm_momentum is not None:
        raise ValueError(
            "decoder_batchnorm_momentum has no effect when only the "
            "low-rank correction trains: the frozen decoder's batch "
            "statistics are held fixed."
        )


def _select_trainable(
    correction: list[Any], decoder: list[Any], trainable: str
) -> tuple[list[Any], list[Any]]:
    """Freeze the half of the model a run does not adapt.

    Freezing is done on the parameters rather than only by leaving them
    out of the optimizer, so that no gradient is computed for them and
    nothing downstream can mistake them for trained.

    Parameters
    ----------
    correction, decoder : list of torch.nn.Parameter
        From :func:`_trainable_groups`. Parameters of the frozen half
        have ``requires_grad`` switched off.
    trainable : str
        One of :data:`TRAINABLE_CHOICES`.

    Returns
    -------
    tuple of list
        The correction's parameters, then the decoder's, with the
        frozen half emptied.
    """
    train_correction = trainable in ("both", "lora")
    train_decoder = trainable in ("both", "decoder")
    for parameter in [
        *([] if train_correction else correction),
        *([] if train_decoder else decoder),
    ]:
        parameter.requires_grad = False
    return (
        correction if train_correction else [],
        decoder if train_decoder else [],
    )


def _parameter_groups(
    correction: list[Any],
    lora_learning_rate: float,
    decoder: list[Any],
    decoder_learning_rate: float,
) -> list[dict[str, Any]]:
    """The optimizer's groups, leaving out a frozen half.

    The trainer logs the rate of the first group as the run's rate, so
    a group with nothing in it must not come first. Each group carries
    its name, because with one left out its position no longer says
    which half it is.

    Parameters
    ----------
    correction : list of torch.nn.Parameter
    lora_learning_rate : float
    decoder : list of torch.nn.Parameter
    decoder_learning_rate : float

    Returns
    -------
    list of dict
    """
    lora_name, decoder_name = PARAMETER_GROUP_NAMES
    groups = [
        {"params": correction, "lr": lora_learning_rate, "name": lora_name},
        {"params": decoder, "lr": decoder_learning_rate, "name": decoder_name},
    ]
    return [group for group in groups if group["params"]]


def _decoder_batchnorm_layers(unetr: Any) -> list[Any]:
    """The decoder's batch-norm layers, excluding the shared encoder.

    Parameters
    ----------
    unetr : torch.nn.Module

    Returns
    -------
    list of torch.nn.Module
    """
    return [
        module for name, module in unetr.named_modules()
        if not name.startswith("encoder")
        and isinstance(module, _BATCHNORM_KINDS)
    ]


def _hold_decoder_batchnorm_in_eval(unetr: Any) -> int:
    """Keep a frozen decoder normalizing the way it does at inference.

    The trainer switches the decoder to training mode at the start of
    every epoch. A frozen decoder would then normalize each step by the
    statistics of its single image - so the correction would be adapted
    to a decoder that behaves differently from the one it is evaluated
    with - and would keep rewriting its running statistics. A hook ahead
    of every forward pass returns the layers to evaluation mode,
    whatever the trainer set.

    Parameters
    ----------
    unetr : torch.nn.Module

    Returns
    -------
    int
        How many layers are held, so a caller can log a zero rather
        than assume it worked.
    """
    layers = _decoder_batchnorm_layers(unetr)

    def hold(_module: Any, _inputs: Any) -> None:
        for layer in layers:
            layer.eval()

    unetr.register_forward_pre_hook(hold)
    return len(layers)


def _without_interactive_training(trainer_class: type) -> type:
    """A trainer that runs the interactive branch but never steps it.

    With the correction frozen, nothing on the interactive side has a
    gradient - the prompt encoder and the mask decoder are frozen as
    well - so its loss cannot be backpropagated and the trainer would
    stop at the first step. The branch is still run, under ``no_grad``,
    so its losses keep being logged and show the interactive model
    staying where it started.

    Parameters
    ----------
    trainer_class : type
        The joint trainer to derive from.

    Returns
    -------
    type

    Raises
    ------
    AttributeError
        If the trainer no longer has a method this overrides, which
        would otherwise turn the override into a silent no-op.
    """
    overridden = (
        "_interactive_train_iteration", "_backprop", "_backprop_mixed",
    )
    missing = [n for n in overridden if not hasattr(trainer_class, n)]
    if missing:
        raise AttributeError(
            f"{trainer_class.__name__} has no {', '.join(missing)}; "
            "the decoder-only override needs updating."
        )

    class DecoderOnlyTrainer(trainer_class):  # type: ignore[valid-type,misc]
        def _interactive_train_iteration(self, x: Any, y: Any) -> Any:
            with torch.no_grad():
                return super()._interactive_train_iteration(x, y)

        def _backprop(self, loss: torch.Tensor) -> None:
            if loss.requires_grad:
                super()._backprop(loss)

        def _backprop_mixed(self, loss: torch.Tensor) -> None:
            if loss.requires_grad:
                super()._backprop_mixed(loss)

    return DecoderOnlyTrainer


def _set_decoder_batchnorm_momentum(unetr: Any, momentum: float) -> int:
    """Slow down how fast the decoder's batch statistics move.

    The decoder normalizes over the batch, and the batch is one image.
    Its running statistics therefore track the handful of images seen
    most recently rather than the training distribution, and a snapshot
    carries whichever images happened to end its epoch. Lowering the
    momentum averages over a far longer stretch, which costs nothing
    during training and makes a snapshot describe the model instead of
    the tail of an epoch.

    Parameters
    ----------
    unetr : torch.nn.Module
    momentum : float

    Returns
    -------
    int
        How many layers were changed, so a caller can log a zero rather
        than assume it worked.
    """
    layers = _decoder_batchnorm_layers(unetr)
    for layer in layers:
        layer.momentum = momentum
    return len(layers)


def train_run(
    name: str,
    *,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    save_root: Path,
    lora_learning_rate: float = LORA_LEARNING_RATE,
    decoder_learning_rate: float = DECODER_LEARNING_RATE,
    lora_rank: int = LORA_RANK,
    update_matrices: Optional[Sequence[str]] = None,
    trainable: str = "both",
    scheduler_horizon_epochs: Optional[int] = None,
    decoder_batchnorm_momentum: Optional[float] = None,
    early_stopping: Optional[int] = None,
    save_every_kth_epoch: Optional[int] = None,
    device: Optional[str] = None,
    tracking: Optional[RunTracking] = None,
) -> None:
    """Run one fine-tuning to completion.

    This assembles the model, the decoder, the optimizer and the
    trainer the way the library's own entry point does, and departs
    from it in exactly four places, all of them deliberate:

    1. **Two learning rates.** The library builds one parameter list at
       one rate. See the module docstring for why the correction and
       the decoder cannot share one.
    2. **Only trainable parameters reach the optimizer.** The library
       hands it the frozen ones too. They cannot move either way, so
       this changes nothing a run produces and keeps the optimizer's
       state from carrying three hundred million parameters that will
       never receive a gradient.
    3. **A schedule with a horizon**, rather than one that reacts to a
       plateau by shaving a tenth off the rate.
    4. **Warnings are not silenced.** The library suppresses them for
       the duration of training by default. This project has already
       lost weeks to a defect that produced no error, so the warnings
       are left where they can be read.

    With ``tracking`` given, the trainer's logger is additionally
    replaced by one that writes to MLflow as well as to TensorBoard.
    It only reads what the trainer hands it, plus one extra forward
    pass per epoch on a validation image under ``no_grad``, so it does
    not change what a run learns.

    The geometry correction is installed here rather than left to the
    caller, because it has to be in place before the model exists and
    a caller that forgot would produce a run that looks healthy and
    trains on distorted images.

    Base weights come from the name alone: a micro-scope generalist
    checkpoint carries a matching instance decoder as a separate file,
    and the library fetches it alongside the encoder and hands it to
    the decoder as its starting state. Passing the decoder's own
    registry name here would instead try to load a decoder in place of
    an encoder.

    Parameters
    ----------
    name : str
        Names the checkpoint directory and the logs.
    model_type : str
        Registry name of the base checkpoint, without a decoder suffix.
    train_loader, val_loader : DataLoader
    n_epochs : int
    save_root : Path
        Directory the checkpoints and logs are written under.
    lora_learning_rate, decoder_learning_rate : float, optional
    lora_rank : int, optional
        Overrides the frozen rank. Exists so that the rank can be
        probed the way the learning rates were, on a short run, without
        editing the frozen configuration. Every comparison that
        attributes anything must leave it at the default.
    update_matrices : sequence of str, optional
        Attention projections that receive the correction. ``None``
        keeps the frozen choice, the query and value projections.
    trainable : {"both", "lora", "decoder"}, optional
        What the run adapts. ``"both"`` is the configuration every
        comparison of policies uses. ``"lora"`` freezes the instance
        decoder and holds its batch statistics at their pretrained
        values; the correction still learns from both losses, the
        instance one reaching it through the frozen decoder.
        ``"decoder"`` freezes the correction, which therefore stays at
        exactly zero, and runs the interactive branch without stepping
        it. The rate of the frozen half is ignored, and the checkpoint
        keeps its usual layout either way.
    scheduler_horizon_epochs : int, optional
        Epochs the rate is annealed over. Defaults to the run's own
        length; a run meant to be stopped early and resumed later has
        to pass the full budget instead, or its rate would fall to zero
        at the point it was interrupted and the resumed half would
        train at nothing.
    decoder_batchnorm_momentum : float, optional
        Overrides how fast the decoder's batch statistics move.
        ``None`` leaves the library's value untouched. Must be ``None``
        when ``trainable`` is ``"lora"``.
    early_stopping : int, optional
        Epochs without improvement before stopping. ``None`` disables
        it, which is what a comparison of fixed budgets needs: a run
        that halted early would leave a shorter curve than the one it
        is compared against.
    save_every_kth_epoch : int, optional
        Keep a numbered checkpoint every this many epochs, alongside
        the running best and last. Without it only two snapshots
        survive a run, both chosen by the library's own validation
        loss - which is not the quantity this study selects models
        on. Keeping the whole series lets the intended metric be
        computed afterwards, on every snapshot, and the best one
        picked by it.
    device : str, optional
    tracking : RunTracking, optional
        Record the run in MLflow as it trains. ``None`` records it in
        TensorBoard only, as every run before this option existed was.

    Raises
    ------
    ValueError
        If ``trainable`` is unknown or conflicts with
        ``decoder_batchnorm_momentum``.
    """
    _check_trainable(trainable, decoder_batchnorm_momentum)

    import torch_em
    from micro_sam.instance_segmentation import get_unetr
    from micro_sam.training.joint_sam_trainer import (JointSamLogger,
                                                      JointSamTrainer)
    from micro_sam.training.training import _check_loader
    from micro_sam.training.util import (ConvertToSamInputs,
                                         get_trainable_sam_model)
    from micro_sam.util import get_device

    prepare_geometry()
    _check_loader(train_loader, True, "train", VERIFY_N_LABELS_IN_LOADER)
    _check_loader(val_loader, True, "val", VERIFY_N_LABELS_IN_LOADER)

    peft_kwargs = dict(PEFT_KWARGS)
    peft_kwargs["rank"] = lora_rank
    if update_matrices is not None:
        peft_kwargs["update_matrices"] = list(update_matrices)
    resolved_device = get_device(device)
    model, state = get_trainable_sam_model(
        model_type=model_type,
        device=resolved_device,
        freeze=list(FREEZE_PARTS),
        return_state=True,
        peft_kwargs=peft_kwargs,
    )
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device=resolved_device,
    )
    if decoder_batchnorm_momentum is not None:
        logger.info(
            "Batch-norm momentum set to %.3g on %d decoder layer(s).",
            decoder_batchnorm_momentum,
            _set_decoder_batchnorm_momentum(
                unetr, decoder_batchnorm_momentum
            ),
        )

    correction, decoder = _select_trainable(
        *_trainable_groups(model, unetr), trainable
    )
    if trainable == "lora":
        logger.info(
            "Batch-norm held in evaluation mode on %d frozen decoder "
            "layer(s).", _hold_decoder_batchnorm_in_eval(unetr),
        )
    optimizer = torch.optim.AdamW(_parameter_groups(
        correction, lora_learning_rate, decoder, decoder_learning_rate,
    ))
    trainer_class = (
        _without_interactive_training(JointSamTrainer)
        if trainable == "decoder" else JointSamTrainer
    )
    horizon = (
        n_epochs if scheduler_horizon_epochs is None
        else scheduler_horizon_epochs
    )
    instance_loss = torch_em.loss.DiceBasedDistanceLoss(
        mask_distances_in_bg=True
    )
    logger.info(
        "Training %r (%s) from %s: %d epoch(s) x %d step(s), rank %d, "
        "correction %d parameter(s) at lr %.0e, decoder %d at %.0e, "
        "annealed over %d epoch(s).",
        name, trainable, model_type, n_epochs, len(train_loader),
        lora_rank, sum(p.numel() for p in correction), lora_learning_rate,
        sum(p.numel() for p in decoder), decoder_learning_rate, horizon,
    )

    trainer = trainer_class(
        name=name,
        save_root=str(save_root),
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=resolved_device,
        lr_scheduler=CosineAnnealingIgnoringMetric(
            optimizer, T_max=horizon
        ),
        logger=JointSamLogger if tracking is None else MlflowJointSamLogger,
        log_image_interval=LOG_IMAGE_INTERVAL,
        mixed_precision=True,
        convert_inputs=ConvertToSamInputs(
            transform=model.transform,
            box_distortion_factor=BOX_DISTORTION_FACTOR,
        ),
        n_objects_per_batch=N_OBJECTS_PER_BATCH,
        n_sub_iteration=N_SUB_ITERATION,
        compile_model=False,
        unetr=unetr,
        instance_loss=instance_loss,
        instance_metric=instance_loss,
        early_stopping=early_stopping,
        mask_prob=MASK_PROB,
    )
    fit_kwargs: dict[str, Any] = {
        "epochs": n_epochs, "overwrite_training": True,
    }
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    if tracking is None:
        trainer.fit(**fit_kwargs)
        return
    params = training_params(
        model_type=model_type,
        n_epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        n_val_images=len(val_loader),
        peft_kwargs=peft_kwargs,
        trainable=trainable,
        lora_learning_rate=lora_learning_rate if correction else None,
        decoder_learning_rate=decoder_learning_rate if decoder else None,
        scheduler_horizon_epochs=horizon,
        decoder_batchnorm_momentum=decoder_batchnorm_momentum,
        early_stopping=early_stopping,
        n_lora_parameters=sum(p.numel() for p in correction),
        n_decoder_parameters=sum(p.numel() for p in decoder),
    )
    checkpoint_dir = Path(save_root) / "checkpoints" / name
    with tracked_run(name, checkpoint_dir, tracking, params):
        trainer.fit(**fit_kwargs)


def training_params(
    *,
    model_type: str,
    n_epochs: int,
    steps_per_epoch: int,
    n_val_images: int,
    peft_kwargs: dict[str, Any],
    trainable: str,
    lora_learning_rate: Optional[float],
    decoder_learning_rate: Optional[float],
    scheduler_horizon_epochs: int,
    decoder_batchnorm_momentum: Optional[float],
    early_stopping: Optional[int],
    n_lora_parameters: int,
    n_decoder_parameters: int,
) -> dict[str, Any]:
    """Everything that defines a run, as the flat record MLflow keeps.

    The fixed values of this module are included alongside the ones a
    caller passed, so a run's page states its full configuration rather
    than only what differed from a default a reader would have to look
    up - and so that a change to a default shows up as a difference
    between two runs' parameters.

    The rate of a half that does not train is recorded as ``None``
    rather than as the value the caller happened to pass, so a run's
    page cannot suggest that something frozen was trained.

    Returns
    -------
    dict of str to Any
    """
    return {
        "model_type": model_type,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "n_val_images": n_val_images,
        "lora_rank": peft_kwargs["rank"],
        "lora_update_matrices": ",".join(peft_kwargs["update_matrices"]),
        "lora_attention_layers": (
            ",".join(map(str, peft_kwargs["attention_layers_to_update"]))
            or "all"
        ),
        "trainable": trainable,
        "lora_learning_rate": lora_learning_rate,
        "decoder_learning_rate": decoder_learning_rate,
        "scheduler": "cosine",
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "decoder_batchnorm_momentum": decoder_batchnorm_momentum,
        "early_stopping": early_stopping,
        "freeze_parts": ",".join(FREEZE_PARTS),
        "batch_size": BATCH_SIZE,
        "n_objects_per_batch": N_OBJECTS_PER_BATCH,
        "n_sub_iteration": N_SUB_ITERATION,
        "mask_prob": MASK_PROB,
        "box_distortion_factor": BOX_DISTORTION_FACTOR,
        "min_fragment_area_px2": A_MIN_FRAGMENT_PX2,
        "n_lora_parameters": n_lora_parameters,
        "n_decoder_parameters": n_decoder_parameters,
    }


@contextmanager
def tracked_run(
    name: str,
    checkpoint_dir: Path,
    tracking: RunTracking,
    params: dict[str, Any],
) -> Iterator[str]:
    """Open the MLflow run a training writes into.

    The checkpoint directory is stored as a tag: it is what scoring the
    snapshots later has in hand, and it is how that scoring finds this
    run to add the instance metrics to.

    Parameters
    ----------
    name : str
    checkpoint_dir : Path
    tracking : RunTracking
    params : dict
        From :func:`training_params`.

    Yields
    ------
    str
        The run's identifier.
    """
    import mlflow

    configure_tracking()
    experiment_id = ensure_experiment(tracking.experiment)
    tags = {
        **dict(tracking.tags),
        CHECKPOINT_DIR_TAG: checkpoint_dir_tag(checkpoint_dir),
    }
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=name, tags=tags,
        log_system_metrics=True,
    ) as run:
        mlflow.log_params({**params, **dict(tracking.params)})
        for artifact in tracking.artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="provenance")
        logger.info(
            "Recording %r in MLflow experiment %r as run %s.",
            name, tracking.experiment, run.info.run_id,
        )
        yield run.info.run_id
        # Asynchronous step metrics are flushed before the run is closed,
        # or the last steps of the run would be missing from its curves.
        mlflow.flush_async_logging()
