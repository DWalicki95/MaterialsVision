"""
The single place a training run is configured.

This study attributes a change in segmentation quality to a change in
the augmentation policy. That inference only holds if the policy is the
only thing that differed between two runs, so everything else - the
backbone, the adaptation, the frozen components, the optimizer rate,
the shape of the loader - has one definition rather than one per script
that happens to start a run.

**What trains and what does not.** The image encoder is frozen except
for low-rank corrections on the query and value projections of every
attention block. The prompt encoder and the mask decoder are frozen
outright, which is not the library default: left alone, the library
keeps the mask decoder trainable, and at four million parameters
against the correction's eight hundred thousand it would dominate the
run and make the name "low-rank fine-tuning" misleading. The instance
decoder trains in full, and at nine and a half million parameters it is
the largest thing learning here - worth stating plainly, because "we
train a fraction of a percent of the model" is true of the low-rank
correction alone and false of the run.

**Two learning rates, not one.** The low-rank correction and the
instance decoder are in different regimes and cannot share a rate. The
correction starts at exactly zero and holds under a million
parameters, so it needs a rate well above the library's default to
move at all. The decoder is ten times larger, starts from pretrained
weights and is trained in full, which is the regime the library's
default was tuned for. Running both at the correction's rate trains
the decoder far too fast: it drives its training loss down threefold
while its validation loss turns upward within three epochs, and the
run peaks long before its budget is spent. One rate per group is the
smallest change that lets each move at its own pace.

**The optimizer is built here rather than by the library.** The
library's training entry point accepts a single rate and builds the
parameter list itself, so two rates are not reachable through it. What
it does around that is reproduced below unchanged - the same frozen
parts, the same prompt conversion, the same losses, the same trainer
and its settings - and the differences are limited to the optimizer,
the schedule, and the two deliberate departures noted at the call
site.

**The schedule has the budget as its horizon.** A rate that stays flat
leaves the model wherever the last few steps happened to put it, and
with one image per step that is a noisy place: neighbouring epochs of
the same run landed up to 0.045 apart in instance F1, more than three
times the spread between independent seeds. Annealing over the run's
budget removes most of that by ending every run at a rate near zero,
which also makes the end of a run a defensible point of comparison
rather than a sample from a moving target.

**One image per step.** Rotations by ninety degrees swap an image's
sides, and images from the two microscopes differ in height because one
of them carries an information panel that is cropped away. Batching
would therefore have to group samples by shape - the encoder's input is
padded to a square and would not care, but the decoder's targets are
built at content resolution and keep it. A batch of one removes the
question. It also matches the reference work, and the gradient of a
step already averages over roughly twenty-five instances.

**The order images arrive in is the sampler's, not the loader's.**
Asking the loader to shuffle draws the permutation from the global
random stream, whose state depends on how much randomness everything
else consumed first - and that differs between augmentation policies.
Two policies at one seed would then see different image orders, and
part of any measured difference would be the ordering rather than the
augmentation. The sampler draws from its own stream, keyed to the run
seed and the epoch alone, and carries the epoch into the index so that
augmentation varies from pass to pass even though samples are prepared
in worker processes.

**The geometry correction is not optional.** The library resizes an
image to the encoder's square canvas by reading two axes of the tensor
that hold the batch size and the channel count rather than the height
and the width. For a three-channel image in a batch of one that puts
every image at 341 by 1024 whatever its real shape, squeezing the
content vertically by a factor of two and a quarter - and a wall two
pixels across survives horizontally while vanishing vertically, so the
model would be taught a directional distortion the material does not
have. Inference does not go through that path, so the damage is a
train-time-only distortion that would never show up as an error. The
correction is installed before the model is built and then verified by
reading the geometry back out of the library, because a correction that
silently failed to install would leave no other trace.

**The loader emits content resolution and nothing else.** No resizing,
no padding, no normalization: the model performs all three inside its
own forward pass, and doing any of them earlier would either duplicate
the normalization or pad before the model scales, shrinking the working
resolution that every visibility criterion in this study was calibrated
against.
"""
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from micro_sam.models.peft_sam import LoRASurgery
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from materials_vision.data import (ProportionalImageSampler, SampleSource,
                                   load_split, read_manifest)
from materials_vision.data.dataset import (InstanceSegmentationDataset,
                                           build_label_transform)
from materials_vision.sam_geometry import (patch_resize_longest_side,
                                           verify_preprocess_geometry)

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
    split_csv: Path, manifest_csv: Path, subset: str
) -> SampleSource:
    """Open one subset of the frozen split for reading.

    Parameters
    ----------
    split_csv, manifest_csv : Path
    subset : str
        ``"train"`` or ``"val"``. The test set is deliberately not
        reachable through here.

    Returns
    -------
    SampleSource
    """
    split = load_split(split_csv, subset=subset)
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
    kinds = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    )
    changed = 0
    for name, module in unetr.named_modules():
        if name.startswith("encoder") or not isinstance(module, kinds):
            continue
        module.momentum = momentum
        changed += 1
    return changed


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
    scheduler_horizon_epochs: Optional[int] = None,
    decoder_batchnorm_momentum: Optional[float] = None,
    early_stopping: Optional[int] = None,
    save_every_kth_epoch: Optional[int] = None,
    device: Optional[str] = None,
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
    scheduler_horizon_epochs : int, optional
        Epochs the rate is annealed over. Defaults to the run's own
        length; a run meant to be stopped early and resumed later has
        to pass the full budget instead, or its rate would fall to zero
        at the point it was interrupted and the resumed half would
        train at nothing.
    decoder_batchnorm_momentum : float, optional
        Overrides how fast the decoder's batch statistics move.
        ``None`` leaves the library's value untouched.
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
    """
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

    correction, decoder = _trainable_groups(model, unetr)
    optimizer = torch.optim.AdamW([
        {"params": correction, "lr": lora_learning_rate},
        {"params": decoder, "lr": decoder_learning_rate},
    ])
    horizon = (
        n_epochs if scheduler_horizon_epochs is None
        else scheduler_horizon_epochs
    )
    instance_loss = torch_em.loss.DiceBasedDistanceLoss(
        mask_distances_in_bg=True
    )
    logger.info(
        "Training %r from %s: %d epoch(s) x %d step(s), rank %d, "
        "correction %d parameter(s) at lr %.0e, decoder %d at %.0e, "
        "annealed over %d epoch(s).",
        name, model_type, n_epochs, len(train_loader), lora_rank,
        sum(p.numel() for p in correction), lora_learning_rate,
        sum(p.numel() for p in decoder), decoder_learning_rate, horizon,
    )

    trainer = JointSamTrainer(
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
        logger=JointSamLogger,
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
    trainer.fit(**fit_kwargs)
