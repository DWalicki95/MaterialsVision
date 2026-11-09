#!/usr/bin/env python3
"""
Train a single PEFT-SAM (LoRA) model with configurable settings.

All training settings are command-line options: epoch count, learning
rates, trainable part (LoRA, the UNETR instance decoder or both),
whether the interactive loss trains LoRA, LoRA rank, augmentation
policy, seed and data subsets. Unset options fall back to the default
recipe - 12 epochs, cosine learning-rate schedule, ``full_b``
augmentation, joint interactive and instance loss, training on TRAIN
with validation loss on VALIDATION.

Runs are logged to a separate MLflow experiment and checkpoint
directory. A checkpoint is saved every epoch; to pick the best one,
score them with ``evaluate_checkpoint.py`` (``best.pt`` is selected
by validation loss only).

Examples
--------
The adopted recipe on TRAIN, as a reference:
    $ python scripts/peft_sam/train_peft_sam.py

Only the low-rank correction, for twenty epochs:
    $ python scripts/peft_sam/train_peft_sam.py --trainable lora \\
        --epochs 20

Only the instance decoder, at three times its adopted rate:
    $ python scripts/peft_sam/train_peft_sam.py \\
        --trainable instance_decoder \\
        --decoder-lr 3e-5

The adopted recipe with LoRA learning from the instance loss alone:
    $ python scripts/peft_sam/train_peft_sam.py --interactive-loss off

Rehearse the whole path in minutes:
    $ python scripts/peft_sam/train_peft_sam.py --epochs 1 --n-train 8 \\
        --n-monitor 2
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from materials_vision.augmentation import AugmentationPolicy
from materials_vision.data.split_io import SplitLoadError, load_split
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import REHEARSAL_EXPERIMENT, RunTracking
from materials_vision.training import (DECODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, LORA_RANK,
                                       TRAINABLE_CHOICES, build_loader,
                                       build_source,
                                       resolve_interactive_loss, train_run)
from scripts.peft_sam.run_e8_full import (BASE_MODEL, N_EPOCHS, RUN_SEED,
                                          VARIANTS, build_full_policy,
                                          spaced_indices)
from scripts.peft_sam.train_cv_fold import write_provenance
from scripts.peft_sam.train_final_model import POLICY

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_EXPERIMENT = "peft_sam_adhoc"

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/adhoc")

# No augmentation at all, the study's baseline condition.
NO_POLICY = "none"

POLICIES = (NO_POLICY, *VARIANTS)


def build_policy(policy: str) -> Optional[AugmentationPolicy]:
    """The augmentation a run trains under.

    Parameters
    ----------
    policy : str
        One of :data:`POLICIES`.

    Returns
    -------
    AugmentationPolicy or None
        ``None`` for :data:`NO_POLICY`.
    """
    return None if policy == NO_POLICY else build_full_policy(policy)


def policy_families(policy: str) -> list[str]:
    """The augmentation families a policy carries.

    Parameters
    ----------
    policy : str

    Returns
    -------
    list of str
    """
    built = build_policy(policy)
    return [] if built is None else list(built.families)


def resolve_rates(args: argparse.Namespace) -> None:
    """Fill in the adopted rate wherever none was given.

    Called once, by :func:`parse_args`, so that every later reader sees
    the rate the run actually uses. Whether a rate was given is kept,
    because a rate given for a frozen half is ignored and should be
    said to be.

    Parameters
    ----------
    args : argparse.Namespace
        Updated in place.
    """
    args.lora_lr_given = args.lora_lr is not None
    args.decoder_lr_given = args.decoder_lr is not None
    if args.lora_lr is None:
        args.lora_lr = LORA_LEARNING_RATE
    if args.decoder_lr is None:
        args.decoder_lr = DECODER_LEARNING_RATE


def trains_lora(trainable: str) -> bool:
    """Whether the low-rank correction is adapted.

    Parameters
    ----------
    trainable : str

    Returns
    -------
    bool
    """
    return trainable in ("both", "lora")


def trains_instance_decoder(trainable: str) -> bool:
    """Whether the instance decoder is adapted.

    Parameters
    ----------
    trainable : str

    Returns
    -------
    bool
    """
    return trainable in ("both", "instance_decoder")


def interactive_loss_of(args: argparse.Namespace) -> bool:
    """Whether the interactive loss trains the low-rank correction.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If it was asked for while the correction is frozen;
        :func:`refusal` reports that before a run starts.
    """
    requested = (
        None if args.interactive_loss is None
        else args.interactive_loss == "on"
    )
    return resolve_interactive_loss(args.trainable, requested)


def horizon_of(args: argparse.Namespace) -> int:
    """Epochs the rate is annealed over.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    int
    """
    return (
        args.epochs if args.scheduler_horizon is None
        else args.scheduler_horizon
    )


def run_name(args: argparse.Namespace) -> str:
    """What the run's checkpoints are filed under.

    Built from every setting that can differ between two runs of this
    script, so that two runs set up differently cannot write into one
    directory. The rate of a frozen half is left out: it does not
    change what the run is.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str
    """
    if args.name is not None:
        return args.name
    parts = [args.policy, args.trainable]
    if trains_lora(args.trainable) and not interactive_loss_of(args):
        parts.append("nointeractive")
    parts.append(f"e{args.epochs}")
    if args.scheduler_horizon is not None:
        parts.append(f"h{args.scheduler_horizon}")
    if trains_lora(args.trainable):
        parts.append(f"lora{args.lora_lr:g}")
    if trains_instance_decoder(args.trainable):
        parts.append(f"dec{args.decoder_lr:g}")
    if args.lora_rank != LORA_RANK:
        parts.append(f"r{args.lora_rank}")
    parts.append(args.train_subset.replace("+", ""))
    if args.n_train > 0:
        parts.append(f"n{args.n_train}")
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def snapshot_paths(args: argparse.Namespace) -> list[Path]:
    """Every numbered snapshot the run is expected to leave.

    The trainer numbers a snapshot whenever the epoch divides by the
    interval it is given.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    list of Path
    """
    directory = args.save_root / "checkpoints" / run_name(args)
    return [
        directory / f"epoch-{k}.pt"
        for k in range(args.save_every, args.epochs + 1, args.save_every)
    ]


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why the run may not be trained as asked, or nothing.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    if args.epochs < 1:
        return f"A run needs at least one epoch, got {args.epochs}."
    if horizon_of(args) < args.epochs:
        # The cosine schedule is periodic: past its horizon the rate
        # climbs back up instead of staying at zero.
        return (
            f"The schedule horizon ({horizon_of(args)}) is shorter than "
            f"the run ({args.epochs}); the rate would rise again after "
            f"reaching zero."
        )
    if not 1 <= args.save_every <= args.epochs:
        return (
            f"--save-every must lie between 1 and the number of epochs "
            f"({args.epochs}), got {args.save_every}."
        )
    if args.interactive_loss == "on" and not trains_lora(args.trainable):
        return (
            f"--interactive-loss on cannot train anything under "
            f"--trainable {args.trainable}: the correction is frozen."
        )
    if trains_lora(args.trainable) and args.lora_lr <= 0:
        return f"--lora-lr must be positive, got {args.lora_lr:g}."
    if trains_instance_decoder(args.trainable) and args.decoder_lr <= 0:
        return f"--decoder-lr must be positive, got {args.decoder_lr:g}."
    if args.lora_rank < 1:
        return f"--lora-rank must be positive, got {args.lora_rank}."
    if args.n_train < 0 or args.n_monitor < 0:
        return "--n-train and --n-monitor cannot be negative."
    for subset in (args.train_subset, args.monitor_subset):
        if "test" in subset.split("+"):
            return (
                "TEST is not trained or monitored on; it is scored once, "
                "by evaluate_checkpoint.py."
            )
    for path in (args.split, args.manifest):
        if not path.exists():
            return f"{path} does not exist."
    return None


def ignored_rates(args: argparse.Namespace) -> list[str]:
    """The rate options given for a half that does not train.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    list of str
    """
    ignored = []
    if args.lora_lr_given and not trains_lora(args.trainable):
        ignored.append("--lora-lr")
    if (args.decoder_lr_given
            and not trains_instance_decoder(args.trainable)):
        ignored.append("--decoder-lr")
    return ignored


def provenance_record(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that produced the run, for the file beside it.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
    """
    return {
        "provenance": run_provenance(),
        "run": run_name(args),
        "base_model": BASE_MODEL,
        "trainable": args.trainable,
        "interactive_loss": interactive_loss_of(args),
        "epochs": args.epochs,
        "scheduler_horizon_epochs": horizon_of(args),
        "lora_learning_rate": (
            args.lora_lr if trains_lora(args.trainable) else None
        ),
        "decoder_learning_rate": (
            args.decoder_lr if trains_instance_decoder(args.trainable)
            else None
        ),
        "lora_rank": args.lora_rank,
        "policy": args.policy,
        "families": policy_families(args.policy),
        "seed": args.seed,
        "train_subset": args.train_subset,
        "monitor_subset": args.monitor_subset,
        "n_train": args.n_train,
        "n_monitor": args.n_monitor,
        "save_every_kth_epoch": args.save_every,
        "early_stopping": None,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }


def build_tracking(
    args: argparse.Namespace, provenance_path: Path
) -> RunTracking:
    """Describe how the run is recorded in MLflow.

    A run on a subset of the training images is a rehearsal of the
    path and goes to the rehearsal experiment, as everywhere else.

    Parameters
    ----------
    args : argparse.Namespace
    provenance_path : Path

    Returns
    -------
    RunTracking
    """
    return RunTracking(
        experiment=(
            REHEARSAL_EXPERIMENT if args.n_train > 0 else args.experiment
        ),
        tags={
            "study": "adhoc",
            "trainable": args.trainable,
            "interactive_loss": "on" if interactive_loss_of(args) else "off",
            "epochs": str(args.epochs),
            "policy": args.policy.upper(),
            "seed": str(args.seed),
            "train_subset": args.train_subset,
            "source": "live",
        },
        params={
            "seed": args.seed,
            "policy": args.policy.upper(),
            "augmentation_families": ",".join(policy_families(args.policy)),
            "save_every_kth_epoch": args.save_every,
        },
        artifacts=(provenance_path,),
    )


def log_monitoring(
    args: argparse.Namespace, n_monitor_images: int
) -> None:
    """Say whether the trainer's validation loss is held out.

    Parameters
    ----------
    args : argparse.Namespace
    n_monitor_images : int
    """
    trained = set(args.train_subset.split("+"))
    if set(args.monitor_subset.split("+")) <= trained:
        logger.warning(
            "The %d monitoring image(s) from %s are also trained on; the "
            "validation loss reported on them measures memorization.",
            n_monitor_images, args.monitor_subset.upper(),
        )
    else:
        logger.info(
            "The %d monitoring image(s) come from %s, which is not "
            "trained on.", n_monitor_images, args.monitor_subset.upper(),
        )


def execute(args: argparse.Namespace) -> None:
    """Record the run and train it.

    The seed is set before anything else is built, because the initial
    state of the low-rank correction is drawn from the global stream.

    Parameters
    ----------
    args : argparse.Namespace
    """
    provenance_path = write_provenance(provenance_record(args), args.save_root)
    name = run_name(args)

    torch.manual_seed(args.seed)
    train_source = build_source(args.split, args.manifest, args.train_subset)
    monitor_source = build_source(
        args.split, args.manifest, args.monitor_subset
    )
    policy = build_policy(args.policy)
    train_loader = build_loader(
        train_source, policy=policy, run_seed=args.seed, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    monitor_loader = build_loader(
        monitor_source, policy=None, run_seed=args.seed, shuffle=False,
        indices=spaced_indices(len(monitor_source), args.n_monitor),
    )
    logger.info(
        "%s: %s trains, interactive loss %s, policy %s, seed %d, %d "
        "training image(s) from %s of %s.", name, args.trainable,
        "on" if interactive_loss_of(args) else "off", args.policy, args.seed,
        len(train_loader), args.train_subset.upper(), args.split.name,
    )
    log_monitoring(args, len(monitor_loader))

    train_run(
        name,
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=monitor_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=args.lora_lr,
        decoder_learning_rate=args.decoder_lr,
        lora_rank=args.lora_rank,
        trainable=args.trainable,
        interactive_loss=interactive_loss_of(args),
        scheduler_horizon_epochs=args.scheduler_horizon,
        early_stopping=None,
        save_every_kth_epoch=args.save_every,
        tracking=build_tracking(args, provenance_path),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help=f"Passes over the training images. Default {N_EPOCHS}.",
    )
    parser.add_argument(
        "--trainable", choices=TRAINABLE_CHOICES, default="both",
        help="What is adapted: the low-rank correction, the UNETR "
             "instance decoder, or both. SAM's prompt encoder and mask "
             "decoder stay frozen in every case. Default both.",
    )
    parser.add_argument(
        "--interactive-loss", choices=("on", "off"), default=None,
        help="Whether the interactive loss, back-propagated through "
             "SAM's frozen mask decoder, trains the low-rank correction "
             "alongside the instance loss. Default: on wherever the "
             "correction trains, as in the library's joint objective. "
             "With it off, best.pt no longer reflects the run; score "
             "the numbered snapshots instead.",
    )
    parser.add_argument(
        "--lora-lr", type=float, default=None,
        help=f"Rate of the low-rank correction. Default "
             f"{LORA_LEARNING_RATE:g}, the adopted one.",
    )
    parser.add_argument(
        "--decoder-lr", type=float, default=None,
        help=f"Rate of the instance decoder. Default "
             f"{DECODER_LEARNING_RATE:g}, the adopted one.",
    )
    parser.add_argument(
        "--scheduler-horizon", type=int, default=None,
        help="Epochs the cosine schedule anneals over. Default: the "
             "run's own length, so the rate ends near zero.",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=LORA_RANK,
        help=f"Rank of the low-rank correction. Default {LORA_RANK}.",
    )
    parser.add_argument(
        "--policy", choices=POLICIES, default=POLICY,
        help=f"Augmentation policy; '{NO_POLICY}' trains unaugmented. "
             f"Default {POLICY}, the adopted one.",
    )
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    parser.add_argument(
        "--train-subset", default="train",
        help="Subset of the split trained on, e.g. 'train' or "
             "'train+val'. Default train.",
    )
    parser.add_argument(
        "--monitor-subset", default="val",
        help="Subset the trainer's validation loss is computed on. "
             "Default val.",
    )
    parser.add_argument(
        "--n-train", type=int, default=0,
        help="Use this many training images; 0 means all of them. Any "
             "other value makes the run a rehearsal.",
    )
    parser.add_argument(
        "--n-monitor", type=int, default=0,
        help="Use this many monitoring images; 0 means all of them.",
    )
    parser.add_argument(
        "--save-every", type=int, default=1,
        help="Keep a numbered snapshot every this many epochs. Default "
             "1, so every epoch can be scored on validation.",
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment. Default {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Checkpoint directory name. Default: built from the "
             "settings.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if every snapshot already exists.",
    )
    args = parser.parse_args(argv)
    resolve_rates(args)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Train one run.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    reason = refusal(args)
    if reason is not None:
        logger.error(reason)
        return EXIT_FAILED
    for option in ignored_rates(args):
        logger.warning(
            "%s is ignored: that half is frozen under --trainable %s.",
            option, args.trainable,
        )
    try:
        # Checked before the GPU is touched: a split drawn from another
        # manifest would train on images it does not describe.
        load_split(
            args.split, args.train_subset.split("+")[0],
            verify_manifest=args.manifest,
        )
    except SplitLoadError as e:
        logger.error("%s", e)
        return EXIT_FAILED
    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    snapshots = snapshot_paths(args)
    if not args.force and all(path.exists() for path in snapshots):
        logger.info(
            "All %d snapshot(s) of %s already exist; nothing to do.",
            len(snapshots), run_name(args),
        )
        return EXIT_OK
    execute(args)

    missing = [path for path in snapshots if not path.exists()]
    if missing:
        logger.error(
            "Training finished but %d snapshot(s) were not written, "
            "first %s.", len(missing), missing[0],
        )
        return EXIT_FAILED
    logger.info(
        "Run finished: %d snapshot(s) in %s.",
        len(snapshots), snapshots[0].parent,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
