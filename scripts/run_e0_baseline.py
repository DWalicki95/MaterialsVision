#!/usr/bin/env python3
"""
The calibration runs every later comparison is measured against.

Everything this study reports takes the form "this augmentation
changed the result by so much". Three things have to exist before such
a sentence can be honest, and all three come from running the
unaugmented baseline.

**A shared training budget.** If one run trained longer than another,
its advantage might be the extra training rather than the treatment.
So the budget is fixed once, in optimizer steps, and every comparison
run gets exactly it. Reading it off requires seeing where the baseline
stops improving.

**A noise floor.** The same baseline is run three times, changing only
the random seed - same data, same configuration, same everything else.
The results still differ, because the initialization of the low-rank
correction, the order images arrive in, and which instances each step
samples are all random. The spread between those three runs is what
this pipeline produces when nothing real has changed. Any later
difference smaller than it is not evidence of anything. Without this
number, every "it improved" is a wish.

**A stopping rule.** How often to measure, and how many measurements
without improvement mean the run is done.

**Why a fourth run, with augmentation.** The budget applies to some
thirty-five later runs, and an augmented run converges later than an
unaugmented one - it is being shown a harder, more varied task on
purpose. A budget read off the baseline alone would cut the augmented
runs short, and would do so in one direction: it would understate
exactly the effect this study exists to measure. One run under the
richest policy shows how much later that happens. It calibrates the
budget and takes no part in attributing anything, in the same way the
base-model pilot took no part in it.

**No early stopping, and every epoch kept.** Stopping early would
leave curves of different lengths, and the point here is to see the
whole shape, including what happens past the peak. Each epoch's
snapshot is kept so that the metric this study selects models on can
be computed afterwards on all of them: the trainer's own notion of a
best checkpoint follows its validation loss, and loss is not what gets
reported.

Examples
--------
Run the calibration as specified:
    $ python scripts/run_e0_baseline.py

Rehearse the whole path in minutes:
    $ python scripts/run_e0_baseline.py --n-train 8 --n-val 4 --epochs 1

Re-run only what is missing after an interruption:
    $ python scripts/run_e0_baseline.py          # finished runs are skipped
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy, BlurConfig,
                                           MaskAwareConfig, OrientationConfig,
                                           PolicyConfig, ScaleConfig,
                                           SeptumConfig, TonalConfig)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.training import (DECODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, build_loader,
                                       build_source, train_run)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/e0")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# Three seeds for the baseline, because two cannot show a spread and
# four would cost a run for little more. The fourth run repeats the
# first seed under the full policy, so the pair differs in the policy
# alone.
BASELINE_SEEDS = (20260907, 20260908, 20260909)

CALIBRATION_SEED = BASELINE_SEEDS[0]

# Four epochs past the latest peak any arm of the learning-rate grid
# reached, which was its eighth. The margin is there because a policy
# shown a harder, more varied task converges later than an unaugmented
# one, and a budget read off the baseline alone would cut the augmented
# runs short - understating, in one known direction, exactly the effect
# this study exists to measure.
#
# Twenty was the earlier value, chosen when the curves appeared to peak
# after two epochs and decline. They did not: that shape came from
# scoring against seeding thresholds calibrated for a decoder that had
# not been fine-tuned. Against calibrated ones the curves rise for two
# to three epochs and then hold flat, so the extra eight epochs bought
# nothing but disk.
N_EPOCHS = 12

# Every epoch, giving twenty snapshots per run - the lower end of the
# twenty to forty full evaluations a run is meant to receive.
CHECKPOINT_EVERY_EPOCHS = 1


@dataclass(frozen=True)
class Run:
    """One training run of the calibration set."""

    name: str
    seed: int
    augmented: bool


def planned_runs() -> tuple[Run, ...]:
    """The four runs, in the order they should happen.

    The baseline seeds come first so that an interrupted session still
    yields the noise floor, which is the part nothing else can be
    derived without.

    Returns
    -------
    tuple of Run
    """
    baseline = tuple(
        Run(f"b0_seed{seed}", seed, augmented=False)
        for seed in BASELINE_SEEDS
    )
    return baseline + (
        Run(f"full_seed{CALIBRATION_SEED}", CALIBRATION_SEED, augmented=True),
    )


def build_full_policy() -> AugmentationPolicy:
    """Every augmentation family at once.

    The most demanding policy that will ever run, which is the one
    whose convergence the budget has to accommodate.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(
        scale=ScaleConfig(),
        orientation=OrientationConfig(),
        mask_aware=MaskAwareConfig(),
        septum=SeptumConfig(),
        tonal=TonalConfig(),
        blur=BlurConfig(),
    ))


def spaced_indices(n_available: int, n_wanted: int) -> Optional[list[int]]:
    """Evenly spaced positions, or every one of them.

    Spaced rather than taken from the front, so that a shortened
    rehearsal still crosses both microscopes and both scale bins.

    Parameters
    ----------
    n_available : int
    n_wanted : int
        Zero or less asks for all of them.

    Returns
    -------
    list of int or None
    """
    if n_wanted <= 0 or n_wanted >= n_available:
        return None
    stride = n_available / n_wanted
    return [int(position * stride) for position in range(n_wanted)]


def is_finished(run: Run, args: argparse.Namespace) -> bool:
    """Whether this run already produced its last epoch's snapshot.

    Lets an interrupted session be restarted without repeating hours
    of finished work. Coarse on purpose: a run is either complete or
    starts again from the beginning, because resuming mid-run would
    have to restore the optimizer, the scheduler and the sampler state
    to be worth anything, and a partly trained model that looks
    finished is worse than an hour lost.

    Parameters
    ----------
    run : Run
    args : argparse.Namespace

    Returns
    -------
    bool
    """
    last = (args.save_root / "checkpoints" / run.name
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(run: Run, args: argparse.Namespace) -> None:
    """Train one run of the calibration set.

    Parameters
    ----------
    run : Run
    args : argparse.Namespace
    """
    torch.manual_seed(run.seed)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_full_policy() if run.augmented else None

    train_loader = build_loader(
        train_source, policy=policy, run_seed=run.seed, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=run.seed, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, %s, %d training image(s), %d validation image(s).",
        run.name, run.seed,
        "full augmentation" if run.augmented else "no augmentation",
        len(train_loader), len(val_loader),
    )
    train_run(
        run.name,
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=LORA_LEARNING_RATE,
        decoder_learning_rate=DECODER_LEARNING_RATE,
        early_stopping=None,
        save_every_kth_epoch=CHECKPOINT_EVERY_EPOCHS,
    )


def write_provenance(args: argparse.Namespace, runs: tuple[Run, ...]) -> None:
    """Record what produced these checkpoints, next to them.

    Parameters
    ----------
    args : argparse.Namespace
    runs : tuple of Run
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / "e0_provenance.json"
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "runs": [
            {"name": run.name, "seed": run.seed, "augmented": run.augmented}
            for run in runs
        ],
        "epochs": args.epochs,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }, indent=2, default=str))
    logger.info("Wrote %s.", destination)


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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Passes over the training split, the same for every run.",
    )
    parser.add_argument(
        "--n-train", type=int, default=0,
        help="Use this many training images; 0 means all of them.",
    )
    parser.add_argument(
        "--n-val", type=int, default=0,
        help="Use this many validation images; 0 means all of them.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run runs that already finished.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train the baseline seeds and the calibration run.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    runs = planned_runs()
    write_provenance(args, runs)
    for run in runs:
        if not args.force and is_finished(run, args):
            logger.info("%s already finished; skipping.", run.name)
            continue
        execute(run, args)

    logger.info(
        "Calibration runs finished. Every epoch's snapshot is under %s; "
        "the budget, the noise floor, the patience and the cadence are "
        "read off by scoring those snapshots, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
