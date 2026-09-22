#!/usr/bin/env python3
"""
The first augmentation family measured on its own: orientation.

Everything this study reports is a sentence of the form "this family of
transformations changed the result by so much". The unaugmented
baseline has been trained three times and fixed the two numbers such a
sentence needs: the training budget every comparison run receives, and
the spread three identical runs produce when nothing real has changed.
This run is the first to put a treatment against that baseline.

**Why orientation goes first.** A foam has no preferred direction, and
which way a sample happened to lie under the microscope is an accident
of mounting. A model can nonetheless learn that accident. Showing it
every quarter turn and every mirror of an image removes the shortcut,
and costs nothing in image quality: none of the eight elements
resamples a single pixel, so no interpolation blur enters that would
later have to be told apart from the effect being measured. It is the
cheapest, least invasive family, and the ones measured after it are
measured on top of whatever this run settles.

**One run, to the whole budget.** A screening budget is six tenths of
the full one, and a candidate that looks good there is promoted to the
rest. With a snapshot kept every epoch, that six tenths is a position
to read the curve at, not a place the process has to stop: the run goes
to the end and both readings come out of the same series of snapshots.
This costs the training time a promoted candidate would have cost
anyway and returns the full-budget comparison in the same run. What it
gives up is the saving on a candidate that turns out to be harmful,
which would have been stopped at six tenths. That is a poor trade for a
family expected to be harmful, and a good one here.

**The seed is the baseline's.** The comparison is paired: the same
seed, the same split, the same image order, the same number of
optimizer steps, the same evaluation schedule, and exactly one
difference - the orientation family. Image order is drawn from a stream
no augmentation can disturb, so the pairing holds even though one side
draws random numbers the other does not.

**No early stopping.** A run that halted early would leave a shorter
curve than the one it is compared against, and a comparison at equal
steps needs equal steps. The baseline ran without it for the same
reason.

**Scoring happens afterwards.** This script only trains. What decides
is instance F1 against the annotation, computed by running each
snapshot the way the model will be used, at the frozen seeding
thresholds - not the validation loss the trainer minimizes, which can
improve while neighbouring pores merge into one.

Examples
--------
Train the run as specified:
    $ python scripts/run_e1_orientation.py

Rehearse the whole path in minutes:
    $ python scripts/run_e1_orientation.py --n-train 8 --n-val 4 \\
        --epochs 1

Score the snapshots afterwards, against the paired baseline:
    $ python scripts/evaluate_checkpoint.py \\
        --checkpoint 'checkpoints/e1/checkpoints/d4_seed20260907/epoch-*.pt'
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy,
                                           OrientationConfig, PolicyConfig)
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

DEFAULT_SAVE_ROOT = Path("checkpoints/e1")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# The baseline run this one is paired against. Sharing the seed is what
# makes the pair differ in the policy alone; naming the run here keeps
# the counterpart identifiable from the provenance file rather than from
# somebody's memory of which baseline was meant.
BASELINE_RUN = "b0_seed20260907"

RUN_SEED = 20260907

RUN_NAME = f"d4_seed{RUN_SEED}"

# The budget every comparison run receives, read off the baseline: the
# curves flatten between the third and sixth epoch, and the run under
# every family at once - which converges later, being shown a harder
# task on purpose - flattens by the eighth and peaks in the tenth.
N_EPOCHS = 12

# Every epoch, so the metric that decides can be computed afterwards on
# the whole series rather than on the two snapshots the trainer keeps by
# its own criterion.
CHECKPOINT_EVERY_EPOCHS = 1

# Share of the budget a screening comparison is read at.
SCREENING_FRACTION = 0.6


def build_orientation_policy() -> AugmentationPolicy:
    """The orientation family alone, with every other family off.

    Isolation is the whole point of the run: a policy carrying a second
    family would attribute that family's effect to this one.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(orientation=OrientationConfig()))


def screening_epoch(
    n_epochs: int, fraction: float = SCREENING_FRACTION
) -> int:
    """Last epoch lying within the screening share of the budget.

    A screening comparison is made on a fraction of the full budget.
    Since a snapshot is kept once per epoch, the comparison can only be
    read where a snapshot exists, so the fraction is rounded down to the
    epoch below it. The rounding shortens the reading slightly, and by
    the same amount on both sides of the comparison, which is what keeps
    it paired.

    Parameters
    ----------
    n_epochs : int
        Length of the full budget.
    fraction : float, optional
        Share of the budget the screening reading is taken at.

    Returns
    -------
    int
        Epoch number, one-based, as the snapshot files are numbered.
        At least one, so a rehearsal short enough to round down to zero
        still names a snapshot that exists.
    """
    return max(1, int(n_epochs * fraction))


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
        ``None`` means no restriction.
    """
    if n_wanted <= 0 or n_wanted >= n_available:
        return None
    stride = n_available / n_wanted
    return [int(position * stride) for position in range(n_wanted)]


def is_finished(args: argparse.Namespace) -> bool:
    """Whether the run already produced its last epoch's snapshot.

    Coarse on purpose: the run is either complete or starts again from
    the beginning. Resuming halfway would have to restore the optimizer,
    the schedule and the sampler's position to be worth anything, and a
    partly trained model that looks finished is worse than a lost hour.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    bool
    """
    last = (args.save_root / "checkpoints" / RUN_NAME
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(args: argparse.Namespace) -> None:
    """Train the run.

    Validation is loaded without a policy and without shuffling: a
    transformed validation image would measure the transformation
    rather than the model, and a reordered one would change nothing
    except the comparability of the logs.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_orientation_policy()

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, %d training image(s), "
        "%d validation image(s).",
        RUN_NAME, RUN_SEED, ", ".join(policy.families),
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Paired against %s over %d step(s), one per image; the "
        "screening reading falls at epoch %d of %d.",
        BASELINE_RUN, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    train_run(
        RUN_NAME,
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


def write_provenance(args: argparse.Namespace) -> Path:
    """Record what produced these checkpoints, next to them.

    The policy is written out family by family rather than as a name,
    so that the file says what ran even if the defaults behind the name
    change afterwards.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
        The file written.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / "e1_provenance.json"
    policy = build_orientation_policy()
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": RUN_NAME,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(OrientationConfig()),
        "epochs": args.epochs,
        "screening_epoch": screening_epoch(args.epochs),
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return destination


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
        help="Passes over the training split; the baseline's budget.",
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
        help="Re-run even if the run already finished.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train the orientation run.

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

    write_provenance(args)
    if not args.force and is_finished(args):
        logger.info("%s already finished; nothing to do.", RUN_NAME)
        return EXIT_OK
    execute(args)

    logger.info(
        "Run finished. Every epoch's snapshot is under %s; the effect "
        "of the family is read by scoring those snapshots against the "
        "paired baseline at equal steps, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
