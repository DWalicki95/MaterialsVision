#!/usr/bin/env python3
"""
The third augmentation family measured: tonal response.

The photographs come off two microscopes whose detectors and
acquisition settings differ, so the same foam is recorded with a
different mapping from physical brightness to grey level. A model
trained on that mixture can tie its notion of a pore to one of those
mappings, and will then read images from the other instrument worse.

**What the family does.** Half the training samples receive one of two
operations on the grey scale, drawn one at a time. Brightness and
contrast shift the scale linearly, making the whole photograph lighter
or darker and its range wider or narrower. Gamma bends the scale
instead, moving the mid and dark tones more than the highlights, which
is the shape a detector's response actually differs by. Neither one
touches geometry, so the annotation is not merely preserved but
provably untouched, and the integrity check asserts exactly that.

**Why one container rather than two runs.** Both members answer the
same weakness, so they are measured together first. Splitting them
apart costs two training runs to answer a question that one run can
answer, and is worth spending only if the joint result comes out
negative or inconclusive.

**When it does, ``--member`` runs one of them on its own.** The
container keeps the probability it was frozen at and draws from one
alternative instead of two, so each member is measured as a candidate
family in its own right. The alternative reading - halving the
probability so that a member fires as often as it did inside the joint
container, on about a quarter of the samples - would ask how the joint
effect divides between them. That is a different question, and not the
one a decomposition is run to answer: what a positive result here buys
is a family admitted to the base, which it can only be if it earns its
place at the probability the family is defined with.

**Why the ranges are where they are.** They were set by what a
reviewer can actually see. A tonal shift under about eleven grey
levels went unrecognized on review panels, so a range whose weak draws
fall below that is a family that fires less often than its own
probability states. The ends of the ranges measure twenty-six grey
levels for brightness and contrast and twenty-one for gamma, and a
floor keeps a draw from landing near the identity in the middle of a
symmetric range. Where the identity belongs is in the container's
probability, not hidden inside its range.

**Why this run carries orientation as well.** Orientation was measured
and kept, so it is part of what the study now calls its baseline. The
scale family was measured and rejected, so it is not. A run under
tonal photometry alone would differ from its counterpart in two
families rather than one, and the difference could not be attributed
to either. Everything except the tonal family is therefore identical
to the run this one is paired against.

**Where the answer will be read.** The pooled metric over all
hundred and seven validation images decides the status, at the frozen
significance floor. Beside it, and fixed before this run started, the
two instruments are read separately: the first covers eighty-nine of
the validation images and the second eighteen. In this split the
instrument cannot be told apart from the material - every image from
the second microscope is a K or a VAB formulation, every image from
the first is an AS - so a difference confined to one of them is
evidence about that subset and not about detectors, and has to be
reported that way.

**One run, to the whole budget.** A snapshot is kept every epoch, so
the screening reading is a position on the curve rather than a place
the process has to stop. No early stopping, because a run halted early
would leave a shorter curve than the one it is compared against.

**Scoring happens afterwards.** This script only trains. What decides
is instance F1 against the annotation, computed by running each
snapshot the way the model will be used, at the frozen seeding
thresholds - not the validation loss the trainer minimizes, which can
improve while neighbouring pores merge into one.

Examples
--------
Train the run as specified:
    $ python scripts/run_e3_tonal.py

Rehearse the whole path in minutes:
    $ python scripts/run_e3_tonal.py --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e3/checkpoints/tonal_seed20260907/epoch-*.pt' \\
        --center-threshold-sweep 0.25 0.30 0.35 \\
        --out checkpoints/e3/e3_curves.json

Read the comparison, pooled and on each instrument:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e3/e3_curves.json \\
        --slice material=AS --slice material=K+VAB

Decompose an inconclusive joint result into its two members:
    $ python scripts/run_e3_tonal.py --member brightness_contrast
    $ python scripts/run_e3_tonal.py --member gamma
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy,
                                           OrientationConfig, PolicyConfig,
                                           TonalConfig)
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

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# The run this one is paired against, and the policy it carries
# forward. Orientation was measured against the unaugmented baseline
# and kept; scale was measured against orientation and rejected, so the
# condition tonal photometry is being added to is orientation alone.
# Naming the run here keeps the counterpart identifiable from the
# provenance file rather than from somebody's memory of which
# comparison was meant.
BASELINE_RUN = "d4_seed20260907"

RUN_SEED = 20260907

# Which alternatives the tonal container may draw, and the short name
# each variant is recorded under. ``both`` is the family as frozen;
# the other two exist for the decomposition, and are worth training
# only once the joint run has come out negative or inconclusive.
VARIANTS = {
    "both": (None, "tonal", "e3"),
    "brightness_contrast": (("brightness_contrast",), "tonal_bc", "e3_bc"),
    "gamma": (("gamma",), "tonal_gamma", "e3_gamma"),
}


def run_name(member: str) -> str:
    """What this variant's checkpoints are filed under.

    Parameters
    ----------
    member : str
        A key of :data:`VARIANTS`.

    Returns
    -------
    str
    """
    return f"{VARIANTS[member][1]}_seed{RUN_SEED}"


def default_save_root(member: str) -> Path:
    """Where this variant's checkpoints and artefacts go.

    One directory per variant rather than one shared by all three. The
    scoring step globs a directory for a run's epochs, and three runs
    of twelve snapshots under one root would have to be told apart by
    filename alone - which works right up until a glob is written
    slightly too wide and a comparison silently averages two policies.

    Parameters
    ----------
    member : str
        A key of :data:`VARIANTS`.

    Returns
    -------
    Path
    """
    return Path("checkpoints") / VARIANTS[member][2]


# The budget every comparison run receives, read off the unaugmented
# baseline and unchanged since: long enough that a policy converging
# later than the baseline still reaches its own peak inside it.
N_EPOCHS = 12

# Every epoch, so the metric that decides can be computed afterwards on
# the whole series rather than on the two snapshots the trainer keeps
# by its own criterion. It is also what makes a screening reading
# possible without stopping the run.
CHECKPOINT_EVERY_EPOCHS = 1

# Share of the budget a screening comparison is read at.
SCREENING_FRACTION = 0.6


def build_tonal_policy(member: str = "both") -> AugmentationPolicy:
    """Orientation and tonal photometry, with every other family off.

    Every parameter other than the membership is taken from the frozen
    configuration rather than restated here, so that no run can quietly
    differ from what the review panels were drawn at and accepted. That
    includes the container's probability, which a decomposition leaves
    alone: the member is being asked whether it earns a place in the
    base as a family, and a family is defined at the probability it was
    frozen with.

    Parameters
    ----------
    member : str, optional
        A key of :data:`VARIANTS`. ``both`` is the container as frozen,
        and is what the joint run is measured under.

    Returns
    -------
    AugmentationPolicy
    """
    members = VARIANTS[member][0]
    tonal = TonalConfig() if members is None else TonalConfig(members=members)
    return AugmentationPolicy(PolicyConfig(
        orientation=OrientationConfig(),
        tonal=tonal,
    ))


def screening_epoch(
    n_epochs: int, fraction: float = SCREENING_FRACTION
) -> int:
    """Last epoch lying within the screening share of the budget.

    A screening comparison is made on a fraction of the full budget.
    Since a snapshot is kept once per epoch, the comparison can only be
    read where a snapshot exists, so the fraction is rounded down to
    the epoch below it. The rounding shortens the reading slightly, and
    by the same amount on both sides of the comparison, which is what
    keeps it paired.

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
    rehearsal still crosses both microscopes. That is what matters for
    this family in particular: the two instruments are the reason it
    exists, and a rehearsal drawn entirely from one of them would
    exercise the transformation over a single tonal response and show
    nothing about the mixture the run is meant to span.

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
    the beginning. Resuming halfway would have to restore the
    optimizer, the schedule and the sampler's position to be worth
    anything, and a partly trained model that looks finished is worse
    than a lost hour.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    bool
    """
    last = (args.save_root / "checkpoints" / run_name(args.member)
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(args: argparse.Namespace) -> None:
    """Train the run.

    Validation is loaded without a policy and without shuffling: a
    validation image with its tones shifted would measure the
    transformation rather than the model, and a reordered one would
    change nothing except the comparability of the logs.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_tonal_policy(args.member)
    name = run_name(args.member)

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, tonal members %s, %d training "
        "image(s), %d validation image(s).",
        name, RUN_SEED, ", ".join(policy.families),
        ", ".join(policy.config.tonal.members),
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Paired against %s over %d step(s), one per image; the "
        "screening reading falls at epoch %d of %d.",
        BASELINE_RUN, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    logger.info(
        "Neither member of this family touches geometry, so the "
        "annotation is checked for being untouched rather than for "
        "being consistent, and a sample whose mask moved at all stops "
        "the run instead of being counted as a controlled retreat.",
    )
    train_run(
        name,
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
    name = run_name(args.member)
    destination = args.save_root / f"{name}_provenance.json"
    policy = build_tonal_policy(args.member)
    config = policy.config
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": name,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(config.orientation),
        "tonal": vars(config.tonal),
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
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--member", choices=sorted(VARIANTS), default="both",
        help="Which alternatives the tonal container may draw. The "
             "default is the family as frozen; naming one member is "
             "the decomposition, and is worth training only once the "
             "joint run has come out negative or inconclusive.",
    )
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
    args = parser.parse_args(argv)
    if args.save_root is None:
        args.save_root = default_save_root(args.member)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Train the tonal run.

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
        logger.info(
            "%s already finished; nothing to do.", run_name(args.member)
        )
        return EXIT_OK
    execute(args)

    logger.info(
        "Run finished. Every epoch's snapshot is under %s; the effect "
        "of the family is read by scoring those snapshots and "
        "comparing against the paired run, pooled and on each "
        "instrument, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
