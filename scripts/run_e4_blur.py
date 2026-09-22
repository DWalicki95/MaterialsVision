#!/usr/bin/env python3
"""
The fourth augmentation family measured: sharpness.

Real acquisitions are not equally sharp. Focus, working distance and
scanning parameters vary from session to session, and the collection
carries that spread: measured on the training split, the local contrast
across the middle of a pore wall runs from eight grey levels at the
tenth percentile through eleven at the median to fifteen at the
ninetieth. A model trained only on what it happens to be given can bind
its notion of a wall to that sharpness and read a softer photograph
worse.

**What the family does.** It blurs the photograph slightly, and only
the photograph - the annotation is never blurred, because a smeared
label would mean something no annotator drew. One sample in five
receives it, which is a weaker exposure than the families measured so
far, and deliberately so: this is the one transformation here that can
destroy evidence rather than merely move it.

**Why that risk is the thing to watch.** A tonal shift cannot remove a
structure, only change its values. A blur can: it erases the thin wall
that separates two neighbouring pores, which is exactly the evidence
the model needs to keep them apart. So the figures that guard this run
are the merge count, the boundary agreement and the recall on the
smallest pores. They are guards, not goals - this family is adopted for
what it does to sharpness robustness, and those three say what it cost.

**Why the range is where it is, and not lower.** The strength is
calibrated against the quantity the blur destroys, measured where it
matters - on the model's own grid, after the panel crop and the
rescaling the encoder applies. That rescaling is itself a low-pass
filter, so a blur can be plainly visible on a review panel and not
exist for the encoder at all. At the old lower end of 0.4 the walls
kept one hundred per cent of their local contrast and the median pixel
did not move: the bottom half of that range was the identity, and a
family whose weak draws do nothing fires less often than its own
probability claims. The range now reproduces the spread the instruments
themselves produce - dropping a typical image to a soft one retains
72.7 per cent of wall contrast, which is what 1.6 achieves, and
dropping it to the lower quartile is what 0.8 achieves.

**The kernel is derived, not chosen.** Held at three pixels, a drawn
sigma of 0.8 was applied as 0.691, so a run recorded a strength the
model never saw. Sizing the kernel from the widest sigma the range can
produce, at three deviations a side, makes the sigma applied equal the
sigma drawn.

**Why this run carries orientation, and nothing else.** Orientation was
measured and kept. Scale was measured and rejected. Tonal photometry
was measured, came out inside the noise band, and was still inside it
after being decomposed into its two members, so it was not adopted
either. The condition this family is added to is therefore orientation
alone, and everything except the blur is identical to the run this one
is paired against.

**What decides it, fixed before the run.** The pooled metric over all
hundred and seven validation images, against the frozen significance
floor. No targeted slice is pre-registered: the subset this family
exists for would be the softer photographs, and sharpness is not an
axis the scoring layer divides the validation set along. Reading a
slice chosen after the fact is how a threshold gets talked down, so
none is read.

**One run, to the whole budget.** A snapshot is kept every epoch, so
the screening reading is a position on the curve rather than a place
the process has to stop. No early stopping, because a run halted early
would leave a shorter curve than the one it is compared against.

**Scoring happens afterwards.** This script only trains. What decides
is instance F1 against the annotation, computed by running each
snapshot the way the model will be used, at the frozen seeding
thresholds - not the validation loss the trainer minimizes, which can
improve while neighbouring pores merge into one. For this family that
gap is not hypothetical: merging is the failure it risks, and the loss
barely notices a merge.

Examples
--------
Train the run as specified:
    $ python scripts/run_e4_blur.py

Rehearse the whole path in minutes:
    $ python scripts/run_e4_blur.py --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e4/checkpoints/blur_seed20260907/epoch-*.pt' \\
        --center-threshold-sweep 0.25 0.30 0.35 \\
        --out checkpoints/e4/e4_curves.json

Read the comparison:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e4/e4_curves.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy, BlurConfig,
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

DEFAULT_SAVE_ROOT = Path("checkpoints/e4")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# The run this one is paired against, and the policy it carries
# forward. Three families have been measured against it so far and only
# orientation was kept, so the base is still orientation alone. Naming
# the run here keeps the counterpart identifiable from the provenance
# file rather than from somebody's memory of which comparison was meant.
BASELINE_RUN = "d4_seed20260907"

RUN_SEED = 20260907

RUN_NAME = f"blur_seed{RUN_SEED}"

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


def build_blur_policy() -> AugmentationPolicy:
    """Orientation and blur, with every other family off.

    Every parameter is taken from the frozen configuration rather than
    restated here, so that this run cannot quietly differ from what the
    review panels were drawn at and accepted. That includes the kernel,
    which the configuration derives from the top of the sigma range
    rather than accepting as a number: a kernel narrower than the widest
    draw truncates it, and the run would then record a strength the
    model never received.

    The septum family stays off even though it is the one this blur is
    suspected of interacting with - a blurred synthetic wall may stop
    being readable. Measuring the two together here would confound the
    interaction with the family, so it is left to its own experiment.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(
        orientation=OrientationConfig(),
        blur=BlurConfig(),
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
    rehearsal still crosses both microscopes and both scale bins. A
    rehearsal drawn from one corner of the collection would exercise the
    transformation over one sharpness and one pore size, which is the
    opposite of what a rehearsal is for.

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
    last = (args.save_root / "checkpoints" / RUN_NAME
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(args: argparse.Namespace) -> None:
    """Train the run.

    Validation is loaded without a policy and without shuffling: a
    blurred validation image would measure the transformation rather
    than the model, and a reordered one would change nothing except the
    comparability of the logs.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_blur_policy()
    blur = policy.config.blur

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, sigma %s px at p = %.2f, kernel %d "
        "px, %d training image(s), %d validation image(s).",
        RUN_NAME, RUN_SEED, ", ".join(policy.families),
        blur.sigma_px, blur.p, blur.kernel_px,
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Paired against %s over %d step(s), one per image; the "
        "screening reading falls at epoch %d of %d.",
        BASELINE_RUN, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    logger.info(
        "The mask is not blurred, so the annotation is checked for "
        "being untouched. What this family can damage is the thin wall "
        "between neighbouring pores, which is why the merge count and "
        "the boundary agreement are read alongside the headline "
        "metric rather than after it.",
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
    change afterwards. The kernel is recorded too, although nobody set
    it: it describes what was actually applied.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
        The file written.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / f"{RUN_NAME}_provenance.json"
    policy = build_blur_policy()
    config = policy.config
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": RUN_NAME,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(config.orientation),
        "blur": vars(config.blur),
        "blur_kernel_px": config.blur.kernel_px,
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
    """Train the blur run.

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
        "paired run, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
