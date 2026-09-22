#!/usr/bin/env python3
"""
The second augmentation family measured: scale.

The photographs this study learns from were not all taken at the same
magnification, and they did not all come off the same microscope. A
pore of a given physical size therefore arrives as a different number
of pixels depending on which image it is in. A model trained on that
mixture can still end up bound to whichever scale dominates it, and
will then read the rarer one worse - which is what the pooled metric
hides, because the rarer scale is a small part of the validation set.

**What the family does.** It cuts a window smaller than the image and
magnifies it back to the image's own dimensions, so that the same
microstructure arrives with its pores covering more pixels than they
did. The window keeps the image's proportions, so nothing is stretched
and no shape statistic is disturbed. Only magnification is allowed,
never reduction, and only on images that were photographed at the
coarser of the two scales present: magnifying those approaches how the
same foam looks on the other instrument, whereas magnifying an image
that is already the finest in the collection would manufacture detail
that nothing photographed.

**Why the ceiling is where it is.** The largest factor drawn is the
ratio between the coarse scale and the finest non-outlier scale
actually measured in the collection. Past that the sample would claim
a resolution no photograph in the study has, and the model would be
learning to expect detail it will never be shown at inference.

**Why this run carries orientation as well.** The first family was
measured and kept, so it is part of what the study now calls its
baseline. A run under scale alone would differ from its counterpart in
two families rather than one, and the difference could not be
attributed. Everything except the scale family is therefore identical
to the run this one is paired against.

**Where the answer will be read.** Thirteen of the validation images
are at the finer scale. They are the images this family exists for,
and the pooled metric over all hundred and seven is dominated by the
rest, so it is expected to be nearly silent whatever happens. The
comparison that decides is therefore taken on each scale bin
separately, and a result that improves the finer one while leaving the
coarser one alone is the shape of a successful outcome here.

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
    $ python scripts/run_e2_scale.py

Rehearse the whole path in minutes:
    $ python scripts/run_e2_scale.py --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e2/checkpoints/scale_seed20260907/epoch-*.pt'

Read the comparison on the bins the family exists for:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e2/e2_curves.json \\
        --slice scale_bin=fine --slice scale_bin=coarse
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
                                           ScaleConfig)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.training import (A_MIN_FRAGMENT_PX2,
                                       DECODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, build_loader,
                                       build_source, train_run)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/e2")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# The run this one is paired against, and the policy it carries
# forward. Orientation was measured against the unaugmented baseline
# and kept, so it is the condition scale is being added to rather than
# a second variable. Naming the run here keeps the counterpart
# identifiable from the provenance file rather than from somebody's
# memory of which comparison was meant.
BASELINE_RUN = "d4_seed20260907"

RUN_SEED = 20260907

RUN_NAME = f"scale_seed{RUN_SEED}"

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


def build_scale_policy() -> AugmentationPolicy:
    """Orientation and scale, with every other family off.

    The fragment threshold is taken from the training stack rather
    than left to this family's own default. Magnifying a window cuts
    the instances the window's edge crosses, and how small a surviving
    piece may be before it stops being an annotation somebody made is
    one property of this dataset, not one property per component that
    happens to need it. The same number already governs the panel crop
    every sample goes through, and the two must not be able to drift.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(
        orientation=OrientationConfig(),
        scale=ScaleConfig(min_fragment_area_px2=A_MIN_FRAGMENT_PX2),
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
    rehearsal still crosses both microscopes and both scale bins. For
    this run that matters more than it did for the last one: a
    rehearsal drawn entirely from the coarse bin would exercise the
    magnifying path only, and never the branch that pins images at the
    finer scale to their original size.

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
    magnified validation image would measure the transformation rather
    than the model, and a reordered one would change nothing except the
    comparability of the logs.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_scale_policy()

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
    logger.info(
        "A sample that cannot find a window holding enough instances "
        "falls back to its original framing and says so at INFO. How "
        "often that happens is part of what this policy turned out to "
        "be, so the rate belongs in the run's log alongside its score.",
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
    destination = args.save_root / "e2_provenance.json"
    policy = build_scale_policy()
    config = policy.config
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": RUN_NAME,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(config.orientation),
        "scale": vars(config.scale),
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
    """Train the scale run.

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
        "of the family is read by scoring those snapshots and "
        "comparing each scale bin against the paired run, not from the "
        "losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
