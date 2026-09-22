#!/usr/bin/env python3
"""
The last augmentation family measured: synthetic septa.

Foams contain pores that are almost divided - a thin wall reaching part
way across, or fully across but faint enough that the model runs two
chambers together. Merging is this segmentation's characteristic
failure: neighbouring pores share their walls, so the evidence
separating them is a few pixels wide, and losing it costs one instance
and doubles a reported pore size.

**What the family does.** It draws a thin bright line across some of
the larger pores and splits the annotation along it, so the sample
carries two instances where the photograph carried one. The model is
then shown, with a label that agrees, that a faint wall of this kind is
a boundary. The reference work applied the same idea under the name of
synthetic scratches and reported it as the single change that most
improved detection of small pores.

**This is the only family that edits the annotation, and that is why
it is checked hardest.** Every other transformation either leaves the
mask bitwise identical or moves it rigidly with the image. This one
adds instances. The integrity check therefore asserts something
different here: that each new instance is a connected region, that the
two halves together cover what the original covered, and that no
fragment smaller than the dataset's own minimum survives as an
annotation nobody drew.

**Why the contrast has a floor in grey levels.** The strength is set
as a share of each image's own tonal range, because the collection runs
from a range of 34 to one of 208 and a fixed grey offset would be
invisible in one image and garish in another. A share alone, though,
lets the faintest images receive a septum below what anyone can see -
and an invisible wall taught as a boundary is worse than no
augmentation, because the model is told to find an edge on evidence
that is not there. Hence the floor of eleven grey levels, which is the
perception threshold the review established.

**What decides it, fixed before the run.** The pooled metric against
the frozen floor, as always. This family also has the second path the
decision rule allows for a candidate inside the noise band, and it is
the family with the clearest case for it: it was adopted against
merges and against missing the smallest pores, so both are named in
advance. Merges must fall below 2.36 per hundred annotated pores and
recall in the smallest size quartile must rise above 0.6991 - in each
case past the whole spread the unaugmented runs showed, which is what
the rule demands.

**Blur stays off.** A blurred synthetic wall may stop being readable,
which would confound the family with the interaction. That interaction
is a separate experiment, and blur is not in the base anyway.

**The realized rate is part of the result.** A septum needs a pore
large enough to divide and a chord that leaves both halves above the
minimum fragment area, so the share of samples that actually receive
one is a property of the data and not only of the configuration. It is
logged, because a comparison against this family means a comparison
against the rate it actually ran at.

Examples
--------
Train the run as specified:
    $ python scripts/run_e6_septum.py

Rehearse the whole path in minutes:
    $ python scripts/run_e6_septum.py --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e6/checkpoints/septum_seed20260907/epoch-*.pt' \\
        --center-threshold-sweep 0.25 0.30 0.35 \\
        --out checkpoints/e6/e6_curves.json

Read the comparison:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e6/e6_curves.json
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
                                           SeptumConfig)
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

DEFAULT_SAVE_ROOT = Path("checkpoints/e6")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

# The run this one is paired against, and the policy it carries
# forward. Four families have been measured against it and none was
# kept: scale hurt, and tonal photometry, blur and mask-aware light all
# landed inside the noise band. The base is therefore still orientation
# alone. Naming the run here keeps the counterpart identifiable from
# the provenance file rather than from somebody's memory.
BASELINE_RUN = "d4_seed20260907"

RUN_SEED = 20260907

RUN_NAME = f"septum_seed{RUN_SEED}"

# The two errors this family was adopted against, and the spread each
# shows across the unaugmented runs. A candidate inside the noise band
# may still be adopted when one of these improves past its whole
# spread, so the figures are written here rather than recalled later:
# a target named after the fact is not a target.
BASELINE_MERGES_PER_100 = (2.36, 4.48)
BASELINE_SMALLEST_QUARTILE_RECALL = (0.6321, 0.6991)

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


def build_septum_policy() -> AugmentationPolicy:
    """Orientation and synthetic septa, with every other family off.

    The fragment threshold is taken from the training stack rather than
    left to this family's own default. How small a piece of annotation
    may be before it stops being something a person drew is one
    property of this dataset, not one property per component that
    happens to need it, and the same number already governs the panel
    crop every sample goes through. Two copies of it could drift apart
    without anything failing.

    Blur stays off deliberately. A synthetic wall that survives at
    source resolution may not survive being blurred and then rescaled
    into the encoder's grid, so running the two together would measure
    the family and the interaction at once. The interaction has its own
    experiment.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(
        orientation=OrientationConfig(),
        septum=SeptumConfig(min_fragment_area_px2=A_MIN_FRAGMENT_PX2),
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

    Spaced rather than taken from the front, and it matters more for
    this family than for any before it: a septum needs a pore large
    enough to divide, so a rehearsal drawn entirely from the finer
    scale could find nothing to act on and report a clean run in which
    the transformation never fired.

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
    validation image carrying a synthetic wall would be scored against
    an annotation this study invented, which measures the
    transformation rather than the model.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_septum_policy()
    septum = policy.config.septum

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, p = %.2f, contrast %s of the "
        "image range with a floor of %.0f grey level(s), %d training "
        "image(s), %d validation image(s).",
        RUN_NAME, RUN_SEED, ", ".join(policy.families), septum.p,
        septum.contrast, septum.min_contrast_grey,
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Paired against %s over %d step(s), one per image; the "
        "screening reading falls at epoch %d of %d.",
        BASELINE_RUN, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    logger.info(
        "This is the only family that edits the annotation, so the "
        "integrity check asserts connected new instances and no "
        "fragment below %.2f px2 rather than an untouched mask. A "
        "sample offering no pore large enough to divide retreats in a "
        "controlled way and says so, and the rate at which that "
        "happens is part of what this policy turned out to be.",
        septum.min_fragment_area_px2,
    )
    logger.info(
        "Adopted against merges (baseline spread %.2f to %.2f per 100) "
        "and against missing the smallest pores (baseline recall %.4f "
        "to %.4f in the smallest quartile). Both fixed before the run.",
        *BASELINE_MERGES_PER_100, *BASELINE_SMALLEST_QUARTILE_RECALL,
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
    change afterwards. The two targeted errors are recorded with the
    spreads they have to beat, so the acceptance rule cannot be
    reconstructed differently after the numbers are in.

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
    policy = build_septum_policy()
    config = policy.config
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": RUN_NAME,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(config.orientation),
        "septum": vars(config.septum),
        "targeted_errors": ["merges_per_100_gt", "size_bin_recall_q1"],
        "baseline_merges_per_100": list(BASELINE_MERGES_PER_100),
        "baseline_q1_recall": list(BASELINE_SMALLEST_QUARTILE_RECALL),
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
    """Train the septum run.

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
        "paired run, with merges and the smallest-quartile recall read "
        "beside the headline metric.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
