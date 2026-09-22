#!/usr/bin/env python3
"""
The fifth augmentation family measured: local light inside pores.

This is the first family in the sequence aimed at a named failure
rather than at general robustness. The decoder decides where one pore
ends and the next begins, and it can be fooled: a slow change of
brightness across a pore interior, or a shadow lying in part of one,
looks like the edge of something new. The model then cuts a single pore
into two. The reference work reported exactly this degradation on
images with strong depth variation and shadowed crevices.

**What the family does.** It reads the annotation and uses it to put
light where a pore actually is, drawing one of two alternatives. The
first lays a low-frequency brightness field over the interiors of some
of the pores - flat, a linear gradient, or a smooth random field on a
small grid. The second darkens a patch inside a pore, as a shadow
would. Either way the model is shown interiors that are not uniformly
lit and told, by an annotation that has not changed, that each is still
one pore.

**The annotation is read, never written.** "Mask-aware" means the
transformation consults the labels to find the interiors; it does not
alter them, and the integrity check asserts they come out untouched.

**Why the field must fade to nothing at the boundary.** The obvious
implementation - add a constant to every pixel inside the mask - puts a
step in brightness exactly along the pore edge. That manufactures the
very cue the family exists to teach the model to disregard, and would
train it to find boundaries that the annotation does not mark. So the
strength is zero at the edge and rises inwards over the first third of
the pore's depth before levelling off, which is a change the interior
carries and the boundary does not.

**What decides it.** The pooled metric over all hundred and seven
validation images, against the frozen significance floor, as for every
family. But this family has a second path that the two before it did
not: the decision rule admits a candidate whose headline result sits
inside the noise band when the error it was adopted against improves
unambiguously, beyond the spread the unaugmented runs show. Here that
error is the false split, and the spread to beat is the baseline range
of 1.60 to 2.97 splits per hundred annotated pores. Both readings are
fixed before the run, because a target named afterwards is not a target.

**One run, to the whole budget.** A snapshot is kept every epoch, so
the screening reading is a position on the curve rather than a place
the process has to stop. No early stopping, because a run halted early
would leave a shorter curve than the one it is compared against.

**Scoring happens afterwards.** This script only trains. What decides
is instance F1 against the annotation, and the split count beside it,
computed by running each snapshot the way the model will be used at the
frozen seeding thresholds - not the validation loss, which is nearly
blind to whether one pore was reported as two.

Examples
--------
Train the run as specified:
    $ python scripts/run_e5_mask_aware.py

Rehearse the whole path in minutes:
    $ python scripts/run_e5_mask_aware.py --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e5/checkpoints/mask_aware_seed20260907/epoch-*.pt' \\
        --center-threshold-sweep 0.25 0.30 0.35 \\
        --out checkpoints/e5/e5_curves.json

Read the comparison:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e5/e5_curves.json

Decompose an inconclusive joint result into its two members:
    $ python scripts/run_e5_mask_aware.py --member field
    $ python scripts/run_e5_mask_aware.py --member darkening
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy,
                                           MaskAwareConfig, OrientationConfig,
                                           PolicyConfig)
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
# forward. Four families have been measured against it and only
# orientation was kept: scale hurt, tonal photometry and blur both
# landed inside the noise band. The base is therefore still orientation
# alone. Naming the run here keeps the counterpart identifiable from
# the provenance file rather than from somebody's memory.
BASELINE_RUN = "d4_seed20260907"

RUN_SEED = 20260907

# Which alternatives the mask-aware container may draw, the short name
# each variant is recorded under, and where its artefacts go. ``both``
# is the family as frozen; the other two exist for the decomposition,
# and are worth training only once the joint run has come out negative
# or inconclusive.
VARIANTS = {
    "both": (None, "mask_aware", "e5"),
    "field": (("field",), "mask_aware_field", "e5_field"),
    "darkening": (("darkening",), "mask_aware_dark", "e5_dark"),
}

# The split rate of the unaugmented runs, per hundred annotated pores.
# The family is adopted against false splits, so this is the spread its
# targeted improvement has to clear, and it is written here rather than
# recalled later because a target named after the fact is not a target.
BASELINE_SPLITS_PER_100 = (1.60, 2.97)

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


def build_mask_aware_policy(member: str = "both") -> AugmentationPolicy:
    """Orientation and mask-aware light, with every other family off.

    Every parameter other than the membership is taken from the frozen
    configuration rather than restated here, so that no run can quietly
    differ from what the review panels were drawn at and accepted. That
    includes the container's probability, which a decomposition leaves
    alone: a member is being asked whether it earns a place in the base
    as a family, and a family is defined at the probability it was
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
    mask_aware = (
        MaskAwareConfig() if members is None
        else MaskAwareConfig(members=members)
    )
    return AugmentationPolicy(PolicyConfig(
        orientation=OrientationConfig(),
        mask_aware=mask_aware,
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

    Spaced rather than taken from the front. This family needs pores
    large enough to hold a field that fades in from their edges, and
    pore size varies with the scale an image was photographed at, so a
    rehearsal drawn from one corner of the collection could exercise
    only the sizes where the transformation has room to act.

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
    validation image with a shadow painted into it would measure the
    transformation rather than the model, and a reordered one would
    change nothing except the comparability of the logs.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_mask_aware_policy(args.member)
    name = run_name(args.member)
    config = policy.config.mask_aware

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, members %s at p = %.2f, %d training "
        "image(s), %d validation image(s).",
        name, RUN_SEED, ", ".join(policy.families),
        ", ".join(config.members), config.p,
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Paired against %s over %d step(s), one per image; the "
        "screening reading falls at epoch %d of %d.",
        BASELINE_RUN, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    logger.info(
        "This family reads the annotation and does not write it, so "
        "the mask is checked for being untouched. It is adopted "
        "against false splits, whose baseline spread is %.2f to %.2f "
        "per hundred annotated pores - that is the figure to read "
        "beside the headline metric, and it was fixed before the run.",
        *BASELINE_SPLITS_PER_100,
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
    policy = build_mask_aware_policy(args.member)
    config = policy.config
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": name,
        "seed": RUN_SEED,
        "paired_baseline_run": BASELINE_RUN,
        "families": list(policy.families),
        "orientation": vars(config.orientation),
        "mask_aware": vars(config.mask_aware),
        "targeted_error": "splits_per_100_gt",
        "baseline_splits_per_100": list(BASELINE_SPLITS_PER_100),
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
        help="Which alternatives the container may draw. The default "
             "is the family as frozen; naming one member is the "
             "decomposition, and is worth training only once the joint "
             "run has come out negative or inconclusive.",
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
    """Train the mask-aware run.

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
        "of the family is read by scoring those snapshots against the "
        "paired run, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
