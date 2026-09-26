#!/usr/bin/env python3
"""
The composite policies: the families that helped, put together.

Six augmentation families have now been measured one at a time, each
added to the base the previous ones left behind. Exactly one of them
cleared the significance floor: orientation, at +0.0138. Scale came out
clearly negative and was rejected. The remaining four landed inside the
noise band, two of them on the positive side - tonal photometry at
+0.0020 and blur at +0.0017 - and two on the negative side.

**Why a composite is measured at all.** A screening sequence that
admits only families which clear the floor on their own cannot answer
whether several individually undetectable gains accumulate into
something detectable. At this noise level a family worth two thousandths
of instance F1 is invisible one at a time no matter how carefully it is
measured, and the sequence would discard it without ever having tested
the hypothesis. So two composites were fixed, by a mechanical rule
reading nothing but the sign of the difference on the headline metric,
and both are measured:

* **FULL_A** - the base plus every family that cleared the floor. That
  is orientation alone.
* **FULL_B** - the base plus every family whose difference against its
  own base was at or above zero, whether or not it cleared the floor.
  That is orientation, tonal photometry and blur.

**The rule was written down before the last two families were
measured**, which is the whole point of stating it as a rule: the
membership of both composites follows from the sign of a number, not
from what looked appealing once the numbers were in. One consequence
was accepted knowingly - blur enters FULL_B although it raised merges
from 2.78 to 3.47 per hundred pores. The rule reads the headline
metric, not the guard metrics, and 3.47 still sits inside the 2.36 to
4.48 spread the unaugmented runs themselves showed, so a rule amended
to exclude it would have been an amendment that changes nothing except
the appearance of rigour.

**FULL_A needs no training.** It is the orientation run that already
exists, with its full curve of twelve snapshots. Training it again
under a new name would produce the same model and invite somebody to
compare two runs that are one run.

**So this script trains two runs, and they are these.**

``full_b`` is the composite itself: orientation, tonal photometry and
blur together. It is paired against the orientation run, which is
FULL_A, and the difference between them is what decides whether
accumulation happened.

``full_b_no_d4`` is the composite with orientation taken out, leaving
tonal photometry and blur. It belongs to the leave-one-family-out
ablation, the second view of attribution: the first asks what a family
adds when it arrives, this one asks what its removal costs. The
ablation needs four arms, and two of them are already on disk - the
sequence added exactly one family to orientation at each stage, so the
run under orientation and tonal is the composite without blur, and the
run under orientation and blur is the composite without tonal
photometry. Only the arm that drops orientation has never been trained,
because orientation was in the base from the moment it was accepted.
Four variants, two new runs.

**Why the ablation arm is worth a full run rather than a short one.**
Orientation is the one family here with a measured effect larger than
the floor, so its removal is the one removal expected to show
something. Reading it at a fraction of the budget would answer a
smaller question with a curve that cannot be laid over the others.

**What decides, fixed before the runs started.** The pooled metric over
all hundred and seven validation images, against the frozen
significance floor. The likely outcome at one seed is stated in advance
rather than discovered afterwards: if tonal photometry contributes
+0.0020 and blur +0.0017 and the two simply add, the difference between
the composites is about +0.0037 against a floor of 0.0050, so a reading
inside the band is the expected result and not a surprise. The
resolution in that case is three seeds a side on the untouched test
set, which brings the spread of the difference of means down from about
0.0040 to about 0.0023 and makes an effect of that size visible.
Writing this down beforehand is what stops an inconclusive validation
reading from being narrated as either a win or a failure.

**One seed a side, and the seed is the one the counterpart ran at.**
Pairing the runs at a common seed removes the part of the difference
that is the draw rather than the policy. Everything except the
augmentation policy is identical to the run each one is compared
against: the same backbone and adaptation, the same budget, the same
sampler, the same rates.

**No early stopping, a snapshot every epoch.** A run halted early would
leave a shorter curve than the one it is compared against, and the
metric that decides is not the loss the trainer minimizes, so the
trainer is in no position to judge when to stop. Keeping every epoch
means the comparison is read as a position on a curve rather than at a
point the process had to stop at.

**Validation is never augmented and the test split is never opened.**
A transformed validation image measures the transformation rather than
the model. The test split stays closed until the policy is frozen; it
is not reachable from this script at all.

**Scoring happens afterwards.** This script only trains. What decides
is instance F1 against the annotation, computed by running each
snapshot the way the model will be used, at the calibrated seeding
thresholds the scoring step already defaults to - not the validation
loss, which can improve while neighbouring pores merge into one. For a
composite carrying blur that gap is not hypothetical: merging is the
failure blur was already shown to worsen, so the merge count and the
boundary agreement are read beside the headline metric rather than
after it.

Examples
--------
Train the composite, then the arm without orientation:
    $ python scripts/run_e8_full.py --run full_b
    $ python scripts/run_e8_full.py --run full_b_no_d4

Rehearse the whole path in minutes:
    $ python scripts/run_e8_full.py --run full_b \\
        --n-train 8 --n-val 4 --epochs 1

Score the snapshots afterwards, at the calibrated thresholds:
    $ python scripts/evaluate_checkpoint.py --checkpoint-glob \\
        'checkpoints/e8/checkpoints/full_b_seed20260907/epoch-*.pt' \\
        --out checkpoints/e8/e8_full_b_curves.json

Read the composite against FULL_A, which is the orientation run:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e8/e8_full_b_curves.json \\
        --baseline checkpoints/e1/e1_curves.json \\
        --baseline-run-prefix d4_
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy, BlurConfig,
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

# Both variants share one root. They are two arms of one comparison and
# are read together, and the checkpoints still land in a directory named
# after the run, so nothing has to be told apart by filename.
DEFAULT_SAVE_ROOT = Path("checkpoints/e8")

# Settled by the base-model pilot, on the metric this study reports.
BASE_MODEL = "vit_l_lm"

RUN_SEED = 20260907

# The run standing for FULL_A: the base plus the one family that
# cleared the floor. It exists already with its full curve, so the
# composite is compared against it rather than against a retrained
# copy of the same policy.
FULL_A_RUN = f"d4_seed{RUN_SEED}"

# Each family enters at the parameters it was frozen and measured at.
# Taking them from the configuration rather than restating them here is
# what keeps a composite from quietly differing from the runs whose
# results decided its membership.
FAMILY_CONFIGS: dict[str, Any] = {
    "orientation": OrientationConfig,
    "tonal": TonalConfig,
    "blur": BlurConfig,
}


class Variant(NamedTuple):
    """One of the two runs this script trains.

    Parameters
    ----------
    families : tuple of str
        Keys of :data:`FAMILY_CONFIGS` this run switches on.
    counterpart : str
        The run this one is read against. Naming it here keeps the
        intended comparison identifiable from the provenance file
        rather than from somebody's memory of which pairing was meant.
    omitted : str or None
        The family this run drops relative to the full composite, for
        the arms that belong to the ablation. ``None`` for the
        composite itself, which drops nothing.
    """

    families: tuple[str, ...]
    counterpart: str
    omitted: Optional[str]


VARIANTS = {
    "full_b": Variant(
        families=("orientation", "tonal", "blur"),
        counterpart=FULL_A_RUN,
        omitted=None,
    ),
    "full_b_no_d4": Variant(
        families=("tonal", "blur"),
        counterpart=f"full_b_seed{RUN_SEED}",
        omitted="orientation",
    ),
}

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


def run_name(variant: str) -> str:
    """What this variant's checkpoints are filed under.

    Parameters
    ----------
    variant : str
        A key of :data:`VARIANTS`.

    Returns
    -------
    str
    """
    return f"{variant}_seed{RUN_SEED}"


def build_full_policy(variant: str) -> AugmentationPolicy:
    """Compose the families this variant carries, and nothing else.

    Composition is the only thing that distinguishes these runs from
    the ones already on disk, so it is the only thing expressed here.
    Every family is instantiated at its frozen defaults: the parameters
    were calibrated and reviewed once, and a composite that re-tuned
    them would no longer be the composition of the families whose
    individual results selected it.

    The families that were measured and rejected stay off, including
    the two that came out on the negative side of the noise band. The
    membership rule reads the sign of the difference, and theirs was
    below zero.

    Parameters
    ----------
    variant : str
        A key of :data:`VARIANTS`.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(**{
        family: FAMILY_CONFIGS[family]()
        for family in VARIANTS[variant].families
    }))


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
    composite carries families answering to different properties of the
    collection - orientation to how a sample was laid on the stage,
    tonal photometry to the two detectors, blur to focus - and a
    rehearsal drawn from one corner would exercise them all over a
    single instrument and a single pore size.

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
    last = (args.save_root / "checkpoints" / run_name(args.run)
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
    variant = VARIANTS[args.run]
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    policy = build_full_policy(args.run)
    name = run_name(args.run)

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: seed %d, families %s, %d training image(s), %d "
        "validation image(s).",
        name, RUN_SEED, ", ".join(policy.families),
        len(train_loader), len(val_loader),
    )
    logger.info(
        "Read against %s over %d step(s), one per image; the screening "
        "reading falls at epoch %d of %d.",
        variant.counterpart, len(train_loader) * args.epochs,
        screening_epoch(args.epochs), args.epochs,
    )
    if variant.omitted is None:
        logger.info(
            "This is the composite itself. Its families arrive at the "
            "parameters each was frozen at, so the run is the "
            "composition of the measured families and not a new "
            "configuration of them.",
        )
    else:
        logger.info(
            "This is the composite without the %s family, the arm of "
            "the leave-one-family-out ablation that has no counterpart "
            "among the runs already trained. What it measures is the "
            "cost of removal, which is a different view of a family's "
            "contribution from the gain it showed on arrival.",
            variant.omitted,
        )
    logger.info(
        "The composite carries blur, which was already shown to raise "
        "merges, so the merge count and the boundary agreement are "
        "read alongside the headline metric rather than after it.",
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
    change afterwards. That matters more for a composite than for a
    single family: the composite's membership was decided by results
    obtained under particular parameters, and a file naming only the
    families would not say which.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
        The file written.

    Notes
    -----
    The parameter blocks are keyed by the configuration's own field
    names rather than by the codes the policy lists its families under:
    the codes carry the family's number in the study and are what the
    record should read as, but they are not what the configuration can
    be asked for.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    name = run_name(args.run)
    variant = VARIANTS[args.run]
    destination = args.save_root / f"{name}_provenance.json"
    policy = build_full_policy(args.run)
    config = policy.config
    record: dict[str, Any] = {
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": name,
        "seed": RUN_SEED,
        "paired_baseline_run": variant.counterpart,
        "omitted_family": variant.omitted,
        "families": list(policy.families),
        "epochs": args.epochs,
        "screening_epoch": screening_epoch(args.epochs),
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }
    for family in variant.families:
        record[family] = vars(getattr(config, family))
    if config.blur is not None:
        record["blur_kernel_px"] = config.blur.kernel_px
    destination.write_text(json.dumps(record, indent=2, default=str))
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
        "--run", choices=sorted(VARIANTS), default="full_b",
        help="Which of the two runs to train. The default is the "
             "composite; the other drops orientation from it and is "
             "the arm of the ablation that has no counterpart among "
             "the runs already on disk.",
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
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train one of the two composite runs.

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
        logger.info("%s already finished; nothing to do.", run_name(args.run))
        return EXIT_OK
    execute(args)

    logger.info(
        "Run finished. Every epoch's snapshot is under %s; what the "
        "composite is worth is read by scoring those snapshots against "
        "the paired run, not from the losses above.",
        args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
