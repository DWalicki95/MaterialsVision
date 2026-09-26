#!/usr/bin/env python3
"""
Part II of the optimization study: the fine-tuning hyperparameters.

The augmentation study froze everything except the policy. Of the
training settings it froze, only one ever moved the metric beyond the
noise: raising the rate of the low-rank correction from 1e-4 to 3e-4
added about a hundredth of instance F1, consistently under both decoder
rates. That grid had two points, so its winner sits on its edge, and a
winner on an edge has not been shown to be an optimum.

**Why two arms and not one.** The library multiplies the correction by
a fixed factor and does not divide it by the rank. At a fixed rate a
higher rank therefore also means a longer effective step on the weight
update, so a probe of rank alone would confound the capacity of the
correction with the size of its step - and step size is the one thing
known to matter here. Two arms separate the two in one measurement:

* ``a1_r32`` - rank 32 at the current rate: capacity *and* step length;
* ``a2_lr1e-3`` - rank 8 at 1e-3: step length alone.

Everything else - the decoder rate, the schedule, the budget and the
policy - is the reference's.

**The policy is FULL_A, orientation alone.** The composite policies
were indistinguishable on VALIDATION and FULL_A is the simpler one. The
reference is three seeds of it; the first already exists as the
orientation run of the augmentation study, and this script trains the
other two.

**The edge points are listed but not launched by default.** Whichever
arm wins lies on the edge of what was tried by construction, so the
pre-registered procedure gives it exactly one point further out -
``edge_lr3e-3`` in the rate branch, ``edge_r64`` in the capacity
branch. They are defined here so that their values are fixed before
any arm is scored, and run only if the screening sends the study down
their branch.

**This script only trains.** Snapshots are scored afterwards, once the
post-processing has been recalibrated on TRAIN, because reading them
now would read them with the measuring stick that is being replaced.
Losses, learning rates and the decoder's raw output are visible in
MLflow while the run trains; instance metrics are added to the same
run by the evaluation.

Examples
--------
Train the two remaining reference seeds, then both arms:
    $ python scripts/run_hparam_arms.py --arm reference --seed 20260908
    $ python scripts/run_hparam_arms.py --arm reference --seed 20260909
    $ python scripts/run_hparam_arms.py --arm a1_r32
    $ python scripts/run_hparam_arms.py --arm a2_lr1e-3

Rehearse the whole path in minutes, into a separate experiment and
directory so the rehearsal can never be mistaken for a run:
    $ python scripts/run_hparam_arms.py --arm a1_r32 --rehearsal
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from materials_vision.augmentation import (AugmentationPolicy,
                                           OrientationConfig, PolicyConfig)
from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.tracking import (REHEARSAL_EXPERIMENT,
                                       TRAINING_EXPERIMENT, RunTracking)
from materials_vision.training import (DECODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, LORA_RANK,
                                       PEFT_KWARGS, build_loader, build_source,
                                       train_run)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/hparams")

REHEARSAL_SAVE_ROOT = Path("checkpoints/rehearsal")

BASE_MODEL = "vit_l_lm"

POLICY_NAME = "FULL_A"

# The three seeds of the augmentation study. The reference is trained at
# all three; an arm is screened at the first and, if it is confirmed,
# trained at the other two, which are then the runs that go forward.
SEEDS = (20260907, 20260908, 20260909)

SCREENING_SEED = SEEDS[0]

# The reference's budget, unchanged: arms differ from it in the
# adaptation alone.
N_EPOCHS = 12

CHECKPOINT_EVERY_EPOCHS = 1

REHEARSAL_N_TRAIN = 8

REHEARSAL_N_VAL = 4

REHEARSAL_EPOCHS = 1


@dataclass(frozen=True)
class Arm:
    """One configuration of the adaptation.

    Parameters
    ----------
    name : str
    lora_rank : int
    lora_learning_rate : float
    role : str
        Where the arm sits in the pre-registered procedure.
    """

    name: str
    lora_rank: int
    lora_learning_rate: float
    role: str


ARMS = {
    arm.name: arm for arm in (
        Arm("reference", LORA_RANK, LORA_LEARNING_RATE, "reference"),
        Arm("a1_r32", 32, LORA_LEARNING_RATE, "screening"),
        Arm("a2_lr1e-3", LORA_RANK, 1e-3, "screening"),
        Arm("edge_lr3e-3", LORA_RANK, 3e-3, "edge, rate branch only"),
        Arm("edge_r64", 64, LORA_LEARNING_RATE,
            "edge, capacity branch only"),
    )
}


def run_name(arm: Arm, seed: int) -> str:
    """Name a run by what it is and which seed it drew.

    The reference keeps the name the orientation run was given, so that
    its three seeds read as one series wherever they are listed.

    Parameters
    ----------
    arm : Arm
    seed : int

    Returns
    -------
    str
    """
    prefix = "d4" if arm.name == "reference" else f"d4_{arm.name}"
    return f"{prefix}_seed{seed}"


def build_policy() -> AugmentationPolicy:
    """FULL_A: the orientation family alone.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(orientation=OrientationConfig()))


def spaced_indices(n_available: int, n_wanted: int) -> Optional[list[int]]:
    """Evenly spaced positions, or every one of them.

    Spaced rather than taken from the front, so that a rehearsal still
    crosses both microscopes and both scale bins.

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


def is_finished(save_root: Path, name: str, epochs: int) -> bool:
    """Whether the run already produced its last epoch's snapshot.

    Parameters
    ----------
    save_root : Path
    name : str
    epochs : int

    Returns
    -------
    bool
    """
    return (save_root / "checkpoints" / name / f"epoch-{epochs}.pt").exists()


def provenance_record(
    args: argparse.Namespace, arm: Arm, name: str
) -> dict:
    """Everything that produced a run, for the file kept beside it.

    Parameters
    ----------
    args : argparse.Namespace
    arm : Arm
    name : str

    Returns
    -------
    dict
    """
    policy = build_policy()
    return {
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": name,
        "arm": arm.name,
        "arm_role": arm.role,
        "seed": args.seed,
        "policy": POLICY_NAME,
        "families": list(policy.families),
        "orientation": vars(OrientationConfig()),
        "epochs": args.epochs,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "lora_rank": arm.lora_rank,
        "lora_update_matrices": list(PEFT_KWARGS["update_matrices"]),
        "lora_learning_rate": arm.lora_learning_rate,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
        "rehearsal": args.rehearsal,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }


def write_provenance(record: dict, save_root: Path, name: str) -> Path:
    """Write the provenance record next to the run's checkpoints.

    Parameters
    ----------
    record : dict
    save_root : Path
    name : str

    Returns
    -------
    Path
    """
    save_root.mkdir(parents=True, exist_ok=True)
    destination = save_root / f"{name}_provenance.json"
    destination.write_text(json.dumps(record, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return destination


def build_tracking(
    args: argparse.Namespace, arm: Arm, provenance_path: Path
) -> RunTracking:
    """Describe how the run is recorded in MLflow.

    Parameters
    ----------
    args : argparse.Namespace
    arm : Arm
    provenance_path : Path

    Returns
    -------
    RunTracking
    """
    return RunTracking(
        experiment=REHEARSAL_EXPERIMENT if args.rehearsal
        else TRAINING_EXPERIMENT,
        tags={
            "study": "model_optimization",
            "part": "II",
            "arm": arm.name,
            "arm_role": arm.role,
            "policy": POLICY_NAME,
            "seed": str(args.seed),
            "train_subset": "train",
            "source": "live",
        },
        params={
            "seed": args.seed,
            "policy": POLICY_NAME,
            "augmentation_families": ",".join(build_policy().families),
        },
        artifacts=(provenance_path,),
    )


def execute(args: argparse.Namespace, arm: Arm, name: str,
            tracking: RunTracking) -> None:
    """Train the run.

    Validation is loaded without a policy and without shuffling, as in
    every run of the augmentation study.

    Parameters
    ----------
    args : argparse.Namespace
    arm : Arm
    name : str
    tracking : RunTracking
    """
    torch.manual_seed(args.seed)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")
    train_loader = build_loader(
        train_source, policy=build_policy(), run_seed=args.seed,
        shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=args.seed, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s (%s): rank %d, correction rate %.0e, decoder rate %.0e, "
        "seed %d, %d training and %d validation image(s), %d epoch(s).",
        name, arm.role, arm.lora_rank, arm.lora_learning_rate,
        DECODER_LEARNING_RATE, args.seed, len(train_loader),
        len(val_loader), args.epochs,
    )
    train_run(
        name,
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=arm.lora_learning_rate,
        decoder_learning_rate=DECODER_LEARNING_RATE,
        lora_rank=arm.lora_rank,
        early_stopping=None,
        save_every_kth_epoch=CHECKPOINT_EVERY_EPOCHS,
        tracking=tracking,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--seed", type=int, default=SCREENING_SEED, choices=SEEDS,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument(
        "--rehearsal", action="store_true",
        help=(
            f"{REHEARSAL_N_TRAIN} training and {REHEARSAL_N_VAL} "
            f"validation images for {REHEARSAL_EPOCHS} epoch, into "
            f"{REHEARSAL_SAVE_ROOT} and the {REHEARSAL_EXPERIMENT} "
            "experiment."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the run already finished.",
    )
    args = parser.parse_args(argv)
    args.n_train = REHEARSAL_N_TRAIN if args.rehearsal else 0
    args.n_val = REHEARSAL_N_VAL if args.rehearsal else 0
    args.epochs = REHEARSAL_EPOCHS if args.rehearsal else N_EPOCHS
    if args.save_root is None:
        args.save_root = (
            REHEARSAL_SAVE_ROOT if args.rehearsal else DEFAULT_SAVE_ROOT
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Train one arm at one seed.

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

    arm = ARMS[args.arm]
    name = run_name(arm, args.seed)
    if arm.name == "reference" and args.seed == SCREENING_SEED:
        logger.error(
            "%s already exists as the orientation run of the "
            "augmentation study (checkpoints/e1); training it again "
            "would produce a second copy of one run.", name,
        )
        return EXIT_FAILED
    if not args.force and is_finished(args.save_root, name, args.epochs):
        logger.info("%s already finished; nothing to do.", name)
        return EXIT_OK

    provenance_path = write_provenance(
        provenance_record(args, arm, name), args.save_root, name
    )
    execute(args, arm, name, build_tracking(args, arm, provenance_path))
    logger.info(
        "Run finished. Snapshots are under %s/checkpoints/%s; they are "
        "scored once the post-processing has been recalibrated.",
        args.save_root, name,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
