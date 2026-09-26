#!/usr/bin/env python3
"""
The final model: the adopted policy, trained on everything except TEST.

**What was decided, and where.** Every choice behind this model was made
on VALIDATION: the augmentation policy, the budget, the learning rates
and the rule for picking a snapshot. The policy adopted is orientation,
tonal photometry and blur together. Among the policies measured it has
no measurable advantage over orientation alone - the two differ by
+0.0017 of instance F1 against a significance floor of 0.0050, and
removing either photometric family from it costs nothing measurable.
It is adopted as the broadest policy with no measurable cost, on the
expectation that the photometric families guard against differences in
acquisition that five validation formulations cannot represent. What
it must not be described as is better than orientation alone.

**Why VALIDATION is trained on now.** Once nothing is left to choose,
VALIDATION is no longer needed as a judge. The model effectively learns
from formulations rather than from images, since every image of one
formulation comes from one synthesis, so five more formulations are a
quarter more independent material - and the delivered model should
have seen all of it.

**Why the snapshot is fixed before training.** With VALIDATION inside
the training set, the only held-out images left to pick an epoch on are
TEST, and picking on TEST would turn the one unbiased measurement of
this study into one more development choice. So the snapshot scored is
the last one, epoch twelve, and it is fixed here, before the run
exists. It is also a safe point to fix: under this policy on TRAIN
alone the validation curve is flat from epoch seven on, and its last
epoch sits within the noise of its peak.

**Twelve epochs, which is more steps.** The budget is kept in passes
over the data rather than in optimizer steps, so that every image is
seen as often as in the runs the policy was chosen on. With about a
fifth more images an epoch is a fifth longer: some 7200 steps instead
of some 5900, with the rate annealed over the same twelve passes.

**A validation set the trainer insists on and nobody reads.** The
trainer computes a validation loss every epoch and keeps a best
snapshot by it. Every image that could serve is now either being
trained on or is TEST, so it is given a handful of VALIDATION images
that are also in training. The loss it reports on them measures
memorization and is read by nothing: the schedule ignores it, early
stopping is off, and the snapshot scored is the one fixed above, not
the trainer's best. A handful rather than all of them, because each
one costs time and buys nothing.

**One seed, and the same one.** The seed is the one every run of the
policy comparison used, so this run differs from the TRAIN-only run of
the same policy in the data it sees and in nothing that was chosen.

**What this model is not for.** It takes no part in the comparison of
augmentation policies. It has learned from more material than the runs
it would be compared against, so any difference would mix the policy
with the amount of data. That comparison is made on TEST with the runs
trained on TRAIN alone. This model's TEST score answers a different
question: how well the model that is handed over segments formulations
it has never seen.

Examples
--------
Train the final model:
    $ python scripts/train_final_model.py

Rehearse the whole path in minutes, into a throwaway directory:
    $ python scripts/train_final_model.py --n-train 8 --epochs 1 \\
        --save-root checkpoints/final_smoke

Score the fixed snapshot on TEST, once, when the moment has come:
    $ python scripts/evaluate_checkpoint.py --subset test --unlock-test \\
        --checkpoint \\
        checkpoints/final/checkpoints/full_b_trainval_seed20260907/epoch-12.pt
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.training import (DECODER_LEARNING_RATE,
                                       LORA_LEARNING_RATE, build_loader,
                                       build_source, train_run)
from scripts.run_e8_full import (BASE_MODEL, RUN_SEED, build_full_policy,
                                 spaced_indices)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/final")

# The adopted policy, composed by the same function that composed it
# for the comparison, so the final model cannot quietly differ from the
# policy whose results selected it.
POLICY = "full_b"

TRAIN_SUBSET = "train+val"

N_EPOCHS = 12

# How many VALIDATION images the trainer is handed to compute a loss
# nobody reads. Spaced across the subset only so that a crash on some
# property of one microscope would still surface here.
MONITOR_IMAGES = 16


def run_name() -> str:
    """What the final model's checkpoints are filed under.

    Returns
    -------
    str
    """
    return f"{POLICY}_trainval_seed{RUN_SEED}"


def scored_snapshot(save_root: Path, n_epochs: int) -> Path:
    """The one snapshot that will be scored, named before it exists.

    Parameters
    ----------
    save_root : Path
    n_epochs : int

    Returns
    -------
    Path
    """
    return save_root / "checkpoints" / run_name() / f"epoch-{n_epochs}.pt"


def execute(args: argparse.Namespace) -> None:
    """Train the final model.

    Only the scored snapshot is kept as a numbered file. The trainer
    numbers a snapshot whenever the epoch count divides by the interval
    it is given, so an interval equal to the budget yields exactly the
    last epoch; the intermediate ones could not be looked at without a
    held-out set to look at them on, and each costs well over a
    gigabyte.

    Parameters
    ----------
    args : argparse.Namespace
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(args.split, args.manifest, TRAIN_SUBSET)
    monitor_source = build_source(args.split, args.manifest, "val")
    policy = build_full_policy(POLICY)

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    monitor_loader = build_loader(
        monitor_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(monitor_source), MONITOR_IMAGES),
    )
    logger.info(
        "%s: seed %d, families %s, %d training image(s) from %s.",
        run_name(), RUN_SEED, ", ".join(policy.families),
        len(train_loader), TRAIN_SUBSET.upper(),
    )
    logger.info(
        "The %d monitoring image(s) are also trained on. The validation "
        "loss reported on them measures memorization and decides "
        "nothing; the snapshot scored is %s, fixed before training.",
        len(monitor_loader), scored_snapshot(args.save_root, args.epochs),
    )
    train_run(
        run_name(),
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=monitor_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=LORA_LEARNING_RATE,
        decoder_learning_rate=DECODER_LEARNING_RATE,
        early_stopping=None,
        save_every_kth_epoch=args.epochs,
    )


def write_provenance(args: argparse.Namespace) -> Path:
    """Record what produced the final model, next to it.

    The snapshot to be scored is written down here, before training,
    so that the record shows it was fixed in advance rather than chosen
    after the scores were in.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
        The file written.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / f"{run_name()}_provenance.json"
    policy = build_full_policy(POLICY)
    config = policy.config
    record: dict[str, Any] = {
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "run": run_name(),
        "policy": POLICY,
        "families": list(policy.families),
        "seed": RUN_SEED,
        "train_subset": TRAIN_SUBSET,
        "epochs": args.epochs,
        "scored_snapshot": str(scored_snapshot(args.save_root, args.epochs)),
        "monitor_images": MONITOR_IMAGES,
        "monitor_note": (
            "Validation images handed to the trainer are also trained "
            "on; its validation loss and best.pt are not used."
        ),
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }
    for family in ("orientation", "tonal", "blur"):
        record[family] = vars(getattr(config, family))
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
        "--epochs", type=int, default=N_EPOCHS,
        help="Passes over TRAIN and VALIDATION together. The last one "
             "is the snapshot scored.",
    )
    parser.add_argument(
        "--n-train", type=int, default=0,
        help="Use this many training images; 0 means all of them.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the scored snapshot already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train the final model.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if args.epochs < 1:
        logger.error("A run needs at least one epoch, got %d.", args.epochs)
        return EXIT_FAILED
    if not torch.cuda.is_available():
        logger.error("No GPU is visible, so there is nothing to train on.")
        return EXIT_FAILED
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    write_provenance(args)
    snapshot = scored_snapshot(args.save_root, args.epochs)
    if not args.force and snapshot.exists():
        logger.info("%s already exists; nothing to do.", snapshot)
        return EXIT_OK
    execute(args)

    if not snapshot.exists():
        logger.error(
            "Training finished but %s was not written; the snapshot "
            "that was fixed in advance does not exist.", snapshot,
        )
        return EXIT_FAILED
    logger.info("Run finished. The snapshot to score is %s.", snapshot)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
