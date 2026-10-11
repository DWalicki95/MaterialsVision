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
from materials_vision.tracking import (REHEARSAL_EXPERIMENT,
                                       TRAINING_EXPERIMENT, RunTracking)
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


def scored_snapshot(
    save_root: Path, n_epochs: int, name: Optional[str] = None
) -> Path:
    """The one snapshot that will be scored, named before it exists.

    Parameters
    ----------
    save_root : Path
    n_epochs : int
    name : str, optional
        The run; the final model's when not given.

    Returns
    -------
    Path
    """
    run = run_name() if name is None else name
    return save_root / "checkpoints" / run / f"epoch-{n_epochs}.pt"


def train_recipe(
    name: str,
    *,
    split: Path,
    manifest: Path,
    train_subset: str,
    monitor_subset: str,
    save_root: Path,
    epochs: int,
    n_train: int = 0,
    tracking: Optional[RunTracking] = None,
) -> None:
    """Train the adopted recipe on the given images.

    The one definition of how the final model is trained, shared by
    every run meant to reproduce it on other data, so that such a run
    can differ from the final model in its images and in nothing that
    was chosen.

    Only the scored snapshot is kept as a numbered file. The trainer
    numbers a snapshot whenever the epoch count divides by the interval
    it is given, so an interval equal to the budget yields exactly the
    last epoch; the intermediate ones could not be looked at without a
    held-out set to look at them on, and each costs well over a
    gigabyte.

    The seed is set before anything else is built, in the order the
    final model was trained in, because the initial state of the
    low-rank correction is drawn from the global stream.

    Parameters
    ----------
    name : str
        Names the checkpoint directory.
    split, manifest : Path
    train_subset : str
        Subset of ``split`` trained on, e.g. ``"train+val"``.
    monitor_subset : str
        Subset the trainer's validation loss is computed on. Its images
        must also be trained on, or they would be withheld for a loss
        nothing reads.
    save_root : Path
    epochs : int
        The last epoch is the snapshot scored.
    n_train : int, optional
        Use this many training images; 0 means all of them.
    tracking : RunTracking, optional
        Record the run in MLflow as it trains.
    """
    torch.manual_seed(RUN_SEED)
    train_source = build_source(split, manifest, train_subset)
    monitor_source = build_source(split, manifest, monitor_subset)
    policy = build_full_policy(POLICY)

    train_loader = build_loader(
        train_source, policy=policy, run_seed=RUN_SEED, shuffle=True,
        indices=spaced_indices(len(train_source), n_train),
    )
    monitor_loader = build_loader(
        monitor_source, policy=None, run_seed=RUN_SEED, shuffle=False,
        indices=spaced_indices(len(monitor_source), MONITOR_IMAGES),
    )
    logger.info(
        "%s: seed %d, families %s, %d training image(s) from %s of %s.",
        name, RUN_SEED, ", ".join(policy.families),
        len(train_loader), train_subset.upper(), split.name,
    )
    logger.info(
        "The %d monitoring image(s) are also trained on. The validation "
        "loss reported on them measures memorization and decides "
        "nothing; the snapshot scored is %s, fixed before training.",
        len(monitor_loader), scored_snapshot(save_root, epochs, name),
    )
    train_run(
        name,
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=monitor_loader,
        n_epochs=epochs,
        save_root=save_root,
        lora_learning_rate=LORA_LEARNING_RATE,
        decoder_learning_rate=DECODER_LEARNING_RATE,
        early_stopping=None,
        save_every_kth_epoch=epochs,
        tracking=tracking,
    )


def build_tracking(
    args: argparse.Namespace, provenance_path: Path
) -> RunTracking:
    """Describe how the run is recorded in MLflow.

    A run on a subset of the images is a rehearsal of the path, not the
    final model, and goes to the rehearsal experiment so that it can
    never be mistaken for the final model when runs are compared.

    Parameters
    ----------
    args : argparse.Namespace
    provenance_path : Path

    Returns
    -------
    RunTracking
    """
    families = ",".join(recipe_record()["families"])
    return RunTracking(
        experiment=REHEARSAL_EXPERIMENT if args.n_train > 0
        else TRAINING_EXPERIMENT,
        tags={
            "study": "final_model",
            "part": "final",
            "policy": POLICY.upper(),
            "seed": str(RUN_SEED),
            "train_subset": TRAIN_SUBSET,
            "source": "live",
        },
        params={
            "seed": RUN_SEED,
            "policy": POLICY.upper(),
            "augmentation_families": families,
        },
        artifacts=(provenance_path,),
    )


def execute(args: argparse.Namespace) -> None:
    """Record the run and train the final model.

    Parameters
    ----------
    args : argparse.Namespace
    """
    provenance_path = write_provenance(args)
    train_recipe(
        run_name(),
        split=args.split,
        manifest=args.manifest,
        train_subset=TRAIN_SUBSET,
        monitor_subset="val",
        save_root=args.save_root,
        epochs=args.epochs,
        n_train=args.n_train,
        tracking=build_tracking(args, provenance_path),
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
    record: dict[str, Any] = {
        "provenance": run_provenance(),
        "run": run_name(),
        "train_subset": TRAIN_SUBSET,
        "epochs": args.epochs,
        "scored_snapshot": str(scored_snapshot(args.save_root, args.epochs)),
        "monitor_note": (
            "Validation images handed to the trainer are also trained "
            "on; its validation loss and best.pt are not used."
        ),
        "split": str(args.split),
        "manifest": str(args.manifest),
        **recipe_record(),
    }
    destination.write_text(json.dumps(record, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return destination


def recipe_record() -> dict[str, Any]:
    """What the recipe is, independently of the data it is trained on.

    Returns
    -------
    dict
    """
    policy = build_full_policy(POLICY)
    config = policy.config
    record: dict[str, Any] = {
        "base_model": BASE_MODEL,
        "policy": POLICY,
        "families": list(policy.families),
        "seed": RUN_SEED,
        "monitor_images": MONITOR_IMAGES,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "early_stopping": None,
    }
    for family in ("orientation", "tonal", "blur"):
        record[family] = vars(getattr(config, family))
    record["blur_kernel_px"] = config.blur.kernel_px
    return record


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
