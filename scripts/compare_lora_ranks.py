#!/usr/bin/env python3
"""
Whether the low-rank correction is given the right number of degrees of
freedom.

Fine-tuning here does not change the image encoder's weights. It
freezes them and adds a correction alongside, written as the product of
two thin matrices. The rank is the shorter dimension of that product,
and it is exactly the number of independent directions in which the
correction can move any one attention matrix. At rank 8 the correction
to a 1024 by 1024 projection holds sixteen thousand numbers instead of
a million; at rank 32, four times that.

The trade-off runs both ways. Too few directions and the correction
cannot express the adaptation the new domain needs. Too many and it
reintroduces the problem it exists to avoid: a great many parameters
fitted to very little data.

**Why this is worth measuring rather than assuming.** The package whose
method this is publishes rank 32 as its recommended setting, and this
study uses 8. Four times is not a rounding difference. But the
recommendation was tuned on that package's own benchmarks, which are
larger and more varied than this one - and capacity is precisely the
kind of choice that does not carry across datasets of different size.
The baseline runs here reach their best score after two epochs and
decline from there, which is the signature of too much capacity for the
data rather than too little, so the argument says leave it at 8. That
is an argument, and arguments about this study's settings have been
wrong before.

**What is compared.** Everything is held to the baseline runs except
the rank: same base weights, same images in the same order, same seed,
same learning rate, same budget, no augmentation. The arm at the
current rank costs nothing, because the baseline runs already provide
it, epoch for epoch, at the same seed.

**Scored on the metric that is reported.** Every epoch is kept and
scored afterwards, the way the baseline and the learning-rate probe
were. The trainer's own validation loss is not used to choose anything
here; it disagreed with the reported metric by four epochs about where
the peak was.

Examples
--------
Probe the rank the method's authors recommend:
    $ python scripts/compare_lora_ranks.py

Then score what it produced:
    $ python scripts/evaluate_checkpoint.py \\
        --checkpoint-glob "checkpoints/rank_probe/checkpoints/*/epoch-*.pt" \\
        --out checkpoints/rank_probe/rank_curves.json

Rehearse the path in minutes:
    $ python scripts/compare_lora_ranks.py --n-train 6 --epochs 1
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

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

DEFAULT_SAVE_ROOT = Path("checkpoints/rank_probe")

BASE_MODEL = "vit_l_lm"

# The value the method's own package publishes as its recommended
# setting. Probing it, rather than the whole ladder, because the
# question is not "which rank is best" but "is the published
# recommendation better here than the smaller value in use".
PROBE_RANKS = (32,)

# Supplied by the baseline runs at this very seed; not re-run here.
CURRENT_RANK = 8

SEED = 20260907

N_EPOCHS = 5

CHECKPOINT_EVERY_EPOCHS = 1


def run_name(rank: int) -> str:
    """Directory name for one rank's run.

    Parameters
    ----------
    rank : int

    Returns
    -------
    str
    """
    return f"rank_{rank}"


def spaced_indices(n_available: int, n_wanted: int) -> Optional[list[int]]:
    """Evenly spaced positions, or every one of them.

    Parameters
    ----------
    n_available : int
    n_wanted : int
        Zero or less asks for all of them.

    Returns
    -------
    list of int or None
    """
    if n_wanted <= 0 or n_wanted >= n_available:
        return None
    stride = n_available / n_wanted
    return [int(position * stride) for position in range(n_wanted)]


def is_finished(rank: int, args: argparse.Namespace) -> bool:
    """Whether this rank's run already produced its last snapshot.

    Parameters
    ----------
    rank : int
    args : argparse.Namespace

    Returns
    -------
    bool
    """
    last = (args.save_root / "checkpoints" / run_name(rank)
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(rank: int, args: argparse.Namespace) -> None:
    """Train one run at one rank.

    Parameters
    ----------
    rank : int
    args : argparse.Namespace
    """
    torch.manual_seed(args.seed)
    train_source = build_source(args.split, args.manifest, "train")
    val_source = build_source(args.split, args.manifest, "val")

    train_loader = build_loader(
        train_source, policy=None, run_seed=args.seed, shuffle=True,
        indices=spaced_indices(len(train_source), args.n_train),
    )
    val_loader = build_loader(
        val_source, policy=None, run_seed=args.seed, shuffle=False,
        indices=spaced_indices(len(val_source), args.n_val),
    )
    logger.info(
        "%s: rank %d, %d training image(s), %d validation image(s).",
        run_name(rank), rank, len(train_loader), len(val_loader),
    )
    train_run(
        run_name(rank),
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=args.lora_learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        lora_rank=rank,
        early_stopping=None,
        save_every_kth_epoch=CHECKPOINT_EVERY_EPOCHS,
    )


def write_provenance(args: argparse.Namespace) -> None:
    """Record what produced these checkpoints, next to them.

    Parameters
    ----------
    args : argparse.Namespace
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / "rank_probe_provenance.json"
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "probe_ranks": list(args.ranks),
        "rank_already_measured_by_baseline": CURRENT_RANK,
        "lora_learning_rate": args.lora_learning_rate,
        "decoder_learning_rate": args.decoder_learning_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }, indent=2, default=str))
    logger.info("Wrote %s.", destination)


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
        "--ranks", type=int, nargs="+", default=list(PROBE_RANKS),
        help="Ranks to try.",
    )
    parser.add_argument(
        "--lora-learning-rate", type=float, default=LORA_LEARNING_RATE,
        help="Correction rate; must match the baseline arm.",
    )
    parser.add_argument(
        "--decoder-learning-rate", type=float,
        default=DECODER_LEARNING_RATE,
        help="Decoder rate; must match the baseline arm.",
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Passes over the training split, the same for every rank.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Run seed; matched to the baseline arm it is compared with.",
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
        help="Re-run ranks that already finished.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train one short run per candidate rank.

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
    for rank in args.ranks:
        if not args.force and is_finished(rank, args):
            logger.info("%s already finished; skipping.", run_name(rank))
            continue
        execute(rank, args)

    logger.info(
        "Probe finished. Score the snapshots under %s against the "
        "baseline run at rank %d, which supplies the arm at the rank "
        "in use.", args.save_root, CURRENT_RANK,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
