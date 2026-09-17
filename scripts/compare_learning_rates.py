#!/usr/bin/env python3
"""
How fast each half of this model should be allowed to learn.

Fine-tuning here moves two things of very different size. A low-rank
correction on the image encoder holds under a million parameters and
starts at exactly zero. An instance decoder of nine and a half million
parameters starts from pretrained weights and trains in full. They were
run at one rate, chosen by an argument about the correction alone, and
the result was a decoder that drove its own training loss down
threefold while its validation loss turned upward within three epochs.
The run peaked after two passes over the data and declined for the
remaining eighteen.

**What this probes.** Both rates at once, as a grid: the correction at
the rate in use and three times it, the decoder at the library's
default and three times that. Four runs, everything else held to the
baseline - same base weights, same images in the same order, same seed,
same adaptation rank, same budget, no augmentation, no early stopping.

**Ten epochs, not five.** An earlier probe of this question ran five
and concluded that the highest rate it tried was best. That was an
artifact of the window. Two processes overlap here and they run at
different speeds: adaptation to a new imaging domain is fast and shows
up within an epoch or two, while the decoder's overfitting is slow and
only becomes visible past epoch five. Five epochs see the first and are
blind to the second, which is exactly the wrong half to measure when
the question is how fast the decoder should go.

**What decides the winner.** Two readings, not one. The peak of
instance F1 at a mask overlap of one half, and the direction of the
decoder's validation loss between epoch five and epoch ten. A rate pair
that peaks high but has already turned upward by epoch five is worse
than its peak suggests, because every later comparison in this study
runs for far longer than ten epochs.

**What is being compared is a schedule, not a bare rate.** Each arm
anneals its rates to nothing over its own ten epochs, exactly as the
full runs will anneal over theirs, so the number named here is where a
schedule starts rather than where it stays. Two consequences worth
holding on to when reading the result: the late epochs train slowly, so
overfitting shows up muted compared with a flat rate, and the eventual
training budget is not yet known - if it lands far from ten epochs, the
winning pair is worth confirming once at that length rather than
assumed to carry.

**Scored on the metric that is reported, not on the loss.** Every epoch
is kept and scored afterwards. The trainer's own validation loss put
the peak four epochs later than the instance score did, which is how
the instability went unnoticed in the first place.

Examples
--------
Probe the grid:
    $ python scripts/compare_learning_rates.py

Then score what it produced:
    $ python scripts/evaluate_checkpoint.py \\
        --checkpoint-glob "checkpoints/lr_probe/checkpoints/*/epoch-*.pt" \\
        --out checkpoints/lr_probe/lr_curves.json

Rehearse the path in minutes:
    $ python scripts/compare_learning_rates.py --n-train 6 --epochs 1
"""
import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from materials_vision.logging_config import setup_logging
from materials_vision.provenance import run_provenance
from materials_vision.training import build_loader, build_source, train_run

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/lr_probe")

BASE_MODEL = "vit_l_lm"

# The rate in use for the correction, and three times it. Both are
# above the library's default, which is where a correction starting
# from zero has to be; the question is only how far above.
LORA_RATES = (1e-4, 3e-4)

# The library's default for the decoder, and three times it. Nothing
# higher: the symptom being chased is a decoder that already learns too
# fast, and the rate it ran at was ten times the upper end of this
# range.
DECODER_RATES = (1e-5, 3e-5)

# Matched to the seed of the first baseline run, so that any arm can be
# read against it.
SEED = 20260907

N_EPOCHS = 10

CHECKPOINT_EVERY_EPOCHS = 1


def run_name(lora_rate: float, decoder_rate: float) -> str:
    """Directory name for one point of the grid.

    Parameters
    ----------
    lora_rate, decoder_rate : float

    Returns
    -------
    str
    """
    parts = (
        f"lora{lora_rate:.0e}".replace("-0", "-"),
        f"dec{decoder_rate:.0e}".replace("-0", "-"),
    )
    return "_".join(parts)


def planned_pairs(
    lora_rates: list[float], decoder_rates: list[float]
) -> list[tuple[float, float]]:
    """The grid, in the order the runs should happen.

    The decoder's rate varies fastest, so an interrupted session still
    leaves both of its values tried at one correction rate rather than
    one value tried at both.

    Parameters
    ----------
    lora_rates, decoder_rates : list of float

    Returns
    -------
    list of tuple
    """
    return list(itertools.product(lora_rates, decoder_rates))


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


def is_finished(
    pair: tuple[float, float], args: argparse.Namespace
) -> bool:
    """Whether this pair's run already produced its last snapshot.

    Parameters
    ----------
    pair : tuple of float
    args : argparse.Namespace

    Returns
    -------
    bool
    """
    last = (args.save_root / "checkpoints" / run_name(*pair)
            / f"epoch-{args.epochs}.pt")
    return last.exists()


def execute(pair: tuple[float, float], args: argparse.Namespace) -> None:
    """Train one run at one pair of rates.

    Parameters
    ----------
    pair : tuple of float
        The correction's rate and the decoder's.
    args : argparse.Namespace
    """
    lora_rate, decoder_rate = pair
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
        "%s: correction %.0e, decoder %.0e, %d training image(s), "
        "%d validation image(s).",
        run_name(*pair), lora_rate, decoder_rate,
        len(train_loader), len(val_loader),
    )
    train_run(
        run_name(*pair),
        model_type=BASE_MODEL,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=lora_rate,
        decoder_learning_rate=decoder_rate,
        early_stopping=None,
        save_every_kth_epoch=CHECKPOINT_EVERY_EPOCHS,
    )


def write_provenance(
    args: argparse.Namespace, pairs: list[tuple[float, float]]
) -> None:
    """Record what produced these checkpoints, next to them.

    Parameters
    ----------
    args : argparse.Namespace
    pairs : list of tuple
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / "lr_probe_provenance.json"
    destination.write_text(json.dumps({
        "provenance": run_provenance(),
        "base_model": BASE_MODEL,
        "runs": [
            {
                "name": run_name(*pair),
                "lora_learning_rate": pair[0],
                "decoder_learning_rate": pair[1],
            }
            for pair in pairs
        ],
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "early_stopping": None,
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
        "--lora-rates", type=float, nargs="+", default=list(LORA_RATES),
        help="Rates to try for the low-rank correction.",
    )
    parser.add_argument(
        "--decoder-rates", type=float, nargs="+",
        default=list(DECODER_RATES),
        help="Rates to try for the instance decoder.",
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Passes over the training split, the same for every arm.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Run seed; matched to the baseline arm it is read against.",
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
        help="Re-run arms that already finished.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Train one short run per point of the grid.

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

    pairs = planned_pairs(args.lora_rates, args.decoder_rates)
    write_provenance(args, pairs)
    for pair in pairs:
        if not args.force and is_finished(pair, args):
            logger.info("%s already finished; skipping.", run_name(*pair))
            continue
        execute(pair, args)

    logger.info(
        "Probe finished. Score the snapshots under %s; the winner is "
        "the pair whose peak is highest and whose decoder validation "
        "loss has not yet turned upward by epoch five.", args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
