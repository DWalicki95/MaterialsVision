#!/usr/bin/env python3
"""
Which pretrained checkpoint this foam segmentation should start from.

The backbone is adapted by low-rank corrections that touch a quarter of
a percent of the image encoder, which makes the encoder's pretraining
close to permanent: whatever it learned about what an image looks like
is what the run will keep. The instance decoder, by contrast, trains in
full, so its starting weights are largely recoverable. The choice of
base checkpoint therefore matters most through the encoder, and it is
not obvious which of two candidates wins.

**The electron microscopy generalist** matches the imaging modality.
These are greyscale electron images with the noise and contrast that
implies, which is what the foam micrographs are. Its instances, though,
are organelles: sparse, separate objects floating inside a cell.

**The light microscopy generalist** matches the structure instead. It
was adapted on densely packed cells that share their boundaries, which
is topologically what foam is - a tessellation of cells separated by
thin shared walls, where the whole difficulty is telling two touching
instances apart. Its images, though, come from an entirely different
physics.

Argument alone favours the first, on the reasoning above. This script
settles it by measurement instead, because a study that has already
taken two consequential decisions on argument rather than data should
not take a third.

**What is held fixed.** Everything but the base checkpoint: the same
images in the same order, the same adaptation rank, the same frozen
components, the same learning rate, the same seed, the same budget, and
no augmentation on either side. Early stopping is off, so both runs
spend the same number of steps.

**What decides the winner.** Instance F1 at a mask overlap of one half,
pooled over instances, measured on the validation split by the same
evaluation code the rest of the study uses. Not the training loss:
loss is what the model optimizes, and the question here is how well the
result segments.

This run doubles as the first end-to-end exercise of the pipeline, so
its early output is worth watching for two further things: whether the
loss descends at all, which is the standing check on the learning rate,
and whether the instances the decoder produces look plausible before
the watershed settings behind them are frozen for good.

Examples
--------
Run the pilot as specified:
    $ python scripts/run_p0_pilot.py

Rehearse the whole path in a few minutes, on a handful of images:
    $ python scripts/run_p0_pilot.py --n-train 8 --n-val 4 --epochs 1
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

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

DEFAULT_SAVE_ROOT = Path("checkpoints/p0")

# Registry names of the two candidates, without a decoder suffix: the
# matching instance decoder is a separate file that the library fetches
# alongside each of these on its own.
CANDIDATES = ("vit_l_em_organelles", "vit_l_lm")

# Five passes over the training split. Expressed as a budget in its own
# right rather than as a fraction of the full training budget, which is
# not known yet and which this pilot does not need to be commensurate
# with: the pilot picks a starting point, it does not take part in
# attributing anything.
N_EPOCHS = 5

# The archive serving the base weights times out often enough that one
# attempt is not enough to distinguish a bad moment from a bad name.
DOWNLOAD_ATTEMPTS = 3

DOWNLOAD_RETRY_WAIT_S = 20.0


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
        "--candidates", nargs="+", default=list(CANDIDATES),
        help="Registry names of the base checkpoints to compare.",
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Passes over the training split, the same for every run.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260907,
        help="Run seed, shared by every candidate so the runs pair up.",
    )
    parser.add_argument(
        "--n-train", type=int, default=0,
        help="Use this many training images; 0 means all of them.",
    )
    parser.add_argument(
        "--n-val", type=int, default=0,
        help="Use this many validation images; 0 means all of them.",
    )
    return parser.parse_args(argv)


def fetch_base_weights(candidates: Sequence[str]) -> None:
    """Download every candidate's weights before any training starts.

    Base weights are pulled from a public archive on first use, and
    that archive times out often enough to matter. Left to happen
    lazily, the download for the second candidate would be attempted an
    hour into the pilot, after the first candidate had already spent
    that hour on the GPU - and a momentary network failure would throw
    the whole run away. Fetching everything up front means a bad
    connection costs seconds instead.

    Each candidate is retried a few times, because the failure this
    guards against is transient by nature.

    Parameters
    ----------
    candidates : sequence of str
        Registry names of the base checkpoints.

    Raises
    ------
    RuntimeError
        If a candidate's weights cannot be fetched.
    """
    from micro_sam.util import models

    registry = models()
    for candidate in candidates:
        # The instance decoder ships as its own file under a suffixed
        # name; a candidate that has one is useless without it.
        wanted = [candidate]
        decoder_name = f"{candidate}_decoder"
        if decoder_name in registry.registry:
            wanted.append(decoder_name)
        for name in wanted:
            _fetch_with_retries(registry, name)


def _fetch_with_retries(registry, name: str) -> None:
    """Fetch one registry entry, retrying a transient failure.

    Parameters
    ----------
    registry : pooch.Pooch
    name : str

    Raises
    ------
    RuntimeError
        If every attempt fails.
    """
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            path = registry.fetch(name, progressbar=False)
            logger.info("Base weights ready: %s -> %s.", name, path)
            return
        except Exception as failure:  # noqa: BLE001 - reported below
            logger.warning(
                "Fetching %s failed on attempt %d of %d: %s",
                name, attempt, DOWNLOAD_ATTEMPTS, failure,
            )
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Could not fetch base weights {name!r} after "
                    f"{DOWNLOAD_ATTEMPTS} attempt(s)."
                ) from failure
            time.sleep(DOWNLOAD_RETRY_WAIT_S)


def spaced_indices(n_available: int, n_wanted: int) -> Optional[list[int]]:
    """Evenly spaced positions, or every one of them.

    Spaced rather than taken from the front so that a shortened
    rehearsal still crosses both microscopes and both scale bins, which
    sit in different stretches of the ordering.

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


def run_candidate(
    model_type: str, args: argparse.Namespace
) -> None:
    """Train one candidate on the shared budget.

    Loaders are rebuilt per candidate so that each run starts from the
    same loader state rather than inheriting a partially consumed one,
    and the seed is reset for the same reason.

    Parameters
    ----------
    model_type : str
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
        "%s: %d training image(s), %d validation image(s).",
        model_type, len(train_loader), len(val_loader),
    )
    train_run(
        model_type,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_root=args.save_root,
        lora_learning_rate=LORA_LEARNING_RATE,
        decoder_learning_rate=DECODER_LEARNING_RATE,
        early_stopping=None,
    )


def write_provenance(args: argparse.Namespace) -> Path:
    """Record what produced these checkpoints, next to them.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Path
        The file written.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    destination = args.save_root / "p0_provenance.json"
    payload = {
        "provenance": run_provenance(),
        "candidates": list(args.candidates),
        "epochs": args.epochs,
        "seed": args.seed,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "decoder_learning_rate": DECODER_LEARNING_RATE,
        "split": str(args.split),
        "manifest": str(args.manifest),
    }
    destination.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s.", destination)
    return destination


def main(argv: Optional[list[str]] = None) -> int:
    """Train each candidate on the same budget.

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
    fetch_base_weights(args.candidates)
    for model_type in args.candidates:
        run_candidate(model_type, args)
    logger.info(
        "Both runs finished. Checkpoints are under %s; the winner is "
        "decided by evaluating them on the validation split, not from "
        "the losses above.", args.save_root,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
