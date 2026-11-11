#!/usr/bin/env python3
"""
Average the low-rank corrections of several snapshots of one run.

The validation curves of every run of this study wobble from epoch to
epoch, and the wobble follows the image order of the seed rather than
the setting under study: with one image per step, the images that end
an epoch pull the encoder their way. A single snapshot is therefore one
noisy draw from the end of a trajectory. Averaging the corrections of
several snapshots asks whether that noise costs segmentation quality,
before anything is changed in training to remove it.

**The average is exact, not an approximation.** A correction changes a
projection by the product of its two factors, ``B @ A``. Averaging the
factors separately would not average the products, since the mean of
products is not the product of means. Stacking them does:

.. code-block:: text

    A_cat = [A_1; ...; A_k]            (k*r x d)
    B_cat = [B_1/k, ..., B_k/k]        (d x k*r)
    B_cat @ A_cat = (B_1 A_1 + ... + B_k A_k) / k

The result is a correction of rank ``k*r``. The evaluation reads the
rank from the weights, so it loads without being told.

**Only snapshots that differ in their corrections alone.** Everything
else - the encoder's own weights and the whole instance decoder with
its batch statistics - must be identical across the snapshots, which
holds for a run that adapted only low-rank corrections with the decoder
frozen. A run that also trained the decoder is refused: averaging
decoders would need its batch statistics recomputed, and that is a
different experiment.

The output is named without ``epoch-`` on purpose: its figures then go
to the JSON report only, never onto a run's per-epoch curve in MLflow.

Examples
--------
Average epochs 25 to 38 of the LoRA-only run, then score it:
    $ PYTHONPATH=. python scripts/peft_sam/average_lora_snapshots.py \\
        --run-dir checkpoints/adhoc/checkpoints/<run> --epochs 25-38 \\
        --out checkpoints/adhoc/checkpoints/<run>/avg-25-38.pt
    $ PYTHONPATH=. python scripts/peft_sam/evaluate_checkpoint.py \\
        --checkpoint checkpoints/adhoc/checkpoints/<run>/avg-25-38.pt \\
        --center-threshold-sweep 0.25 0.30 0.35 --out average.json
"""
import argparse
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

FIRST_FACTOR = ".w_a_linear_"

SECOND_FACTOR = ".w_b_linear_"

EPOCH_IN_NAME = re.compile(r"epoch-\d+")

MIN_SNAPSHOTS = 2


def parse_epochs(text: str) -> list[int]:
    """Read which epochs to average.

    Parameters
    ----------
    text : str
        A range such as ``25-38``, epochs such as ``25,30,38``, or both
        joined by commas.

    Returns
    -------
    list of int
        Sorted, without repeats.

    Raises
    ------
    argparse.ArgumentTypeError
        For anything else, including a reversed range or an epoch below
        one.
    """
    epochs: set[int] = set()
    try:
        for part in text.split(","):
            if "-" in part:
                first, last = (int(v) for v in part.split("-"))
                if first > last:
                    raise ValueError(part)
                epochs.update(range(first, last + 1))
            else:
                epochs.add(int(part))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a range like 25-38 or a list like 25,30,38."
        ) from None
    if min(epochs) < 1:
        raise argparse.ArgumentTypeError("Epochs are numbered from 1.")
    return sorted(epochs)


def snapshot_paths(run_dir: Path, epochs: Sequence[int]) -> list[Path]:
    """The snapshot files of the given epochs.

    Parameters
    ----------
    run_dir : Path
    epochs : sequence of int

    Returns
    -------
    list of Path
        In epoch order. Not checked for existence.
    """
    return [run_dir / f"epoch-{epoch}.pt" for epoch in epochs]


def _is_factor(key: str) -> bool:
    return FIRST_FACTOR in key or SECOND_FACTOR in key


def _first_difference(
    states: Sequence[Mapping[str, Any]], skip_factors: bool
) -> Optional[str]:
    """The first entry that is not identical across all states.

    Parameters
    ----------
    states : sequence of mapping
    skip_factors : bool
        Leave the low-rank factors out of the comparison.

    Returns
    -------
    str or None
        The key, ``"<keys>"`` when the states do not even hold the same
        entries, or ``None`` when they agree.
    """
    keys = list(states[0])
    for other in states[1:]:
        if list(other) != keys:
            return "<keys>"
    for key in keys:
        if skip_factors and _is_factor(key):
            continue
        reference = states[0][key]
        for other in states[1:]:
            if not torch.equal(reference, other[key]):
                return key
    return None


def average_low_rank(
    model_states: Sequence[Mapping[str, torch.Tensor]]
) -> "OrderedDict[str, torch.Tensor]":
    """Average several snapshots' corrections into one exact correction.

    Parameters
    ----------
    model_states : sequence of mapping
        The ``model_state`` of each snapshot. Not modified.

    Returns
    -------
    OrderedDict
        The first snapshot's weights, with every pair of factors
        replaced by the stacked pair whose product is the mean of the
        snapshots' products.

    Raises
    ------
    ValueError
        For fewer than two snapshots, snapshots holding different
        entries or differing anywhere outside their factors, a factor
        without its partner, or no factors at all.
    """
    if len(model_states) < MIN_SNAPSHOTS:
        raise ValueError(
            f"Averaging needs at least {MIN_SNAPSHOTS} snapshots, got "
            f"{len(model_states)}."
        )
    differing = _first_difference(model_states, skip_factors=True)
    if differing == "<keys>":
        raise ValueError("The snapshots hold different entries.")
    if differing is not None:
        raise ValueError(
            f"{differing} differs between snapshots: something other "
            f"than the low-rank corrections trained, and averaging only "
            f"those would misrepresent the run."
        )

    first = model_states[0]
    count = len(model_states)
    factors = [key for key in first if _is_factor(key)]
    if not factors:
        raise ValueError("The snapshots hold no low-rank factors.")
    for key in factors:
        partner = (
            key.replace(FIRST_FACTOR, SECOND_FACTOR) if FIRST_FACTOR in key
            else key.replace(SECOND_FACTOR, FIRST_FACTOR)
        )
        if partner not in first:
            raise ValueError(f"{key} has no partner factor {partner}.")

    averaged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in first.items():
        if FIRST_FACTOR in key:
            averaged[key] = torch.cat(
                [state[key] for state in model_states], dim=0
            )
        elif SECOND_FACTOR in key:
            averaged[key] = torch.cat(
                [state[key] / count for state in model_states], dim=1
            )
        else:
            averaged[key] = value
    return averaged


def build_average(paths: Sequence[Path]) -> dict[str, Any]:
    """Load the snapshots and assemble the averaged checkpoint.

    Parameters
    ----------
    paths : sequence of Path

    Returns
    -------
    dict
        ``model_state`` and ``decoder_state`` - all the evaluation reads
        - and the paths averaged, for provenance. No optimizer state:
        the result cannot be trained on.

    Raises
    ------
    ValueError
        From :func:`average_low_rank`, or when the instance decoders
        differ between snapshots.
    """
    states = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for path in paths
    ]
    decoders = [state["decoder_state"] for state in states]
    differing = _first_difference(decoders, skip_factors=False)
    if differing is not None:
        raise ValueError(
            f"The instance decoder differs between snapshots (first at "
            f"{differing}): it trained in this run, and averaging it "
            f"would need its batch statistics recomputed."
        )
    return {
        "model_state": average_low_rank(
            [state["model_state"] for state in states]
        ),
        "decoder_state": decoders[0],
        "averaged_from": [str(path) for path in paths],
        "averaging": "mean of the low-rank products, factors stacked",
    }


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why the average may not be built as asked, or nothing.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    if len(args.epochs) < MIN_SNAPSHOTS:
        return f"Averaging needs at least {MIN_SNAPSHOTS} epochs."
    if EPOCH_IN_NAME.search(args.out.name):
        return (
            f"{args.out.name} looks like a training snapshot, and its "
            f"figures would land on the run's per-epoch curve. Name it "
            f"e.g. avg-25-38.pt."
        )
    if args.out.exists() and not args.force:
        return f"{args.out} exists; pass --force to replace it."
    missing = [
        path for path in snapshot_paths(args.run_dir, args.epochs)
        if not path.exists()
    ]
    if missing:
        return (
            f"{len(missing)} snapshot(s) missing, first {missing[0]}. The "
            f"run may keep only every k-th epoch."
        )
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory holding the run's epoch-<k>.pt snapshots.",
    )
    parser.add_argument(
        "--epochs", type=parse_epochs, required=True,
        help="Epochs to average, e.g. 25-38 or 25,30,38.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Where to write the averaged checkpoint.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Build and save the average.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    reason = refusal(args)
    if reason is not None:
        logger.error(reason)
        return EXIT_FAILED
    paths = snapshot_paths(args.run_dir, args.epochs)
    try:
        average = build_average(paths)
    except ValueError as error:
        logger.error("%s", error)
        return EXIT_FAILED

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(average, args.out)
    rank = next(
        int(value.shape[0]) for key, value in average["model_state"].items()
        if FIRST_FACTOR in key
    )
    logger.info(
        "Averaged %d snapshot(s) of %s (epochs %s) into %s, a correction "
        "of rank %d.", len(paths), args.run_dir.name,
        ",".join(map(str, args.epochs)), args.out, rank,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
