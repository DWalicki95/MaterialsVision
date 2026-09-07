#!/usr/bin/env python3
"""
Which learning rate makes the low-rank adaptation actually learn.

Both frameworks this pipeline is built on default to 1e-5, and that
default was chosen for a different regime: fine-tuning whole decoders,
tens of millions of parameters moving at once. Here the image encoder
is frozen except for low-rank corrections that start at exactly zero
and hold well under a million parameters. A correction starting from
zero with orders of magnitude fewer parameters moves far less per step
at the same rate, so the inherited default is not evidence about this
setup. Neither is the 1e-4 reported by work using low-rank adaptation
on the same kind of material, since that work had no instance decoder
attached. The question has to be settled by measurement.

**What the comparison holds fixed.** Everything except the rate: the
same images, the same subset of them, the same adaptation rank, the
same frozen parts, the same seed, the same number of epochs, no
augmentation on either side. Early stopping is switched off on purpose
- a run that halted early would produce a shorter curve, and the two
curves have to be read side by side.

**What is read out.** The validation metric after every epoch. One
number per epoch is enough to answer the only question being asked,
which is whether the curve descends at all: a rate too low to move the
correction produces noise around a constant, and no amount of detail
about its composition changes that reading.

**How it is read out.** The trainer keeps its per-epoch metric to
itself - it goes to a progress bar and to a checkpoint, neither of
which is a usable interface. But the trainer does save a checkpoint it
calls "latest" after every epoch, and hands that epoch's validation
metric to the call. Wrapping that one method records the whole curve
without reproducing any of the trainer's internals, which would break
the next time the library changes.

**A subset, not the whole training set.** Fifty-odd images, sampled
across both microscopes so neither is missing, because the question is
about the shape of a curve rather than about the quality of a model.
A rate that cannot descend on a small set will not descend on a large
one, and the small set costs minutes instead of hours.

**Why the geometry correction is installed.** Without it the library
resizes every three-channel image to a fixed 341 by 1024 - computed
from the batch and channel axes of the tensor instead of its height and
width - which squeezes the content vertically by a factor of two and a
quarter. A model trained that way learns a directional distortion the
material does not have, so a curve measured without the correction
describes a model nobody intends to train.

Examples
--------
Compare the two candidate rates:
    $ python scripts/compare_learning_rates.py

Compare a different set of rates:
    $ python scripts/compare_learning_rates.py --rates 3e-5 1e-4 3e-4
"""
import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.data.dataset import (InstanceSegmentationDataset,
                                           build_label_transform)
from materials_vision.logging_config import setup_logging
from materials_vision.sam_geometry import (patch_resize_longest_side,
                                           verify_preprocess_geometry)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAVE_ROOT = Path("checkpoints/lr_comparison")

A_MIN_FRAGMENT_PX2 = 388.43

DEFAULT_RATES = (1e-5, 1e-4)

# Training images per microscope. The set is dominated by one of them,
# so an evenly spaced sample of the whole would leave the other with a
# handful of images and no way to tell a rate that fails on it from one
# that fails everywhere.
N_TRAIN_PER_MICROSCOPE = {"M1": 32, "M2": 16}

N_VAL_PER_MICROSCOPE = {"M1": 16, "M2": 8}

N_EPOCHS = 10

MODEL_TYPE = "vit_l"

N_OBJECTS_PER_BATCH = 25

# Low-rank corrections on the query and value projections of every
# attention block, with the prompt encoder and the mask decoder left
# untouched. The instance decoder trains in full.
PEFT_KWARGS = {
    "rank": 8,
    "update_matrices": ["q", "v"],
    "attention_layers_to_update": [],
    "quantize": False,
}

FREEZE_PARTS = ["prompt_encoder", "mask_decoder"]

NUM_WORKERS = 4

# Name the trainer uses for the checkpoint it writes after every epoch,
# as opposed to the one it writes only on an improvement. Recording
# both would put two entries in the curve for every epoch that improved
# on its predecessor.
PER_EPOCH_CHECKPOINT = "latest"


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
        "--rates", type=float, nargs="+", default=list(DEFAULT_RATES),
        help="Learning rates to compare.",
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Epochs per run; the same for every rate.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260907,
        help="Run seed, shared by every rate so the runs are paired.",
    )
    return parser.parse_args(argv)


def select_by_microscope(
    source: SampleSource, quotas: dict[str, int]
) -> list[int]:
    """Take an evenly spaced sample from each microscope's images.

    Evenly spaced rather than drawn at random, so that re-running the
    comparison later compares the same images, and so that each quota
    spreads across the formulations rather than landing on whichever
    ones happen to sit together in the ordering.

    Parameters
    ----------
    source : SampleSource
    quotas : dict
        How many images to take per microscope.

    Returns
    -------
    list of int
        Indices into the source, in its own order.

    Raises
    ------
    ValueError
        If a microscope has fewer images than its quota asks for.
    """
    chosen: list[int] = []
    for microscope, quota in quotas.items():
        available = [
            record.index for record in source.records
            if record.microscope == microscope
        ]
        if len(available) < quota:
            raise ValueError(
                f"Microscope {microscope} has {len(available)} image(s), "
                f"which is fewer than the {quota} asked for."
            )
        stride = len(available) / quota
        chosen.extend(
            available[int(position * stride)] for position in range(quota)
        )
    return sorted(chosen)


def build_loader(
    split_csv: Path, manifest_csv: Path, subset: str,
    quotas: dict[str, int], run_seed: int,
) -> DataLoader:
    """Build one loader over an evenly sampled slice of a split subset.

    No augmentation on either side of the comparison: the question is
    about the learning rate, and a transformation that fired on one run
    and not the other would answer a different one.

    Parameters
    ----------
    split_csv, manifest_csv : Path
    subset : str
        ``"train"`` or ``"validation"``.
    quotas : dict
        Images per microscope.
    run_seed : int

    Returns
    -------
    DataLoader
    """
    split = load_split(split_csv, subset=subset)
    manifest = read_manifest(manifest_csv)
    source = SampleSource(
        split, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )
    dataset = InstanceSegmentationDataset(
        source,
        label_transform=build_label_transform(),
        transform=None,
        run_seed=run_seed,
    )
    indices = select_by_microscope(source, quotas)
    logger.info(
        "%s: %d of %d image(s), %s.",
        subset, len(indices), len(source),
        ", ".join(f"{key} {value}" for key, value in quotas.items()),
    )
    return DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=(subset == "train"),
        num_workers=NUM_WORKERS,
    )


@contextmanager
def recording_validation():
    """Collect the validation metric the trainer reports each epoch.

    Yields
    ------
    list of float
        Filled in as training proceeds, one entry per finished epoch.
    """
    from micro_sam.training.joint_sam_trainer import JointSamTrainer

    curve: list[float] = []
    original = JointSamTrainer.save_checkpoint

    def recording(self, name, current_metric, best_metric, **kwargs):
        if name == PER_EPOCH_CHECKPOINT:
            curve.append(float(current_metric))
        return original(self, name, current_metric, best_metric, **kwargs)

    JointSamTrainer.save_checkpoint = recording
    try:
        yield curve
    finally:
        JointSamTrainer.save_checkpoint = original


def run_one_rate(
    rate: float, args: argparse.Namespace, device: torch.device
) -> list[float]:
    """Train once at one rate and return its validation curve.

    Parameters
    ----------
    rate : float
    args : argparse.Namespace
    device : torch.device

    Returns
    -------
    list of float
        Validation metric after each epoch, lower being better.
    """
    import micro_sam.training as sam_training

    torch.manual_seed(args.seed)
    train_loader = build_loader(
        args.split, args.manifest, "train",
        N_TRAIN_PER_MICROSCOPE, args.seed,
    )
    val_loader = build_loader(
        args.split, args.manifest, "validation",
        N_VAL_PER_MICROSCOPE, args.seed,
    )

    logger.info("Training at lr = %.0e for %d epoch(s).", rate, args.epochs)
    with recording_validation() as curve:
        sam_training.train_sam(
            name=f"lr_{rate:.0e}",
            model_type=MODEL_TYPE,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.epochs,
            # Off on purpose: a run that stopped early would leave a
            # curve shorter than the one it is compared against.
            early_stopping=None,
            n_objects_per_batch=N_OBJECTS_PER_BATCH,
            with_segmentation_decoder=True,
            freeze=list(FREEZE_PARTS),
            device=device,
            lr=rate,
            peft_kwargs=dict(PEFT_KWARGS),
            save_root=str(args.save_root),
        )
    return curve


def report(curves: dict[float, list[float]]) -> None:
    """Write the curves side by side and say what they show.

    Parameters
    ----------
    curves : dict
        Validation curve per learning rate.
    """
    n_epochs = max(len(curve) for curve in curves.values())
    header = "  ".join(f"{epoch + 1:>6d}" for epoch in range(n_epochs))
    logger.info("epoch      %s", header)
    for rate, curve in curves.items():
        values = "  ".join(f"{value:6.3f}" for value in curve)
        logger.info("lr %.0e  %s", rate, values)

    for rate, curve in curves.items():
        if len(curve) < 2:
            continue
        drop = curve[0] - curve[-1]
        spread = max(curve) - min(curve)
        verdict = (
            "descends" if drop > spread / 2 else "flat, within its own noise"
        )
        logger.info(
            "lr %.0e: %.3f -> %.3f, change %+.3f, spread %.3f - %s.",
            rate, curve[0], curve[-1], -drop, spread, verdict,
        )


def main(argv: Optional[list[str]] = None) -> int:
    """Train once per learning rate and compare the curves.

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
    device = torch.device("cuda")
    logger.info("Training on %s.", torch.cuda.get_device_name(0))

    patch_resize_longest_side()
    for geometry_px, content_px in verify_preprocess_geometry().items():
        logger.info(
            "Content %dx%d reaches the encoder as %dx%d.",
            geometry_px[0], geometry_px[1], content_px[0], content_px[1],
        )

    curves = {
        rate: run_one_rate(rate, args, device) for rate in args.rates
    }
    report(curves)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
