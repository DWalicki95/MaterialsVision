#!/usr/bin/env python3
"""
CLI entry point for measuring the walls a synthetic one has to imitate.

One augmentation family draws a wall across a large pore, so that the
pore becomes two instances and the model gets an example of two pores
that touch. A wall drawn from invented numbers would teach the model
what an invented wall looks like. The two numbers that decide the
appearance - how many pixels across a wall is, and how far its
brightness sits above the pore beside it - are therefore measured on
the annotated images first.

The population mirrors the one the minimum fragment area and the pore
size classes were measured on: TRAIN images of the frozen split,
cropped to their content region, with the handful of close-ups left
out. Measuring on TRAIN keeps the evaluation sets out of a quantity
the training depends on, and the close-ups are three to thirteen times
finer than the rest, so their walls are not measured in comparable
pixels.

The result is printed, not written. The two figures belong in the
augmentation package's frozen parameters, next to every other number
that defines a family, and are copied there by hand so that changing
them is a visible edit rather than a re-run.

Examples
--------
Measure on the whole training set:
    $ python scripts/calibrate_septum.py

Measure on a quick sample while developing:
    $ python scripts/calibrate_septum.py --n-images 40
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from materials_vision.augmentation.walls import measure_walls, summarize_walls
from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v2/manifest_v2.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

A_MIN_FRAGMENT_PX2 = 432.0

# What the model sees relative to the source files: the longer side of
# every image is resized to the encoder's input width. Wall widths are
# reported at both scales because the augmentation draws in source
# pixels while the question of whether a wall survives at all is a
# question about the resized image.
WORKING_SCALE = 0.8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="measure an evenly spaced sample; 0 measures all of them",
    )
    return parser.parse_args(argv)


def measure(source: SampleSource, n_images: int) -> list:
    """Measure the walls of the chosen training images.

    Parameters
    ----------
    source : SampleSource
    n_images : int
        How many images to measure, evenly spaced; 0 measures all.

    Returns
    -------
    list of WallSample
    """
    step = 1 if n_images <= 0 else max(1, len(source) // n_images)
    samples = []
    for index in range(0, len(source), step):
        prepared = source.load(index)
        samples.append(
            measure_walls(prepared.image, prepared.labels)
        )
    return samples


def report(summary, samples: list) -> None:
    """Write the measured figures to the log."""
    working = tuple(
        round(value * WORKING_SCALE, 2) for value in summary.thickness_px
    )
    logger.info(
        "Measured %d wall-centre pixel(s) across %d image(s).",
        summary.n_ridge_px, summary.n_images,
    )
    logger.info(
        "Wall width in source pixels: %s",
        json.dumps(summary.thickness_percentiles),
    )
    logger.info(
        "Frozen width range: %.2f-%.2f source px, "
        "%.2f-%.2f at the scale the model sees",
        summary.thickness_px[0], summary.thickness_px[1],
        working[0], working[1],
    )
    contrasts = np.array(
        [sample.contrast for sample in samples if
         np.isfinite(sample.contrast)]
    )
    logger.info(
        "Wall contrast above the pore interior, as a share of each "
        "image's tonal range: median %.4f, p10 %.4f, p90 %.4f",
        summary.contrast,
        float(np.percentile(contrasts, 10)),
        float(np.percentile(contrasts, 90)),
    )
    logger.info(
        "Copy into SeptumConfig: thickness_px=(%.2f, %.2f), "
        "contrast=%.4f",
        summary.thickness_px[0], summary.thickness_px[1],
        summary.contrast,
    )


def main(argv: list[str] | None = None) -> int:
    """Measure the walls and report the figures to freeze.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    split = load_split(args.split, subset="train")
    manifest = read_manifest(args.manifest)
    source = SampleSource(
        split, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )
    samples = measure(source, args.n_images)

    try:
        summary = summarize_walls(samples)
    except ValueError as error:
        logger.error("Wall calibration failed: %s", error)
        return EXIT_FATAL

    report(summary, samples)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
