#!/usr/bin/env python3
"""
CLI entry point for freezing the pore size classes used in evaluation.

Recall is reported separately for four size classes so that a model
which finds the large pores and misses the small ones cannot hide
behind a single figure. The class boundaries are quartiles of the
annotated instance areas, and they are measured once here and then
read back by every evaluation: two policies compared on classes with
different boundaries would not be compared at all.

The population mirrors the one ``A_min_fragment`` was calibrated on:
TRAIN images of the frozen split, cropped to ``load_crop_bbox``, with
the handful of close-up images excluded. Measuring on TRAIN keeps the
evaluation sets out of a quantity the evaluation depends on. Areas
come from the frozen instance masks rather than from the Label Studio
export, because those masks are what training and evaluation actually
read.

Areas are converted to square micrometres with each image's own pixel
size before the quartiles are taken. The reason is in
``materials_vision.evaluation.size_bins``: in pixels a class would
encode which microscope took the picture as much as how large the pore
is, and microscope and foam family coincide in this dataset.

Examples
--------
Measure and print without writing anything:
    $ python scripts/calibrate_size_bins.py --dry-run

Freeze the classes:
    $ python scripts/calibrate_size_bins.py
"""
import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from materials_vision.data.instances import apply_content_crop, parse_crop_bbox
from materials_vision.evaluation.size_bins import calibrate_size_bins
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_OUTPUT = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

A_MIN_FRAGMENT_PX2 = 388.43

SIZE_BINS_ID = "size_bins_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="measure and report without writing the artifact",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "replace an existing artifact; the classes are frozen, so "
            "recalibrating invalidates comparisons with earlier runs"
        ),
    )
    return parser.parse_args(argv)


def select_calibration_images(
    manifest: pd.DataFrame, split: pd.DataFrame
) -> pd.DataFrame:
    """Pick the TRAIN images the classes are measured on.

    Parameters
    ----------
    manifest : pd.DataFrame
    split : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Manifest rows for evaluated TRAIN images, close-ups removed.
    """
    train = split[split["split"] == "train"]
    if "used" in train.columns:
        train = train[train["used"].astype(bool)]
    rows = manifest[manifest["image_id"].isin(set(train["image_id"]))]
    return rows[rows["scale_bin"] != "outlier"]


def collect_instance_areas_um2(rows: pd.DataFrame) -> np.ndarray:
    """Measure every annotated instance area, in square micrometres.

    Each mask is cropped to its content region first, because that is
    the geometry instances have by the time any evaluation sees them.

    Parameters
    ----------
    rows : pd.DataFrame
        Manifest rows of the calibration images.

    Returns
    -------
    np.ndarray
        One area per annotated instance, across all images.
    """
    areas: list[np.ndarray] = []
    for _, row in rows.iterrows():
        labels = tifffile.imread(row["mask_path"]).astype(np.int32)
        cropped = apply_content_crop(
            np.zeros(labels.shape, dtype=np.uint8), labels,
            parse_crop_bbox(row["load_crop_bbox"]),
            min_fragment_area_px2=A_MIN_FRAGMENT_PX2,
        )
        counts = np.bincount(
            cropped.labels.ravel(), minlength=cropped.n_instances + 1
        )[1:]
        areas.append(counts.astype(float) * row["pixel_size_um"] ** 2)
    if not areas:
        return np.empty(0, dtype=float)
    return np.concatenate(areas)


def file_sha256(path: Path) -> str:
    """Hash a file so the artifact records what it was built from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    bins, args: argparse.Namespace, n_images: int
) -> dict:
    """Assemble the record that travels with the frozen classes."""
    return {
        "size_bins_id": SIZE_BINS_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "split_path": str(args.split),
        "split_sha256": file_sha256(args.split),
        "population": {
            "subset": "train",
            "excludes": ["scale_outlier"],
            "includes_border_instances": True,
            "measured_after": "content crop to load_crop_bbox",
            "source": "frozen instance masks",
            "n_images": n_images,
        },
        "a_min_fragment_px2": A_MIN_FRAGMENT_PX2,
        "bins": bins.as_metadata(),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the size classes and write the frozen artifact.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    if args.output.exists() and not (args.overwrite or args.dry_run):
        logger.error(
            "%s already exists; the size classes are frozen. Pass "
            "--overwrite only if you accept that earlier runs are no "
            "longer comparable.", args.output,
        )
        return EXIT_FATAL

    manifest = pd.read_csv(args.manifest)
    split = pd.read_csv(args.split)
    rows = select_calibration_images(manifest, split)
    logger.info("Calibrating on %d TRAIN image(s).", len(rows))

    areas = collect_instance_areas_um2(rows)
    try:
        bins = calibrate_size_bins(areas)
    except ValueError as error:
        logger.error("Size class calibration failed: %s", error)
        return EXIT_FATAL

    metadata = build_metadata(bins, args, len(rows))
    logger.info(
        "Class populations: %s",
        np.bincount(bins.assign(areas), minlength=4).tolist(),
    )

    if args.dry_run:
        logger.info("Dry run: %s", json.dumps(metadata["bins"], indent=2))
        return EXIT_OK

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    logger.info("Frozen size classes written to %s", args.output)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
