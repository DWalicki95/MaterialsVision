#!/usr/bin/env python3
"""
Fix the TRAIN images the post-processing is calibrated on.

Reads the split and nothing else - no image, no mask, no prediction - so
the sample is settled before any result could influence it. Every image
of the two small materials is taken and 45 of the dominant one, drawn
in proportion to its formulations and scale bins; close-ups are left
out. The file written here is what every later calibration step reads.

Examples
--------
    $ python scripts/select_calibration_sample.py
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from materials_vision.evaluation.calibration_sample import (
    N_SUBSAMPLED, SAMPLE_SEED, SUBSAMPLED_MATERIAL, select_calibration_sample)
from materials_vision.logging_config import setup_logging
from materials_vision.training import build_source

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_OUT = Path("checkpoints/postprocessing/calibration_sample.json")

SUBSET = "train"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing sample. Only before any calibration "
             "has read it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Draw the sample and write it down.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    if args.out.exists() and not args.force:
        logger.error(
            "%s already exists. The sample is fixed once; pass --force "
            "only if nothing has been calibrated on it yet.", args.out,
        )
        return EXIT_FAILED

    source = build_source(args.split, args.manifest, SUBSET)
    chosen = select_calibration_sample(source.records)
    counts = Counter(record.material for record in chosen)
    payload = {
        "subset": SUBSET,
        "rule": (
            f"every scored image of every material except "
            f"{SUBSAMPLED_MATERIAL}; {N_SUBSAMPLED} of "
            f"{SUBSAMPLED_MATERIAL}, stratified by formulation and scale "
            f"bin; close-ups excluded"
        ),
        "seed": SAMPLE_SEED,
        "split": str(args.split),
        "manifest": str(args.manifest),
        "n_images": len(chosen),
        "n_per_material": dict(sorted(counts.items())),
        "images": [
            {
                "index": record.index,
                "image_id": record.image_id,
                "material": record.material,
                "formulation": record.formulation,
                "scale_bin": record.scale_bin,
                "microscope": record.microscope,
            }
            for record in chosen
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Calibration sample: %d image(s), %s. Wrote %s.",
        len(chosen), dict(sorted(counts.items())), args.out,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
