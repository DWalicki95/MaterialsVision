#!/usr/bin/env python3
"""
CLI entry point for the instance mask builder.

Rasterizes every annotated image's polygons into an instance label
file, so training reads masks instead of re-deriving them from the
Label Studio export on every run. All domain logic lives in
``data_prep/masks/``.

The masks are a frozen artifact: built once per annotation export and
then reused unchanged, which is why rebuilding over existing files
takes an explicit ``--overwrite``.

Examples
--------
Rasterize twenty images to check the wiring, writing elsewhere:
    $ python scripts/build_instance_masks.py --limit 20 \\
          --output-root /tmp/masks_smoke_test

Build the full set:
    $ python scripts/build_instance_masks.py

Rebuild after a new annotation export:
    $ python scripts/build_instance_masks.py --overwrite
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data_prep.annotations import AnnotationLookupError
from data_prep.inventory.config import ConfigError, load_config
from data_prep.masks.build import (MaskBuildError, build_masks, summarize,
                                   write_build_artifacts)
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = (
    REPO_ROOT / "data_prep" / "inventory" / "inventory_config.yaml"
)

DEFAULT_MANIFEST = Path(
    "/home/dwalicki/dane/manifests/v2/manifest_v2.csv"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize Label Studio polygons into instance mask files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "Smoke-test twenty images into a scratch directory:\n"
            "  python scripts/build_instance_masks.py --limit 20 "
            "--output-root /tmp/masks_smoke_test\n\n"
            "Build the full set:\n"
            "  python scripts/build_instance_masks.py\n\n"
            "Rebuild after a new annotation export:\n"
            "  python scripts/build_instance_masks.py --overwrite\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Configuration naming the Label Studio exports and the "
            f"mask output root (default: {DEFAULT_CONFIG})."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Manifest listing the images to rasterize (default: "
            f"{DEFAULT_MANIFEST})."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override the configured mask root. Masks are written to "
            "<root>/<series>/<image_id>_masks.tif."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N images, for smoke-testing the wiring.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing mask files that already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main() -> int:
    """CLI entry point.

    Returns
    -------
    int
        0 on success, 2 when nothing could be built.
    """
    args = build_parser().parse_args()
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    try:
        config = load_config(args.config)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
        return EXIT_FATAL

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return EXIT_FATAL

    output_root = args.output_root or config.mask_root
    exports = {
        source.series: source.label_studio_json
        for source in config.sources
    }
    manifest = pd.read_csv(args.manifest)
    logger.info(
        "Building masks for %d image(s) into %s.",
        len(manifest) if args.limit is None
        else min(args.limit, len(manifest)),
        output_root,
    )

    try:
        result = build_masks(
            manifest, exports, output_root,
            overwrite=args.overwrite, limit=args.limit,
        )
    except (MaskBuildError, AnnotationLookupError) as e:
        logger.error("%s", e)
        return EXIT_FATAL

    write_build_artifacts(result, args.manifest, exports)
    logger.info("Mask build summary:\n%s", summarize(result))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
