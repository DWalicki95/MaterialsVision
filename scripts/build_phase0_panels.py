#!/usr/bin/env python3
"""
CLI entry point for rendering the Phase 0 review panels.

Every augmentation family is judged by eye before it may enter a
training comparison, and this produces the material that judgement is
made on: one figure per gallery image, family and strength setting,
together with the encoder's own view of the result and the values the
transformation drew. Reasoning behind the panels is in
``materials_vision.phase0.panels``; behind the settings, in
``materials_vision.phase0.levels``.

The run is a function of the frozen gallery, the frozen configuration
and one seed, so re-rendering reproduces the same panels. What changes
a panel is a change to a family's parameters, and that is exactly what
its fingerprint records - a verdict stored against the old fingerprint
does not carry over to the new panel.

**The preprocessing correction is installed and verified here**, not
assumed: a panel showing the encoder's view is worthless if the
geometry it shows is not the one the model receives.

Examples
--------
Render everything:
    $ python scripts/build_phase0_panels.py

Render one family while its parameters are still being tuned:
    $ python scripts/build_phase0_panels.py --families F5_septum \\
        --overwrite

Rehearse on two images per level:
    $ python scripts/build_phase0_panels.py --max-images 2 --dry-run
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.logging_config import setup_logging
from materials_vision.phase0.levels import review_levels
from materials_vision.phase0.panels import (PanelRecord, render_panel,
                                            write_index)
from materials_vision.phase0.preview import MODE_ISOTROPIC, MODES
from materials_vision.phase0.viewer import write_viewer
from materials_vision.provenance import run_provenance
from materials_vision.sam_geometry import (patch_resize_longest_side,
                                           verify_preprocess_geometry)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_GALLERY = Path(
    "/home/dwalicki/dane/galleries/golden_gallery_v1_metadata.json"
)

DEFAULT_OUTPUT_DIR = Path("/home/dwalicki/dane/faza0")

A_MIN_FRAGMENT_PX2 = 388.43

RUN_SEED = 20260905


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--families", nargs="*", default=None,
        help="family codes to render; all of them by default",
    )
    parser.add_argument(
        "--levels", nargs="*", default=None,
        help="level names to render; all of them by default",
    )
    parser.add_argument("--run-seed", type=int, default=RUN_SEED)
    parser.add_argument(
        "--preprocess", choices=MODES, default=MODE_ISOTROPIC,
        help=(
            "geometry the encoder's view is rendered at; the corrected "
            "one unless you are comparing it with the defect"
        ),
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="images per level; 0 renders every image of the family",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be rendered without writing anything",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "re-render into an existing review directory; verdicts "
            "already recorded stay, and those whose fingerprint "
            "changed return to the queue"
        ),
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    """Hash a file so the run records what it was built from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_levels(args: argparse.Namespace):
    """Levels this run renders, after the command line's filters.

    Returns
    -------
    tuple of ReviewLevel
    """
    levels = review_levels()
    if args.families:
        levels = tuple(
            level for level in levels if level.family in args.families
        )
    if args.levels:
        levels = tuple(
            level for level in levels if level.level in args.levels
        )
    return levels


def build_metadata(
    args: argparse.Namespace,
    gallery: dict,
    levels,
    records: list[PanelRecord],
    seconds: float,
) -> dict:
    """Assemble the record describing this rendering run."""
    return {
        "phase0_run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_seed": args.run_seed,
            "preprocess_mode": args.preprocess,
            "seconds": round(seconds, 1),
        },
        "gallery": {
            "gallery_id": gallery["gallery_id"],
            "rules_version": gallery["rules_version"],
            "metadata_path": str(args.gallery),
            "metadata_sha256": file_sha256(args.gallery),
        },
        "inputs": {
            "manifest_path": str(args.manifest),
            "manifest_sha256": file_sha256(args.manifest),
            "split_path": str(args.split),
            "split_sha256": file_sha256(args.split),
            "a_min_fragment_px2": A_MIN_FRAGMENT_PX2,
        },
        "levels": [
            {
                "key": level.key,
                "family": level.family,
                "level": level.level,
                "kind": level.kind,
                "fingerprint": level.fingerprint,
                "note": level.note,
                "repeats": level.repeats,
                "image_offset": level.image_offset,
                "image_stride": level.image_stride,
                "parameters": level.parameters,
            }
            for level in levels
        ],
        "n_panels": len(records),
        "provenance": run_provenance(),
    }


def render_all(
    args: argparse.Namespace,
    levels,
    families: dict[str, list[str]],
    source: SampleSource,
) -> list[PanelRecord]:
    """Render every panel of every selected level."""
    by_id = {
        record.image_id: index
        for index, record in enumerate(source.records)
    }
    cache: dict[str, object] = {}
    records: list[PanelRecord] = []

    for level in levels:
        image_ids = level.images(tuple(families.get(level.family, ())))
        if args.max_images:
            image_ids = image_ids[:args.max_images]
        for image_id in image_ids:
            if image_id not in by_id:
                logger.error(
                    "%s is in the gallery but not in TRAIN of this "
                    "split; skipping.", image_id,
                )
                continue
            if image_id not in cache:
                cache[image_id] = source.load(by_id[image_id])
            for repeat in range(level.repeats):
                records.append(render_panel(
                    cache[image_id], level,
                    run_seed=args.run_seed,
                    repeat=repeat,
                    output_dir=args.output_dir,
                    preprocess_mode=args.preprocess,
                ))
        logger.info(
            "%s: %d image(s) x %d repeat(s).",
            level.key, len(image_ids), level.repeats,
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Render the panels and write the review directory.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    index_path = args.output_dir / "panels.json"
    if index_path.exists() and not (args.overwrite or args.dry_run):
        logger.error(
            "%s already exists. Pass --overwrite to re-render; "
            "verdicts already recorded are kept, and any whose "
            "parameters changed return to the queue.", index_path,
        )
        return EXIT_FATAL

    patch_resize_longest_side()
    verify_preprocess_geometry()

    with open(args.gallery, encoding="utf-8") as handle:
        gallery = json.load(handle)
    families = gallery["families"]

    levels = selected_levels(args)
    if not levels:
        logger.error(
            "no level matches --families %s --levels %s",
            args.families, args.levels,
        )
        return EXIT_FATAL

    planned = sum(
        len(level.images(tuple(families.get(level.family, ()))))
        * level.repeats
        for level in levels
    )
    logger.info(
        "%d level(s), %d panel(s) to render at %s geometry.",
        len(levels), planned, args.preprocess,
    )
    if args.dry_run:
        for level in levels:
            images = level.images(
                tuple(families.get(level.family, ()))
            )
            logger.info(
                "  %-28s %s x%d  %s",
                level.key, len(images), level.repeats,
                level.fingerprint,
            )
        return EXIT_OK

    started = time.time()
    subset = load_split(args.split, "train")
    manifest = read_manifest(args.manifest)
    source = SampleSource(
        subset, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = render_all(args, levels, families, source)
    seconds = time.time() - started

    write_index(records, args.output_dir)
    metadata = build_metadata(
        args, gallery, levels, records, seconds
    )
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    viewer_path = write_viewer(args.output_dir)

    fallbacks = sum(1 for record in records if not record.applied)
    logger.info(
        "Rendered %d panel(s) in %.0f s (%.2f s each); %d did not "
        "fire. Families: %s.",
        len(records), seconds,
        seconds / max(1, len(records)), fallbacks,
        dict(Counter(record.family for record in records)),
    )
    logger.info(
        "Review with: python scripts/review_phase0.py --review-dir %s "
        "(viewer at %s)", args.output_dir, viewer_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
