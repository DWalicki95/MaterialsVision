#!/usr/bin/env python3
"""
CLI entry point for freezing the golden gallery of Phase 0.

Every augmentation family is judged by eye on the same set of training
images, and this builds that set. The choice is made by a rule over
measured properties rather than by hand: strata by microscope and scale
bin, and within each stratum the extremes of wall thickness, wall
contrast, pore density and pore size, hardest first, with one typical
image always kept. Reasoning is in ``materials_vision.phase0.gallery``.

The result is the fifth frozen artifact, after the manifest, the split,
the instance masks and the size classes. It is frozen for the same
reason they are: a family accepted on one set of images and a family
rejected on another have not been compared, and the reviewer's verdict
is only evidence if the evidence it was given can be named.

Nothing here is random. Rebuilding on the same manifest and split
yields the same gallery; a different one means the inputs changed.

Examples
--------
Measure and report without writing anything:
    $ python scripts/build_golden_gallery.py --dry-run

Freeze the gallery:
    $ python scripts/build_golden_gallery.py

Rehearse quickly on part of TRAIN:
    $ python scripts/build_golden_gallery.py --dry-run --n-images 80
"""
import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.logging_config import setup_logging
from materials_vision.phase0 import (FAMILY_EXCLUDED_BINS, FAMILY_SIZES,
                                     FORCED_IMAGES, GALLERY_RULES_VERSION,
                                     SELECTION_REASONS, STRATUM_QUOTAS,
                                     GalleryError, assign_families,
                                     check_coverage, gallery_table,
                                     measure_axes, select_gallery)
from materials_vision.phase0.gallery import THIN_WALL_PX

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_OUTPUT_DIR = Path("/home/dwalicki/dane/galleries")

GALLERY_ID = "golden_gallery_v1"

A_MIN_FRAGMENT_PX2 = 388.43


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
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--gallery-id", default=GALLERY_ID)
    parser.add_argument(
        "--n-images", type=int, default=0,
        help=(
            "measure an evenly spaced sample of TRAIN; 0 measures all "
            "of it. A gallery built on a sample is a rehearsal, not "
            "the artifact"
        ),
    )
    parser.add_argument(
        "--no-outlier", action="store_true",
        help="leave the diagnostic close-up out of the gallery",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the gallery without writing it",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "replace an existing gallery; verdicts recorded against "
            "the old one no longer describe the new one"
        ),
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    """Hash a file so the artifact records what it was built from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    args: argparse.Namespace,
    table: pd.DataFrame,
    assignment: dict[str, tuple[str, ...]],
    n_candidates: int,
) -> dict:
    """Assemble the record that travels with the frozen gallery."""
    return {
        "gallery_id": args.gallery_id,
        "rules_version": GALLERY_RULES_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "split_path": str(args.split),
        "split_sha256": file_sha256(args.split),
        "population": {
            "subset": "train",
            "n_candidates": n_candidates,
            "measured_after": "content crop to load_crop_bbox",
            "a_min_fragment_px2": A_MIN_FRAGMENT_PX2,
        },
        "rules": {
            "stratum_quotas": {
                f"{microscope}_{scale_bin}": quota
                for (microscope, scale_bin), quota
                in STRATUM_QUOTAS.items()
            },
            "selection_reasons": list(SELECTION_REASONS),
            "thin_wall_px": THIN_WALL_PX,
            "forced_images": dict(FORCED_IMAGES),
            "family_sizes": dict(FAMILY_SIZES),
            "family_excluded_bins": {
                family: sorted(bins)
                for family, bins in FAMILY_EXCLUDED_BINS.items()
            },
            "randomness": "none; selection is a function of the inputs",
        },
        "n_images": int(len(table)),
        "families": {
            family: list(image_ids)
            for family, image_ids in assignment.items()
        },
        "n_panels_estimated": sum(
            len(image_ids) for image_ids in assignment.values()
        ),
    }


def build_report(
    table: pd.DataFrame,
    assignment: dict[str, tuple[str, ...]],
    candidates: pd.DataFrame,
) -> str:
    """Write the gallery out in the form a person checks it in."""
    lines = [
        "# Golden gallery (Phase 0)",
        "",
        f"{len(table)} image(s), chosen from {len(candidates)} TRAIN "
        f"candidate(s) by rule, without randomness.",
        "",
        "## Coverage",
        "",
        "| microscope | scale_bin | geometry | images |",
        "|---|---|---|---:|",
    ]
    grouped = table.groupby(
        ["microscope", "scale_bin", "height_px", "width_px"]
    )
    for (microscope, scale_bin, height, width), rows in grouped:
        lines.append(
            f"| {microscope} | {scale_bin} | {height}x{width} | "
            f"{len(rows)} |"
        )

    lines += [
        "",
        "## Why each image is in",
        "",
        "| image_id | material | stratum | role | reason | walls "
        "[px] | contrast | pores/mm2 | d50 [um] |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['image_id']} | {row['material']} | "
            f"{row['microscope']}/{row['scale_bin']} | {row['role']} | "
            f"{row['reason']} | {row['wall_thickness_mean_px']} | "
            f"{row['wall_contrast']} | {row['density_per_mm2']} | "
            f"{row['pore_diameter_median_um']} |"
        )

    lines += [
        "",
        "## Review load",
        "",
        "| family | evaluated | close-ups | microscopes | scale bins |",
        "|---|---:|---:|---|---|",
    ]
    indexed = table.set_index("image_id")
    for family, image_ids in assignment.items():
        rows = indexed.loc[list(image_ids)]
        evaluated = rows[rows["scale_bin"] != "outlier"]
        lines.append(
            f"| {family} | {len(evaluated)} | "
            f"{len(rows) - len(evaluated)} | "
            f"{','.join(sorted(set(evaluated['microscope'])))} | "
            f"{','.join(sorted(set(evaluated['scale_bin'])))} |"
        )

    lines += [
        "",
        "## Selected against the population",
        "",
        "The gallery is meant to sit at the edges of the training set, "
        "not at its centre; the columns below are how far it does.",
        "",
        "| axis | candidates min | candidates median | candidates max "
        "| gallery min | gallery max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    axes = (
        "wall_thickness_mean_px",
        "wall_contrast",
        "density_per_mm2",
        "pore_diameter_median_um",
    )
    for axis in axes:
        whole = candidates[axis].dropna()
        picked = table[axis].dropna()
        lines.append(
            f"| {axis} | {whole.min():.2f} | {whole.median():.2f} | "
            f"{whole.max():.2f} | {picked.min():.2f} | "
            f"{picked.max():.2f} |"
        )
    return "\n".join(lines) + "\n"


def candidate_frame(axes) -> pd.DataFrame:
    """Lay the measured candidates out for the report's comparison."""
    return pd.DataFrame([vars(entry) for entry in axes])


def main(argv: list[str] | None = None) -> int:
    """Build the gallery and write the frozen artifact.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    csv_path = args.output_dir / f"{args.gallery_id}.csv"
    if csv_path.exists() and not (args.overwrite or args.dry_run):
        logger.error(
            "%s already exists; the gallery is frozen. Pass "
            "--overwrite only if you accept that verdicts recorded "
            "against it describe a different set of images.", csv_path,
        )
        return EXIT_FATAL

    subset = load_split(args.split, "train")
    manifest = read_manifest(args.manifest)
    source = SampleSource(
        subset, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )

    indices = None
    if args.n_images:
        step = max(1, len(source) // args.n_images)
        indices = list(range(0, len(source), step))[:args.n_images]
        logger.warning(
            "Measuring %d of %d TRAIN image(s); this is a rehearsal, "
            "not the artifact.", len(indices), len(source),
        )

    axes = measure_axes(source, indices)
    try:
        gallery = select_gallery(
            axes, include_outlier=not args.no_outlier
        )
        check_coverage(gallery)
        assignment = assign_families(gallery)
    except GalleryError as error:
        logger.error("Gallery construction failed: %s", error)
        return EXIT_FATAL

    table = gallery_table(gallery, assignment)
    logger.info(
        "Roles: %s", dict(Counter(table["role"]).most_common())
    )
    logger.info(
        "Panels to review: %d over %d family/families.",
        sum(len(ids) for ids in assignment.values()), len(assignment),
    )

    metadata = build_metadata(args, table, assignment, len(axes))
    report = build_report(table, assignment, candidate_frame(axes))

    if args.dry_run:
        logger.info("Dry run; nothing written.\n%s", report)
        return EXIT_OK

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    metadata_path = args.output_dir / f"{args.gallery_id}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    report_path = args.output_dir / f"{args.gallery_id}_report.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info(
        "Golden gallery written to %s, %s and %s.",
        csv_path, metadata_path, report_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
