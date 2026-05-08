#!/usr/bin/env python3
"""
Group material images and masks into separate dataset directories
based on user-defined material type groups and magnification filters.

Supported naming conventions
-----------------------------
Regular  : AS1A_40_29_jpg.rf.<hash>_image / _masks
AS2-like : 0ab7de9d-AS2_40_10_jpg.rf.<hash>_image / _masks
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Matches optional hex-uuid prefix, then MATERIAL_TYPE and MAGNIFICATION
_FILENAME_RE = re.compile(
    r"^(?:[a-f0-9]{8}-)?([A-Z]+\d+[A-Z]?)_(\d+)_"
)


def parse_filename(name):
    """Extract material type and magnification from a filename stem.

    Parameters
    ----------
    name : str
        Filename (stem or full name) to parse.

    Returns
    -------
    material : str or None
        Material type identifier, e.g. ``"AS1A"``, ``"AS2"``.
    magnification : str or None
        Magnification value as a string, e.g. ``"40"``.
    """
    match = _FILENAME_RE.match(name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def collect_group_files(library_path, material_types, magnifications):
    """Collect image and mask files that match the given criteria.

    Parameters
    ----------
    library_path : Path
        Root directory that contains image and mask files.
    material_types : list of str
        Material type identifiers to include, e.g. ``["AS1A", "AS1B"]``.
    magnifications : list of str
        Magnifications to include, e.g. ``["40", "50"]``.
        Pass an empty list to accept all magnifications.

    Returns
    -------
    matched : list of Path
        Absolute paths of files that satisfy both filters.
    skipped : int
        Number of files that could not be parsed.
    """
    mag_filter = set(magnifications)
    type_filter = set(material_types)
    matched = []
    skipped = 0

    for file_path in sorted(library_path.iterdir()):
        if not file_path.is_file():
            continue

        material, magnification = parse_filename(file_path.name)

        if material is None:
            logger.debug("Cannot parse: %s", file_path.name)
            skipped += 1
            continue

        if material not in type_filter:
            continue

        if mag_filter and magnification not in mag_filter:
            continue

        matched.append(file_path)

    return matched, skipped


def copy_files(files, output_dir, dry_run=False):
    """Copy files to the output directory, preserving original names.

    Parameters
    ----------
    files : list of Path
        Source file paths to copy.
    output_dir : Path
        Destination directory.
    dry_run : bool, optional
        When ``True``, log what would be copied without touching disk.
    """
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = output_dir / src.name
        if dry_run:
            logger.info("[dry-run] %s -> %s", src.name, output_dir)
        else:
            shutil.copy2(src, dst)
            logger.debug("Copied %s -> %s", src.name, output_dir)


def build_groups(raw_groups):
    """Parse raw group strings into lists of material type identifiers.

    Parameters
    ----------
    raw_groups : list of list of str
        Each inner list is one ``--group`` argument value, which may
        contain multiple space-separated or comma-separated tokens.

    Returns
    -------
    groups : list of list of str
        Cleaned list of material type lists.
    """
    groups = []
    for tokens in raw_groups:
        types = []
        for token in tokens:
            # Support comma-separated values within a single token
            types.extend(t.strip() for t in token.split(",") if t.strip())
        if types:
            groups.append(types)
    return groups


def run(library, output, groups, magnifications, dry_run):
    """Orchestrate the grouping workflow.

    Parameters
    ----------
    library : Path
        Source directory with all images and masks.
    output : Path
        Root output directory; sub-directories will be created per group.
    groups : list of list of str
        Each entry defines one dataset group's material types.
    magnifications : list of str
        Magnifications accepted for all groups; empty means accept all.
    dry_run : bool
        When ``True``, no files are written to disk.
    """
    if not library.exists():
        logger.error("Library directory does not exist: %s", library)
        return

    total_copied = 0

    for idx, material_types in enumerate(groups, start=1):
        group_label = f"dataset_{idx}"
        output_dir = output / group_label

        logger.info(
            "Group %d (%s): materials=%s, magnifications=%s",
            idx,
            group_label,
            material_types,
            magnifications if magnifications else "all",
        )

        files, skipped = collect_group_files(
            library, material_types, magnifications
        )

        if skipped:
            logger.warning(
                "Group %d: %d file(s) skipped (unparseable names).",
                idx,
                skipped,
            )

        if not files:
            logger.warning(
                "Group %d: no files matched the given criteria.", idx
            )
            continue

        copy_files(files, output_dir, dry_run=dry_run)

        logger.info(
            "Group %d: %d file(s) %s %s",
            idx,
            len(files),
            "would be copied to" if dry_run else "copied to",
            output_dir,
        )
        total_copied += len(files)

    logger.info(
        "Done. Total files %s: %d",
        "that would be copied" if dry_run else "copied",
        total_copied,
    )


def build_parser():
    """Build and return the argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Copy images and masks into separate dataset directories "
            "grouped by material type and filtered by magnification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "Two groups, magnifications 40 and 50:\n"
            "  python group_material_into_datasets.py \\\n"
            "      --library /data/library \\\n"
            "      --output /data/out \\\n"
            "      --group AS1A AS1B \\\n"
            "      --group AS2 AS3 \\\n"
            "      --magnifications 40 50\n"
        ),
    )
    parser.add_argument(
        "--library",
        required=True,
        help="Path to the source directory containing images and masks.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Root output directory. Sub-directories dataset_1, dataset_2, "
            "... are created automatically."
        ),
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        nargs="+",
        metavar="MATERIAL_TYPE",
        required=True,
        help=(
            "List of material types forming one dataset group. "
            "Repeat the flag for each group, e.g. "
            "--group AS1A AS1B --group AS2 AS3."
        ),
    )
    parser.add_argument(
        "--magnifications",
        nargs="+",
        metavar="MAG",
        default=[],
        help=(
            "Magnifications to include (e.g. 40 50). "
            "Applies to all groups. Omit to accept all magnifications."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned actions without copying any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main():
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    groups = build_groups(args.groups)
    if not groups:
        logger.error("No valid material groups provided.")
        return

    run(
        library=Path(args.library),
        output=Path(args.output),
        groups=groups,
        magnifications=args.magnifications,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
