#!/usr/bin/env python3
"""
CLI entry point for the data inventory manifest builder.

Thin wrapper around ``data_prep.inventory``: parses arguments, resolves
the configuration, calls the pipeline, and translates its result into
the appropriate process exit code. All domain logic lives in
``data_prep/inventory/``

Examples
--------
Validate without writing anything:
    $ python scripts/build_inventory_manifest.py --dry-run

Build and freeze v1:
    $ python scripts/build_inventory_manifest.py \\
          --manifest-version v1

Debug VAB sidecar mapping only:
    $ python scripts/build_inventory_manifest.py \\
          --series VAB --dry-run --verbose
"""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from data_prep.inventory.config import ConfigError, load_config
from data_prep.inventory.issues import IssueLevel, ManifestBuildAborted
from data_prep.inventory.manifest import build_manifest, write_artifacts
from data_prep.inventory.reporting import (write_dataset_summary,
                                           write_run_metadata,
                                           write_thumbnails,
                                           write_validation_report)
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERRORS = 1
EXIT_FATAL = 2

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "data_prep" / "inventory" / "inventory_config.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen data inventory manifest from SEM "
            "images, Label Studio exports and SEM sidecar metadata."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "Validate without writing anything:\n"
            "  python scripts/build_inventory_manifest.py --dry-run"
            "\n\n"
            "Build and freeze v1:\n"
            "  python scripts/build_inventory_manifest.py "
            "--manifest-version v1\n\n"
            "Debug VAB sidecar mapping only:\n"
            "  python scripts/build_inventory_manifest.py "
            "--series VAB --dry-run --verbose\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Path to the inventory configuration YAML (default: "
            f"{DEFAULT_CONFIG})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    parser.add_argument(
        "--manifest-version",
        type=str,
        default=None,
        help="Override the configured manifest version tag.",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=None,
        help=(
            "Restrict the run to one configured series; repeatable. "
            "Default: all configured sources."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full validation but write no artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing manifest of the same "
            "version."
        ),
    )
    parser.add_argument(
        "--thumbnails",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Save N visual-verification thumbnails of the detected "
            "non-image region (0 disables, the default)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def _apply_overrides(config, args):
    """Apply CLI overrides on top of a loaded ``InventoryConfig``.

    Parameters
    ----------
    config : InventoryConfig
    args : argparse.Namespace

    Returns
    -------
    InventoryConfig

    Raises
    ------
    ConfigError
        If ``--series`` names a series not present in ``config``.
    """
    updates = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.manifest_version is not None:
        updates["manifest_version"] = args.manifest_version
    if args.series:
        selected = set(args.series)
        known = {s.series for s in config.sources}
        missing = selected - known
        if missing:
            raise ConfigError(
                f"--series requested unknown series: "
                f"{sorted(missing)} (known: {sorted(known)})"
            )
        updates["sources"] = tuple(
            s for s in config.sources if s.series in selected
        )
    if updates:
        config = dataclasses.replace(config, **updates)
    return config


def main() -> int:
    """CLI entry point.

    Returns
    -------
    int
        Process exit code: 0 success, 1 completed with ERROR-level
        issues (some images dropped), 2 fatal (no artifacts written).
        See the inventory plan, section 8.3.
    """
    args = build_parser().parse_args()
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    try:
        config = load_config(args.config)
        config = _apply_overrides(config, args)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
        return EXIT_FATAL

    try:
        result = build_manifest(config)
    except ManifestBuildAborted as e:
        logger.error(
            "Run aborted: %d fatal issue(s). No artifacts written.",
            len(e.fatal_issues),
        )
        return EXIT_FATAL

    n_errors = sum(
        1 for i in result.issues if i.level == IssueLevel.ERROR
    )
    n_warnings = sum(
        1 for i in result.issues if i.level == IssueLevel.WARNING
    )
    logger.info(
        "Processed: %d row(s), %d rejection(s), %d error(s), "
        "%d warning(s).",
        len(result.manifest), len(result.rejected), n_errors,
        n_warnings,
    )

    if args.dry_run:
        logger.info("Dry run: no artifacts written.")
        return EXIT_ERRORS if n_errors else EXIT_OK

    try:
        paths = write_artifacts(
            result, config, overwrite=args.overwrite
        )
    except FileExistsError as e:
        logger.error(str(e))
        return EXIT_FATAL

    write_validation_report(result, config)
    write_dataset_summary(result, config)
    write_run_metadata(result, config, paths["manifest"])
    if args.thumbnails > 0:
        write_thumbnails(result, config, args.thumbnails)

    return EXIT_ERRORS if n_errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
