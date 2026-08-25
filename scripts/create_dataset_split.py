#!/usr/bin/env python3
"""
CLI entry point for the grouped TRAIN/VALIDATION/TEST dataset split.

Thin wrapper around ``data_prep.split``: parses arguments, resolves the
configuration, runs the search, calibrates ``A_min_fragment`` on TRAIN
and writes the artifacts. All domain logic lives in
``data_prep/split/``.

The split is frozen: it is produced once and then reused unchanged by
every run of the augmentation experiment, so writing over an existing
split_id requires an explicit ``--overwrite``.

This supersedes ``data_prep/split_dataset_into_subsets.py``, which
splits per image and therefore scatters images of one formulation
across sets. Since those images come from a single synthesis and are
strongly correlated, that inflates evaluation scores, and results
obtained on such a split are not comparable with this experiment.

Examples
--------
Search and report without writing anything:
    $ python scripts/create_dataset_split.py --dry-run

Freeze the split, skipping the slow A_min_fragment pass:
    $ python scripts/create_dataset_split.py --skip-fragment-area

Freeze the split in full:
    $ python scripts/create_dataset_split.py
"""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from data_prep.split.config import SplitConfigError, load_split_config
from data_prep.split.constraints import check_constraints
from data_prep.split.fragment_area import (FragmentAreaError,
                                           compute_min_fragment_area)
from data_prep.split.profiles import (SplitDataError,
                                      build_formulation_profiles,
                                      load_manifest)
from data_prep.split.reporting import (SplitExistsError,
                                       format_console_summary,
                                       write_split_metadata,
                                       write_split_report, write_split_table)
from data_prep.split.search import SplitSearchError, search_split
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONSTRAINTS = 1
EXIT_FATAL = 2

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "data_prep" / "split" / "split_config.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen, formulation-grouped TRAIN/VALIDATION/"
            "TEST split from the inventory manifest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "Search and report without writing anything:\n"
            "  python scripts/create_dataset_split.py --dry-run\n\n"
            "Freeze the split, skipping the slow A_min_fragment "
            "pass:\n"
            "  python scripts/create_dataset_split.py "
            "--skip-fragment-area\n\n"
            "Freeze the split in full:\n"
            "  python scripts/create_dataset_split.py\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Path to the split configuration YAML (default: "
            f"{DEFAULT_CONFIG})."
        ),
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default=None,
        help="Override the configured split identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the configured split seed. Changing it produces "
            "a different split and requires a new --split-id."
        ),
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Override the configured number of candidate splits.",
    )
    parser.add_argument(
        "--skip-fragment-area",
        action="store_true",
        help=(
            "Skip the A_min_fragment calibration (the slow pass: it "
            "rasterizes every TRAIN annotation). The split itself is "
            "unaffected."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the search and print the summary, write nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing split of the same id.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def _apply_overrides(config, args):
    """Apply CLI overrides on top of a loaded ``SplitConfig``.

    Parameters
    ----------
    config : SplitConfig
    args : argparse.Namespace

    Returns
    -------
    SplitConfig
    """
    updates = {}
    if args.split_id is not None:
        updates["split_id"] = args.split_id
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.n_candidates is not None:
        updates["n_candidates"] = args.n_candidates
    if args.skip_fragment_area:
        updates["min_fragment_area"] = None
    if updates:
        config = dataclasses.replace(config, **updates)
    return config


def main() -> int:
    """CLI entry point.

    Returns
    -------
    int
        Process exit code: 0 success, 1 the chosen split violates a
        hard condition (artifacts are still written, so the violation
        can be inspected, but the split must not be frozen), 2 fatal
        (nothing written).
    """
    args = build_parser().parse_args()
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    try:
        config = load_split_config(args.config)
        config = _apply_overrides(config, args)
    except SplitConfigError as e:
        logger.error("Configuration error: %s", e)
        return EXIT_FATAL

    try:
        manifest = load_manifest(config.manifest_path)
        profiles = build_formulation_profiles(manifest)
        result = search_split(profiles, config)
    except (SplitDataError, SplitSearchError) as e:
        logger.error("%s", e)
        return EXIT_FATAL

    violations = check_constraints(result.stats, config.constraints)
    if violations:
        logger.error(
            "The chosen split violates %d hard condition(s):",
            len(violations),
        )
        for violation in violations:
            logger.error("  - %s", violation)

    logger.info(
        "Split %s (cost %.4f):\n%s",
        config.split_id, result.cost, format_console_summary(result),
    )

    fragment_area = None
    if config.min_fragment_area is not None:
        try:
            fragment_area = compute_min_fragment_area(
                manifest, result.assignment, config.min_fragment_area
            )
        except FragmentAreaError as e:
            logger.error("A_min_fragment calibration failed: %s", e)
            return EXIT_FATAL

    if args.dry_run:
        logger.info("Dry run: no artifacts written.")
        return EXIT_CONSTRAINTS if violations else EXIT_OK

    try:
        table_path = write_split_table(
            manifest, result, config, overwrite=args.overwrite
        )
    except SplitExistsError as e:
        logger.error("%s", e)
        return EXIT_FATAL

    write_split_report(result, config, fragment_area)
    write_split_metadata(result, config, table_path, fragment_area)

    return EXIT_CONSTRAINTS if violations else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
