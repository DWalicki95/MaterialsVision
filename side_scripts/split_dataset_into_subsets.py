#!/usr/bin/env python3
"""
Split a flat directory of image/mask pairs into train/, test/,
and optionally post_train/ subdirectories using material-type
stratified sampling.

Naming conventions handled
--------------------------
Regular  : AS1A_40_29_jpg.rf.<hash>_image.jpg / _masks.tif
AS2-like : 0ab7de9d-AS2_40_10_jpg.rf.<hash>_image.jpg / _masks.tif

Split logic
-----------
1. post_train share is taken from the full set first (per material).
2. The remaining files are split into train and test according to
   the provided ratio (e.g. 80:20).

Example
-------
--post-train 10 --train 80 --test 20
  => 10 % -> post_train/
     72 % -> train/   (80 % of the remaining 90 %)
     18 % -> test/    (20 % of the remaining 90 %)
"""

import argparse
import logging
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(
    r"^(?:[a-f0-9]{8}-)?([A-Z]+\d+[A-Z]?)_(\d+)_"
)
_ROLE_RE = re.compile(r"_(image|masks)$")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_material(stem):
    """Extract material type identifier from a filename stem.

    Parameters
    ----------
    stem : str
        Filename stem (without extension), e.g.
        ``"AS1A_40_29_jpg.rf.abc123_image"``.

    Returns
    -------
    str or None
        Material type, e.g. ``"AS1A"``, or ``None`` if not parseable.
    """
    match = _FILENAME_RE.match(stem)
    return match.group(1) if match else None


def pair_key(stem):
    """Strip the trailing ``_image`` or ``_masks`` role suffix.

    Parameters
    ----------
    stem : str
        Filename stem containing a role suffix.

    Returns
    -------
    str
        Stem without the role suffix, used as a pairing key.
    """
    return _ROLE_RE.sub("", stem)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_pairs(directory):
    """Scan *directory* and return image/mask pairs grouped by material.

    Parameters
    ----------
    directory : Path
        Flat directory containing image and mask files.

    Returns
    -------
    pairs_by_material : dict[str, list[dict]]
        Keys are material type strings; values are lists of dicts with
        keys ``"image"`` and ``"masks"``, each holding a ``Path``.
    unparseable : int
        Count of files skipped because their name could not be parsed.
    incomplete : int
        Count of pair keys that had an image but no mask, or vice versa.
    """
    files_by_key = defaultdict(dict)
    unparseable = 0

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue

        stem = file_path.stem
        role_match = _ROLE_RE.search(stem)

        if not role_match:
            logger.debug(
                "No _image/_masks suffix, skipping: %s", file_path.name
            )
            unparseable += 1
            continue

        role = role_match.group(1)
        key = pair_key(stem)
        files_by_key[key][role] = file_path

    pairs_by_material = defaultdict(list)
    incomplete = 0

    for key, roles in files_by_key.items():
        material = parse_material(key)

        if material is None:
            logger.debug("Cannot parse material from: %s", key)
            unparseable += 1
            continue

        if "image" not in roles or "masks" not in roles:
            logger.warning("Incomplete pair (missing counterpart): %s", key)
            incomplete += 1
            continue

        pairs_by_material[material].append(roles)

    return pairs_by_material, unparseable, incomplete


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def _split_one_material(pairs, n_post, train_ratio, test_ratio):
    """Split a single material's pairs into three buckets.

    Parameters
    ----------
    pairs : list of dict
        Shuffled list of ``{"image": Path, "masks": Path}`` dicts.
    n_post : int
        Number of pairs to reserve for post_train.
    train_ratio : float
        Relative weight for train from the remainder.
    test_ratio : float
        Relative weight for test from the remainder.

    Returns
    -------
    post : list of dict
    train : list of dict
    test : list of dict
    """
    post = pairs[:n_post]
    remainder = pairs[n_post:]

    n_remaining = len(remainder)
    total_ratio = train_ratio + test_ratio
    n_train = math.floor(n_remaining * train_ratio / total_ratio)

    train = remainder[:n_train]
    test = remainder[n_train:]

    return post, train, test


def stratified_split(
    pairs_by_material,
    train_ratio,
    test_ratio,
    post_train_ratio,
    seed,
):
    """Split all pairs with per-material stratification.

    Parameters
    ----------
    pairs_by_material : dict[str, list[dict]]
        Output of :func:`collect_pairs`.
    train_ratio : float
        Relative weight for train split (e.g. ``80``).
    test_ratio : float
        Relative weight for test split (e.g. ``20``).
    post_train_ratio : float
        Fraction of the *total* set reserved for post_train (0–1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    splits : dict[str, list[dict]]
        Keys ``"train"``, ``"test"``, ``"post_train"``; values are
        lists of ``{"image": Path, "masks": Path}`` dicts.
    """
    rng = random.Random(seed)
    splits = {"train": [], "test": [], "post_train": []}

    for material in sorted(pairs_by_material):
        pairs = list(pairs_by_material[material])
        rng.shuffle(pairs)

        n = len(pairs)
        n_post = math.floor(n * post_train_ratio)

        post, train, test = _split_one_material(
            pairs, n_post, train_ratio, test_ratio
        )

        logger.info(
            "%-8s  total=%-4d  post_train=%-4d  train=%-4d  test=%d",
            material,
            n,
            len(post),
            len(train),
            len(test),
        )

        splits["train"].extend(train)
        splits["test"].extend(test)
        splits["post_train"].extend(post)

    return splits


# ---------------------------------------------------------------------------
# File moving
# ---------------------------------------------------------------------------


def move_pairs(pairs, output_dir, dry_run=False):
    """Move image/mask pairs into *output_dir*.

    Parameters
    ----------
    pairs : list of dict
        Each dict has ``"image"`` and ``"masks"`` ``Path`` values.
    output_dir : Path
        Destination directory.
    dry_run : bool, optional
        When ``True`` log planned moves without touching disk.
    """
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        for file_path in pair.values():
            dst = output_dir / file_path.name
            if dry_run:
                logger.info(
                    "[dry-run] move %s -> %s", file_path.name, output_dir
                )
            else:
                shutil.move(str(file_path), dst)
                logger.debug(
                    "Moved %s -> %s", file_path.name, output_dir
                )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_ratios(train, test, post_train):
    """Raise ``ValueError`` when split ratios are invalid.

    Parameters
    ----------
    train : float
        Relative train weight (must be > 0).
    test : float
        Relative test weight (must be > 0).
    post_train : float
        Fraction of total for post_train (0 <= value < 1).
    """
    if train <= 0 or test <= 0:
        raise ValueError("--train and --test must both be positive.")
    if not 0.0 <= post_train < 1.0:
        raise ValueError(
            "--post-train must be in [0, 100) (percentage of total)."
        )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(input_dir, train_ratio, test_ratio, post_train_pct, seed, dry_run):
    """Orchestrate the full split workflow.

    Parameters
    ----------
    input_dir : Path
        Flat directory containing image and mask files.
    train_ratio : float
        Relative weight for train (e.g. ``80``).
    test_ratio : float
        Relative weight for test (e.g. ``20``).
    post_train_pct : float
        Percentage of total files for post_train (e.g. ``10``).
    seed : int
        Random seed for reproducibility.
    dry_run : bool
        When ``True`` no files are moved.
    """
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    post_train_ratio = post_train_pct / 100.0
    validate_ratios(train_ratio, test_ratio, post_train_ratio)

    logger.info("Scanning: %s", input_dir)
    pairs_by_material, unparseable, incomplete = collect_pairs(input_dir)

    if unparseable:
        logger.warning("%d file(s) could not be parsed.", unparseable)
    if incomplete:
        logger.warning("%d pair(s) were incomplete.", incomplete)

    if not pairs_by_material:
        logger.error("No valid image/mask pairs found. Aborting.")
        return

    total_pairs = sum(len(v) for v in pairs_by_material.values())
    logger.info(
        "Found %d pair(s) across %d material type(s).",
        total_pairs,
        len(pairs_by_material),
    )

    splits = stratified_split(
        pairs_by_material,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        post_train_ratio=post_train_ratio,
        seed=seed,
    )

    subset_dirs = {
        "train": input_dir / "train",
        "test": input_dir / "test",
    }
    if splits["post_train"]:
        subset_dirs["post_train"] = input_dir / "post_train"

    for subset, pairs in splits.items():
        if not pairs:
            continue
        move_pairs(pairs, subset_dirs[subset], dry_run=dry_run)
        logger.info(
            "%-12s %d file pair(s) %s %s",
            subset + ":",
            len(pairs),
            "would be moved to" if dry_run else "moved to",
            subset_dirs[subset],
        )

    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split a flat image/mask directory into train/, test/, and "
            "optionally post_train/ using material-type stratification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "80/20 split, no post_train:\n"
            "  python split_dataset_into_subsets.py \\\n"
            "      --input /data/out/dataset_1 \\\n"
            "      --train 80 --test 20\n\n"
            "80/20 split with 10 % post_train:\n"
            "  python split_dataset_into_subsets.py \\\n"
            "      --input /data/out/dataset_1 \\\n"
            "      --train 80 --test 20 --post-train 10\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Flat directory of image/mask files to split.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=80.0,
        metavar="WEIGHT",
        help=(
            "Relative weight for the train split from the non-post_train "
            "pool (default: 80)."
        ),
    )
    parser.add_argument(
        "--test",
        type=float,
        default=20.0,
        metavar="WEIGHT",
        help=(
            "Relative weight for the test split from the non-post_train "
            "pool (default: 20)."
        ),
    )
    parser.add_argument(
        "--post-train",
        type=float,
        default=0.0,
        metavar="PCT",
        dest="post_train",
        help=(
            "Percentage of ALL files reserved for post_train (0–99). "
            "Omit or set to 0 to disable post_train. "
            "This is carved out before the train/test split."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned moves without touching disk.",
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

    run(
        input_dir=Path(args.input),
        train_ratio=args.train,
        test_ratio=args.test,
        post_train_pct=args.post_train,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
