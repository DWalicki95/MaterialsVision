"""
Cross-validation folds over formulations, derived from a frozen split.

The single frozen split measures how a training recipe transfers to new
formulations on one draw of them. For the materials of the second
microscope that draw is one formulation per evaluation set, so a figure
for K or VAB there describes one synthesis rather than the material.
Folds answer the same question on every formulation in turn.

**The unit is the formulation, as in the split.** Every image of one
formulation comes from a single synthesis and is strongly correlated
with the others, so a formulation is either held out whole or trained
on whole.

**The first folds are the sets of the frozen split, unchanged.** Fold
one holds out exactly the split's TEST and fold two its VALIDATION. A
model already trained on everything except TEST is then fold one's
model, so it needs no retraining, and its existing TEST score is fold
one's score.

**The remaining folds share out what the split trained on.** Within
each material its formulations are divided among the remaining folds
in groups as equal in size as possible, and among all such divisions
the one whose groups differ least in image count is taken. Ties are
broken by a seeded draw. For a material with as many formulations as
there are remaining folds every division ties, so each fold receives
one formulation at random. Formulations the split forced into TRAIN
stay in training in every fold, for the reason they were forced there:
too few evaluable images for a cross-section anyone could read.

**The held-out set is labelled ``test``.** A fold file has the columns
of the split and only two sets, ``train`` and ``test``. With nothing
left to choose, a fold needs no validation set. Labelling the held-out
set ``test`` puts it behind the lock every loader already honours, and
lets the one-snapshot-per-run rule of the evaluation apply unchanged.
Close-up images of a held-out formulation are marked unused, by the
split's own rule: they follow their formulation, and are neither moved
into training nor scored.
"""
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_FOLDS = 5

# Fold number to the set of the frozen split it holds out as it stands.
FROZEN_SET_FOLDS: Mapping[int, str] = {1: "test", 2: "val"}

TRAIN = "train"
HELD_OUT = "test"
SCALE_OUTLIER_BIN = "outlier"

# Enumerating divisions is exact but grows combinatorially. The largest
# case here, fourteen formulations into three folds, is about a quarter
# of a million; the cap turns a much larger input into an error rather
# than an afternoon.
MAX_DIVISIONS = 2_000_000


class FoldError(ValueError):
    """Raised when folds cannot be built from the given split table."""


def formulation_profile(table: pd.DataFrame) -> pd.DataFrame:
    """One row per formulation: its material, set and image count.

    Parameters
    ----------
    table : pandas.DataFrame
        The frozen split table, all rows.

    Returns
    -------
    pandas.DataFrame
        Indexed by formulation, sorted, with columns ``material``,
        ``split`` and ``n_images``.

    Raises
    ------
    FoldError
        If a formulation spans several materials or several sets.
    """
    grouped = table.groupby("formulation")
    for column in ("material", "split"):
        spread = grouped[column].nunique()
        if (spread > 1).any():
            names = sorted(spread[spread > 1].index)
            raise FoldError(
                f"Formulation(s) {names} span more than one {column}; "
                f"a formulation must be a single unit."
            )
    profile = grouped.agg(
        material=("material", "first"),
        split=("split", "first"),
        n_images=("image_id", "size"),
    )
    return profile.sort_index()


def group_sizes(n_items: int, n_groups: int) -> tuple[int, ...]:
    """Sizes as equal as possible, larger groups first.

    Parameters
    ----------
    n_items : int
    n_groups : int

    Returns
    -------
    tuple of int

    Raises
    ------
    FoldError
        If ``n_groups`` is not positive.
    """
    if n_groups < 1:
        raise FoldError(f"Need at least one group, got {n_groups}.")
    base, extra = divmod(n_items, n_groups)
    return tuple(base + 1 if i < extra else base for i in range(n_groups))


def count_divisions(sizes: Sequence[int]) -> int:
    """How many ordered divisions into groups of these sizes exist.

    Parameters
    ----------
    sizes : sequence of int

    Returns
    -------
    int
    """
    total = math.factorial(sum(sizes))
    for size in sizes:
        total //= math.factorial(size)
    return total


def _divisions(
    names: tuple[str, ...], sizes: tuple[int, ...]
) -> Iterator[tuple[tuple[str, ...], ...]]:
    """Every ordered division of ``names`` into groups of ``sizes``.

    Parameters
    ----------
    names : tuple of str
    sizes : tuple of int

    Yields
    ------
    tuple of tuple of str
    """
    if not sizes:
        yield ()
        return
    for first in itertools.combinations(names, sizes[0]):
        rest = tuple(name for name in names if name not in first)
        for tail in _divisions(rest, sizes[1:]):
            yield (first,) + tail


def balanced_division(
    counts: Mapping[str, int],
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[str, ...], ...], int]:
    """Divide items into groups whose totals differ least.

    Parameters
    ----------
    counts : mapping of str to int
        Item to its weight, here a formulation to its image count.
    n_groups : int
    rng : numpy.random.Generator
        Breaks ties among equally balanced divisions.

    Returns
    -------
    division : tuple of tuple of str
        One group per fold, in fold order, names sorted within a group.
    n_tied : int
        How many divisions reached the same smallest spread; the draw
        was made among them.

    Raises
    ------
    FoldError
        If the number of divisions exceeds ``MAX_DIVISIONS``.
    """
    names = tuple(sorted(counts))
    sizes = group_sizes(len(names), n_groups)
    n_divisions = count_divisions(sizes)
    if n_divisions > MAX_DIVISIONS:
        raise FoldError(
            f"{n_divisions} divisions of {len(names)} item(s) into "
            f"{n_groups} group(s) exceed the cap of {MAX_DIVISIONS}."
        )
    best_spread: Optional[int] = None
    tied: list[tuple[tuple[str, ...], ...]] = []
    for division in _divisions(names, sizes):
        totals = [sum(counts[name] for name in group) for group in division]
        spread = max(totals) - min(totals)
        if best_spread is None or spread < best_spread:
            best_spread, tied = spread, [division]
        elif spread == best_spread:
            tied.append(division)
    chosen = tied[int(rng.integers(len(tied)))]
    return chosen, len(tied)


def assign_folds(
    table: pd.DataFrame,
    *,
    forced_train: Sequence[str],
    seed: int,
    n_folds: int = N_FOLDS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Decide which fold holds out each formulation.

    Parameters
    ----------
    table : pandas.DataFrame
        The frozen split table, all rows.
    forced_train : sequence of str
        Formulations kept in training in every fold.
    seed : int
        Seeds the tie-breaking draw. Materials are drawn in sorted
        order from one generator, so the result depends on nothing
        else.
    n_folds : int, optional

    Returns
    -------
    assignment : dict of str to int
        Formulation to fold number, ``1`` to ``n_folds``. Formulations
        in ``forced_train`` are absent.
    n_tied : dict of str to int
        Material to the number of equally balanced divisions the draw
        was made among.

    Raises
    ------
    FoldError
        If a forced formulation is unknown, or if there are no folds
        left to share TRAIN among.
    """
    profile = formulation_profile(table)
    unknown = sorted(set(forced_train) - set(profile.index))
    if unknown:
        raise FoldError(f"forced_train names unknown formulation(s) {unknown}")
    free_folds = [
        k for k in range(1, n_folds + 1) if k not in FROZEN_SET_FOLDS
    ]
    if not free_folds:
        raise FoldError(
            f"{n_folds} fold(s) leave none beyond the frozen sets to "
            f"share TRAIN among."
        )

    assignment: dict[str, int] = {}
    for fold, subset in FROZEN_SET_FOLDS.items():
        for name in profile.index[profile["split"] == subset]:
            assignment[name] = fold

    rng = np.random.default_rng(seed)
    free = profile[
        (profile["split"] == TRAIN) & ~profile.index.isin(forced_train)
    ]
    n_tied: dict[str, int] = {}
    for material, rows in free.groupby("material", sort=True):
        division, n_tied[str(material)] = balanced_division(
            rows["n_images"].to_dict(), len(free_folds), rng
        )
        for fold, group in zip(free_folds, division):
            for name in group:
                assignment[name] = fold
    return assignment, n_tied


def held_out(assignment: Mapping[str, int], fold: int) -> frozenset[str]:
    """Formulations one fold holds out.

    Parameters
    ----------
    assignment : mapping of str to int
    fold : int

    Returns
    -------
    frozenset of str
    """
    return frozenset(name for name, k in assignment.items() if k == fold)


def fold_table(
    table: pd.DataFrame, held_out_formulations: frozenset[str]
) -> pd.DataFrame:
    """The split table rewritten for one fold.

    Parameters
    ----------
    table : pandas.DataFrame
        The frozen split table, all rows. Not modified.
    held_out_formulations : frozenset of str

    Returns
    -------
    pandas.DataFrame
        Same columns, ``split`` set to ``test`` for the held-out
        formulations and ``train`` for the rest, ``used`` recomputed,
        rows ordered as the split writes them.

    Raises
    ------
    FoldError
        If a held-out formulation is not in the table.
    """
    missing = sorted(held_out_formulations - set(table["formulation"]))
    if missing:
        raise FoldError(f"Held-out formulation(s) {missing} not in table")
    fold = table.copy()
    is_held = fold["formulation"].isin(held_out_formulations)
    fold["split"] = np.where(is_held, HELD_OUT, TRAIN)
    fold["used"] = ~(is_held & (fold["scale_bin"] == SCALE_OUTLIER_BIN))
    return fold.sort_values(
        ["split", "formulation", "image_id"]
    ).reset_index(drop=True)


def check_folds(
    table: pd.DataFrame,
    assignment: Mapping[str, int],
    *,
    forced_train: Sequence[str],
    n_folds: int = N_FOLDS,
) -> list[str]:
    """Every property the folds are meant to have, as violations.

    Parameters
    ----------
    table : pandas.DataFrame
        The frozen split table, all rows.
    assignment : mapping of str to int
    forced_train : sequence of str
    n_folds : int, optional

    Returns
    -------
    list of str
        Empty when the folds are sound.
    """
    violations: list[str] = []
    profile = formulation_profile(table)
    expected = set(profile.index) - set(forced_train)

    unassigned = sorted(expected - set(assignment))
    if unassigned:
        violations.append(f"never held out: {unassigned}")
    forced_held = sorted(set(forced_train) & set(assignment))
    if forced_held:
        violations.append(f"forced into training but held out: {forced_held}")
    out_of_range = sorted(
        name for name, k in assignment.items() if not 1 <= k <= n_folds
    )
    if out_of_range:
        violations.append(f"fold number out of range for {out_of_range}")

    for fold, subset in FROZEN_SET_FOLDS.items():
        rows = table[table["split"] == subset]
        frozen_ids = set(rows.loc[rows["used"].astype(bool), "image_id"])
        rewritten = fold_table(table, held_out(assignment, fold))
        fold_ids = set(
            rewritten.loc[
                (rewritten["split"] == HELD_OUT) & rewritten["used"],
                "image_id",
            ]
        )
        if fold_ids != frozen_ids:
            violations.append(
                f"fold {fold} does not hold out exactly the images of "
                f"{subset.upper()}"
            )

    materials = set(profile["material"])
    for fold in range(1, n_folds + 1):
        names = held_out(assignment, fold)
        present = set(profile.loc[sorted(names), "material"])
        absent = sorted(materials - present)
        if absent:
            violations.append(f"fold {fold} holds out no {absent} formulation")
    return violations


def fold_summary(fold: pd.DataFrame) -> dict[str, Any]:
    """Counts describing one fold, for the report and the metadata.

    Parameters
    ----------
    fold : pandas.DataFrame
        A table returned by ``fold_table``.

    Returns
    -------
    dict
        Held-out formulations per material, and image and pore counts
        of the evaluable held-out part and of the training part.
    """
    used = fold[fold["used"].astype(bool)]
    test = used[used["split"] == HELD_OUT]
    train = used[used["split"] == TRAIN]
    return {
        "held_out": {
            str(material): sorted(rows["formulation"].unique())
            for material, rows in test.groupby("material", sort=True)
        },
        "n_test_images": len(test),
        "n_test_pores": int(test["n_instances"].sum()),
        "n_test_images_by_material": {
            str(k): int(v) for k, v in
            test.groupby("material", sort=True).size().items()
        },
        "n_test_pores_by_material": {
            str(k): int(v) for k, v in
            test.groupby("material", sort=True)["n_instances"].sum().items()
        },
        "n_test_fine_images": int((test["scale_bin"] == "fine").sum()),
        "n_train_images": len(train),
        "n_unused": int((~fold["used"].astype(bool)).sum()),
    }


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fold(
    fold: pd.DataFrame,
    metadata: Mapping[str, Any],
    out_dir: Path,
    name: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one fold as a split table with its metadata sidecar.

    The sidecar is named the way ``load_split`` looks for it, so a fold
    file loads, locks and verifies against its manifest exactly as the
    frozen split does.

    Parameters
    ----------
    fold : pandas.DataFrame
    metadata : mapping
        Must carry ``split_id`` and ``manifest_sha256``.
    out_dir : Path
    name : str
        File stem, e.g. ``"fold3"``.
    overwrite : bool, optional

    Returns
    -------
    Path
        The table written.

    Raises
    ------
    FoldError
        If the table exists and ``overwrite`` is False, or if the
        metadata lacks a required key.
    """
    for key in ("split_id", "manifest_sha256"):
        if key not in metadata:
            raise FoldError(f"Fold metadata lacks {key!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    if path.exists() and not overwrite:
        raise FoldError(
            f"{path} already exists. Folds are fixed before any model is "
            f"trained on them and are not rewritten; pass overwrite only "
            f"to replace a draft nothing has used."
        )
    fold.to_csv(path, index=False, na_rep="")
    record = dict(metadata)
    record["split_table_sha256"] = file_sha256(path)
    meta_path = out_dir / f"{name}_metadata.json"
    meta_path.write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote %s and %s.", path, meta_path.name)
    return path
