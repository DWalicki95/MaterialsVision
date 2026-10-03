"""Tests for the cross-validation folds over formulations.

The miniature frozen split they run on is ``frozen_table`` /
``frozen_split`` in ``tests/conftest.py``.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

import scripts.create_cv_folds as cli
from materials_vision.data.cv_folds import (FoldError, assign_folds,
                                            balanced_division, check_folds,
                                            count_divisions, fold_table,
                                            group_sizes, held_out, write_fold)
from materials_vision.data.split_io import LockedTestSetError, load_split

# As the metadata of the miniature split records it.
FORCED = ("VAB11",)
SEED = 11


def test_group_sizes_put_the_larger_groups_first():
    assert group_sizes(14, 3) == (5, 5, 4)
    assert group_sizes(3, 3) == (1, 1, 1)


def test_the_real_as_division_is_within_the_cap():
    assert count_divisions((5, 5, 4)) == 252252


def test_balanced_division_minimises_the_spread_of_totals():
    counts = {"a": 5, "b": 1, "c": 4, "d": 2}

    division, n_tied = balanced_division(
        counts, 2, np.random.default_rng(0)
    )

    totals = [sum(counts[name] for name in group) for group in division]
    assert totals == [6, 6]
    assert n_tied == 2


def test_every_formulation_is_held_out_exactly_once(frozen_table):
    assignment, _ = assign_folds(
        frozen_table, forced_train=FORCED, seed=SEED
    )

    names = set(frozen_table["formulation"]) - set(FORCED)
    assert set(assignment) == names
    assert set(assignment.values()) == {1, 2, 3, 4, 5}
    assert check_folds(frozen_table, assignment, forced_train=FORCED) == []


def test_first_folds_are_the_frozen_sets(frozen_table):
    assignment, _ = assign_folds(
        frozen_table, forced_train=FORCED, seed=SEED
    )

    assert held_out(assignment, 1) == {"AS3", "K3", "VAB6"}
    assert held_out(assignment, 2) == {"AS2", "K2", "VAB5"}


def test_each_remaining_fold_holds_out_one_k_and_one_vab(frozen_table):
    assignment, n_tied = assign_folds(
        frozen_table, forced_train=FORCED, seed=SEED
    )

    for fold in (3, 4, 5):
        names = held_out(assignment, fold)
        assert sum(name.startswith("K") for name in names) == 1
        assert sum(name.startswith("VAB") for name in names) == 1
    assert n_tied["K"] == 6
    assert n_tied["VAB"] == 6


def test_the_draw_depends_on_the_seed_alone(frozen_table):
    first, _ = assign_folds(frozen_table, forced_train=FORCED, seed=SEED)
    shuffled = frozen_table.sample(frac=1.0, random_state=3)
    second, _ = assign_folds(shuffled, forced_train=FORCED, seed=SEED)

    assert first == second


def test_unknown_forced_formulation_is_refused(frozen_table):
    with pytest.raises(FoldError, match="unknown"):
        assign_folds(frozen_table, forced_train=("VAB99",), seed=SEED)


def test_fold_table_drops_only_held_out_closeups(frozen_table):
    before = frozen_table.copy()

    fold = fold_table(frozen_table, frozenset({"K1", "AS3"}))

    pd.testing.assert_frame_equal(frozen_table, before)
    unused = set(fold.loc[~fold["used"], "image_id"])
    assert unused == {"K1_2"}
    assert set(fold["split"]) == {"train", "test"}
    held = set(fold.loc[fold["split"] == "test", "formulation"])
    assert held == {"K1", "AS3"}


def test_check_folds_reports_what_is_wrong(frozen_table):
    assignment, _ = assign_folds(
        frozen_table, forced_train=FORCED, seed=SEED
    )
    broken = dict(assignment)
    del broken["AS10"]
    broken["VAB11"] = 3
    for name, fold in assignment.items():
        if fold == 4 and name.startswith("K"):
            broken[name] = 5

    violations = check_folds(frozen_table, broken, forced_train=FORCED)

    text = " | ".join(violations)
    assert "never held out: ['AS10']" in text
    assert "forced into training but held out: ['VAB11']" in text
    assert "fold 4 holds out no ['K'] formulation" in text


def test_check_folds_notices_a_frozen_set_not_reproduced(frozen_table):
    assignment, _ = assign_folds(
        frozen_table, forced_train=FORCED, seed=SEED
    )
    broken = dict(assignment)
    broken["AS3"] = 3

    violations = check_folds(frozen_table, broken, forced_train=FORCED)

    assert any("fold 1" in v and "TEST" in v for v in violations)


def test_a_written_fold_loads_locks_and_verifies(
    tmp_path, frozen_table, frozen_split
):
    _, manifest = frozen_split
    fold = fold_table(frozen_table, frozenset({"K1", "AS10"}))
    metadata = {
        "split_id": "cv_fold3",
        "manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
    }
    path = write_fold(fold, metadata, tmp_path / "cv", "fold3")

    train = load_split(path, "train", verify_manifest=manifest)
    assert train.split_id == "cv_fold3"
    assert not {"K1", "AS10"} & set(train.table["formulation"])
    with pytest.raises(LockedTestSetError):
        load_split(path, "test")
    test = load_split(path, "test", allow_test=True)
    assert set(test.table["formulation"]) == {"K1", "AS10"}
    assert test.n_excluded_unused == 1


def test_an_existing_fold_is_not_overwritten(tmp_path, frozen_table):
    fold = fold_table(frozen_table, frozenset({"K1"}))
    metadata = {"split_id": "cv_fold3", "manifest_sha256": "0" * 64}
    write_fold(fold, metadata, tmp_path, "fold3")

    with pytest.raises(FoldError, match="already exists"):
        write_fold(fold, metadata, tmp_path, "fold3")
    write_fold(fold, metadata, tmp_path, "fold3", overwrite=True)


def test_the_written_folds_cover_every_evaluable_image_once(
    tmp_path, frozen_table, frozen_split
):
    split_csv, manifest = frozen_split
    out_dir = tmp_path / "cv5"

    code = cli.main([
        "--split", str(split_csv), "--out-dir", str(out_dir),
        "--seed", str(SEED),
    ])

    assert code == cli.EXIT_OK
    scored: list[str] = []
    for k in range(1, 6):
        path = out_dir / f"fold{k}.csv"
        test = load_split(
            path, "test", allow_test=True, verify_manifest=manifest
        )
        train = load_split(path, "train", verify_manifest=manifest)
        assert not set(test.image_ids) & set(train.image_ids)
        scored.extend(test.image_ids)
    assert len(scored) == len(set(scored))
    evaluable = frozen_table[
        ~frozen_table["formulation"].isin(FORCED)
        & (frozen_table["scale_bin"] != "outlier")
    ]
    assert set(scored) == set(evaluable["image_id"])
    frozen_test = frozen_table.loc[
        frozen_table["split"] == "test", "image_id"
    ]
    fold1 = load_split(out_dir / "fold1.csv", "test", allow_test=True)
    assert set(fold1.image_ids) == set(frozen_test)
    assert (out_dir / "cv5_v1_assignment.json").exists()


def test_dry_run_writes_nothing(tmp_path, frozen_split):
    split_csv, _ = frozen_split
    out_dir = tmp_path / "cv5"

    code = cli.main([
        "--split", str(split_csv), "--out-dir", str(out_dir), "--dry-run",
    ])

    assert code == cli.EXIT_OK
    assert not out_dir.exists()
