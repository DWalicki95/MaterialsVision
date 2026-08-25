"""Tests for reading the frozen split and for the TEST lock."""
import pytest

from materials_vision.data.split_io import (LockedTestSetError, SplitLoadError,
                                            load_split)


def test_train_subset_is_loaded_and_sorted(split_csv):
    subset = load_split(split_csv, "train")

    assert subset.subset == "train"
    assert subset.split_id == "split_test"
    assert len(subset) == 5
    assert subset.image_ids == tuple(sorted(subset.image_ids))
    assert subset.n_excluded_unused == 0


def test_row_order_in_the_file_does_not_change_the_subset(
    tmp_path, split_csv, split_rows
):
    shuffled_path = tmp_path / "shuffled.csv"
    split_rows.sample(frac=1.0, random_state=7).to_csv(
        shuffled_path, index=False
    )

    first = load_split(split_csv, "train")
    second = load_split(shuffled_path, "train")

    assert first.image_ids == second.image_ids


def test_unused_images_are_excluded(split_csv):
    subset = load_split(split_csv, "test", allow_test=True)

    assert "AS3_1" not in subset.image_ids
    assert subset.n_excluded_unused == 1
    assert len(subset) == 2


def test_reading_test_without_the_flag_is_refused(split_csv):
    with pytest.raises(LockedTestSetError, match="TEST is locked"):
        load_split(split_csv, "test")


def test_reading_test_with_the_flag_is_logged(split_csv, caplog):
    with caplog.at_level("WARNING"):
        load_split(split_csv, "test", allow_test=True)

    assert any(
        "TEST UNLOCKED" in record.message for record in caplog.records
    )


def test_allow_test_does_not_affect_the_other_subsets(split_csv):
    without = load_split(split_csv, "val")
    with_flag = load_split(split_csv, "val", allow_test=True)

    assert without.image_ids == with_flag.image_ids


def test_unknown_subset_is_refused(split_csv):
    with pytest.raises(SplitLoadError, match="Unknown subset"):
        load_split(split_csv, "training")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(SplitLoadError, match="not found"):
        load_split(tmp_path / "absent.csv", "train")


def test_missing_column_is_reported(tmp_path, split_rows):
    path = tmp_path / "broken.csv"
    split_rows.drop(columns=["used"]).to_csv(path, index=False)

    with pytest.raises(SplitLoadError, match="missing column"):
        load_split(path, "train")


def test_matching_manifest_passes_verification(
    split_csv, manifest_file
):
    subset = load_split(
        split_csv, "train", verify_manifest=manifest_file
    )

    assert len(subset) == 5


def test_manifest_that_changed_is_refused(split_csv, manifest_file):
    manifest_file.write_text("image_id\nCHANGED\n", encoding="utf-8")

    with pytest.raises(SplitLoadError, match="Manifest mismatch"):
        load_split(split_csv, "train", verify_manifest=manifest_file)


def test_exposure_shares_sum_to_one(split_csv):
    subset = load_split(split_csv, "train")

    shares = subset.exposure("material")

    assert shares["AS"] == pytest.approx(3 / 5)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_exposure_on_an_unknown_column_is_refused(split_csv):
    subset = load_split(split_csv, "train")

    with pytest.raises(SplitLoadError, match="not in the split table"):
        subset.exposure("nonexistent")
