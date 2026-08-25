"""Tests for manifest loading, grouping integrity and profiling."""
import pandas as pd
import pytest

from data_prep.split.profiles import (SplitDataError,
                                      build_formulation_profiles,
                                      check_grouping_integrity, load_manifest)


def test_build_profiles_aggregates_per_formulation(manifest):
    profiles = build_formulation_profiles(manifest)

    by_name = {p.formulation: p for p in profiles}
    assert set(by_name) == {"AS1", "AS2", "K1", "VAB1"}
    assert by_name["AS1"].n_images == 4
    assert by_name["AS1"].n_coarse == 3
    assert by_name["AS1"].n_fine == 1
    assert by_name["AS1"].n_eval_images == 4
    assert by_name["VAB1"].n_outlier == 1
    assert by_name["VAB1"].n_eval_images == 1
    assert by_name["VAB1"].microscope == "M2"


def test_profiles_are_sorted_regardless_of_row_order(manifest):
    shuffled = manifest.sample(frac=1.0, random_state=0)

    names = [p.formulation for p in build_formulation_profiles(shuffled)]

    assert names == sorted(names)


def test_formulation_spanning_two_materials_is_rejected(manifest):
    manifest.loc[0, "material"] = "K"

    violations = check_grouping_integrity(manifest)

    assert any("spans multiple material" in v for v in violations)
    with pytest.raises(SplitDataError, match="material"):
        build_formulation_profiles(manifest)


def test_shared_file_hash_across_formulations_is_rejected(manifest):
    shared = manifest.loc[manifest["formulation"] == "AS2"].index[0]
    manifest.loc[shared, "file_hash"] = manifest.loc[0, "file_hash"]

    violations = check_grouping_integrity(manifest)

    assert any("file_hash" in v for v in violations)


def test_unset_scale_bin_is_rejected(manifest):
    manifest.loc[0, "scale_bin"] = None

    with pytest.raises(SplitDataError, match="scale_bin"):
        build_formulation_profiles(manifest)


def test_missing_microscope_is_rejected(manifest):
    manifest.loc[manifest["formulation"] == "K1", "microscope"] = None

    with pytest.raises(SplitDataError, match="microscope"):
        build_formulation_profiles(manifest)


def test_load_manifest_reports_missing_file(tmp_path):
    with pytest.raises(SplitDataError, match="not found"):
        load_manifest(tmp_path / "absent.csv")


def test_load_manifest_reports_missing_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["a"]}).to_csv(path, index=False)

    with pytest.raises(SplitDataError, match="missing required column"):
        load_manifest(path)
