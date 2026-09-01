"""Tests for the frozen pore size classes and per-class recall.

The unit test that matters most is the scale one: the same physical
pore photographed on either microscope must land in the same class.
That is the whole reason the boundaries are measured in square
micrometres rather than in pixels, so it is asserted directly rather
than left to the docstring.
"""
import json

import numpy as np
import pytest

from materials_vision.evaluation.size_bins import (SIZE_BIN_LABELS, SizeBins,
                                                   SizeBinsLoadError,
                                                   calibrate_size_bins,
                                                   instance_areas_um2,
                                                   load_size_bins,
                                                   recall_per_size_bin)

COARSE_UM_PER_PX = 3.24023

FINE_UM_PER_PX = 2.480469


def _bins():
    return SizeBins(edges_um2=(100.0, 200.0, 400.0),
                    n_calibration_instances=1000)


def _strip_labels(sizes):
    """A row of square instances of the given side lengths, in order."""
    width = sum(size + 1 for size in sizes) + 1
    height = max(sizes) + 2
    labels = np.zeros((height, width), dtype=np.int32)
    column = 1
    for index, size in enumerate(sizes, start=1):
        labels[1:1 + size, column:column + size] = index
        column += size + 1
    return labels


def test_calibration_returns_the_quartiles():
    areas = np.arange(1, 1001, dtype=float)

    bins = calibrate_size_bins(areas)

    assert bins.n_calibration_instances == 1000
    assert bins.edges_um2 == pytest.approx((250.75, 500.5, 750.25))


def test_calibration_splits_a_sample_into_four_equal_parts():
    rng = np.random.default_rng(0)
    areas = rng.lognormal(mean=10.0, sigma=1.0, size=40000)

    counts = np.bincount(calibrate_size_bins(areas).assign(areas),
                         minlength=4)

    assert counts.min() / counts.max() == pytest.approx(1.0, abs=0.01)


def test_assignment_boundaries_are_half_open_from_below():
    bins = _bins()

    assigned = bins.assign(np.array([99.9, 100.0, 199.9, 200.0, 400.0]))

    assert assigned.tolist() == [0, 1, 1, 2, 3]


@pytest.mark.parametrize("areas, message", [
    (np.array([1.0, 2.0]), "at least 4"),
    (np.array([1.0, 2.0, 3.0, np.nan]), "finite and positive"),
    (np.array([1.0, 2.0, 3.0, 0.0]), "finite and positive"),
    (np.ones(100), "not distinct"),
])
def test_calibration_rejects_samples_it_cannot_describe(areas, message):
    with pytest.raises(ValueError, match=message):
        calibrate_size_bins(areas)


@pytest.mark.parametrize("edges, message", [
    ((100.0, 50.0, 400.0), "ascend"),
    ((-1.0, 50.0, 400.0), "positive"),
])
def test_malformed_edges_are_rejected(edges, message):
    with pytest.raises(ValueError, match=message):
        SizeBins(edges_um2=edges, n_calibration_instances=10)


def test_areas_convert_pixels_to_square_micrometres():
    labels = _strip_labels([2, 3])

    areas = instance_areas_um2(labels, pixel_size_um=2.0)

    assert areas.tolist() == [4 * 4.0, 9 * 4.0]


def test_areas_reject_a_labelling_with_gaps():
    labels = _strip_labels([2, 3])
    labels[labels == 1] = 0

    with pytest.raises(ValueError, match="densely numbered"):
        instance_areas_um2(labels, pixel_size_um=1.0)


@pytest.mark.parametrize("pixel_size_um", [0.0, -1.0, np.nan])
def test_areas_reject_an_impossible_pixel_size(pixel_size_um):
    labels = _strip_labels([2])

    with pytest.raises(ValueError, match="pixel_size_um"):
        instance_areas_um2(labels, pixel_size_um=pixel_size_um)


def test_the_same_physical_pore_lands_in_the_same_class_on_both_scales():
    """A pore of one physical size, imaged at either calibration.

    On the coarse scale it covers fewer pixels than on the fine one -
    that ratio is what pixel-based edges would encode instead of size.
    """
    side_um = 40.0
    coarse_side_px = round(side_um / COARSE_UM_PER_PX)
    fine_side_px = round(side_um / FINE_UM_PER_PX)
    assert coarse_side_px != fine_side_px

    bins = SizeBins(edges_um2=(400.0, 1600.0, 3600.0),
                    n_calibration_instances=1000)
    coarse = instance_areas_um2(_strip_labels([coarse_side_px]),
                                pixel_size_um=COARSE_UM_PER_PX)
    fine = instance_areas_um2(_strip_labels([fine_side_px]),
                              pixel_size_um=FINE_UM_PER_PX)

    assert bins.assign(coarse) == bins.assign(fine)


def test_pixel_edges_would_have_put_that_pore_in_different_classes():
    """The failure the micrometre unit is there to prevent."""
    side_um = 40.0
    coarse_px2 = round(side_um / COARSE_UM_PER_PX) ** 2
    fine_px2 = round(side_um / FINE_UM_PER_PX) ** 2
    pixel_edges = np.array([100.0, 180.0, 400.0])

    assert (np.digitize(coarse_px2, pixel_edges)
            != np.digitize(fine_px2, pixel_edges))


def test_recall_is_reported_per_class_with_its_population():
    labels = _strip_labels([2, 4, 8, 16])
    bins = SizeBins(edges_um2=(10.0, 40.0, 150.0),
                    n_calibration_instances=1000)

    result = recall_per_size_bin(labels, np.array([1, 3]), bins,
                                 pixel_size_um=1.0)

    assert [entry.label for entry in result] == list(SIZE_BIN_LABELS)
    assert [entry.n_gt for entry in result] == [1, 1, 1, 1]
    assert [entry.recall for entry in result] == [1.0, 0.0, 1.0, 0.0]


def test_class_recalls_decompose_the_overall_recall():
    labels = _strip_labels([2, 3, 5, 8, 13, 21])
    bins = SizeBins(edges_um2=(15.0, 60.0, 200.0),
                    n_calibration_instances=1000)
    matched = np.array([1, 3, 4, 6])

    result = recall_per_size_bin(labels, matched, bins, pixel_size_um=1.0)

    assert sum(entry.n_gt for entry in result) == 6
    assert sum(entry.n_matched for entry in result) == matched.size


def test_an_empty_class_reports_nan_rather_than_a_failed_detection():
    labels = _strip_labels([2, 3])
    bins = SizeBins(edges_um2=(100.0, 200.0, 400.0),
                    n_calibration_instances=1000)

    result = recall_per_size_bin(labels, np.array([1]), bins,
                                 pixel_size_um=1.0)

    assert result[0].n_gt == 2
    assert all(np.isnan(entry.recall) for entry in result[1:])


def test_nothing_matched_gives_zero_recall_not_nan():
    labels = _strip_labels([2, 3])
    bins = _bins()

    result = recall_per_size_bin(labels, np.empty(0, dtype=np.int64),
                                 bins, pixel_size_um=1.0)

    assert result[0].recall == 0.0


def test_a_matched_id_outside_the_ground_truth_is_rejected():
    labels = _strip_labels([2, 3])

    with pytest.raises(ValueError, match="outside"):
        recall_per_size_bin(labels, np.array([5]), _bins(),
                            pixel_size_um=1.0)


def test_metadata_records_the_unit_and_the_population():
    bins = _bins()

    metadata = bins.as_metadata()

    assert metadata["unit"] == "um2"
    assert metadata["edges_um2"] == [100.0, 200.0, 400.0]
    assert metadata["n_calibration_instances"] == 1000


def test_frozen_classes_survive_a_round_trip(tmp_path):
    bins = _bins()
    path = tmp_path / "size_bins.json"
    path.write_text(json.dumps({"bins": bins.as_metadata()}))

    assert load_size_bins(path) == bins


def test_edges_recorded_in_pixels_are_refused(tmp_path):
    path = tmp_path / "size_bins.json"
    metadata = _bins().as_metadata()
    metadata["unit"] = "px2"
    path.write_text(json.dumps({"bins": metadata}))

    with pytest.raises(SizeBinsLoadError, match="square micrometres"):
        load_size_bins(path)


@pytest.mark.parametrize("payload", ['{}', 'not json', '{"bins": {}}'])
def test_an_unusable_artifact_is_refused(tmp_path, payload):
    path = tmp_path / "size_bins.json"
    path.write_text(payload)

    with pytest.raises(SizeBinsLoadError):
        load_size_bins(path)


def test_a_missing_artifact_is_refused(tmp_path):
    with pytest.raises(SizeBinsLoadError, match="Cannot read"):
        load_size_bins(tmp_path / "absent.json")
