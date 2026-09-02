"""Tests for the assembly of per-image measurements into a report.

The properties worth pinning are the ones a reader of the report will
rely on without checking: that micro and macro really differ when the
images differ in size, that close-ups never reach a subset, and that
distribution figures are pooled from the pores rather than averaged
from per-image summaries.
"""
from pathlib import Path

import numpy as np
import pytest

from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.aggregate import (aggregate,
                                                   cross_sections,
                                                   evaluate_image,
                                                   scale_outlier_report)
from materials_vision.evaluation.size_bins import SizeBins

PIXEL_SIZE_UM = 3.24023

BINS = SizeBins(edges_um2=(38505.0, 98891.0, 243941.0),
                n_calibration_instances=25247)


def _record(image_id="AS1_40_1", formulation="AS1", material="AS",
            microscope="M1", scale_bin="coarse"):
    return SampleRecord(
        index=0, image_id=image_id, formulation=formulation,
        material=material, microscope=microscope, scale_bin=scale_bin,
        pixel_size_um=PIXEL_SIZE_UM, q_max_i=1.3,
        source_path=Path("image.tif"), mask_path=Path("mask.tif"),
        crop_bbox=(0, 0, 128, 128), n_instances_expected=0,
    )


def _grid_labels(n_pores, frame=(128, 128), side=20, spacing=32):
    """A row-major grid of square pores of a fixed size."""
    labels = np.zeros(frame, dtype=np.int32)
    per_row = frame[1] // spacing
    for index in range(n_pores):
        row = 4 + spacing * (index // per_row)
        col = 4 + spacing * (index % per_row)
        labels[row:row + side, col:col + side] = index + 1
    return labels


def _evaluate(gt, pred, record=None):
    return evaluate_image(record or _record(), gt, pred, size_bins=BINS)


def test_a_perfect_prediction_aggregates_to_one():
    gt = _grid_labels(4)
    result = aggregate([_evaluate(gt, gt.copy())])

    assert result.n_images == 1
    assert result.f1 == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)


def test_counts_are_pooled_across_images():
    gt = _grid_labels(4)
    evaluations = [_evaluate(gt, gt.copy()) for _ in range(3)]

    result = aggregate(evaluations)

    assert result.n_images == 3
    assert result.n_gt == 12
    assert result.n_pred == 12


def test_micro_and_macro_differ_when_the_images_do():
    """A crowded image should weigh more in micro than in macro.

    The dense image is scored perfectly and the sparse one badly, so
    pooling the counts favours the dense one while averaging the
    scores treats them alike.
    """
    dense = _grid_labels(12)
    sparse = _grid_labels(2)
    sparse_pred = _grid_labels(2)
    sparse_pred[sparse_pred == 2] = 0

    result = aggregate([
        _evaluate(dense, dense.copy()),
        _evaluate(sparse, sparse_pred),
    ])

    assert result.f1 > result.macro_f1


def test_close_ups_never_reach_a_subset():
    gt = _grid_labels(4)
    normal = _evaluate(gt, gt.copy())
    close_up = _evaluate(gt, gt.copy(),
                         record=_record(scale_bin="outlier"))

    result = aggregate([normal, close_up])

    assert result.n_images == 1
    assert result.n_scale_outliers_excluded == 1


def test_close_ups_are_reported_on_their_own():
    gt = _grid_labels(4)
    close_up = _evaluate(gt, gt.copy(),
                         record=_record(scale_bin="outlier"))

    result = scale_outlier_report([_evaluate(gt, gt.copy()), close_up])

    assert result.label == "scale_outlier"
    assert result.n_images == 1


def test_an_empty_subset_reports_undefined_rather_than_zero():
    gt = _grid_labels(4)
    close_up = _evaluate(gt, gt.copy(),
                         record=_record(scale_bin="outlier"))

    result = aggregate([close_up])

    assert result.n_images == 0
    assert np.isnan(result.f1)
    assert np.isnan(result.macro_f1)


def test_cross_sections_split_by_the_requested_key():
    gt = _grid_labels(4)
    evaluations = [
        _evaluate(gt, gt.copy(), record=_record(material="AS")),
        _evaluate(gt, gt.copy(), record=_record(material="K",
                                                microscope="M2")),
        _evaluate(gt, gt.copy(), record=_record(material="K",
                                                microscope="M2")),
    ]

    by_material = cross_sections(evaluations, "material")

    assert [item.label for item in by_material] == [
        "material=AS", "material=K"
    ]
    assert [item.n_images for item in by_material] == [1, 2]


def test_cross_sections_carry_their_own_close_up_count():
    gt = _grid_labels(4)
    evaluations = [
        _evaluate(gt, gt.copy(), record=_record(material="AS")),
        _evaluate(gt, gt.copy(),
                  record=_record(material="AS", scale_bin="outlier")),
    ]

    by_material = cross_sections(evaluations, "material")

    assert by_material[0].n_images == 1
    assert by_material[0].n_scale_outliers_excluded == 1


def test_an_unknown_cross_section_key_is_rejected():
    with pytest.raises(ValueError, match="cross-section key"):
        cross_sections([], "pixel_size_um")


def test_merge_and_split_counts_are_pooled_and_normalised():
    gt = _grid_labels(4, spacing=24, side=20)
    merged = gt.copy()
    merged[merged == 2] = 1

    result = aggregate([_evaluate(gt, merged)])

    assert result.n_merges == 1
    assert result.merges_per_100_gt == pytest.approx(25.0)


def test_the_signed_count_error_is_pooled_and_the_absolute_one_is_not():
    """Opposite per-image errors cancel pooled but not in the mean.

    This is why the tie-break reads the macro absolute error: a subset
    whose images err in both directions would otherwise look correct.
    """
    gt = _grid_labels(4)
    too_few = gt.copy()
    too_few[too_few == 4] = 0
    too_many = gt.copy()
    too_many[100:110, 100:110] = 5

    result = aggregate([_evaluate(gt, too_few), _evaluate(gt, too_many)])

    assert result.pore_count_error == pytest.approx(0.0)
    assert result.macro_abs_pore_count_error > 0.0


def test_boundary_scores_are_reported_both_pooled_and_averaged():
    gt = _grid_labels(4)
    result = aggregate([_evaluate(gt, gt.copy())])

    assert set(result.boundary_f1) == set(result.macro_boundary_f1)
    assert all(np.isclose(value, 1.0)
               for value in result.boundary_f1.values())


def test_pooling_refuses_images_scored_at_different_tolerances():
    gt = _grid_labels(4)
    a = evaluate_image(_record(), gt, gt.copy(), size_bins=BINS,
                       boundary_scales=(0.1,))
    b = evaluate_image(_record(), gt, gt.copy(), size_bins=BINS,
                       boundary_scales=(0.2,))

    with pytest.raises(ValueError, match="different boundary tolerances"):
        aggregate([a, b])


def test_size_bin_recalls_are_pooled_and_still_decompose_recall():
    gt = _grid_labels(6)
    partial = gt.copy()
    partial[partial == 3] = 0

    result = aggregate([_evaluate(gt, partial), _evaluate(gt, gt.copy())])

    assert sum(item.n_gt for item in result.size_bin_recall) == result.n_gt
    matched = sum(item.n_matched for item in result.size_bin_recall)
    assert matched == result.n_true_positives


def test_number_density_pools_counts_over_total_area():
    gt = _grid_labels(4)
    one = _evaluate(gt, gt.copy())

    single = aggregate([one])
    doubled = aggregate([one, one])

    assert doubled.gt_per_mm2 == pytest.approx(single.gt_per_mm2)


def test_the_diameter_distribution_is_pooled_not_averaged():
    """Two images whose per-image drifts cancel still drift as a pool.

    One image predicts every pore too large, the other too small. The
    mean of the two per-image medians would be near zero; the pooled
    distributions are genuinely wider than the annotation, and the
    interquartile drift shows it.
    """
    gt = _grid_labels(4, side=20, spacing=32)
    larger = _grid_labels(4, side=26, spacing=32)
    smaller = _grid_labels(4, side=14, spacing=32)

    result = aggregate([_evaluate(gt, larger), _evaluate(gt, smaller)])

    assert result.median_diameter_drift_um == pytest.approx(0.0, abs=15.0)
    assert result.iqr_drift_um > 0.0
    assert result.wasserstein_um > 0.0
