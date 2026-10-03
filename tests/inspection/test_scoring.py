"""Tests for the material metrics expressed relative to the annotation.

The cases are chosen so that each expected percentage can be written
down by hand: square pores of side 10 annotated, predicted at side 12,
make every predicted diameter exactly 1.2 times the annotated one.
"""
from dataclasses import astuple

import numpy as np
import pytest

from materials_vision.evaluation.aggregate import aggregate
from materials_vision.evaluation.size_bins import SizeBins
from materials_vision.inspection.scoring import (material_drift,
                                                 scores_by_material,
                                                 size_bias_by_bin)

# Annotated areas 49, 100, 169 and 225 um2 fall in Q1, Q2, Q3 and Q4 of
# the synthetic size classes, one pore each.
ONE_PORE_PER_CLASS = (7, 10, 13, 15)


def test_uniformly_enlarged_pores(evaluate, pores):
    evaluation = evaluate(pores(10), pores(12))

    drift = material_drift(aggregate([evaluation]), [evaluation])

    assert drift.pore_count_pct == pytest.approx(0.0)
    assert drift.median_diameter_pct == pytest.approx(20.0)
    assert drift.wasserstein_pct == pytest.approx(20.0)
    assert drift.porosity_pct == pytest.approx(100.0 * (144 - 100) / 100)


def test_spread_of_identical_pores_has_no_relative_value(evaluate, pores):
    evaluation = evaluate(pores(10), pores(12))

    drift = material_drift(aggregate([evaluation]), [evaluation])

    assert np.isnan(drift.iqr_diameter_pct)


def test_spurious_pore_raises_the_count_by_a_quarter(evaluate, pores):
    pred = pores(10)
    pred[20:30, 20:30] = 5
    evaluation = evaluate(pores(10), pred)

    drift = material_drift(aggregate([evaluation]), [evaluation])

    assert drift.pore_count_pct == pytest.approx(25.0)


def test_close_ups_leave_the_denominators_alone(evaluate, pores):
    kept = evaluate(pores(10), pores(12))
    close_up = evaluate(pores(20), pores(20), index=1, scale_bin="outlier")

    with_close_up = material_drift(aggregate([kept, close_up]),
                                   [kept, close_up])
    without = material_drift(aggregate([kept]), [kept])

    np.testing.assert_array_equal(astuple(with_close_up), astuple(without))


def test_empty_annotation_gives_undefined_drift(evaluate, pores):
    empty = np.zeros_like(pores(10))
    evaluation = evaluate(empty, pores(10))

    drift = material_drift(aggregate([evaluation]), [evaluation])

    assert np.isnan(drift.porosity_pct)
    assert np.isnan(drift.median_diameter_pct)


def test_scores_by_material_lists_overall_then_materials(
    evaluate, pores, size_bins
):
    evaluations = [
        evaluate(pores(10), pores(10), index=0, material="K"),
        evaluate(pores(10), pores(12), index=1, material="AS"),
    ]

    scores = scores_by_material(evaluations, size_bins)

    assert list(scores) == ["overall", "AS", "K"]
    assert scores["overall"].result.n_gt == 8
    assert scores["K"].drift.median_diameter_pct == pytest.approx(0.0)
    assert scores["AS"].drift.median_diameter_pct == pytest.approx(20.0)


class TestSizeBiasByBin:

    def test_each_pair_lands_in_the_class_of_its_annotated_pore(
        self, evaluate, pores, size_bins
    ):
        gt = pores(ONE_PORE_PER_CLASS)
        pred = pores((7, 12, 13, 15))
        evaluation = evaluate(gt, pred)

        bias = size_bias_by_bin([evaluation], size_bins)

        assert [item.n_pairs for item in bias] == [1, 1, 1, 1]
        assert [item.median_pct for item in bias] == pytest.approx(
            [0.0, 20.0, 0.0, 0.0]
        )

    def test_classes_together_hold_every_shape_pair(
        self, evaluate, pores, size_bins
    ):
        evaluation = evaluate(pores(ONE_PORE_PER_CLASS),
                              pores((8, 11, 13, 16)))

        bias = size_bias_by_bin([evaluation], size_bins)

        assert sum(item.n_pairs for item in bias) == aggregate(
            [evaluation]
        ).n_shape_pairs

    def test_empty_class_has_no_value(self, evaluate, pores, size_bins):
        evaluation = evaluate(pores(10), pores(10))

        bias = size_bias_by_bin([evaluation], size_bins)

        assert bias[0].n_pairs == 0
        assert np.isnan(bias[0].median_pct)

    def test_pairs_touching_the_frame_edge_are_left_out(
        self, evaluate, pores, size_bins
    ):
        corners = ((0, 0), (5, 35), (35, 5), (35, 35))
        evaluation = evaluate(pores(10, corners), pores(10, corners))

        bias = size_bias_by_bin([evaluation], size_bins)

        assert sum(item.n_pairs for item in bias) == 3

    def test_close_ups_are_left_out(self, evaluate, pores, size_bins):
        close_up = evaluate(pores(10), pores(12), scale_bin="outlier")

        bias = size_bias_by_bin([close_up], size_bins)

        assert sum(item.n_pairs for item in bias) == 0

    def test_pore_on_a_class_edge_is_classed_as_for_recall(
        self, evaluate, pores
    ):
        pixel_size_um = 0.7
        # The middle edge is exactly the area of an 11 x 11 pore, computed
        # the way the evaluation computes areas for recall.
        bins = SizeBins(
            edges_um2=(10.0, 121.0 * pixel_size_um ** 2, 150.0),
            n_calibration_instances=1,
        )
        evaluation = evaluate(pores(11), pores(11), bins=bins,
                              pixel_size_um=pixel_size_um)

        bias = size_bias_by_bin([evaluation], bins)

        assert [item.n_pairs for item in bias] == [
            item.n_matched for item in evaluation.size_bins
        ]
