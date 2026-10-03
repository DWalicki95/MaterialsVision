"""Tests for the outcome overlay: paired colours, drawing, per-image counts.

The colouring has one job - a pair shares a colour across the two
sides, anything unpaired is red - so the cases are the outcomes of the
matching: a perfect prediction, a missed pore, a spurious one, a merge
and an empty frame. The matching itself is the real one, not a stand-in,
so that the colours are checked against the pairs the metrics use.
"""
import numpy as np
import pytest
from matplotlib.colors import rgb_to_hsv

from materials_vision.evaluation.matching import match_instances
from materials_vision.inspection.overlay import (PAIR_HUE_RANGE, UNPAIRED_RGB,
                                                 outcome_overlay, pair_palette,
                                                 paired_colours,
                                                 summarise_match)

FRAME = (20, 30)


def _labels(instances):
    """Build a label image from ``{id: (rows_slice, cols_slice)}``."""
    labels = np.zeros(FRAME, dtype=np.int32)
    for instance_id, (rows, cols) in instances.items():
        labels[rows, cols] = instance_id
    return labels


def _two_pores():
    return _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(12, 18)),
    })


def _merged_pair():
    """Two equal touching pores, one prediction covering both.

    Each pore overlaps the prediction by exactly one half, which is
    enough to pair: the merge keeps one pair.
    """
    gt = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
    })
    return gt, _labels({1: (slice(2, 8), slice(2, 14))})


def _merged_triple():
    """Three equal touching pores, one prediction covering all three.

    Each overlaps the prediction by a third, so nothing pairs.
    """
    gt = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
        3: (slice(2, 8), slice(14, 20)),
    })
    return gt, _labels({1: (slice(2, 8), slice(2, 20))})


def _colour_of(rgb, labels, instance_id):
    """The single colour an instance was painted in."""
    colours = np.unique(rgb[labels == instance_id], axis=0)
    assert len(colours) == 1
    return colours[0]


class TestPairPalette:

    def test_one_colour_per_pair(self):
        assert pair_palette(7).shape == (7, 3)

    def test_no_pairs_gives_an_empty_palette(self):
        assert pair_palette(0).shape == (0, 3)

    def test_negative_count_is_rejected(self):
        with pytest.raises(ValueError):
            pair_palette(-1)

    def test_hues_stay_inside_the_band_that_excludes_red(self):
        hues = rgb_to_hsv(pair_palette(50))[:, 0]
        low, high = PAIR_HUE_RANGE
        assert np.all((hues >= low - 1e-9) & (hues <= high + 1e-9))

    def test_same_seed_same_colours(self):
        np.testing.assert_array_equal(
            pair_palette(10, seed=3), pair_palette(10, seed=3)
        )


class TestPairedColours:

    def test_perfect_prediction_pairs_share_colours_and_none_is_red(self):
        gt = _two_pores()
        pred = gt.copy()
        gt_rgb, pred_rgb = paired_colours(gt, pred, match_instances(gt, pred))

        for instance_id in (1, 2):
            colour = _colour_of(gt_rgb, gt, instance_id)
            np.testing.assert_array_equal(
                colour, _colour_of(pred_rgb, pred, instance_id)
            )
            assert not np.array_equal(colour, UNPAIRED_RGB)

    def test_pairs_follow_the_match_not_the_label_numbers(self):
        gt = _two_pores()
        pred = np.where(gt > 0, 3 - gt, 0)
        gt_rgb, pred_rgb = paired_colours(gt, pred, match_instances(gt, pred))

        np.testing.assert_array_equal(
            _colour_of(gt_rgb, gt, 1), _colour_of(pred_rgb, pred, 2)
        )

    def test_missed_pore_is_red_on_the_annotation_side(self):
        gt = _two_pores()
        pred = np.where(gt == 2, 0, gt)
        gt_rgb, _ = paired_colours(gt, pred, match_instances(gt, pred))

        np.testing.assert_array_equal(_colour_of(gt_rgb, gt, 2), UNPAIRED_RGB)

    def test_spurious_pore_is_red_on_the_prediction_side(self):
        gt = _two_pores()
        pred = gt.copy()
        pred[12:18, 20:26] = 3
        _, pred_rgb = paired_colours(gt, pred, match_instances(gt, pred))

        np.testing.assert_array_equal(
            _colour_of(pred_rgb, pred, 3), UNPAIRED_RGB
        )

    def test_merge_of_three_is_red_on_both_sides(self):
        gt, pred = _merged_triple()
        gt_rgb, pred_rgb = paired_colours(gt, pred, match_instances(gt, pred))

        np.testing.assert_array_equal(
            _colour_of(pred_rgb, pred, 1), UNPAIRED_RGB
        )
        for instance_id in (1, 2, 3):
            np.testing.assert_array_equal(
                _colour_of(gt_rgb, gt, instance_id), UNPAIRED_RGB
            )

    def test_merge_of_two_keeps_one_pair_and_one_red_pore(self):
        gt, pred = _merged_pair()
        gt_rgb, pred_rgb = paired_colours(gt, pred, match_instances(gt, pred))

        merged_colour = _colour_of(pred_rgb, pred, 1)
        gt_colours = [_colour_of(gt_rgb, gt, i) for i in (1, 2)]
        assert not np.array_equal(merged_colour, UNPAIRED_RGB)
        assert sum(np.array_equal(c, merged_colour) for c in gt_colours) == 1
        assert sum(np.array_equal(c, UNPAIRED_RGB) for c in gt_colours) == 1

    def test_background_stays_black(self):
        gt = _two_pores()
        gt_rgb, pred_rgb = paired_colours(gt, gt, match_instances(gt, gt))

        assert np.all(gt_rgb[gt == 0] == 0)
        assert np.all(pred_rgb[gt == 0] == 0)

    def test_empty_frames_are_all_background(self):
        empty = np.zeros(FRAME, dtype=np.int32)
        gt_rgb, pred_rgb = paired_colours(
            empty, empty, match_instances(empty, empty)
        )

        assert not gt_rgb.any() and not pred_rgb.any()

    def test_frames_of_different_shape_are_rejected(self):
        gt = _two_pores()
        with pytest.raises(ValueError):
            paired_colours(gt, gt[:, :-1], match_instances(gt, gt))

    def test_inputs_are_left_untouched(self):
        gt = _two_pores()
        pred = gt.copy()
        gt_before, pred_before = gt.copy(), pred.copy()
        paired_colours(gt, pred, match_instances(gt, pred))

        np.testing.assert_array_equal(gt, gt_before)
        np.testing.assert_array_equal(pred, pred_before)


class TestOutcomeOverlay:

    GREY = 128 / 255

    def _drawn(self, alpha=0.4):
        labels = _labels({1: (slice(4, 14), slice(4, 14))})
        image = np.full(FRAME, 128, dtype=np.uint8)
        colours = np.zeros((*FRAME, 3))
        colours[labels == 1] = (0.0, 1.0, 0.0)
        return image, labels, colours, outcome_overlay(
            image, labels, colours, alpha=alpha
        )

    def test_background_keeps_the_image_grey(self):
        _, labels, _, drawn = self._drawn()
        np.testing.assert_allclose(drawn[labels == 0], self.GREY)

    def test_outline_is_opaque(self):
        _, _, _, drawn = self._drawn()
        np.testing.assert_array_equal(drawn[4, 9], (0.0, 1.0, 0.0))

    def test_interior_is_blended(self):
        _, _, _, drawn = self._drawn(alpha=0.4)
        np.testing.assert_allclose(
            drawn[9, 9],
            (0.6 * self.GREY, 0.6 * self.GREY + 0.4, 0.6 * self.GREY),
        )

    def test_inputs_are_left_untouched(self):
        image, labels, colours, _ = self._drawn()
        assert np.all(image == 128)
        assert set(np.unique(labels)) == {0, 1}
        np.testing.assert_array_equal(colours[9, 9], (0.0, 1.0, 0.0))

    def test_mismatched_shapes_are_rejected(self):
        image, labels, colours, _ = self._drawn()
        with pytest.raises(ValueError):
            outcome_overlay(image, labels[:, :-1], colours)

    def test_colour_image_is_rejected(self):
        image, labels, colours, _ = self._drawn()
        with pytest.raises(ValueError):
            outcome_overlay(np.dstack([image] * 3), labels, colours)

    def test_opacity_outside_the_unit_interval_is_rejected(self):
        image, labels, colours, _ = self._drawn()
        with pytest.raises(ValueError):
            outcome_overlay(image, labels, colours, alpha=1.5)


class TestSummariseMatch:

    def test_counts_of_a_spurious_and_a_missed_pore(self):
        gt = _two_pores()
        pred = np.where(gt == 2, 0, gt)
        pred[12:18, 20:26] = 3

        summary = summarise_match(match_instances(gt, pred))

        assert (summary.n_true_positives, summary.n_false_positives,
                summary.n_false_negatives) == (1, 1, 1)
        assert summary.f1 == pytest.approx(0.5)

    def test_merge_is_counted_even_when_it_keeps_a_pair(self):
        gt, pred = _merged_pair()

        summary = summarise_match(match_instances(gt, pred))

        assert summary.n_merges == 1
        assert (summary.n_true_positives, summary.n_false_negatives) == (1, 1)
        assert summary.f1 == pytest.approx(2 / 3)

    def test_empty_frame_has_no_f1(self):
        empty = np.zeros(FRAME, dtype=np.int32)
        assert np.isnan(summarise_match(match_instances(empty, empty)).f1)
