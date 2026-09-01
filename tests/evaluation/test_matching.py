"""Tests for instance matching, merge/split counting and count error.

The cases follow the synthetic battery agreed for this metric layer:
a perfect prediction, one missing instance, one spurious instance, a
merge, a split, shifted boundaries, and the two empty frames. Each has
its expected outcome written down here rather than derived at run
time, so a change in the implementation cannot quietly redefine what
the metric means.
"""
import numpy as np
import pytest

from materials_vision.evaluation.matching import (match_instances,
                                                  pore_count_error)

FRAME = (20, 30)


def _labels(instances):
    """Build a label image from ``{id: (rows_slice, cols_slice)}``."""
    labels = np.zeros(FRAME, dtype=np.int32)
    for instance_id, (rows, cols) in instances.items():
        labels[rows, cols] = instance_id
    return labels


def _three_pores():
    """Three well-separated square pores, ids 1..3."""
    return _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(12, 18)),
        3: (slice(12, 18), slice(2, 8)),
    })


def test_perfect_prediction_scores_one():
    gt = _three_pores()

    match = match_instances(gt, gt.copy())

    assert match.true_positives == 3
    assert match.false_positives == 0
    assert match.false_negatives == 0
    assert match.precision == 1.0
    assert match.recall == 1.0
    assert match.f1 == 1.0
    assert match.mean_pair_iou == pytest.approx(1.0)
    assert match.n_merges == 0
    assert match.n_splits == 0


def test_perfect_prediction_pairs_carry_original_ids():
    gt = _three_pores()
    pred = _labels({
        50: (slice(2, 8), slice(2, 8)),
        60: (slice(2, 8), slice(12, 18)),
        70: (slice(12, 18), slice(2, 8)),
    })

    match = match_instances(gt, pred)

    assert sorted((p.gt_id, p.pred_id) for p in match.pairs) == [
        (1, 50), (2, 60), (3, 70)
    ]


def test_one_missing_instance_is_a_false_negative():
    gt = _three_pores()
    pred = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(12, 18)),
    })

    match = match_instances(gt, pred)

    assert match.true_positives == 2
    assert match.false_negatives == 1
    assert match.false_positives == 0
    assert match.unmatched_gt_ids.tolist() == [3]
    assert match.precision == 1.0
    assert match.recall == pytest.approx(2 / 3)


def test_one_spurious_instance_is_a_false_positive():
    gt = _three_pores()
    pred = _three_pores()
    pred[12:18, 12:18] = 4

    match = match_instances(gt, pred)

    assert match.true_positives == 3
    assert match.false_positives == 1
    assert match.false_negatives == 0
    assert match.unmatched_pred_ids.tolist() == [4]
    assert match.recall == 1.0
    assert match.precision == pytest.approx(3 / 4)


def test_merged_prediction_is_counted_once_as_a_merge():
    gt = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
    })
    pred = _labels({1: (slice(2, 8), slice(2, 14))})

    match = match_instances(gt, pred)

    assert match.n_merges == 1
    assert match.merged_pred_ids.tolist() == [1]
    assert match.n_splits == 0


def test_split_ground_truth_is_counted_once_as_a_split():
    gt = _labels({1: (slice(2, 8), slice(2, 14))})
    pred = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
    })

    match = match_instances(gt, pred)

    assert match.n_splits == 1
    assert match.split_gt_ids.tolist() == [1]
    assert match.n_merges == 0


def test_merge_and_split_are_reported_per_hundred_instances():
    gt = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
        3: (slice(12, 18), slice(2, 8)),
        4: (slice(12, 18), slice(12, 18)),
    })
    pred = _labels({
        1: (slice(2, 8), slice(2, 14)),
        3: (slice(12, 18), slice(2, 8)),
        4: (slice(12, 18), slice(12, 18)),
    })

    match = match_instances(gt, pred)

    assert match.n_merges == 1
    assert match.per_hundred_gt(match.n_merges) == pytest.approx(25.0)


def test_shifted_boundaries_keep_the_pairing_but_lower_the_overlap():
    gt = _three_pores()
    pred = np.roll(gt, 1, axis=0)

    match = match_instances(gt, pred)

    assert match.true_positives == 3
    assert match.f1 == 1.0
    assert match.mean_pair_iou < 1.0


def test_overlap_below_the_threshold_is_not_a_match():
    gt = _labels({1: (slice(2, 8), slice(2, 8))})
    pred = _labels({1: (slice(6, 12), slice(2, 8))})

    match = match_instances(gt, pred)

    assert match.true_positives == 0
    assert match.false_positives == 1
    assert match.false_negatives == 1


def test_empty_prediction_scores_zero_recall_and_undefined_precision():
    gt = _three_pores()
    pred = np.zeros(FRAME, dtype=np.int32)

    match = match_instances(gt, pred)

    assert match.false_negatives == 3
    assert match.recall == 0.0
    assert np.isnan(match.precision)
    assert match.f1 == 0.0


def test_empty_ground_truth_scores_zero_precision_and_undefined_recall():
    gt = np.zeros(FRAME, dtype=np.int32)
    pred = _three_pores()

    match = match_instances(gt, pred)

    assert match.false_positives == 3
    assert match.precision == 0.0
    assert np.isnan(match.recall)
    assert match.f1 == 0.0


def test_two_empty_frames_give_nan_so_the_image_leaves_the_average():
    empty = np.zeros(FRAME, dtype=np.int32)

    match = match_instances(empty, empty.copy())

    assert match.true_positives == 0
    assert np.isnan(match.f1)
    assert np.isnan(match.precision)
    assert np.isnan(match.recall)
    assert np.isnan(match.mean_pair_iou)


def test_swapping_sides_swaps_errors_instead_of_erasing_them():
    gt = _labels({
        1: (slice(2, 8), slice(2, 8)),
        2: (slice(2, 8), slice(8, 14)),
        3: (slice(12, 18), slice(2, 14)),
    })
    pred = _labels({
        1: (slice(2, 8), slice(2, 14)),
        2: (slice(12, 18), slice(2, 8)),
        3: (slice(12, 18), slice(8, 14)),
    })

    forward = match_instances(gt, pred)
    reverse = match_instances(pred, gt)

    assert forward.n_merges == 1 and forward.n_splits == 1
    assert reverse.n_merges == forward.n_splits
    assert reverse.n_splits == forward.n_merges
    assert reverse.false_positives == forward.false_negatives
    assert reverse.false_negatives == forward.false_positives
    assert reverse.precision == pytest.approx(forward.recall)
    assert reverse.recall == pytest.approx(forward.precision)


def test_sparse_prediction_labels_are_handled_and_reported_as_given():
    gt = _three_pores()
    pred = _labels({
        900: (slice(2, 8), slice(2, 8)),
        7: (slice(2, 8), slice(12, 18)),
        41: (slice(12, 18), slice(2, 8)),
    })

    match = match_instances(gt, pred)

    assert match.true_positives == 3
    assert sorted(p.pred_id for p in match.pairs) == [7, 41, 900]


@pytest.mark.parametrize("n_gt, n_pred, signed", [
    (100, 110, 0.10),
    (100, 90, -0.10),
    (100, 100, 0.0),
])
def test_pore_count_error_is_signed_and_absolute(n_gt, n_pred, signed):
    got_signed, got_absolute = pore_count_error(n_gt, n_pred)

    assert got_signed == pytest.approx(signed)
    assert got_absolute == pytest.approx(abs(signed))


def test_pore_count_error_without_annotation_is_undefined():
    signed, absolute = pore_count_error(0, 5)

    assert np.isnan(signed)
    assert np.isnan(absolute)


@pytest.mark.parametrize("gt, pred, message", [
    (np.zeros((4, 4), np.int32), np.zeros((5, 5), np.int32),
     "different frames"),
    (np.zeros((4, 4, 2), np.int32), np.zeros((4, 4, 2), np.int32),
     "must be 2-D"),
    (np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32),
     "integer labels"),
    (-np.ones((4, 4), np.int32), np.zeros((4, 4), np.int32),
     "negative labels"),
])
def test_malformed_input_is_rejected(gt, pred, message):
    with pytest.raises(ValueError, match=message):
        match_instances(gt, pred)


def test_threshold_outside_the_unit_interval_is_rejected():
    frame = np.zeros((4, 4), np.int32)

    with pytest.raises(ValueError, match="iou_threshold"):
        match_instances(frame, frame.copy(), iou_threshold=0.0)
