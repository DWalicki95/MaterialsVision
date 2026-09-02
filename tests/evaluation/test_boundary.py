"""Tests for the boundary agreement score.

The central test is the oracle: on frames where no instance touches
the edge, this implementation must reproduce
``cellpose.metrics.boundary_scores`` exactly, because it is meant to
be that computation with one masking step inserted. Where instances do
touch the edge the two are expected to disagree, and a second test
asserts that too - otherwise the masking step could be silently
inert.
"""
import numpy as np
import pytest
from cellpose.metrics import boundary_scores as cellpose_boundary_scores

from materials_vision.evaluation.boundary import (BOUNDARY_SCALES,
                                                  DECISION_SCALE,
                                                  DIAGNOSTIC_SCALES,
                                                  boundary_scores)

FRAME = (60, 80)


def _pores_away_from_the_edge():
    """Four pores, none of them reaching the frame border."""
    labels = np.zeros(FRAME, dtype=np.int32)
    labels[8:26, 8:30] = 1
    labels[8:26, 36:60] = 2
    labels[34:52, 8:30] = 3
    labels[34:52, 36:60] = 4
    return labels


def _pores_touching_the_edge():
    labels = _pores_away_from_the_edge()
    labels[0:26, 0:30] = 1
    return labels


def test_decision_scale_is_one_of_the_reported_scales():
    assert DECISION_SCALE in BOUNDARY_SCALES


def test_identical_masks_score_one():
    gt = _pores_away_from_the_edge()

    scores = boundary_scores(gt, gt.copy())

    for score in scores.values():
        assert score.f1 == pytest.approx(1.0)
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(1.0)


def test_score_falls_as_the_boundary_drifts_further():
    gt = _pores_away_from_the_edge()

    drifts = [
        boundary_scores(gt, np.roll(gt, shift, axis=0))[DECISION_SCALE].f1
        for shift in (1, 2, 4)
    ]

    assert drifts == sorted(drifts, reverse=True)
    assert drifts[0] < 1.0


def test_a_looser_tolerance_never_scores_worse():
    gt = _pores_away_from_the_edge()
    pred = np.roll(gt, 3, axis=0)

    scores = boundary_scores(gt, pred, scales=DIAGNOSTIC_SCALES)
    ordered = [scores[scale].f1 for scale in sorted(scores)]

    assert ordered == sorted(ordered)


def test_matches_cellpose_when_nothing_touches_the_frame_edge():
    gt = _pores_away_from_the_edge()
    pred = np.roll(gt, 2, axis=0)

    ours = boundary_scores(gt, pred, scales=DIAGNOSTIC_SCALES)
    reference = cellpose_boundary_scores([gt], [pred],
                                         list(DIAGNOSTIC_SCALES))

    for index, scale in enumerate(DIAGNOSTIC_SCALES):
        assert ours[scale].precision == pytest.approx(reference[0][index, 0])
        assert ours[scale].recall == pytest.approx(reference[1][index, 0])
        assert ours[scale].f1 == pytest.approx(reference[2][index, 0])


def test_frame_edge_outlines_are_actually_dropped():
    gt = _pores_touching_the_edge()
    pred = np.roll(gt, 2, axis=0)

    ours = boundary_scores(gt, pred)[DECISION_SCALE].f1
    reference = cellpose_boundary_scores(
        [gt], [pred], [DECISION_SCALE]
    )[2][0, 0]

    assert ours != pytest.approx(reference)
    assert ours < reference


def test_tolerance_is_reported_and_scales_with_pore_size():
    gt = _pores_away_from_the_edge()

    scores = boundary_scores(gt, gt.copy(), scales=DIAGNOSTIC_SCALES)
    tolerances = [scores[scale].tolerance_px for scale in sorted(scores)]

    assert tolerances == sorted(tolerances)
    assert all(tolerance >= 1.0 for tolerance in tolerances)


def test_a_run_scores_one_tolerance_and_it_is_the_decision_one():
    """The default is what an evaluation pays for on every image.

    Scoring the wider sweep on every evaluation would cost several
    times the training it is meant to judge, so the default holds the
    decision tolerance alone and the sweep is a separate study.
    """
    assert BOUNDARY_SCALES == (DECISION_SCALE,)
    assert DECISION_SCALE in DIAGNOSTIC_SCALES


def test_tolerance_comes_from_the_annotation_not_the_prediction():
    gt = _pores_away_from_the_edge()
    merged = gt.copy()
    merged[merged == 2] = 1

    from_gt = boundary_scores(gt, merged)[DECISION_SCALE].tolerance_px
    from_merged = boundary_scores(merged, gt)[DECISION_SCALE].tolerance_px

    assert from_gt < from_merged


def test_empty_prediction_scores_zero_recall():
    gt = _pores_away_from_the_edge()
    pred = np.zeros(FRAME, dtype=np.int32)

    score = boundary_scores(gt, pred)[DECISION_SCALE]

    assert score.recall == 0.0
    assert np.isnan(score.precision)
    assert np.isnan(score.f1)


def test_empty_annotation_gives_no_tolerance_to_score_at():
    empty = np.zeros(FRAME, dtype=np.int32)

    score = boundary_scores(empty, _pores_away_from_the_edge())[DECISION_SCALE]

    assert np.isnan(score.tolerance_px)
    assert np.isnan(score.f1)


def test_two_empty_frames_give_nan():
    empty = np.zeros(FRAME, dtype=np.int32)

    score = boundary_scores(empty, empty.copy())[DECISION_SCALE]

    assert np.isnan(score.f1)


def test_labels_are_not_modified_by_scoring():
    gt = _pores_touching_the_edge()
    before = gt.copy()

    boundary_scores(gt, gt.copy())

    assert np.array_equal(gt, before)


@pytest.mark.parametrize("gt, pred, message", [
    (np.zeros((4, 4), np.int32), np.zeros((5, 5), np.int32),
     "different frames"),
    (np.zeros((4, 4, 2), np.int32), np.zeros((4, 4, 2), np.int32),
     "must be 2-D"),
    (np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32),
     "integer labels"),
])
def test_malformed_input_is_rejected(gt, pred, message):
    with pytest.raises(ValueError, match=message):
        boundary_scores(gt, pred)


@pytest.mark.parametrize("scales", [(), (0.0,), (0.1, -0.2)])
def test_invalid_scales_are_rejected(scales):
    frame = _pores_away_from_the_edge()

    with pytest.raises(ValueError):
        boundary_scores(frame, frame.copy(), scales=scales)
