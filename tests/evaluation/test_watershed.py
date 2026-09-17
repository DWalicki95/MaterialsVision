"""Tests for the settings that separate touching pores.

These values decide how many instances a prediction contains, so two
runs compared to attribute anything to augmentation have to share them.
The tests below pin the frozen defaults and the shape of the robustness
cross-section, so that changing either requires changing a test and
noticing.
"""
import pytest

from materials_vision.evaluation.watershed import (
    FROZEN_WATERSHED, LIBRARY_DEFAULT_WATERSHED, ROBUSTNESS_CENTER_THRESHOLDS,
    WatershedParams, robustness_series)


def test_the_library_defaults_are_restated_not_inherited():
    """What every run before the first calibration was scored under.

    Restated here so that a dependency changing its mind cannot
    silently change what an old result meant.
    """
    assert LIBRARY_DEFAULT_WATERSHED == WatershedParams(
        center_distance_threshold=0.5,
        boundary_distance_threshold=0.5,
        foreground_threshold=0.5,
        foreground_smoothing=1.0,
        distance_smoothing=1.6,
        min_size=0,
    )


def test_the_frozen_setting_is_the_calibrated_one():
    """Seeding is tighter than the library's, in both thresholds.

    Fine-tuning sharpens the distance maps these thresholds are read
    against, so the library's values seed too freely for a trained
    decoder and cut single pores in two. Loosening them back would not
    merely lower a score: it moves the epoch a run appears to peak at
    from its fifth to its first, which is a different conclusion about
    how long to train rather than a different number.
    """
    assert FROZEN_WATERSHED.center_distance_threshold == 0.30
    assert FROZEN_WATERSHED.boundary_distance_threshold == 0.40
    assert (
        FROZEN_WATERSHED.center_distance_threshold
        < LIBRARY_DEFAULT_WATERSHED.center_distance_threshold
    )


def test_calibration_left_everything_but_the_seeding_alone():
    """Only the two thresholds that decide where seeds appear moved."""
    assert (
        FROZEN_WATERSHED.foreground_threshold
        == LIBRARY_DEFAULT_WATERSHED.foreground_threshold
    )
    assert (
        FROZEN_WATERSHED.distance_smoothing
        == LIBRARY_DEFAULT_WATERSHED.distance_smoothing
    )
    assert (
        FROZEN_WATERSHED.foreground_smoothing
        == LIBRARY_DEFAULT_WATERSHED.foreground_smoothing
    )


def test_nothing_is_filtered_out_of_a_prediction_by_size():
    """The annotation keeps its small pores, so predictions must too.

    Dropping predicted instances below a size the ground truth still
    counts would score the model as having missed pores it found.
    """
    assert FROZEN_WATERSHED.min_size == 0


def test_the_setting_renders_as_the_segmenter_expects_it():
    kwargs = FROZEN_WATERSHED.to_kwargs()

    assert kwargs["center_distance_threshold"] == 0.30
    assert set(kwargs) == {
        "center_distance_threshold", "boundary_distance_threshold",
        "foreground_threshold", "foreground_smoothing",
        "distance_smoothing", "min_size",
    }


def test_the_frozen_setting_is_labelled_as_such():
    assert FROZEN_WATERSHED.label() == "frozen"


def test_a_varied_centre_threshold_names_itself():
    varied = WatershedParams(
        center_distance_threshold=0.5,
        boundary_distance_threshold=(
            FROZEN_WATERSHED.boundary_distance_threshold
        ),
    )

    assert varied.label() == "cdt=0.5"


def test_a_setting_varied_elsewhere_gets_a_fuller_name():
    """Two rows of a results table must never read as the same thing."""
    varied = WatershedParams(
        center_distance_threshold=(
            FROZEN_WATERSHED.center_distance_threshold
        ),
        boundary_distance_threshold=(
            FROZEN_WATERSHED.boundary_distance_threshold
        ),
        distance_smoothing=3.0,
    )

    assert varied.label() == "distance_smoothing=3"


def test_the_cross_section_varies_one_thing_only():
    series = robustness_series()

    assert len(series) == len(ROBUSTNESS_CENTER_THRESHOLDS)
    assert [s.center_distance_threshold for s in series] == list(
        ROBUSTNESS_CENTER_THRESHOLDS
    )
    assert {s.distance_smoothing for s in series} == {
        FROZEN_WATERSHED.distance_smoothing
    }


def test_the_cross_section_keeps_everything_else_from_its_base():
    """A calibrated setting must carry into its own robustness check.

    Otherwise the cross-section would compare the calibrated model
    against settings it was never calibrated under, and answer a
    question nobody asked.
    """
    base = WatershedParams(
        center_distance_threshold=0.35, distance_smoothing=2.4
    )

    series = robustness_series(base, (0.3, 0.5))

    assert {s.distance_smoothing for s in series} == {2.4}


def test_the_base_is_never_scored_twice():
    series = robustness_series(
        WatershedParams(center_distance_threshold=0.4), (0.4, 0.4, 0.5)
    )

    assert [s.center_distance_threshold for s in series] == [0.4, 0.5]


def test_a_setting_can_be_used_as_a_dictionary_key():
    """Scoring several settings in one pass keys results by them."""
    same = WatershedParams(
        center_distance_threshold=(
            FROZEN_WATERSHED.center_distance_threshold
        ),
        boundary_distance_threshold=(
            FROZEN_WATERSHED.boundary_distance_threshold
        ),
    )
    assert len({FROZEN_WATERSHED, same}) == 1

    with pytest.raises(Exception):
        FROZEN_WATERSHED.center_distance_threshold = 0.5
