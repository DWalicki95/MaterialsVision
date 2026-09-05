"""Tests for the wall drawn across a pore.

This is the only family that changes what the annotation says, so the
tests are about the annotation more than about the picture. A division
must leave exactly two pores where there was one, both of them whole,
both large enough to be instances, and the numbering must stay dense -
a gap in it silently shifts every per-instance array downstream. The
wall must also survive being seen at the scale the model works at,
which is the whole point of measuring real walls to draw it.
"""
import random

import numpy as np
import pytest
from skimage.measure import label as connected_components

from materials_vision.augmentation.config import (FAMILY_SEPTUM, PolicyConfig,
                                                  SeptumConfig)
from materials_vision.augmentation.policy import AugmentationPolicy
from materials_vision.augmentation.walls import measure_walls, summarize_walls

SEEDS = tuple(range(301, 331))


class _FakeRecord:
    """Stands in for SampleRecord; the policy reads three fields."""

    def __init__(
        self, image_id="AS1_40_x1", scale_bin="coarse", q_max_i=1.31
    ):
        self.image_id = image_id
        self.scale_bin = scale_bin
        self.q_max_i = q_max_i


def _sample(height=120, width=160, rows=2, cols=2):
    """A frame of pores wide enough to be worth dividing."""
    rng = np.random.default_rng(7)
    image = rng.integers(40, 120, (height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int32)
    cell_height, cell_width = height // rows, width // cols
    for index in range(rows * cols):
        row, column = divmod(index, cols)
        top = row * cell_height + 4
        left = column * cell_width + 4
        labels[
            top:top + cell_height - 8, left:left + cell_width - 8
        ] = index + 1
    image[labels == 0] = 200
    return image, labels


def _config(**overrides):
    """A septum configuration sized for the small test frames."""
    settings = {"p": 1.0, "min_fragment_area_px2": 200.0}
    settings.update(overrides)
    return SeptumConfig(**settings)


def _policy(**overrides):
    return AugmentationPolicy(PolicyConfig(septum=_config(**overrides)))


def _entry(result):
    """The one transformation record a septum-only policy produces."""
    return result.record.transforms[0]


def _divided(policy, image, labels, seeds=SEEDS):
    """Every sample of the given seeds where a wall was drawn."""
    produced = []
    for seed in seeds:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )
        if _entry(result).params["changed_mask"]:
            produced.append(result)
    return produced


def test_a_division_adds_exactly_one_instance():
    """One pore becomes two; nothing else may appear or vanish."""
    image, labels = _sample()
    policy = _policy()

    produced = _divided(policy, image, labels)

    assert produced
    for result in produced:
        assert int(result.labels.max()) == int(labels.max()) + 1


def test_the_numbering_stays_dense_after_a_division():
    """A gap shifts every per-instance array that reads the mask."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        present = np.unique(result.labels)

        assert np.array_equal(present, np.arange(present.max() + 1))


def test_both_halves_are_whole():
    """An instance in two pieces teaches the model to split a pore."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        components = connected_components(
            result.labels, background=0, connectivity=1
        )

        assert int(components.max()) == int(result.labels.max())


def test_neither_half_is_a_sliver():
    """A wall that shaved the rim has not divided anything."""
    image, labels = _sample()
    policy = _policy(fragment_ratio=0.25)

    for result in _divided(policy, image, labels):
        params = _entry(result).params

        assert params["fragment_ratio"] >= 0.25
        assert min(params["fragment_areas_px2"]) >= 200.0


def test_only_the_divided_pore_changes():
    """Every other instance keeps its identity and its every pixel."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instance"]
        untouched = (labels > 0) & (labels != divided)

        assert np.array_equal(
            result.labels[untouched], labels[untouched]
        )


def test_the_wall_is_background_in_the_annotation():
    """A wall belongs to no pore, so its pixels belong to no instance."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instance"]
        was_pore = labels == divided
        halves = result.labels[was_pore]

        assert np.count_nonzero(halves == 0) > 0


def test_the_wall_is_drawn_only_inside_the_pore_it_divides():
    """It joins the walls already there; it does not paint over them."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instance"]
        elsewhere = labels != divided

        assert np.array_equal(
            result.image[elsewhere], image[elsewhere]
        )


def test_the_wall_is_brighter_than_the_pore_it_divides():
    """Drawn as a dark line it would teach the model to hunt for one.

    The wall is painted towards a brightness, not added on top of what
    was there, so a pixel of pore texture that happened to be brighter
    than a wall comes down to meet it. What has to rise is the region
    as a whole, and its centre has to arrive at the brightness that
    was measured on real walls.
    """
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        params = _entry(result).params
        was_pore = labels == params["divided_instance"]
        changed = was_pore & (result.image != image)
        core = was_pore & (result.labels == 0)

        assert changed.any()
        assert np.median(result.image[changed]) > np.median(
            image[changed]
        )
        assert float(
            np.median(result.image[core])
        ) == pytest.approx(params["target_intensity"], abs=2.0)


def test_the_wall_is_as_wide_as_a_real_one():
    """Its width is a measurement, so it has to come out as measured."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        assert 2.0 <= _entry(result).params["thickness_px"] <= 4.0


def test_the_wall_survives_the_scale_the_model_works_at():
    """A wall lost to the resize would divide the annotation only.

    The model reads the image at four fifths of its size. A wall that
    disappears there leaves a pore that looks whole but is labelled as
    two, which is worse than not augmenting the sample at all.
    """
    from skimage.transform import resize

    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instance"]
        was_pore = labels == divided
        shape = (
            round(image.shape[0] * 0.8), round(image.shape[1] * 0.8)
        )
        before = resize(
            np.where(was_pore, image, 0), shape, order=1,
            preserve_range=True, anti_aliasing=True,
        )
        after = resize(
            np.where(was_pore, result.image, 0), shape, order=1,
            preserve_range=True, anti_aliasing=True,
        )

        assert float(np.abs(after - before).max()) > 5.0


def test_only_the_larger_pores_are_ever_divided():
    """Dividing a small pore invents two nobody would have drawn."""
    image, _ = _sample()
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[10:70, 10:70] = 1
    labels[90:100, 90:100] = 2
    policy = _policy(candidate_fraction=(0.5, 0.5))

    for result in _divided(policy, image, labels):
        assert _entry(result).params["divided_instance"] == 1


def test_a_frame_with_no_pore_large_enough_is_left_alone():
    image, _ = _sample()
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[10:20, 10:20] = 1
    policy = _policy()

    result = policy.apply(image, labels, record=_FakeRecord(), seed=3)
    entry = _entry(result)

    assert entry.fallback == "no_pore_large_enough"
    assert np.array_equal(result.labels, labels)
    assert np.array_equal(result.image, image)


def test_a_frame_with_no_pore_at_all_is_reported_not_skipped():
    image, _ = _sample()
    labels = np.zeros(image.shape, dtype=np.int32)
    policy = _policy()

    result = policy.apply(image, labels, record=_FakeRecord(), seed=3)

    assert _entry(result).fallback == "frame_holds_no_pore"
    assert np.array_equal(result.labels, labels)


def test_the_family_gives_up_after_a_bounded_number_of_draws():
    """A wall that cannot divide anything must not be retried forever."""
    image, labels = _sample()
    policy = _policy(fragment_ratio=0.5, max_retries=3)

    entry = _entry(
        policy.apply(image, labels, record=_FakeRecord(), seed=3)
    )

    if entry.fallback is not None:
        assert entry.fallback == "no_wall_divided_the_pore_in_two"
        assert entry.attempts == 4


def test_a_sample_left_undivided_is_still_recorded():
    """How often the family fires is part of what it is compared on."""
    image, labels = _sample()
    policy = _policy(p=0.0)

    entry = _entry(
        policy.apply(image, labels, record=_FakeRecord(), seed=3)
    )

    assert entry.applied is False
    assert entry.family == FAMILY_SEPTUM


def test_the_same_seed_reproduces_the_division():
    image, labels = _sample()
    policy = _policy()

    first = policy.apply(image, labels, record=_FakeRecord(), seed=42)
    second = policy.apply(image, labels, record=_FakeRecord(), seed=42)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.labels, second.labels)


def test_the_divided_arrays_never_reach_the_record():
    """A record is written to the log; an image cannot go there."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels, SEEDS[:8]):
        params = _entry(result).params

        assert "walled_image" not in params
        assert "divided_labels" not in params
        assert "shape" not in params
        assert all(not isinstance(value, np.ndarray)
                   for value in params.values())


def test_walls_are_drawn_both_straight_and_curved():
    image, labels = _sample()
    policy = _policy()

    sags = {
        round(_entry(result).params["sag"], 3)
        for result in _divided(policy, image, labels)
    }

    assert len(sags) > 1


def test_the_family_does_not_touch_the_global_random_state():
    """Image order is drawn elsewhere and must stay independent."""
    image, labels = _sample()
    policy = _policy()
    np.random.seed(0)
    random.seed(0)
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    for seed in SEEDS:
        policy.apply(image, labels, record=_FakeRecord(), seed=seed)

    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state


def test_a_wall_is_measured_at_the_width_it_was_drawn():
    """What the calibration measures and what it feeds must agree.

    A frame with walls of a known width is measured by the same code
    that produced the frozen numbers, so an error in the measurement
    would show up as a width nobody drew.
    """
    labels = np.zeros((60, 90), dtype=np.int32)
    labels[10:50, 10:40] = 1
    labels[10:50, 46:76] = 2
    image = np.full(labels.shape, 80, dtype=np.uint8)
    image[labels == 0] = 180

    sample = measure_walls(image, labels)
    summary = summarize_walls([sample])

    assert summary.thickness_px == pytest.approx((6.0, 6.0), abs=1.0)
    assert sample.contrast > 0.0


def test_settings_that_could_not_describe_a_draw_are_refused():
    with pytest.raises(ValueError, match="fragment_ratio"):
        SeptumConfig(fragment_ratio=0.8)

    with pytest.raises(ValueError, match="thickness_px"):
        SeptumConfig(thickness_px=(4.0, 2.0))

    with pytest.raises(ValueError, match="min_chord_share"):
        SeptumConfig(min_chord_share=0.0)

    with pytest.raises(ValueError, match="edge_softness_px"):
        SeptumConfig(edge_softness_px=0.0)
