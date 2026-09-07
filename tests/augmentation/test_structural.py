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

        assert min(params["fragment_ratios"]) >= 0.25
        assert params["smallest_fragment_px2"] >= 200.0


def test_only_the_divided_pores_change():
    """Every other instance keeps its identity and its every pixel."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instances"]
        untouched = (labels > 0) & ~np.isin(labels, divided)

        assert np.array_equal(
            result.labels[untouched], labels[untouched]
        )


def test_the_wall_is_background_in_the_annotation():
    """A wall belongs to no pore, so its pixels belong to no instance."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        for divided in _entry(result).params["divided_instances"]:
            halves = result.labels[labels == divided]

            assert np.count_nonzero(halves == 0) > 0


def test_the_walls_are_drawn_only_inside_the_pores_they_divide():
    """They join the walls already there; they do not paint over them."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instances"]
        elsewhere = ~np.isin(labels, divided)

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
        for divided, target in zip(
            params["divided_instances"], params["target_intensities"]
        ):
            was_pore = labels == divided
            changed = was_pore & (result.image != image)
            core = was_pore & (result.labels == 0)

            assert changed.any()
            assert np.median(result.image[changed]) > np.median(
                image[changed]
            )
            assert float(
                np.median(result.image[core])
            ) == pytest.approx(target, abs=2.0)


def test_the_walls_are_as_wide_as_real_ones():
    """Their width is a measurement, so it has to come out as measured."""
    image, labels = _sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        for thickness in _entry(result).params["thickness_px"]:
            assert 2.0 <= thickness <= 4.0


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
        for divided in _entry(result).params["divided_instances"]:
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
        assert _entry(result).params["divided_instances"] == (1,)


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


def _dense_sample():
    """A frame of twenty-five pores, so a pool worth sharing exists."""
    return _sample(height=300, width=300, rows=5, cols=5)


def test_a_sample_receives_several_walls():
    """The reference work divides many pores per image; one wall on an
    image of dozens leaves the error being trained against too rare in
    the sample to be learned from."""
    image, labels = _dense_sample()
    policy = _policy()

    counts = {
        _entry(result).params["n_septa_drawn"]
        for result in _divided(policy, image, labels)
    }

    assert max(counts) > 1


def test_no_pore_is_divided_twice():
    """Cutting a fragment again would make slivers in a chain."""
    image, labels = _dense_sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        divided = _entry(result).params["divided_instances"]

        assert len(set(divided)) == len(divided)


def test_the_instance_count_grows_by_the_number_of_walls():
    """Each wall turns one pore into two and nothing else changes."""
    image, labels = _dense_sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        params = _entry(result).params

        assert params["n_instances_after"] - params[
            "n_instances_before"
        ] == params["n_septa_drawn"]
        assert int(result.labels.max()) == params["n_instances_after"]


def test_the_numbering_stays_dense_across_several_walls():
    """A gap silently shifts every per-instance array downstream, and
    each wall adds an id, so the risk grows with the count."""
    image, labels = _dense_sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        present = np.unique(result.labels)

        assert np.array_equal(present, np.arange(present.max() + 1))


def test_every_half_of_every_wall_is_whole():
    image, labels = _dense_sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        components = connected_components(
            result.labels, background=0, connectivity=1
        )

        assert int(components.max()) == int(result.labels.max())


def test_the_count_follows_the_size_of_the_pool():
    """A share of the pool, so a sparse and a dense image are treated
    alike relative to what each of them offers."""
    policy = _policy(rate=(0.5, 0.5))
    sparse = _sample(height=300, width=300, rows=2, cols=2)
    dense = _dense_sample()

    def drawn(sample):
        results = _divided(policy, *sample)
        return max(
            _entry(result).params["n_septa_drawn"] for result in results
        )

    assert drawn(dense) > drawn(sparse)


def test_the_area_cap_bounds_what_a_sample_may_lose():
    """Walls go into the biggest pores, so a count alone does not say
    how much of the image changed."""
    image, labels = _dense_sample()
    policy = _policy(max_divided_area_share=0.15, rate=(1.0, 1.0))

    for result in _divided(policy, image, labels):
        params = _entry(result).params

        assert params["divided_area_share"] <= 0.15
        assert params["n_septa_drawn"] < params["n_septa_requested"]


def test_the_count_never_exceeds_its_cap():
    image, labels = _dense_sample()
    policy = _policy(rate=(1.0, 1.0), count_cap=2)

    for result in _divided(policy, image, labels):
        assert _entry(result).params["n_septa_drawn"] <= 2


def test_the_record_says_what_was_asked_for_and_what_was_drawn():
    """Without both, a sample short of walls cannot be told from one
    that never wanted them."""
    image, labels = _dense_sample()
    policy = _policy()

    for result in _divided(policy, image, labels):
        params = _entry(result).params

        assert params["n_septa_requested"] >= params["n_septa_drawn"]
        assert params["candidate_pool"] >= params["n_septa_requested"]
        assert 0.0 < params["divided_area_share"] <= 1.0


def test_walls_are_drawn_both_straight_and_curved():
    image, labels = _sample()
    policy = _policy()

    sags = {
        round(sag, 3)
        for result in _divided(policy, image, labels)
        for sag in _entry(result).params["sags"]
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


class TestTheWallContrast:
    """Drawn from the walls the images contain, floored where it would
    stop being a wall.

    Holding the contrast at one value made every synthetic wall equally
    bright, which no micrograph is; holding it at the ninetieth
    percentile of the measured walls was the alternative and would have
    taught the bright membrane the model already separates instead of
    the faint one it misses. What the range cannot be allowed to do is
    produce a wall too faint to survive the model's resize, because
    that sample's annotation would claim two pores over a picture
    showing one - a wrong label rather than a hard example.
    """

    @staticmethod
    def _flat_sample():
        """A frame whose whole tonal range is narrow."""
        image, labels = _sample()
        image = (image // 8 + 100).astype(np.uint8)
        image[labels == 0] = 112
        return image, labels

    def test_the_contrast_varies_between_walls(self) -> None:
        image, labels = _sample()
        policy = _policy(contrast=(0.111, 0.280))

        drawn = {
            contrast
            for result in _divided(policy, image, labels)
            for contrast in _entry(result).params["contrasts"]
        }

        assert len(drawn) > 1

    def test_every_draw_stays_inside_the_measured_range(self) -> None:
        image, labels = _sample()
        policy = _policy(contrast=(0.111, 0.280), min_contrast_grey=0.0)

        for result in _divided(policy, image, labels):
            for contrast in _entry(result).params["contrasts"]:
                assert 0.111 <= contrast <= 0.280

    def test_a_flat_image_has_its_wall_raised_to_the_floor(self) -> None:
        """On an image spanning a few dozen grey levels the faint end of
        the range is not a wall at all, and the floor is what says so."""
        image, labels = self._flat_sample()
        low, high = np.percentile(image, (5.0, 95.0))
        span = float(high) - float(low)
        policy = _policy(
            contrast=(0.111, 0.111), min_contrast_grey=11.0
        )

        produced = _divided(policy, image, labels)

        assert produced
        assert span * 0.111 < 11.0
        for result in produced:
            for contrast in _entry(result).params["contrasts"]:
                # The record rounds the contrast to four places, so the
                # floor is met to within that rounding rather than
                # exactly.
                assert contrast * span >= 11.0 - 1e-4 * span

    def test_the_floor_leaves_a_contrasty_image_alone(self) -> None:
        """A guard that binds everywhere has replaced the rule."""
        image, labels = _sample()
        policy = _policy(
            contrast=(0.280, 0.280), min_contrast_grey=11.0
        )

        for result in _divided(policy, image, labels):
            for contrast in _entry(result).params["contrasts"]:
                assert contrast == pytest.approx(0.280)


def test_settings_that_could_not_describe_a_draw_are_refused():
    with pytest.raises(ValueError, match="fragment_ratio"):
        SeptumConfig(fragment_ratio=0.8)

    with pytest.raises(ValueError, match="contrast"):
        SeptumConfig(contrast=(0.3, 0.1))

    with pytest.raises(ValueError, match="min_contrast_grey"):
        SeptumConfig(min_contrast_grey=-1.0)

    with pytest.raises(ValueError, match="thickness_px"):
        SeptumConfig(thickness_px=(4.0, 2.0))

    with pytest.raises(ValueError, match="min_chord_share"):
        SeptumConfig(min_chord_share=0.0)

    with pytest.raises(ValueError, match="edge_softness_px"):
        SeptumConfig(edge_softness_px=0.0)
