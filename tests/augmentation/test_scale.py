"""Tests for the crop that varies how large a pore looks.

Two properties carry the weight here. The annotation must survive the
magnification exactly - no id may be interpolated into existence, none
may be left in two pieces, and the numbering must stay dense - and a
window that cannot be made to work must give up in a way that is
visible in the record rather than silently.
"""
import random

import numpy as np
import pytest

from materials_vision.augmentation.config import (FAMILY_SCALE, PolicyConfig,
                                                  ScaleConfig)
from materials_vision.augmentation.policy import AugmentationPolicy

SEEDS = tuple(range(101, 141))


class _FakeRecord:
    """Stands in for SampleRecord; the policy reads three fields."""

    def __init__(
        self, image_id="AS1_40_x1", scale_bin="coarse", q_max_i=1.31
    ):
        self.image_id = image_id
        self.scale_bin = scale_bin
        self.q_max_i = q_max_i


def _sample(height=64, width=96, rows=4, cols=6):
    """A frame tiled with instances, so a window holds several."""
    rng = np.random.default_rng(7)
    image = rng.integers(30, 220, (height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int32)
    cell_height, cell_width = height // rows, width // cols
    for index in range(rows * cols):
        row, column = divmod(index, cols)
        top = row * cell_height + 2
        left = column * cell_width + 2
        labels[
            top:top + cell_height - 4, left:left + cell_width - 4
        ] = index + 1
    return image, labels


def _config(**overrides):
    """A scale configuration sized for the small test frames.

    The real minimum fragment area is measured on full-resolution
    micrographs and is larger than a whole instance here, which would
    make every cut instance disappear and hide what the tests are
    about.
    """
    settings = {"min_fragment_area_px2": 20.0}
    settings.update(overrides)
    return ScaleConfig(**settings)


def _policy(**overrides):
    return AugmentationPolicy(PolicyConfig(scale=_config(**overrides)))


def _entry(result):
    """The one transformation record a scale-only policy produces."""
    return result.record.transforms[0]


def test_the_magnifications_drawn_follow_the_frozen_distribution():
    """Half the draws are the identity; none exceeds the maximum."""
    image, labels = _sample()
    policy = _policy()

    drawn = [
        _entry(
            policy.apply(image, labels, record=_FakeRecord(), seed=seed)
        ).params["q"]
        for seed in SEEDS
    ]

    assert all(q == 1.0 or 1.05 <= q <= 1.30 for q in drawn)
    assert 0.25 < drawn.count(1.0) / len(drawn) < 0.75
    assert any(q > 1.15 for q in drawn)


def test_an_image_already_at_the_finest_scale_is_never_magnified():
    """There is nothing finer for it to be magnified towards."""
    image, labels = _sample()
    policy = _policy()

    for seed in SEEDS:
        result = policy.apply(
            image, labels,
            record=_FakeRecord(scale_bin="fine"), seed=seed,
        )

        assert _entry(result).params["q"] == 1.0
        assert np.array_equal(result.image, image)
        assert np.array_equal(result.labels, labels)


def test_the_sample_comes_back_the_size_it_went_in():
    """The window is magnified back to the frame it was cut from."""
    image, labels = _sample()
    policy = _policy()

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert result.image.shape == image.shape
        assert result.labels.shape == labels.shape


def test_the_window_keeps_the_proportions_of_the_frame():
    """A window of a different shape would stretch the pores."""
    image, labels = _sample()
    policy = _policy()
    height, width = labels.shape

    for seed in SEEDS:
        params = _entry(
            policy.apply(image, labels, record=_FakeRecord(), seed=seed)
        ).params
        if params["window"] is None:
            continue
        x0, y0, x1, y1 = params["window"]

        assert abs((x1 - x0) / (y1 - y0) - width / height) < 0.02


def test_the_mask_never_gains_a_value_that_was_not_an_id():
    """An interpolated label image holds instances nobody annotated."""
    image, labels = _sample()
    policy = _policy()
    annotated = set(np.unique(labels).tolist())

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert result.labels.dtype == labels.dtype
        assert set(np.unique(result.labels).tolist()) <= annotated


def test_the_numbering_stays_dense_after_a_window_removes_instances():
    """A gap would shift every per-instance array that follows."""
    image, labels = _sample()
    policy = _policy()

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )
        present = np.unique(result.labels)
        expected = np.arange(present.max() + 1)

        assert np.array_equal(present, expected)


def test_magnifying_cannot_break_an_instance_into_two_pieces():
    """The reason the connectivity check should never fire here.

    Every magnification is an enlargement, and enlarging by nearest
    neighbour only repeats pixels that were already neighbours. The
    window itself can leave an instance in pieces, but the crop
    resolves that before the enlargement happens, so a sample that
    reaches the end of this family is always whole.
    """
    from skimage.measure import label as connected_components

    image, labels = _sample()
    policy = _policy()

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )
        components = connected_components(
            result.labels, background=0, connectivity=1
        )

        assert int(components.max()) == int(result.labels.max())


def test_an_instance_the_window_did_not_touch_survives_at_any_size():
    """The minimum area suppresses slivers the cut manufactured.

    Applied to instances the window left whole it would delete the
    smallest real annotations instead, on every single sample.
    """
    image, labels = _sample()
    policy = _policy(min_fragment_area_px2=1e6)

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )
        params = _entry(result).params
        if params["window"] is None:
            continue

        assert params["n_instances_after"] > 0
        assert params["n_dropped_below_min_area"] == params[
            "n_cut_by_crop"
        ]


def test_a_frame_with_too_few_instances_is_left_alone():
    """No window can satisfy a requirement the whole frame fails."""
    image, labels = _sample(rows=1, cols=2)
    policy = _policy()

    result = policy.apply(
        image, labels, record=_FakeRecord(), seed=3
    )

    assert _entry(result).fallback == "frame_holds_too_few_instances"
    assert _entry(result).attempts == 1
    assert np.array_equal(result.labels, labels)


def test_the_family_gives_up_after_a_bounded_number_of_draws():
    """A failed draw is expected; it must not become an endless one."""
    image, labels = _sample()
    policy = _policy(
        bands=((1.0, 1.25, 1.25),), min_instances=24, max_retries=5,
    )

    result = policy.apply(
        image, labels, record=_FakeRecord(), seed=3
    )
    entry = _entry(result)

    assert entry.fallback == "no_window_held_enough_instances"
    assert entry.attempts == 6
    assert entry.params["q"] == 1.0
    assert np.array_equal(result.image, image)
    assert np.array_equal(result.labels, labels)


def test_the_identity_draw_leaves_the_sample_bitwise_identical():
    """Half of all samples take this path, so it has to be exact."""
    image, labels = _sample()
    policy = _policy(bands=((1.0, 1.00, 1.00),))

    result = policy.apply(
        image, labels, record=_FakeRecord(), seed=3
    )
    entry = _entry(result)

    assert entry.applied is True
    assert entry.fallback is None
    assert entry.params["changed_mask"] is False
    assert np.array_equal(result.image, image)
    assert np.array_equal(result.labels, labels)


def test_the_same_seed_reproduces_the_crop():
    image, labels = _sample()
    policy = _policy()

    first = policy.apply(image, labels, record=_FakeRecord(), seed=42)
    second = policy.apply(image, labels, record=_FakeRecord(), seed=42)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.labels, second.labels)


def test_the_record_says_what_the_window_cost():
    """Instances lost to a window are part of what a run reports."""
    image, labels = _sample()
    policy = _policy(bands=((1.0, 1.30, 1.30),))

    params = _entry(
        policy.apply(image, labels, record=_FakeRecord(), seed=3)
    ).params

    assert params["n_instances_before"] == 24
    assert params["n_instances_after"] < 24
    assert params["n_instances_after"] >= 3
    assert params["n_border_instances"] >= 0
    assert params["changed_mask"] is True


def test_the_magnified_arrays_never_reach_the_record():
    """A record is written to the log; an image cannot go there."""
    image, labels = _sample()
    policy = _policy(bands=((1.0, 1.30, 1.30),))

    params = _entry(
        policy.apply(image, labels, record=_FakeRecord(), seed=3)
    ).params

    assert "scaled_image" not in params
    assert "scaled_labels" not in params
    assert "shape" not in params
    assert all(not isinstance(value, np.ndarray)
               for value in params.values())


def test_magnifying_past_an_image_own_headroom_is_recorded():
    """The per-image calibration and the per-bin policy can disagree.

    They do not on the data the policy was calibrated on, but a new
    image at an unexpected scale would be magnified beyond anything
    that was ever photographed, and that has to be visible rather than
    inferred later from a confusing result.
    """
    image, labels = _sample()
    policy = _policy(bands=((1.0, 1.30, 1.30),))

    generous = _entry(
        policy.apply(
            image, labels, record=_FakeRecord(q_max_i=1.31), seed=3
        )
    ).params
    tight = _entry(
        policy.apply(
            image, labels, record=_FakeRecord(q_max_i=1.05), seed=3
        )
    ).params

    assert generous["q_above_image_headroom"] is False
    assert tight["q_above_image_headroom"] is True


def test_the_crop_does_not_touch_the_global_random_state():
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


def test_the_family_is_recorded_under_its_own_code():
    image, labels = _sample()
    policy = _policy()

    entry = _entry(
        policy.apply(image, labels, record=_FakeRecord(), seed=3)
    )

    assert entry.family == FAMILY_SCALE
    assert entry.name == "MultiScaleCrop"


def test_a_distribution_that_does_not_describe_a_draw_is_refused():
    """A malformed distribution would produce plausible-looking runs."""
    with pytest.raises(ValueError, match="sum to 1"):
        ScaleConfig(bands=((0.5, 1.0, 1.0), (0.2, 1.1, 1.2)))

    with pytest.raises(ValueError, match="q must be >= 1"):
        ScaleConfig(bands=((1.0, 0.9, 1.2),))

    with pytest.raises(ValueError, match="range"):
        ScaleConfig(bands=((1.0, 1.2, 1.1),))
