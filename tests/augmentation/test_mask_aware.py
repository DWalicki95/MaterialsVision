"""Tests for the shading painted inside pores.

The property everything else rests on is that the annotation comes
back untouched: this family reads the mask on every sample and must
never write to it. After that come the two properties that decide
whether the shading is usable at all - it must fade to nothing exactly
at a pore's edge, and it must leave the texture it passes over intact.
A shading that fails either would be teaching the model the boundary
it is supposed to teach it to ignore.
"""
import random

import numpy as np
import pytest

from materials_vision.augmentation.config import (FAMILY_MASK_AWARE,
                                                  MaskAwareConfig,
                                                  PolicyConfig)
from materials_vision.augmentation.mask_aware import (PoreBrightnessField,
                                                      PoreDarkening)
from materials_vision.augmentation.policy import AugmentationPolicy

SEEDS = tuple(range(201, 225))


class _FakeRecord:
    """Stands in for SampleRecord; the policy reads three fields."""

    def __init__(
        self, image_id="AS1_40_x1", scale_bin="coarse", q_max_i=1.31
    ):
        self.image_id = image_id
        self.scale_bin = scale_bin
        self.q_max_i = q_max_i


def _sample(height=80, width=120, rows=4, cols=6):
    """A frame of pores wide enough to hold shading and a patch."""
    rng = np.random.default_rng(7)
    image = rng.integers(40, 210, (height, width), dtype=np.uint8)
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


def _outline(labels):
    """Pore pixels lying directly against something outside the pore."""
    padded = np.pad(labels, 1, constant_values=0)
    same = np.ones(labels.shape, dtype=bool)
    for rows, columns in ((0, 1), (2, 1), (1, 0), (1, 2)):
        neighbour = padded[
            rows:rows + labels.shape[0],
            columns:columns + labels.shape[1],
        ]
        same &= neighbour == labels
    return (labels > 0) & ~same


def _apply(transform, image, labels, seed):
    transform.set_random_seed(seed)
    return transform(image=image, mask=labels)


def test_shading_leaves_the_annotation_bitwise_identical():
    """The family reads the mask on every sample and writes to none."""
    image, labels = _sample()
    transform = PoreBrightnessField(MaskAwareConfig())

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.array_equal(result["mask"], labels)
        assert result["mask"].dtype == labels.dtype


def test_darkening_leaves_the_annotation_bitwise_identical():
    image, labels = _sample()
    transform = PoreDarkening(MaskAwareConfig())

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.array_equal(result["mask"], labels)
        assert result["mask"].dtype == labels.dtype


def test_the_shading_is_exactly_zero_on_a_pore_edge():
    """A step in brightness on the boundary is a boundary drawn in.

    The whole point of fading the shading out is that the edge of a
    pore looks no different afterwards than it did before, so the
    model has nothing new to mistake for one.
    """
    image, labels = _sample()
    transform = PoreBrightnessField(
        MaskAwareConfig(strength=(0.5, 0.5), pore_fraction=(1.0, 1.0))
    )
    edge = _outline(labels)

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.array_equal(result["image"][edge], image[edge])


def test_nothing_outside_a_pore_is_shaded():
    """The walls between pores are not this family's business."""
    image, labels = _sample()
    transform = PoreBrightnessField(
        MaskAwareConfig(strength=(0.5, 0.5), pore_fraction=(1.0, 1.0))
    )
    outside = labels == 0

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.array_equal(
            result["image"][outside], image[outside]
        )


def test_a_dark_patch_keeps_clear_of_the_boundary():
    """A patch that reached the edge would deform the edge."""
    image, labels = _sample()
    transform = PoreDarkening(MaskAwareConfig())
    edge = _outline(labels)
    outside = labels == 0

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.array_equal(result["image"][edge], image[edge])
        assert np.array_equal(
            result["image"][outside], image[outside]
        )


def test_a_dark_patch_only_ever_darkens():
    """It stands for seeing further into the material, never less."""
    image, labels = _sample()
    transform = PoreDarkening(MaskAwareConfig())

    for seed in SEEDS:
        result = _apply(transform, image, labels, seed)

        assert np.all(result["image"] <= image)


def test_a_dark_patch_actually_darkens_something():
    image, labels = _sample()
    transform = PoreDarkening(
        MaskAwareConfig(darkening_factor=(0.6, 0.6))
    )

    changed = [
        int(np.count_nonzero(
            _apply(transform, image, labels, seed)["image"] != image
        ))
        for seed in SEEDS
    ]

    assert all(count > 0 for count in changed)


def test_shading_leaves_the_texture_it_passes_over_intact():
    """Shading is added, so the detail underneath keeps its contrast.

    Measured as the spread of differences between neighbouring pixels:
    a transformation that flattened or sharpened the texture would
    change it, and one that only slides whole neighbourhoods up and
    down cannot.
    """
    image, labels = _sample()
    transform = PoreBrightnessField(
        MaskAwareConfig(strength=(0.15, 0.15), pore_fraction=(1.0, 1.0))
    )

    before = float(np.std(np.diff(image.astype(np.float32), axis=1)))
    for seed in SEEDS[:8]:
        shaded = _apply(transform, image, labels, seed)["image"]
        after = float(
            np.std(np.diff(shaded.astype(np.float32), axis=1))
        )

        assert abs(after - before) / before < 0.05


def test_shading_moves_the_brightness_of_pore_interiors():
    """The transformation has to do something, not merely be safe."""
    image, labels = _sample()
    transform = PoreBrightnessField(
        MaskAwareConfig(strength=(0.15, 0.15), pore_fraction=(1.0, 1.0))
    )

    for seed in SEEDS[:8]:
        shaded = _apply(transform, image, labels, seed)["image"]

        assert not np.array_equal(shaded, image)


def test_the_strength_is_measured_against_the_image_own_range():
    """A fixed number of grey levels means different things per image.

    The two microscopes expose differently, so an amplitude that is a
    gentle shading on one would be a heavy one on the other.
    """
    image, labels = _sample()
    transform = PoreBrightnessField(MaskAwareConfig())

    _apply(transform, image, labels, 3)
    params = transform.params
    low, high = np.percentile(image, (5.0, 95.0))

    assert params["tonal_span"] == pytest.approx(float(high - low))
    assert 0.08 <= params["strength"] <= 0.15
    assert params["amplitude"] == pytest.approx(
        params["strength"] * params["tonal_span"], abs=1e-2
    )


def test_pores_too_small_to_hold_shading_are_left_alone():
    """Below a few pixels of depth the shading would amount to nothing."""
    image, _ = _sample()
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[10:12, 10:13] = 1
    labels[20:22, 20:23] = 2
    transform = PoreBrightnessField(MaskAwareConfig())

    result = _apply(transform, image, labels, 3)

    assert transform.params["n_pores_eligible"] == 0
    assert transform.params["fallback"] == (
        "no_pore_is_deep_enough_to_shade"
    )
    assert np.array_equal(result["image"], image)


def test_a_frame_with_no_pore_at_all_is_reported_not_skipped():
    """A sample nothing happened to still counts in the comparison."""
    image, _ = _sample()
    labels = np.zeros(image.shape, dtype=np.int32)
    transform = PoreDarkening(MaskAwareConfig())

    result = _apply(transform, image, labels, 3)

    assert transform.params["fallback"] is not None
    assert transform.params["n_pores_darkened"] == 0
    assert np.array_equal(result["image"], image)


def test_a_patch_that_cannot_be_fitted_gives_up_after_bounded_tries():
    """Asking for a patch larger than the pore can never succeed."""
    image, labels = _sample()
    transform = PoreDarkening(
        MaskAwareConfig(
            darkened_area=(0.99, 0.99), darkening_max_attempts=4,
        )
    )

    result = _apply(transform, image, labels, 3)

    assert transform.params["fallback"] == (
        "no_patch_fitted_clear_of_a_boundary"
    )
    assert transform.params["attempts"] >= 4
    assert np.array_equal(result["image"], image)


def test_the_same_seed_reproduces_the_shading():
    image, labels = _sample()
    field = PoreBrightnessField(MaskAwareConfig())
    patch = PoreDarkening(MaskAwareConfig())

    for transform in (field, patch):
        first = _apply(transform, image, labels, 42)["image"]
        second = _apply(transform, image, labels, 42)["image"]

        assert np.array_equal(first, second)


def test_every_shape_of_shading_is_drawn():
    image, labels = _sample()
    transform = PoreBrightnessField(MaskAwareConfig())

    kinds = set()
    for seed in SEEDS:
        _apply(transform, image, labels, seed)
        kinds.add(transform.params["kind"])

    assert kinds == {"constant", "gradient", "random"}


def test_the_painted_fields_never_reach_the_record():
    """A record is written to the log; an image cannot go there."""
    image, labels = _sample()
    policy = AugmentationPolicy(
        PolicyConfig(mask_aware=MaskAwareConfig(p=1.0))
    )

    for seed in SEEDS:
        entry = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        ).record.transforms[0]

        assert entry.family == FAMILY_MASK_AWARE
        assert entry.name in {"PoreBrightnessField", "PoreDarkening"}
        assert "delta" not in entry.params
        assert "attenuation" not in entry.params
        assert "shape" not in entry.params
        assert all(not isinstance(value, np.ndarray)
                   for value in entry.params.values())


def test_both_members_are_drawn_and_never_both_at_once():
    """Compounded in one pore they would show an impossible interior."""
    image, labels = _sample()
    policy = AugmentationPolicy(
        PolicyConfig(mask_aware=MaskAwareConfig(p=1.0))
    )

    drawn = [
        policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        ).record.transforms
        for seed in SEEDS
    ]

    assert all(len(entry) == 1 for entry in drawn)
    assert {entry[0].name for entry in drawn} == {
        "PoreBrightnessField", "PoreDarkening",
    }


def test_the_family_does_not_touch_the_global_random_state():
    """Image order is drawn elsewhere and must stay independent."""
    image, labels = _sample()
    policy = AugmentationPolicy(
        PolicyConfig(mask_aware=MaskAwareConfig(p=1.0))
    )
    np.random.seed(0)
    random.seed(0)
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    for seed in SEEDS:
        policy.apply(image, labels, record=_FakeRecord(), seed=seed)

    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state


def test_settings_that_could_not_describe_a_draw_are_refused():
    with pytest.raises(ValueError, match="runs backwards"):
        MaskAwareConfig(strength=(0.2, 0.1))

    with pytest.raises(ValueError, match="unknown field kind"):
        MaskAwareConfig(field_kinds=("constant", "swirl"))

    with pytest.raises(ValueError, match="no room for a patch"):
        MaskAwareConfig(
            min_core_distance_px=2.0, darkening_margin_px=2.0
        )

    with pytest.raises(ValueError, match="darkening_edge_softness"):
        MaskAwareConfig(darkening_edge_softness=0.0)
