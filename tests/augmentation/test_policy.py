"""Tests for the policy that binds the families into one pipeline.

The load-bearing tests here are
``test_the_same_seed_reproduces_the_sample`` and
``test_augmentation_does_not_touch_the_global_random_state``. Together
they are what allows two policies to be compared: the first says a run
can be repeated, the second says the augmentation cannot disturb
anything else that draws random numbers, so the only difference between
two runs is the policy itself.
"""
import random

import numpy as np
import pytest

from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_TONAL,
                                                  BlurConfig,
                                                  OrientationConfig,
                                                  PolicyConfig, ScaleConfig,
                                                  TonalConfig)
from materials_vision.augmentation.integrity import IntegrityError
from materials_vision.augmentation.policy import AugmentationPolicy

SEEDS = (11, 12, 13, 14, 15, 16, 17, 18)


class _FakeRecord:
    """Stands in for SampleRecord; the policy reads three fields."""

    def __init__(
        self, image_id="AS1_40_x1", scale_bin="coarse", q_max_i=1.31
    ):
        self.image_id = image_id
        self.scale_bin = scale_bin
        self.q_max_i = q_max_i


def _sample(height=40, width=64):
    """A rectangular frame with three instances of different sizes."""
    rng = np.random.default_rng(7)
    image = rng.integers(30, 220, (height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int32)
    labels[4:12, 4:14] = 1
    labels[16:30, 20:40] = 2
    labels[6:10, 48:60] = 3
    return image, labels


def _policy(**families):
    return AugmentationPolicy(PolicyConfig(**families))


def test_a_policy_with_no_family_is_refused():
    """No augmentation is expressed by no policy, not an idle one."""
    with pytest.raises(ValueError, match="at least one"):
        AugmentationPolicy(PolicyConfig())


def test_families_are_applied_in_a_fixed_order():
    """Order follows the pipeline, not the order they were named."""
    policy = AugmentationPolicy(PolicyConfig(
        blur=BlurConfig(), orientation=OrientationConfig(),
        tonal=TonalConfig(), scale=ScaleConfig(),
    ))

    assert policy.families == (
        FAMILY_SCALE, FAMILY_ORIENTATION, FAMILY_TONAL, FAMILY_BLUR,
    )


def test_the_policy_and_its_configuration_agree_on_the_order():
    """Two places state the order; they must not be able to diverge."""
    config = PolicyConfig(
        orientation=OrientationConfig(), tonal=TonalConfig(),
        blur=BlurConfig(), scale=ScaleConfig(),
    )

    assert AugmentationPolicy(config).families == config.families


def test_the_same_seed_reproduces_the_sample():
    image, labels = _sample()
    policy = _policy(
        orientation=OrientationConfig(), tonal=TonalConfig(p=1.0),
        blur=BlurConfig(p=1.0),
    )

    first = policy.apply(image, labels, record=_FakeRecord(), seed=42)
    second = policy.apply(image, labels, record=_FakeRecord(), seed=42)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.labels, second.labels)


def test_a_different_seed_changes_the_sample():
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    drawn = {
        policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        ).record.transforms[0].params["group_element"]
        for seed in SEEDS
    }

    assert len(drawn) > 1


def test_brightness_changes_leave_the_mask_bitwise_identical():
    image, labels = _sample()
    policy = _policy(tonal=TonalConfig(p=1.0), blur=BlurConfig(p=1.0))

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert np.array_equal(result.labels, labels)
        assert result.labels.dtype == labels.dtype
        assert not np.array_equal(result.image, image)


def test_orientation_preserves_every_instance_area():
    """A quarter turn moves pixels; it may not resample them."""
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())
    expected = np.sort(np.bincount(labels.ravel())[1:])

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert np.array_equal(
            np.sort(np.bincount(result.labels.ravel())[1:]), expected
        )


def test_a_crop_may_change_instance_areas_and_a_turn_may_not():
    """The checks after a sample follow the sample, not the policy.

    Cutting a window legitimately takes area off the instances it
    crosses, so the rule that every area survives a transformation
    cannot apply to a sample that was cut. It still applies to the
    samples the same policy left uncut, which is why the choice is
    made per sample.
    """
    image, labels = _sample(height=64, width=64)
    labels[:] = 0
    labels[8:56, 8:24] = 1
    labels[8:56, 28:44] = 2
    labels[8:56, 48:60] = 3
    policy = _policy(
        scale=ScaleConfig(
            bands=((1.0, 1.30, 1.30),), min_fragment_area_px2=20.0,
        ),
        orientation=OrientationConfig(),
    )
    before = np.sort(np.bincount(labels.ravel())[1:])

    result = policy.apply(image, labels, record=_FakeRecord(), seed=4)
    after = np.sort(np.bincount(result.labels.ravel())[1:])

    assert not np.array_equal(before, after)


def test_a_policy_that_can_cut_still_guards_the_samples_it_did_not():
    """An identity draw leaves the mask exactly as it was."""
    image, labels = _sample()
    policy = _policy(
        scale=ScaleConfig(bands=((1.0, 1.00, 1.00),)),
        tonal=TonalConfig(p=1.0),
    )

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert np.array_equal(result.labels, labels)
        assert result.labels.dtype == labels.dtype


def test_the_mask_never_gains_an_id_nobody_annotated():
    image, labels = _sample()
    policy = _policy(
        orientation=OrientationConfig(), tonal=TonalConfig(p=1.0),
        blur=BlurConfig(p=1.0),
    )
    annotated = set(np.unique(labels).tolist())

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert set(np.unique(result.labels).tolist()) == annotated


def test_a_quarter_turn_transposes_the_frame():
    """Both orientations occur and neither is stretched back."""
    image, labels = _sample(height=40, width=64)
    policy = _policy(orientation=OrientationConfig())

    shapes = {
        policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        ).image.shape
        for seed in SEEDS
    }

    assert shapes == {(40, 64), (64, 40)}


def test_the_image_and_mask_come_out_the_same_shape():
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    for seed in SEEDS:
        result = policy.apply(
            image, labels, record=_FakeRecord(), seed=seed
        )

        assert result.image.shape == result.labels.shape


def test_a_family_that_did_not_fire_is_still_recorded():
    """The firing rate is not recoverable from the samples that fired."""
    image, labels = _sample()
    policy = _policy(tonal=TonalConfig(p=0.0))

    result = policy.apply(image, labels, record=_FakeRecord(), seed=3)

    assert len(result.record.transforms) == 1
    assert result.record.transforms[0].family == FAMILY_TONAL
    assert result.record.transforms[0].applied is False
    assert result.record.transforms[0].name is None
    assert result.record.applied_families == ()


def test_the_record_carries_the_value_that_was_drawn():
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    entry = policy.apply(
        image, labels, record=_FakeRecord(), seed=5
    ).record.transforms[0]

    assert entry.applied is True
    assert entry.name == "D4"
    assert entry.params["group_element"] in {
        "e", "r90", "r180", "r270", "v", "hvt", "h", "t",
    }


def test_the_record_drops_the_frame_size_the_library_adds():
    """Every transformation reports it; none of them drew it."""
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    entry = policy.apply(
        image, labels, record=_FakeRecord(), seed=5
    ).record.transforms[0]

    assert "shape" not in entry.params


def test_blur_records_the_width_it_actually_applied():
    """Holding the kernel fixed truncates the widest draws.

    The drawn sigma and the applied one are therefore different
    numbers, and only the second describes what the model saw.
    """
    image, labels = _sample()
    policy = _policy(blur=BlurConfig(p=1.0))

    entry = policy.apply(
        image, labels, record=_FakeRecord(), seed=5
    ).record.transforms[0]

    assert entry.params["kernel_px"] == 3
    assert 0.0 < entry.params["sigma_effective_px"] < 0.8
    assert "kernel" not in entry.params


def test_the_record_carries_the_image_and_the_seed():
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    record = policy.apply(
        image, labels, record=_FakeRecord("K5_30_x2"), seed=77
    ).record

    assert record.image_id == "K5_30_x2"
    assert record.seed == 77


def test_the_call_form_returns_only_the_arrays():
    """What the dataset uses; the record cannot cross a worker."""
    image, labels = _sample()
    policy = _policy(orientation=OrientationConfig())

    augmented_image, augmented_labels = policy(
        image, labels, record=_FakeRecord(), seed=9
    )

    assert augmented_image.shape == augmented_labels.shape
    assert augmented_labels.dtype == labels.dtype


def test_augmentation_does_not_touch_the_global_random_state():
    """The property that keeps two policies comparable.

    Image order is drawn from a separate stream. If augmentation
    consumed the global one, two policies at the same seed would see
    different image orders and part of the measured difference would be
    the ordering rather than the augmentation.
    """
    image, labels = _sample()
    policy = _policy(
        scale=ScaleConfig(min_fragment_area_px2=20.0),
        orientation=OrientationConfig(), tonal=TonalConfig(p=1.0),
        blur=BlurConfig(p=1.0),
    )
    np.random.seed(0)
    random.seed(0)
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    for seed in SEEDS:
        policy.apply(image, labels, record=_FakeRecord(), seed=seed)

    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state


def test_an_empty_frame_survives_a_policy_that_keeps_the_mask():
    """An unannotated frame is not an integrity failure."""
    image, _ = _sample()
    empty = np.zeros(image.shape, dtype=np.int32)
    policy = _policy(orientation=OrientationConfig())

    result = policy.apply(image, empty, record=_FakeRecord(), seed=1)

    assert int(result.labels.max()) == 0


def test_a_mask_with_gaps_in_its_numbering_stops_the_run():
    """The check fires on the way out, naming the policy that ran."""
    image, labels = _sample()
    labels[labels == 2] = 5
    policy = _policy(orientation=OrientationConfig())

    with pytest.raises(IntegrityError, match="leave gaps"):
        policy.apply(image, labels, record=_FakeRecord(), seed=1)


def test_a_policy_satisfies_the_contract_the_dataset_calls_it_with():
    """Pins the two sides of the seam against each other.

    The policy is only ever reached through the dataset, so a change to
    either signature has to fail here rather than at the first training
    step.
    """
    from materials_vision.data.dataset import InstanceSegmentationDataset

    image, labels = _sample()

    class _Source:
        def __len__(self):
            return 2

        def load(self, index):
            class _Prepared:
                pass

            prepared = _Prepared()
            prepared.image = image
            prepared.labels = labels
            prepared.record = _FakeRecord(f"img_{index}")
            return prepared

    dataset = InstanceSegmentationDataset(
        _Source(),
        label_transform=lambda labels: np.stack([labels, labels]),
        transform=_policy(orientation=OrientationConfig()),
        run_seed=5,
    )

    x, y = dataset[1]

    assert x.shape[0] == 3
    assert x.shape[1:] == y.shape[1:]
