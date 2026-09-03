"""Tests for the frozen family parameters and how a run records them."""
from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL,
                                                  MASK_CHANGING_FAMILIES,
                                                  BlurConfig,
                                                  OrientationConfig,
                                                  PolicyConfig, ScaleConfig,
                                                  TonalConfig,
                                                  policy_run_metadata)


def test_the_starting_values_are_the_ones_that_were_approved():
    """Changing any of these changes every run that uses them."""
    tonal = TonalConfig()
    blur = BlurConfig()
    scale = ScaleConfig()

    assert tonal.brightness_limit == (-0.10, 0.10)
    assert tonal.contrast_limit == (-0.15, 0.15)
    assert tonal.gamma_limit == (90, 110)
    assert tonal.p == 0.5
    assert blur.kernel_px == 3
    assert blur.sigma_px == (0.2, 0.8)
    assert blur.p == 0.2
    assert OrientationConfig().p == 1.0
    assert scale.bands == (
        (0.50, 1.00, 1.00), (0.30, 1.05, 1.15), (0.20, 1.15, 1.30),
    )
    assert scale.q_max == 1.30
    assert scale.magnified_bins == ("coarse",)
    assert scale.min_instances == 3
    assert scale.max_retries == 5
    assert scale.min_fragment_area_px2 == 432.0
    assert scale.p == 1.0


def test_an_empty_policy_enables_nothing():
    assert PolicyConfig().families == ()


def test_families_are_listed_in_the_order_they_apply():
    """The order is the pipeline's, not the order they were named.

    Cutting a window out of the frame comes first: adjusting the
    brightness of a frame the model never sees would measure the wrong
    statistics, and turning the sample first would only mean turning
    it twice.
    """
    config = PolicyConfig(
        blur=BlurConfig(), tonal=TonalConfig(),
        orientation=OrientationConfig(), scale=ScaleConfig(),
    )

    assert config.families == (
        FAMILY_SCALE, FAMILY_ORIENTATION, FAMILY_TONAL, FAMILY_BLUR,
    )


def test_a_policy_that_cuts_a_window_can_change_the_mask():
    """What the integrity checks key off after every sample."""
    assert PolicyConfig(scale=ScaleConfig()).changes_mask is True


def test_only_cutting_and_dividing_can_change_the_mask():
    """What the integrity checks key off after every sample."""
    assert MASK_CHANGING_FAMILIES == {FAMILY_SCALE, FAMILY_SEPTUM}


def test_a_photometric_policy_does_not_change_the_mask():
    config = PolicyConfig(tonal=TonalConfig(), blur=BlurConfig())

    assert config.changes_mask is False


def test_the_run_record_holds_every_number_of_every_family():
    """The half of reproducibility the seed does not cover."""
    config = PolicyConfig(
        orientation=OrientationConfig(), blur=BlurConfig(),
    )

    metadata = policy_run_metadata(config)

    assert metadata["families"] == [FAMILY_ORIENTATION, FAMILY_BLUR]
    assert metadata["order"] == [FAMILY_ORIENTATION, FAMILY_BLUR]
    assert metadata["changes_mask"] is False
    assert metadata["parameters"][FAMILY_BLUR] == {
        "kernel_px": 3, "sigma_px": (0.2, 0.8), "p": 0.2,
    }


def test_a_family_that_is_off_is_absent_from_the_record():
    metadata = policy_run_metadata(PolicyConfig(tonal=TonalConfig()))

    assert list(metadata["parameters"]) == [FAMILY_TONAL]


def test_the_record_does_not_alias_the_configuration():
    """A run's record must not change when a config object is reused."""
    config = PolicyConfig(blur=BlurConfig())

    metadata = policy_run_metadata(config)
    metadata["parameters"][FAMILY_BLUR]["p"] = 999

    assert config.blur.p == 0.2
