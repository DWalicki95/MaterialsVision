"""Tests for the frozen family parameters and how a run records them."""
from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL,
                                                  MASK_CHANGING_FAMILIES,
                                                  BlurConfig, MaskAwareConfig,
                                                  OrientationConfig,
                                                  PolicyConfig, ScaleConfig,
                                                  SeptumConfig, TonalConfig,
                                                  policy_run_metadata)


def test_the_starting_values_are_the_ones_that_were_approved():
    """Changing any of these changes every run that uses them."""
    tonal = TonalConfig()
    blur = BlurConfig()
    scale = ScaleConfig()

    assert tonal.brightness_limit == (-0.07, 0.07)
    assert tonal.contrast_limit == (-0.105, 0.105)
    assert tonal.gamma_limit == (75, 125)
    assert tonal.min_magnitude_share == 0.5
    assert tonal.p == 0.5
    assert blur.sigma_px == (0.8, 1.6)
    assert blur.kernel_px == 11
    assert blur.p == 0.2
    assert OrientationConfig().p == 1.0
    assert scale.bands == (
        (0.50, 1.00, 1.00), (0.30, 1.05, 1.15), (0.20, 1.15, 1.30),
    )
    assert scale.q_max == 1.30
    assert scale.magnified_bins == ("coarse",)
    assert scale.min_instances == 3
    assert scale.max_retries == 5
    assert scale.min_fragment_area_px2 == 388.43
    assert scale.p == 1.0

    septum = SeptumConfig()
    assert septum.p == 0.20
    assert septum.candidate_fraction == (0.30, 0.45)
    assert septum.fragment_ratio == 0.25
    assert septum.thickness_px == (2.0, 4.0)
    assert septum.contrast == (0.111, 0.280)
    assert septum.min_contrast_grey == 11.0

    mask_aware = MaskAwareConfig()
    assert mask_aware.strength == (0.22, 0.40)
    assert mask_aware.field_edge_fade_share == 0.35
    assert mask_aware.darkened_rate == (0.04, 0.10)
    assert mask_aware.darkened_cap == 8


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
        mask_aware=MaskAwareConfig(), septum=SeptumConfig(),
    )

    assert config.families == (
        FAMILY_SCALE, FAMILY_ORIENTATION, FAMILY_MASK_AWARE,
        FAMILY_SEPTUM, FAMILY_TONAL, FAMILY_BLUR,
    )


def test_a_policy_that_cuts_a_window_can_change_the_mask():
    """What the integrity checks key off after every sample."""
    assert PolicyConfig(scale=ScaleConfig()).changes_mask is True


def test_shading_inside_pores_does_not_count_as_changing_the_mask():
    """It reads the annotation to place itself and writes to none."""
    config = PolicyConfig(mask_aware=MaskAwareConfig())

    assert config.changes_mask is False


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
        "sigma_px": (0.8, 1.6), "p": 0.2, "kernel_px": 11,
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
