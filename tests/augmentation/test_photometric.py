"""Tests for the transformations that only change brightness."""
import numpy as np
import pytest

from materials_vision.augmentation.config import BlurConfig, TonalConfig
from materials_vision.augmentation.photometric import (build_blur, build_tonal,
                                                       summarize_blur_params)


def _image():
    return np.random.default_rng(3).integers(
        30, 220, (24, 32), dtype=np.uint8
    )


def test_the_tonal_family_draws_one_member_at_a_time():
    """Compounding both would leave the approved range behind."""
    tonal = build_tonal(TonalConfig(p=1.0))
    tonal.set_random_seed(4)

    tonal(image=_image())
    fired = [t for t in tonal.transforms if t.params]

    assert len(fired) == 1


def test_the_two_tonal_members_carry_equal_weight():
    """Each member gets half, spread over its two directions.

    A member is offered once per direction rather than once overall,
    which is what lets the magnitude floor exclude the middle of a
    range that straddles the identity. Four alternatives of equal
    weight still leave each member with half of the container.
    """
    tonal = build_tonal(TonalConfig())

    assert tonal.transforms_ps == [0.25, 0.25, 0.25, 0.25]
    members = [type(t).__name__ for t in tonal.transforms]
    assert members.count("RandomBrightnessContrast") == 2
    assert members.count("RandomGamma") == 2


def test_both_tonal_members_are_reachable():
    tonal = build_tonal(TonalConfig(p=1.0))
    drawn = set()

    for seed in range(12):
        tonal.set_random_seed(seed)
        tonal(image=_image())
        drawn.update(
            type(t).__name__ for t in tonal.transforms if t.params
        )

    assert drawn == {"RandomBrightnessContrast", "RandomGamma"}


def test_pinning_the_magnitude_leaves_only_the_direction_to_chance():
    """A symmetric range drawn from uniformly puts a fair share of the
    draws near the identity, so a panel labelled with the strong
    setting showed something close to the original about as often as
    not."""
    tonal = build_tonal(TonalConfig(
        brightness_limit=(-0.10, 0.10),
        contrast_limit=(-0.15, 0.15),
        members=("brightness_contrast",),
        pin_magnitude=True,
        p=1.0,
    ))
    drawn = set()

    for seed in range(16):
        tonal.set_random_seed(seed)
        tonal(image=_image())
        params = next(t.params for t in tonal.transforms if t.params)
        drawn.add(
            (round(params["alpha"], 4), round(params["beta"], 2))
        )

    assert drawn == {(0.85, -25.5), (1.15, 25.5)}


class TestTheMagnitudeFloor:
    """No draw lands where the reviewer could not see it.

    A range symmetric about the identity spends part of its mass on
    shifts too small to change anything, and a family whose weakest
    draws are the identity fires less often than the probability it
    reports. The floor moves the identity back into ``p``, where it can
    be counted.
    """

    @staticmethod
    def _drawn(config, key, seeds=48):
        tonal = build_tonal(config)
        values = []
        for seed in range(seeds):
            tonal.set_random_seed(seed)
            tonal(image=_image())
            params = next(
                t.params for t in tonal.transforms if t.params
            )
            values.append(params[key])
        return values

    def test_no_brightness_draw_falls_inside_the_excluded_band(
        self,
    ) -> None:
        frozen = TonalConfig(members=("brightness_contrast",), p=1.0)
        floor = frozen.min_magnitude_share * frozen.brightness_limit[1]

        for beta in self._drawn(frozen, "beta"):
            # Albumentations carries brightness as a shift of the value
            # scale, so the configured share is scaled by 255.
            assert abs(beta) >= floor * 255 - 1e-6

    def test_no_gamma_draw_falls_inside_the_excluded_band(self) -> None:
        frozen = TonalConfig(members=("gamma",), p=1.0)
        floor = frozen.min_magnitude_share * (
            frozen.gamma_limit[1] - 100
        )

        for gamma in self._drawn(frozen, "gamma"):
            assert abs(gamma - 100) >= floor - 1e-6

    def test_brightness_and_contrast_never_oppose_each_other(
        self,
    ) -> None:
        """In opposition they cancel: at the ends of the frozen ranges
        a brightness of +0.07 lifts a mid grey by about eighteen levels
        and a contrast of -0.105 takes thirteen back, which would put
        the strongest setting the family has under what anyone can
        see."""
        tonal = build_tonal(
            TonalConfig(members=("brightness_contrast",), p=1.0)
        )

        for seed in range(48):
            tonal.set_random_seed(seed)
            tonal(image=_image())
            params = next(
                t.params for t in tonal.transforms if t.params
            )
            assert params["beta"] * (params["alpha"] - 1.0) >= 0.0

    def test_a_share_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="min_magnitude_share"):
            TonalConfig(min_magnitude_share=1.5)


def test_the_unpinned_range_still_draws_across_it():
    """Pinning is for review; a training run wants the whole range
    above the floor, because that variety is the augmentation."""
    tonal = build_tonal(TonalConfig(
        members=("brightness_contrast",), p=1.0
    ))
    drawn = set()

    for seed in range(16):
        tonal.set_random_seed(seed)
        tonal(image=_image())
        drawn.add(round(
            next(t.params for t in tonal.transforms if t.params)["beta"],
            3,
        ))

    assert len(drawn) > 2


def test_a_pinned_gamma_draws_both_ends_and_nothing_between():
    tonal = build_tonal(TonalConfig(
        gamma_limit=(90, 110), members=("gamma",),
        pin_magnitude=True, p=1.0,
    ))
    drawn = set()

    for seed in range(16):
        tonal.set_random_seed(seed)
        tonal(image=_image())
        drawn.add(round(
            next(t.params for t in tonal.transforms if t.params)["gamma"],
            3,
        ))

    assert drawn == {0.9, 1.1}


def test_the_blur_kernel_is_held_at_one_width_across_draws():
    """Left free the library derives it from each individual sigma, and
    the weakest draws would then produce a one-pixel kernel, which is
    no blur."""
    blur = build_blur(BlurConfig(p=1.0))
    widths = set()

    for seed in range(12):
        blur.set_random_seed(seed)
        blur(image=_image())
        widths.add(int(np.asarray(blur.params["kernel"]).size))

    assert widths == {BlurConfig().kernel_px}


@pytest.mark.parametrize(
    "sigma_max, expected", [(0.8, 7), (1.2, 9), (1.6, 11)]
)
def test_the_kernel_is_sized_from_the_widest_sigma(sigma_max, expected):
    """Three standard deviations either side of centre, so the tail the
    kernel cuts off is below a thousandth of its mass."""
    assert BlurConfig(sigma_px=(0.8, sigma_max)).kernel_px == expected


def test_a_sigma_below_the_sampling_grid_is_rejected():
    """A Gaussian this narrow keeps all its weight on the centre pixel
    at any kernel size, so it returns the image unchanged and the
    family would fire less often than its probability states."""
    with pytest.raises(ValueError, match="pixel grid can represent"):
        BlurConfig(sigma_px=(0.2, 0.8))


def test_a_sigma_that_does_not_survive_the_model_resize_is_rejected():
    """The stricter of the two floors, and the measured one.

    A blur can be plainly visible at source resolution and gone by the
    time the encoder reads the image, because reducing to 0.8 is itself
    a low-pass filter. Measured on the training set, a source sigma of
    0.4 leaves the walls between pores with all of their local contrast
    and moves the median pixel of the model's input by nothing at all.
    """
    with pytest.raises(ValueError, match="survives the model's resize"):
        BlurConfig(sigma_px=(0.4, 1.6))


def test_a_range_running_backwards_is_rejected():
    with pytest.raises(ValueError, match="increasing range"):
        BlurConfig(sigma_px=(0.8, 0.4))


def test_blur_changes_the_image():
    blur = build_blur(BlurConfig(p=1.0))
    blur.set_random_seed(2)
    image = _image()

    assert not np.array_equal(blur(image=image)["image"], image)


@pytest.mark.parametrize("sigma", [0.8, 1.2, 1.6])
def test_a_wider_draw_gives_a_wider_applied_blur(sigma):
    """Strength stays monotone in sigma.

    Three strength settings are only meaningful if the middle one sits
    between the other two.
    """
    config = BlurConfig(sigma_px=(sigma, sigma), p=1.0)
    blur = build_blur(config)
    blur.set_random_seed(0)
    blur(image=_image())

    applied = summarize_blur_params(blur.params)["sigma_effective_px"]

    assert applied == pytest.approx(
        _expected_effective(sigma, config.kernel_px), abs=1e-9
    )


def _expected_effective(sigma, kernel):
    """Second moment of the Gaussian kernel, computed independently."""
    offsets = np.arange(kernel) - (kernel - 1) / 2.0
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return float(np.sqrt(np.sum(weights * offsets ** 2)))


def test_the_widest_draw_gets_the_width_it_asked_for():
    """The point of sizing the kernel from the range.

    Pinned at three pixels this sigma was applied as 0.69, so a run
    recorded a strength the model never saw.
    """
    config = BlurConfig()
    widest = config.sigma_px[1]
    applied = _expected_effective(widest, config.kernel_px)

    assert applied == pytest.approx(widest, abs=0.01)


def test_the_summary_replaces_the_kernel_with_readable_numbers():
    config = BlurConfig(p=1.0)
    blur = build_blur(config)
    blur.set_random_seed(1)
    blur(image=_image())

    summary = summarize_blur_params(blur.params)

    assert set(summary) == {"sigma_effective_px", "kernel_px"}
    assert summary["kernel_px"] == config.kernel_px
