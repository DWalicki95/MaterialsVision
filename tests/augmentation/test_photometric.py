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
    tonal = build_tonal(TonalConfig())

    assert tonal.transforms_ps == [0.5, 0.5]


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


def test_the_unpinned_range_still_draws_across_it():
    """Pinning is for review; a training run wants the whole range,
    weak draws included, because that variety is the augmentation."""
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
    "sigma_max, expected", [(0.5, 5), (0.8, 7), (1.2, 9)]
)
def test_the_kernel_is_sized_from_the_widest_sigma(sigma_max, expected):
    """Three standard deviations either side of centre, so the tail the
    kernel cuts off is below a thousandth of its mass."""
    assert BlurConfig(sigma_px=(0.4, sigma_max)).kernel_px == expected


def test_a_sigma_below_the_sampling_grid_is_rejected():
    """A Gaussian this narrow keeps all its weight on the centre pixel
    at any kernel size, so it returns the image unchanged and the
    family would fire less often than its probability states."""
    with pytest.raises(ValueError, match="pixel grid can represent"):
        BlurConfig(sigma_px=(0.2, 0.8))


def test_a_range_running_backwards_is_rejected():
    with pytest.raises(ValueError, match="increasing range"):
        BlurConfig(sigma_px=(0.8, 0.4))


def test_blur_changes_the_image():
    blur = build_blur(BlurConfig(p=1.0))
    blur.set_random_seed(2)
    image = _image()

    assert not np.array_equal(blur(image=image)["image"], image)


@pytest.mark.parametrize("sigma", [0.4, 0.5, 0.8])
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
    config = BlurConfig(sigma_px=(0.4, 0.8))
    applied = _expected_effective(0.8, config.kernel_px)

    assert applied == pytest.approx(0.8, abs=0.01)


def test_the_summary_replaces_the_kernel_with_readable_numbers():
    config = BlurConfig(p=1.0)
    blur = build_blur(config)
    blur.set_random_seed(1)
    blur(image=_image())

    summary = summarize_blur_params(blur.params)

    assert set(summary) == {"sigma_effective_px", "kernel_px"}
    assert summary["kernel_px"] == config.kernel_px
