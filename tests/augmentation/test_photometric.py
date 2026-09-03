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


def test_the_blur_kernel_is_held_at_the_configured_width():
    """Left free the library derives it from sigma, and the weakest
    draws would then produce a one-pixel kernel, which is no blur."""
    blur = build_blur(BlurConfig(p=1.0))
    widths = set()

    for seed in range(12):
        blur.set_random_seed(seed)
        blur(image=_image())
        widths.add(int(np.asarray(blur.params["kernel"]).size))

    assert widths == {3}


def test_blur_changes_the_image():
    blur = build_blur(BlurConfig(p=1.0))
    blur.set_random_seed(2)
    image = _image()

    assert not np.array_equal(blur(image=image)["image"], image)


@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.8])
def test_a_wider_draw_gives_a_wider_applied_blur(sigma):
    """Strength stays monotone in sigma despite the fixed kernel.

    Three strength settings are only meaningful if the middle one sits
    between the other two, which truncation could have broken.
    """
    blur = build_blur(BlurConfig(sigma_px=(sigma, sigma), p=1.0))
    blur.set_random_seed(0)
    blur(image=_image())

    applied = summarize_blur_params(blur.params)["sigma_effective_px"]

    assert applied == pytest.approx(_expected_effective(sigma), abs=1e-9)


def _expected_effective(sigma):
    """Second moment of a three-tap Gaussian, computed independently."""
    weights = np.exp(-0.5 * (np.array([-1.0, 0.0, 1.0]) / sigma) ** 2)
    weights /= weights.sum()
    return float(np.sqrt(np.sum(weights * np.array([1.0, 0.0, 1.0]))))


def test_the_widest_draw_is_narrower_than_it_asked_for():
    """The documented price of pinning the kernel."""
    applied = _expected_effective(0.8)

    assert applied < 0.8
    assert applied == pytest.approx(0.69, abs=0.01)


def test_the_summary_replaces_the_kernel_with_readable_numbers():
    blur = build_blur(BlurConfig(p=1.0))
    blur.set_random_seed(1)
    blur(image=_image())

    summary = summarize_blur_params(blur.params)

    assert set(summary) == {"sigma_effective_px", "kernel_px"}
    assert summary["kernel_px"] == 3
