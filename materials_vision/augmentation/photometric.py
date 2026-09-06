"""
Transformations that change brightness and nothing else.

They operate on the single working channel and never receive the mask,
so the annotation comes out bitwise identical by construction rather
than by care. That property is still asserted after every sample: it is
cheap, and a mask that changed here would mean the pipeline had been
rewired in a way nobody intended.

Two families live here and they answer different questions. The tonal
one asks whether the model can recognize a pore when the whole image
sits at a different point on the intensity scale - the situation
created by two microscopes with different detectors. The blur asks
whether it can recognize one that is slightly less sharp, as happens
with focus and working distance.

The two are kept apart rather than merged, because their risks differ.
A tonal change cannot destroy a structure, only shift its values. A
blur can: it erases the thin wall separating two pores, which is
precisely the evidence the model needs to keep them apart.
"""
from typing import Any, Mapping

import albumentations as A
import numpy as np

from materials_vision.augmentation.config import BlurConfig, TonalConfig


def build_tonal(config: TonalConfig) -> A.OneOf:
    """Build the tonal transformation.

    The two members are alternatives rather than a sequence. Applying
    both would compound their effects, putting the sample further from
    a plausible image than either range allows on its own.

    Parameters
    ----------
    config : TonalConfig

    Returns
    -------
    A.OneOf
        Fires with the configured probability and then draws one of the
        configured members with equal weight. A configuration naming
        one member still returns a container, so what the pipeline
        reports - and therefore what a record says fired - does not
        depend on how many members were left in.
    """
    if config.pin_magnitude:
        return _pinned_tonal(config)

    members = {
        "brightness_contrast": lambda: A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.contrast_limit,
            p=1.0,
        ),
        "gamma": lambda: A.RandomGamma(
            gamma_limit=config.gamma_limit, p=1.0
        ),
    }
    return A.OneOf(
        [members[name]() for name in config.members], p=config.p
    )


def _pinned_tonal(config: TonalConfig) -> A.OneOf:
    """Build the tonal container at the ends of its ranges only.

    Each member becomes two alternatives of equal weight, one for each
    end of its range, and each of those is a degenerate range - a
    single value - so the magnitude cannot come out weaker than the
    setting claims. The direction is what the draw decides.

    This exists for the visual gate. Reviewing a symmetric range by
    drawing from it uniformly means a panel labelled with the strong
    setting frequently shows something close to the identity, and a
    reviewer reporting "no difference" is then describing the draw and
    not the range they were asked about.
    """
    alternatives: list[A.ImageOnlyTransform] = []
    if "brightness_contrast" in config.members:
        alternatives.extend(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(contrast, contrast),
                p=1.0,
            )
            for brightness, contrast in zip(
                config.brightness_limit, config.contrast_limit
            )
        )
    if "gamma" in config.members:
        alternatives.extend(
            A.RandomGamma(gamma_limit=(gamma, gamma), p=1.0)
            for gamma in config.gamma_limit
        )
    return A.OneOf(alternatives, p=config.p)


def build_blur(config: BlurConfig) -> A.GaussianBlur:
    """Build the blur transformation.

    Parameters
    ----------
    config : BlurConfig

    Returns
    -------
    A.GaussianBlur
        Kernel sized to the range, sigma drawn from the configured
        range.

    Notes
    -----
    The kernel is passed as a degenerate range so that it is held at
    one value across every draw, and that value is sized from the
    widest sigma the range can produce rather than picked. Two
    failures are avoided at once. Left free, the library derives the
    kernel from each individual draw and returns a single pixel for
    the weakest of them, which is the identity - the family would then
    fire less often than its own probability states. Held at three
    pixels, the widest draws are truncated instead, and a sigma of 0.8
    is applied as 0.69, so the strength recorded for a run would not be
    the strength the model saw.
    """
    return A.GaussianBlur(
        blur_limit=(config.kernel_px, config.kernel_px),
        sigma_limit=config.sigma_px,
        p=config.p,
    )


def summarize_blur_params(
    params: Mapping[str, Any]
) -> dict[str, float]:
    """Turn a drawn blur kernel into the two numbers describing it.

    The blur reports the kernel it built, not the sigma it drew, and a
    row of floating-point weights is not something anyone can read in a
    log or compare between runs. What matters is how wide the blur
    actually came out, so the kernel is summarized by the standard
    deviation it realizes - its second moment about the centre.

    That number is the honest one to record. Holding the kernel at a
    fixed width truncates the widest draws, so the sigma that was drawn
    and the sigma that was applied are not the same, and only the
    latter describes what the model saw.

    Parameters
    ----------
    params : Mapping
        Parameters reported by the blur; expects a 1-D ``kernel``.

    Returns
    -------
    dict
        ``sigma_effective_px`` and ``kernel_px``.
    """
    kernel = np.asarray(params["kernel"], dtype=np.float64).ravel()
    weights = kernel / kernel.sum()
    offsets = np.arange(kernel.size) - (kernel.size - 1) / 2.0
    variance = float(np.sum(weights * offsets ** 2))
    return {
        "sigma_effective_px": float(np.sqrt(variance)),
        "kernel_px": int(kernel.size),
    }
