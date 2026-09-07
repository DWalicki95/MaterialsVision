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

    Notes
    -----
    Each member is offered as one alternative per direction rather than
    as one transformation spanning both, which is what lets a magnitude
    floor exist at all: a range running from one sign through zero to
    the other cannot exclude its own middle. Split at the identity, the
    two halves are ordinary ranges and the floor is their inner bound.

    **Brightness and contrast keep a common sign.** They are drawn
    independently inside their band but never in opposition, because
    in opposition they cancel: at the ends of the frozen ranges a
    brightness of +0.07 moves a mid grey by about eighteen levels and a
    contrast of -0.105 moves it back by about thirteen, leaving five -
    under what anyone can see, at the setting that is supposed to be
    the strongest the family has. Coupling the sign is what makes the
    floor mean what it says.
    """
    share = 1.0 if config.pin_magnitude else config.min_magnitude_share
    alternatives: list[A.ImageOnlyTransform] = []
    if "brightness_contrast" in config.members:
        alternatives.extend(
            A.RandomBrightnessContrast(
                brightness_limit=_band(brightness, share),
                contrast_limit=_band(contrast, share),
                p=1.0,
            )
            for brightness, contrast in zip(
                config.brightness_limit, config.contrast_limit
            )
        )
    if "gamma" in config.members:
        alternatives.extend(
            A.RandomGamma(
                gamma_limit=_gamma_band(gamma, share), p=1.0
            )
            for gamma in config.gamma_limit
        )
    return A.OneOf(alternatives, p=config.p)


def _band(end: float, share: float) -> tuple[float, float]:
    """The part of one half-range a draw is allowed to land in.

    Runs from ``share`` of the way out to the end itself, so a share of
    one is the end alone - the setting a review panel uses, where a
    draw near the identity would review the draw instead of the range -
    and a share of zero is the whole half.

    Parameters
    ----------
    end : float
        One end of a range symmetric about the identity; either sign.
    share : float

    Returns
    -------
    tuple of float
        Increasing, whichever sign the end has.
    """
    inner = end * share
    return (inner, end) if end >= 0.0 else (end, inner)


def _gamma_band(end: int, share: float) -> tuple[int, int]:
    """The same band for gamma, whose identity is 100 rather than 0.

    Parameters
    ----------
    end : int
        One end of the gamma range, in percent.
    share : float

    Returns
    -------
    tuple of int
        Increasing, and never crossing back over the identity.
    """
    offset = end - 100
    inner = int(round(100 + offset * share))
    return (inner, end) if offset >= 0 else (end, inner)


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
