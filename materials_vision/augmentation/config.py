"""
Frozen parameters of the augmentation families.

Every number that defines a transformation family lives here and
nowhere else. Two things depend on that: a run has to record what was
actually applied to it, and the record of why a family was accepted or
rejected has to cite the ranges that were judged. If the same number
also sat inside the transformation using it, the two copies could drift
apart and neither record would be evidence of anything.

Families carry short codes - ``F1_orientation``, ``F3b_blur`` and so on
- used unchanged in log lines, run metadata and decision records, so a
value seen in a log traces back to the family that produced it without
a lookup table.

**These are starting values for qualification and tuning, not
results.** A family is inspected by eye before it is allowed into a
training comparison, and one that helps may afterwards be retried with
a stronger or weaker range. A number changed here changes every run
that uses it, which makes it a decision rather than a convenience.
"""
from dataclasses import dataclass
from typing import Any, Optional

FAMILY_ORIENTATION = "F1_orientation"
FAMILY_SCALE = "F2_scale"
FAMILY_TONAL = "F3a_tonal"
FAMILY_BLUR = "F3b_blur"
FAMILY_MASK_AWARE = "F4_mask_aware"
FAMILY_SEPTUM = "F5_septum"

# Families that can change which pixel carries which instance id.
# Everything else has to leave the label image either bitwise identical
# (photometry) or a pure rearrangement of it (orientation), and that is
# what the integrity checks verify after every sample.
MASK_CHANGING_FAMILIES = frozenset({FAMILY_SCALE, FAMILY_SEPTUM})


@dataclass(frozen=True)
class OrientationConfig:
    """The eight symmetries of the square, drawn uniformly.

    Segmentation of a foam should not depend on how the sample happened
    to be oriented under the microscope. Rotating by multiples of a
    quarter turn and mirroring costs nothing in image quality, because
    neither resamples a single pixel, and it removes the orientation
    shortcut a model can otherwise learn.

    Parameters
    ----------
    p : float
        Probability of drawing an orientation. The default applies the
        family to every sample; the identity is one of the eight
        elements, so a sample can still come through unchanged.

    Notes
    -----
    Independent flips and rotations are deliberately not composed. The
    two together produce duplicates - a horizontal flip followed by a
    vertical one is the same element as a half turn - which would make
    some orientations twice as likely as others. The group of eight is
    drawn from directly instead.
    """

    p: float = 1.0


@dataclass(frozen=True)
class TonalConfig:
    """Brightness/contrast or gamma, drawn one at a time.

    Both members target the same weakness: the images come from two
    microscopes whose detectors and acquisition settings differ, so a
    model must not tie its notion of a pore to one tonal response.
    Brightness and contrast shift the scale linearly; gamma bends it,
    moving the mid and dark tones more than the highlights. Neither
    touches geometry, so the mask stays untouched.

    They are treated as one family and only split apart if the joint
    result turns out negative or inconclusive - separating them first
    would spend two training runs to answer a question that one can
    answer.

    Parameters
    ----------
    brightness_limit : tuple of float
        Additive brightness range, as a fraction of the value range.
    contrast_limit : tuple of float
        Multiplicative contrast range, as a fraction.
    gamma_limit : tuple of int
        Gamma range in percent; 100 is the identity.
    p : float
        Probability that the container fires. The two members carry
        equal weight inside it, written out rather than left implicit.
    """

    brightness_limit: tuple[float, float] = (-0.10, 0.10)
    contrast_limit: tuple[float, float] = (-0.15, 0.15)
    gamma_limit: tuple[int, int] = (90, 110)
    p: float = 0.5


@dataclass(frozen=True)
class BlurConfig:
    """A very light Gaussian blur of the image only.

    Real acquisitions differ in sharpness - working distance, focus and
    scan settings all move it - so a slight loss of definition is a
    variation the model should tolerate rather than a defect. The mask
    is never blurred: the annotation is exact regardless of how sharp
    the image behind it is.

    Parameters
    ----------
    kernel_px : int
        Kernel side in source pixels, held fixed.
    sigma_px : tuple of float
        Standard deviation range, in source pixels.
    p : float

    Notes
    -----
    Kernel size and sigma are not independent in the library used here.
    Left to itself it derives the kernel from sigma as
    ``int(sigma * 3.5) * 2 + 1``, which at the lower end of the range
    below yields a kernel of 1 - no blur at all - so the family would
    fire measurably less often than its own ``p`` claims. The kernel is
    therefore pinned. The price is that the largest sigma is truncated
    one pixel either side of centre, giving an effective width nearer
    0.69 than 0.8; the kernel is renormalized, so no brightness is
    lost, and strength stays monotone in sigma, which is what comparing
    a weak, a nominal and a strong setting requires.

    Scale matters for judging safety. The model sees the image at 0.8
    of its source resolution, so a source sigma of 0.8 acts like 0.64
    there, and the smallest annotated pore - about 5.5 pixels across at
    that scale - survives it comfortably. The structure genuinely at
    risk is the thin wall between two pores, whose thickness is not
    recorded anywhere in the dataset and can only be judged by looking.
    """

    kernel_px: int = 3
    sigma_px: tuple[float, float] = (0.2, 0.8)
    p: float = 0.2


@dataclass(frozen=True)
class PolicyConfig:
    """The families making up one augmentation policy.

    A field left as ``None`` means the family is switched off. The
    baseline condition - no augmentation whatsoever - is not expressed
    by an empty policy but by giving the dataset no policy at all, so
    that a policy object always stands for something that happens.

    Parameters
    ----------
    orientation : OrientationConfig, optional
    tonal : TonalConfig, optional
    blur : BlurConfig, optional
    """

    orientation: Optional[OrientationConfig] = None
    tonal: Optional[TonalConfig] = None
    blur: Optional[BlurConfig] = None

    @property
    def families(self) -> tuple[str, ...]:
        """Codes of the families this policy switches on.

        Ordered as the pipeline applies them rather than as the fields
        are declared, so the value doubles as a record of the order.

        Returns
        -------
        tuple of str
        """
        enabled = (
            (FAMILY_ORIENTATION, self.orientation),
            (FAMILY_TONAL, self.tonal),
            (FAMILY_BLUR, self.blur),
        )
        return tuple(name for name, cfg in enabled if cfg is not None)

    @property
    def changes_mask(self) -> bool:
        """Whether any enabled family can change instance ids.

        Returns
        -------
        bool
        """
        return bool(MASK_CHANGING_FAMILIES.intersection(self.families))


def policy_run_metadata(config: PolicyConfig) -> dict[str, Any]:
    """Describe a policy for a run's configuration record.

    The result is what a run stores as its parameter configuration:
    which families are on, in which order they are applied, and every
    number each of them was given. It is the half of reproducibility
    that the seed alone does not cover.

    Parameters
    ----------
    config : PolicyConfig

    Returns
    -------
    dict
    """
    parameters: dict[str, Any] = {}
    for name, cfg in (
        (FAMILY_ORIENTATION, config.orientation),
        (FAMILY_TONAL, config.tonal),
        (FAMILY_BLUR, config.blur),
    ):
        if cfg is not None:
            parameters[name] = vars(cfg).copy()
    return {
        "families": list(config.families),
        "order": list(config.families),
        "changes_mask": config.changes_mask,
        "parameters": parameters,
    }
