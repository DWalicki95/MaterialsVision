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

# Shapes the shading inside a pore may take.
FIELD_KINDS = frozenset({"constant", "gradient", "random"})


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
class ScaleConfig:
    """A window cut from the frame and magnified back to fill it.

    The dataset was photographed at two scales that differ by a factor
    of about 1.3, and the coarser one accounts for 88% of it. Left
    alone, a model learns the pore size typical of that majority.
    Cutting a smaller window and magnifying it back shows the same
    foam at the finer scale without inventing detail: every pixel of
    the window is real, only spread further apart.

    Parameters
    ----------
    bands : tuple of tuple
        The distribution of the magnification ``q``, as
        ``(weight, low, high)`` triples. A band is drawn by weight and
        ``q`` then uniformly from its range; a band whose bounds are
        equal contributes that value exactly. The first band is the
        identity, so half of the samples come through untouched.
    magnified_bins : tuple of str
        Scale bins allowed to magnify. Only the coarse bin is: an
        image already at the finest scale in the dataset has nothing
        to be magnified towards, and the six close-ups are three to
        thirteen times finer than everything else, so their headroom
        is an artefact of their being a different imaging task.
    min_instances : int
        Instances a window must contain to be accepted.
    max_retries : int
        Re-draws allowed after a rejected window, so a sample costs at
        most ``max_retries + 1`` draws before it gives up.
    min_fragment_area_px2 : float
        Smallest area, in source pixels squared, a fragment cut by the
        window may keep and still count as an instance. The first
        percentile of the real annotated areas, so the window never
        manufactures a label smaller than anything an annotator drew.
    p : float
        Probability the family fires. One, because the identity is a
        member of the distribution rather than the alternative to it -
        the same arrangement as the orientation family.

    Notes
    -----
    ``q >= 1`` throughout: the window is always smaller than the frame
    and is always magnified, never reduced. Reduction would ask the
    model to recognize pores finer than any that were photographed.

    Only the window position is re-drawn on a rejection, never ``q``.
    Re-drawing ``q`` would let a rejection push the sample towards the
    weaker bands, and the measured distribution would then differ from
    the frozen one by an amount nobody chose.
    """

    bands: tuple[tuple[float, float, float], ...] = (
        (0.50, 1.00, 1.00),
        (0.30, 1.05, 1.15),
        (0.20, 1.15, 1.30),
    )
    magnified_bins: tuple[str, ...] = ("coarse",)
    min_instances: int = 3
    max_retries: int = 5
    min_fragment_area_px2: float = 432.0
    p: float = 1.0

    def __post_init__(self) -> None:
        """Reject a distribution that does not describe a draw.

        The other families hold single numbers, which are wrong only
        by being wrong values. This one holds a distribution, which
        can also be malformed - weights that do not sum to one, a
        range running backwards, a magnification below one - and each
        of those would produce samples silently unlike the frozen
        policy rather than an error.

        Raises
        ------
        ValueError
        """
        if not self.bands:
            raise ValueError("bands must describe at least one band")
        total = sum(weight for weight, _, _ in self.bands)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"band weights must sum to 1, got {total}"
            )
        for weight, low, high in self.bands:
            if weight < 0:
                raise ValueError(f"band weight {weight} is negative")
            if low < 1.0:
                raise ValueError(
                    f"q must be >= 1; band lower bound is {low}"
                )
            if high < low:
                raise ValueError(f"band range [{low}, {high}] is empty")
        if self.min_instances < 1:
            raise ValueError(
                f"min_instances must be >= 1, got {self.min_instances}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.min_fragment_area_px2 < 0:
            raise ValueError(
                f"min_fragment_area_px2 must be >= 0, got "
                f"{self.min_fragment_area_px2}"
            )

    @property
    def q_max(self) -> float:
        """Largest magnification the distribution can produce.

        Returns
        -------
        float
        """
        return max(high for _, _, high in self.bands)


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
class MaskAwareConfig:
    """Shading painted inside pores, leaving the annotation alone.

    Two things inside a pore can be mistaken for its edge: a slow
    change of brightness across the interior, and a dark patch where
    the surface has fallen away and the image looks deeper into the
    material. Neither is a boundary. A model that reads them as one
    reports a single pore as two, which is the error this family
    exists to suppress.

    The two are alternatives, never both on the same sample: applied
    together they would compound inside the same pore and produce an
    interior no photograph would show.

    Parameters
    ----------
    p : float
        Probability that one of the two fires.
    pore_fraction : tuple of float
        Share of the eligible pores the shading covers, drawn per
        sample. Not all of them: an image where every pore is shaded
        teaches the shading as a property of the material rather than
        as a variation to be ignored.
    strength : tuple of float
        Amplitude of the shading, as a fraction of the image's own
        tonal range. Measured against the image rather than against
        the full scale because the images come from two microscopes
        whose exposures differ, and a fixed number of grey levels
        would be a strong effect on one and invisible on the other.
    field_kinds : tuple of str
        Shapes the shading may take: one value across the pore, a
        linear gradient, or a smooth random surface.
    field_grid_sides : tuple of int
        Side of the coarse grid the random surface is drawn on before
        being smoothed out to the pore's size. Small on purpose - the
        shading has to stay lower in frequency than anything the
        annotation calls a boundary.
    min_core_distance_px : float
        A pore is left alone unless some pixel of it lies at least
        this far from anything outside it. Below that the shading
        would be squeezed into the fade-out and amount to nothing,
        and the dark patch would have nowhere to sit clear of the
        boundary.
    darkened_pores : tuple of int
        How many pores may receive a dark patch, drawn inclusively.
    darkened_area : tuple of float
        Area of the patch as a share of the pore holding it.
    darkening_factor : tuple of float
        What the patch multiplies the image by at its centre. Below
        one, so the patch is always darker than its surroundings.
    darkening_margin_px : float
        Distance the patch keeps from the pore's boundary. Touching
        it would deform the edge the annotation describes, which is
        the one thing this family must not do.
    darkening_edge_softness : float
        Width of the patch's fade-out, as a share of its radius. A
        hard-edged patch is a new boundary drawn inside a pore, which
        would teach exactly the mistake the family means to prevent.
    darkening_max_attempts : int
        Placements tried before a pore is left alone.

    Notes
    -----
    The strength range is the setting for ordinary training. Judging
    the family by eye is done with a weaker and a stronger one as
    well, and a deliberately punishing variant - larger patches at a
    lower factor - exists to find where recognizability breaks. All
    three are expressed by building this object with other numbers,
    which is why none of them is hard-coded anywhere else.
    """

    p: float = 0.3
    pore_fraction: tuple[float, float] = (0.30, 0.50)
    strength: tuple[float, float] = (0.08, 0.15)
    field_kinds: tuple[str, ...] = ("constant", "gradient", "random")
    field_grid_sides: tuple[int, ...] = (2, 3)
    min_core_distance_px: float = 3.0
    darkened_pores: tuple[int, int] = (1, 2)
    darkened_area: tuple[float, float] = (0.05, 0.20)
    darkening_factor: tuple[float, float] = (0.60, 0.85)
    darkening_margin_px: float = 2.0
    darkening_edge_softness: float = 0.25
    darkening_max_attempts: int = 8

    def __post_init__(self) -> None:
        """Reject settings that could not describe a draw.

        Raises
        ------
        ValueError
        """
        _check_range("pore_fraction", self.pore_fraction, 0.0, 1.0)
        _check_range("strength", self.strength, 0.0, 1.0)
        _check_range("darkened_area", self.darkened_area, 0.0, 1.0)
        _check_range(
            "darkening_factor", self.darkening_factor, 0.0, 1.0
        )
        if not self.field_kinds:
            raise ValueError("field_kinds must offer at least one")
        unknown = set(self.field_kinds) - FIELD_KINDS
        if unknown:
            raise ValueError(
                f"unknown field kind(s) {sorted(unknown)}; known kinds "
                f"are {sorted(FIELD_KINDS)}"
            )
        if any(side < 2 for side in self.field_grid_sides):
            raise ValueError(
                "a random field needs a grid of at least 2 per side"
            )
        low, high = self.darkened_pores
        if low < 1 or high < low:
            raise ValueError(
                f"darkened_pores must be an increasing range of at "
                f"least one pore, got {self.darkened_pores}"
            )
        if self.darkening_margin_px >= self.min_core_distance_px:
            raise ValueError(
                f"a pore eligible at {self.min_core_distance_px} px "
                f"from its boundary has no room for a patch kept "
                f"{self.darkening_margin_px} px clear of it"
            )
        if not 0.0 < self.darkening_edge_softness <= 1.0:
            raise ValueError(
                f"darkening_edge_softness must be a share of the "
                f"radius in (0, 1], got "
                f"{self.darkening_edge_softness}"
            )
        if self.darkening_max_attempts < 1:
            raise ValueError(
                "darkening_max_attempts must allow at least one try"
            )


def _check_range(
    name: str,
    value: tuple[float, float],
    lowest: float,
    highest: float,
) -> None:
    """Reject a range that runs backwards or leaves its bounds."""
    low, high = value
    if low > high:
        raise ValueError(f"{name} range {value} runs backwards")
    if low < lowest or high > highest:
        raise ValueError(
            f"{name} must lie within [{lowest}, {highest}], got {value}"
        )


@dataclass(frozen=True)
class PolicyConfig:
    """The families making up one augmentation policy.

    A field left as ``None`` means the family is switched off. The
    baseline condition - no augmentation whatsoever - is not expressed
    by an empty policy but by giving the dataset no policy at all, so
    that a policy object always stands for something that happens.

    Parameters
    ----------
    scale : ScaleConfig, optional
    orientation : OrientationConfig, optional
    mask_aware : MaskAwareConfig, optional
    tonal : TonalConfig, optional
    blur : BlurConfig, optional

    Notes
    -----
    The fields are declared in the order the pipeline applies them, so
    reading the class is reading the order.
    """

    scale: Optional[ScaleConfig] = None
    orientation: Optional[OrientationConfig] = None
    mask_aware: Optional[MaskAwareConfig] = None
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
        return tuple(name for name, _ in enabled_families(self))

    @property
    def changes_mask(self) -> bool:
        """Whether any enabled family can change instance ids.

        Returns
        -------
        bool
        """
        return bool(MASK_CHANGING_FAMILIES.intersection(self.families))


def enabled_families(
    config: PolicyConfig,
) -> tuple[tuple[str, Any], ...]:
    """Pair each enabled family's code with its configuration.

    One statement of the pipeline order, in the one place both the
    policy's own listing and a run's record read it from. Written out
    twice, the two could drift, and a run would then record an order
    different from the one it applied.

    Parameters
    ----------
    config : PolicyConfig

    Returns
    -------
    tuple of tuple
        ``(family_code, family_config)``, in application order,
        omitting families that are switched off.
    """
    declared = (
        (FAMILY_SCALE, config.scale),
        (FAMILY_ORIENTATION, config.orientation),
        (FAMILY_MASK_AWARE, config.mask_aware),
        (FAMILY_TONAL, config.tonal),
        (FAMILY_BLUR, config.blur),
    )
    return tuple(
        (name, cfg) for name, cfg in declared if cfg is not None
    )


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
    parameters: dict[str, Any] = {
        name: vars(cfg).copy() for name, cfg in enabled_families(config)
    }
    return {
        "families": list(config.families),
        "order": list(config.families),
        "changes_mask": config.changes_mask,
        "parameters": parameters,
    }
