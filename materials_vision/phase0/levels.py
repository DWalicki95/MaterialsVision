"""
The settings each family is shown at, and what binds a verdict to them.

A family is not one transformation but a range of them, and a range is
accepted or rejected at its ends. So every parametric family is
reviewed at three settings - the weak end of its frozen range, the
middle, and the strong end - and the verdict that matters is the one
at the strong end, because that is the setting the acceptance criterion
speaks about: the maximum strength has to remain plausible.

**Why the levels come from inside the frozen range.** The alternative
would be to bracket the range from outside, which measures where the
transformation stops being believable rather than whether the numbers
in use are. Those are different questions and only the second one gates
the experiment. Two settings from outside the range are kept anyway, as
diagnostics rather than gates: a punishing dark patch and a faint wall.
They cost four panels each and they are the evidence a revision would
otherwise be argued without.

**Why a member is pinned.** Two families draw one of two alternatives
that look nothing alike. Left to the draw, a panel shows whichever came
up, and three panels labelled weak, nominal and strong could easily be
three different transformations. Each member is therefore reviewed on
its own half of the family's images: pinning the member and keeping
every image would double an already long review for no extra coverage.

**Why every level fires with certainty.** A family's own probability
governs how often it acts during training and is measured there. In a
panel it would only produce reviews of the identity, so ``p`` is one
throughout. The one exception is the crop's weakest level, whose
magnification of 1.00 is the identity by definition; it is kept because
it is the only check that the short circuit really does leave a sample
untouched.

**What a verdict is attached to.** The fingerprint below hashes the
parameters a level was rendered with, not the name of the level. Widen
a range afterwards and the fingerprint changes, so the old verdict
stops applying to the new panels instead of silently carrying over - a
sequence that would otherwise turn a reviewed decision into an
unreviewed one without anybody noticing.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL, BlurConfig,
                                                  MaskAwareConfig,
                                                  OrientationConfig,
                                                  PolicyConfig, ScaleConfig,
                                                  SeptumConfig, TonalConfig,
                                                  policy_run_metadata)

KIND_GATE = "gate"
KIND_DIAGNOSTIC = "diagnostic"

# Length of the parameter fingerprint. Twelve hex characters is far
# more than enough to keep a few dozen settings apart and short enough
# to appear in a file name.
FINGERPRINT_CHARS = 12


@dataclass(frozen=True)
class ReviewLevel:
    """One family at one setting, as it will be put in front of a
    reviewer.

    Parameters
    ----------
    family : str
        Family code, e.g. ``F5_septum``.
    level : str
        Name of the setting within the family, e.g. ``high``.
    kind : str
        ``gate`` if the verdict decides whether the family is admitted,
        ``diagnostic`` if it is evidence for a revision instead.
    config : PolicyConfig
        A policy enabling this family alone. Panels isolate a family:
        a promise like "the annotation is untouched" says nothing once
        another family has legitimately changed it.
    note : str
        What this setting is meant to show, shown beside the panel so
        the reviewer knows what to look for.
    repeats : int
        Draws per image. More than one only where a single draw would
        not represent the family - the orientation group has eight
        members and one panel shows one of them.
    image_offset, image_stride : int
        Which of the family's images this level uses, as a slice of the
        review order. The default takes all of them; a pinned member
        takes every second one, so the two members between them still
        cover the whole subset.
    """

    family: str
    level: str
    kind: str
    config: PolicyConfig
    note: str
    repeats: int = 1
    image_offset: int = 0
    image_stride: int = 1

    @property
    def key(self) -> str:
        """Identifier of this family and level together.

        Returns
        -------
        str
        """
        return f"{self.family}__{self.level}"

    @property
    def parameters(self) -> dict[str, Any]:
        """Every number this level was rendered with.

        Returns
        -------
        dict
        """
        metadata = policy_run_metadata(self.config)
        return metadata["parameters"][self.family]

    @property
    def fingerprint(self) -> str:
        """Hash of the parameters, which a verdict is tied to.

        Returns
        -------
        str
        """
        canonical = json.dumps(
            self.parameters, sort_keys=True, default=str
        )
        digest = hashlib.sha256(canonical.encode("utf-8"))
        return digest.hexdigest()[:FINGERPRINT_CHARS]

    def images(self, image_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Pick this level's share of a family's images.

        Parameters
        ----------
        image_ids : tuple of str
            The family's images, in review order.

        Returns
        -------
        tuple of str
        """
        return image_ids[self.image_offset::self.image_stride]


def review_levels() -> tuple[ReviewLevel, ...]:
    """Build every setting Phase 0 reviews.

    Returns
    -------
    tuple of ReviewLevel
        In family order, gates before diagnostics.
    """
    return (
        *_orientation_levels(),
        *_scale_levels(),
        *_tonal_levels(),
        *_blur_levels(),
        *_mask_aware_levels(),
        *_septum_levels(),
    )


def levels_for(family: str) -> tuple[ReviewLevel, ...]:
    """Settings belonging to one family.

    Parameters
    ----------
    family : str

    Returns
    -------
    tuple of ReviewLevel
    """
    return tuple(
        level for level in review_levels() if level.family == family
    )


def level_by_key(key: str) -> Optional[ReviewLevel]:
    """Look a setting up by ``family__level``.

    Parameters
    ----------
    key : str

    Returns
    -------
    ReviewLevel or None
    """
    return next(
        (level for level in review_levels() if level.key == key), None
    )


def _orientation_levels() -> tuple[ReviewLevel, ...]:
    """The orientation group has no strength, only members.

    Rotating by a quarter turn and mirroring resample nothing, so there
    is no weak or strong version of it and nothing that could degrade
    with strength. What a panel has to show instead is that the
    annotation travelled with the image and that a rectangular frame
    came out rectangular the other way round, which two draws per image
    demonstrate as well as eight would.
    """
    return (
        ReviewLevel(
            family=FAMILY_ORIENTATION,
            level="nominal",
            kind=KIND_GATE,
            config=PolicyConfig(
                orientation=OrientationConfig(p=1.0)
            ),
            note=(
                "one of the eight symmetries, drawn; check that the "
                "mask travelled with the image and that a quarter "
                "turn left the geometry 890 or 960 by 1280"
            ),
            repeats=2,
        ),
    )


def _scale_levels() -> tuple[ReviewLevel, ...]:
    """The crop at the three magnifications the plan names.

    1.30 is where the frozen distribution ends and is the setting the
    acceptance criterion is about: whether a wall survives being
    magnified and then reduced again by the model. 1.00 is the identity
    and is reviewed for one reason only - it is the sole check that the
    short circuit taken at that value really does leave the sample
    alone.
    """
    return tuple(
        ReviewLevel(
            family=FAMILY_SCALE,
            level=level,
            kind=KIND_GATE,
            config=PolicyConfig(scale=ScaleConfig(
                bands=((1.0, q, q),), p=1.0
            )),
            note=note,
        )
        for level, q, note in (
            (
                "low", 1.00,
                "the identity; nothing may differ from the original",
            ),
            (
                "nominal", 1.15,
                "a window of 87% of the frame, magnified back",
            ),
            (
                "high", 1.30,
                "the end of the frozen range; the thin walls here are "
                "the ones the model will see reduced by another 0.8",
            ),
        )
    )


def _tonal_levels() -> tuple[ReviewLevel, ...]:
    """Brightness and contrast, then gamma, each on its own images.

    The strong settings sit at the ends of the frozen ranges. Neither
    member can destroy a structure - both are monotone maps of the
    intensity scale - so what a panel is judged on is plausibility:
    whether the result still looks like a micrograph from this
    material rather than a processed copy of one.
    """
    ranges = (
        ("low", (-0.05, 0.05), (-0.075, 0.075), (95, 105)),
        ("nominal", (-0.10, 0.10), (-0.15, 0.15), (90, 110)),
        ("high", (-0.15, 0.15), (-0.25, 0.25), (85, 115)),
    )
    levels = []
    for name, brightness, contrast, gamma in ranges:
        levels.append(ReviewLevel(
            family=FAMILY_TONAL,
            level=f"bc_{name}",
            kind=KIND_GATE,
            config=PolicyConfig(tonal=TonalConfig(
                brightness_limit=brightness,
                contrast_limit=contrast,
                members=("brightness_contrast",),
                p=1.0,
            )),
            note=f"brightness and contrast, {name} end of the range",
            image_offset=0,
            image_stride=2,
        ))
        levels.append(ReviewLevel(
            family=FAMILY_TONAL,
            level=f"gamma_{name}",
            kind=KIND_GATE,
            config=PolicyConfig(tonal=TonalConfig(
                gamma_limit=gamma,
                members=("gamma",),
                p=1.0,
            )),
            note=f"gamma, {name} end of the range; the mid and dark "
                 f"tones move most",
            image_offset=1,
            image_stride=2,
        ))
    return tuple(levels)


def _blur_levels() -> tuple[ReviewLevel, ...]:
    """The blur at three widths, pinned rather than drawn.

    This is the family the thin-wall criterion was written for. A
    source sigma of 0.8 acts like 0.64 at the resolution the model
    works in, which no pore is troubled by; a wall three pixels across
    might be.
    """
    return tuple(
        ReviewLevel(
            family=FAMILY_BLUR,
            level=level,
            kind=KIND_GATE,
            config=PolicyConfig(blur=BlurConfig(
                sigma_px=(sigma, sigma), p=1.0
            )),
            note=note,
            )
        for level, sigma, note in (
            ("low", 0.2, "the weakest blur the family can draw"),
            ("nominal", 0.5, "the middle of the frozen range"),
            (
                "high", 0.8,
                "the strongest; the kernel truncates it to about 0.69 "
                "effective, and the walls have to survive it",
            ),
        )
    )


def _mask_aware_levels() -> tuple[ReviewLevel, ...]:
    """Shading and dark patches, each on its own half of the images.

    The two members fail differently. A shading that reaches the
    boundary draws a step where the annotation says there is none; a
    patch with a hard edge draws a boundary inside a pore. Both are the
    error the family exists to suppress, so both are looked for.
    """
    field = tuple(
        ReviewLevel(
            family=FAMILY_MASK_AWARE,
            level=f"field_{level}",
            kind=KIND_GATE,
            config=PolicyConfig(mask_aware=MaskAwareConfig(
                strength=(strength, strength),
                members=("field",),
                p=1.0,
            )),
            note=(
                f"shading at {strength:.3f} of the tonal range, "
                f"{level} end; it must fade to nothing at the boundary"
            ),
            image_offset=0,
            image_stride=2,
        )
        for level, strength in (
            ("low", 0.08), ("nominal", 0.115), ("high", 0.15)
        )
    )
    patch = tuple(
        ReviewLevel(
            family=FAMILY_MASK_AWARE,
            level=f"patch_{level}",
            kind=KIND_GATE,
            config=PolicyConfig(mask_aware=MaskAwareConfig(
                darkened_area=(area, area),
                darkening_factor=(factor, factor),
                members=("darkening",),
                p=1.0,
            )),
            note=(
                f"a patch over {area:.0%} of the pore at {factor:.2f} "
                f"of its brightness, {level} end; its edge must stay "
                f"soft and clear of the boundary"
            ),
            image_offset=1,
            image_stride=2,
        )
        for level, area, factor in (
            ("low", 0.05, 0.85),
            ("nominal", 0.125, 0.725),
            ("high", 0.20, 0.60),
        )
    )
    stress = (
        ReviewLevel(
            family=FAMILY_MASK_AWARE,
            level="patch_stress",
            kind=KIND_DIAGNOSTIC,
            config=PolicyConfig(mask_aware=MaskAwareConfig(
                darkened_area=(0.15, 0.30),
                darkening_factor=(0.45, 0.70),
                members=("darkening",),
                p=1.0,
            )),
            note=(
                "deliberately beyond the frozen range: where does a "
                "dark patch stop being a shadow and start being a "
                "second pore"
            ),
            image_offset=1,
            image_stride=4,
        ),
    )
    return field + patch + stress


def _septum_levels() -> tuple[ReviewLevel, ...]:
    """The synthetic wall at three widths, plus a faint one.

    Width is the whole question here. Two source pixels is 1.6 at the
    resolution the model works in, and a wall that disappears there has
    taught the model to divide a pore on evidence it cannot see. The
    faint variant pairs the thinnest wall with the tenth percentile of
    the measured contrast, which is the hardest case the images
    actually contain.
    """
    gates = tuple(
        ReviewLevel(
            family=FAMILY_SEPTUM,
            level=level,
            kind=KIND_GATE,
            config=PolicyConfig(septum=SeptumConfig(
                thickness_px=(thickness, thickness), p=1.0
            )),
            note=(
                f"a wall {thickness:.1f} source pixels across, "
                f"{thickness * 0.8:.1f} as the model sees it"
            ),
        )
        for level, thickness in (
            ("low", 2.0), ("nominal", 3.0), ("high", 4.0)
        )
    )
    faint = (
        ReviewLevel(
            family=FAMILY_SEPTUM,
            level="faint",
            kind=KIND_DIAGNOSTIC,
            config=PolicyConfig(septum=SeptumConfig(
                thickness_px=(2.0, 2.0), contrast=0.111, p=1.0
            )),
            note=(
                "the thinnest wall at the faintest contrast measured "
                "in the training set; the hardest case the data holds"
            ),
            image_offset=0,
            image_stride=4,
        ),
    )
    return gates + faint
