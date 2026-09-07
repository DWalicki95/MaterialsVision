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
                "jedna z osmiu symetrii, wylosowana; sprawdz, czy "
                "maska pojechala razem z obrazem i czy cwierc obrotu "
                "zostawila geometrie 890 albo 960 na 1280"
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
                "identycznosc - NIC nie moze sie roznic od oryginalu; "
                "to jedyna kontrola, ze skrot przy q = 1.00 naprawde "
                "zostawia probke nietknieta, wiec brak roznicy jest "
                "tu wynikiem poprawnym, nie usterka",
            ),
            (
                "nominal", 1.15,
                "okno 87% boku kadru (76% powierzchni), powiekszone "
                "z powrotem do pelnego rozmiaru",
            ),
            (
                "high", 1.30,
                "koniec zamrozonego zakresu: okno 77% boku (59% "
                "powierzchni); cienkie sciany, ktore tu widac, model "
                "zobaczy jeszcze raz pomniejszone o 0.8",
            ),
        )
    )


def _tonal_levels() -> tuple[ReviewLevel, ...]:
    """Brightness and contrast, then gamma, each on its own images.

    **The magnitude is pinned and only the direction is drawn.** These
    ranges are symmetric about the identity, so a panel that drew from
    one uniformly showed the strong setting as something near the
    original about as often as not - and a reviewer reporting no
    difference was then describing that draw rather than the range they
    were asked to judge. Pinned, every panel carries the full strength
    of its level and the two directions appear across the level's
    images.

    **The strong setting is the end of the frozen range, not past it.**
    The criterion asks whether the numbers in use are believable, so
    the levels are read off V.3 - brightness 0.10, contrast 0.15,
    gamma 90 to 110 - and the weaker two are fractions of those.
    Bracketing from outside would measure where the family stops being
    believable, which is a different question and gates nothing.

    Neither member can destroy a structure - both are monotone maps of
    the intensity scale - so what a panel is judged on is plausibility:
    whether the result still looks like a micrograph from this material
    rather than a processed copy of one.
    """
    frozen = TonalConfig()
    brightness_max = frozen.brightness_limit[1]
    contrast_max = frozen.contrast_limit[1]
    gamma_max = frozen.gamma_limit[1] - 100
    # The weakest level is the weakest draw the family can now make,
    # not an arbitrary fraction: the range carries a floor of its own,
    # and reviewing below it would review a setting no run produces.
    ranges = (
        ("low", frozen.min_magnitude_share),
        ("nominal", (1.0 + frozen.min_magnitude_share) / 2.0),
        ("high", 1.0),
    )
    levels = []
    for name, share in ranges:
        brightness = round(brightness_max * share, 4)
        contrast = round(contrast_max * share, 4)
        gamma = int(round(gamma_max * share))
        levels.append(ReviewLevel(
            family=FAMILY_TONAL,
            level=f"bc_{name}",
            kind=KIND_GATE,
            config=PolicyConfig(tonal=TonalConfig(
                brightness_limit=(-brightness, brightness),
                contrast_limit=(-contrast, contrast),
                members=("brightness_contrast",),
                pin_magnitude=True,
                p=1.0,
            )),
            note=(
                f"jasnosc {brightness:+.3f} i kontrast {contrast:+.3f} "
                f"zakresu, poziom {name}; kierunek losowany, sila "
                f"przypieta"
            ),
            image_offset=0,
            image_stride=2,
        ))
        levels.append(ReviewLevel(
            family=FAMILY_TONAL,
            level=f"gamma_{name}",
            kind=KIND_GATE,
            config=PolicyConfig(tonal=TonalConfig(
                gamma_limit=(100 - gamma, 100 + gamma),
                members=("gamma",),
                pin_magnitude=True,
                p=1.0,
            )),
            note=(
                f"gamma {100 - gamma} albo {100 + gamma}, poziom "
                f"{name}; najmocniej ruszaja sie tony srednie i ciemne"
            ),
            image_offset=1,
            image_stride=2,
        ))
    # The gamma candidate that used to sit here has been promoted: it
    # showed 22 grey levels against the 9 the frozen range reached, the
    # reviewer saw the first and not the second, and V.3 was widened to
    # (75, 125) accordingly. A diagnostic exists to argue for a change,
    # so it retires when the change is made - kept on, it would review
    # a range identical to the gate beside it.
    return tuple(levels)


def _blur_levels() -> tuple[ReviewLevel, ...]:
    """The blur at three widths, pinned rather than drawn.

    This is the family the thin-wall criterion was written for, and it
    is the family whose panels were misleading twice over. A blur is
    judged on what reaches the encoder, and the encoder reads the image
    at 0.8 of the size a panel shows it at - a reduction that is itself
    a low-pass filter. Measured through it, the range this family used
    to carry disappeared: at a source sigma of 0.4 the walls kept
    100.0 per cent of their local contrast and the median pixel did not
    move at all. The panel was not wrong; it was rendered at a
    resolution the model never sees.

    The range now spans what the microscopes themselves vary over.
    Sharpness across the training set runs from 8 to 15 grey levels of
    wall contrast in the model's grid, median 11, so the strong end is
    the sigma that takes a typical image down to a soft one - 1.6 - and
    the weak end the one that takes it to the lower quartile - 0.8.
    """
    frozen = BlurConfig()
    low_sigma, high_sigma = frozen.sigma_px
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
            (
                "low", low_sigma,
                "najslabsze rozmycie, jakie rodzina moze wylosowac; "
                "odpowiada zejsciu obrazu typowego do dolnego kwartyla "
                "ostrosci zbioru",
            ),
            (
                "nominal", round((low_sigma + high_sigma) / 2.0, 2),
                "srodek zamrozonego zakresu",
            ),
            (
                "high", high_sigma,
                "najmocniejsze: odtwarza zejscie z obrazu typowego na "
                "najmiekszy dziesiaty percentyl zbioru; ocena nalezy "
                "do zblizenia na najciensza sciane, bo tam rodzina "
                "dziala, a nie na calym kadrze",
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
    frozen = MaskAwareConfig()
    weak, strong = frozen.strength
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
                f"cieniowanie o sile {strength:.3f} rozpietosci "
                f"tonalnej obrazu, poziom {level}; musi zanikac do "
                f"zera na granicy pora. Amplituda w poziomach "
                f"szarosci zalezy od obrazu i jest podana nizej"
            ),
            image_offset=0,
            image_stride=2,
        )
        for level, strength in (
            ("low", weak),
            ("nominal", round((weak + strong) / 2.0, 3)),
            ("high", strong),
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
                f"plama na {area:.0%} powierzchni pora, przy "
                f"{factor:.2f} jego jasnosci, poziom {level}; jej "
                f"krawedz ma zostac miekka i nie dotykac granicy"
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
    # The three amplitude candidates that used to sit here have been
    # withdrawn rather than resolved, because the question they asked
    # was the wrong one. They bounded the amplitude; the amplitude was
    # never what reached the image. Under the fade this family used to
    # apply, full strength was reached at one pixel per pore and the
    # median pixel carried a third of it, so all three candidates
    # measured within one grey level of each other and of the setting
    # they were meant to improve on - which is exactly what the
    # reviewer reported seeing. The fade is now a share of the pore's
    # depth, the amplitude arrives, and the choice between a floor, a
    # ceiling and a larger share is not a choice anyone needs to make.
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
                "celowo poza zamrozonym zakresem: gdzie ciemna plama "
                "przestaje byc cieniem, a zaczyna byc drugim porem; "
                "poza bramka"
            ),
            image_offset=1,
            image_stride=4,
        ),
    )
    return field + patch + stress


def _septum_levels() -> tuple[ReviewLevel, ...]:
    """The synthetic wall at three contrasts, plus the hardest corner.

    **The strength axis is contrast, not width.** The criterion for
    this family is that the wall is still visible once the model has
    reduced the image, and width barely moves that: reviewed at two,
    three and four source pixels, the measured visibility of one wall
    ran 45, 34 and 31 grey levels - it fell as the wall got wider. What
    sets visibility is how far the wall's brightness sits from the pore
    it divides, and that is the contrast. Varying width therefore
    produced three levels that differed in something other than the
    quantity being judged, and a reviewer reporting that the strong
    setting was still too faint was reporting exactly that.

    The three contrasts are the tenth percentile, the mean and the
    ninetieth percentile of the contrast measured on real walls in the
    training set, so the range reviewed is the range the images
    themselves contain. Width is pinned at the middle of its calibrated
    range throughout, which keeps it out of the comparison.

    The diagnostic pairs the faintest contrast with the thinnest wall -
    the hardest corner of both ranges at once - and runs on the same
    images as the gates. Given its own subset it landed on five images
    whose walls were brighter than average, so the setting labelled the
    hardest case in the data measured easier than the settings it was
    meant to bracket.
    """
    measured = SeptumConfig()
    thin, thick = measured.thickness_px
    middle = round((thin + thick) / 2.0, 2)
    faintest, brightest = measured.contrast
    gates = tuple(
        ReviewLevel(
            family=FAMILY_SEPTUM,
            level=level,
            kind=KIND_GATE,
            config=PolicyConfig(septum=SeptumConfig(
                thickness_px=(middle, middle),
                contrast=(contrast, contrast),
                p=1.0,
            )),
            note=(
                f"kontrast sciany {contrast:.3f} rozpietosci tonalnej "
                f"({description}); szerokosc {middle:.1f} px "
                f"zrodlowych, {middle * 0.8:.1f} px tak, jak widzi to "
                f"model. Na obrazie o malej rozpietosci podloga "
                f"{measured.min_contrast_grey:.0f} poziomow szarosci "
                f"podniesie ten kontrast - i tak ma byc"
            ),
        )
        for level, contrast, description in (
            ("low", faintest, "p10 zmierzonych scian"),
            (
                "nominal", round((faintest + brightest) / 2.0, 4),
                "srodek zmierzonego zakresu",
            ),
            ("high", brightest, "p90 zmierzonych scian"),
        )
    )
    # What the floor keeps out, rendered so that keeping it out can be
    # judged rather than asserted. The thinnest wall at the faintest
    # measured contrast, with the floor switched off: this is the
    # sample whose annotation would say two pores while its image, once
    # the model had reduced it, showed one.
    faint = (
        ReviewLevel(
            family=FAMILY_SEPTUM,
            level="faint",
            kind=KIND_DIAGNOSTIC,
            config=PolicyConfig(septum=SeptumConfig(
                thickness_px=(thin, thin),
                contrast=(faintest, faintest),
                min_contrast_grey=0.0,
                p=1.0,
            )),
            note=(
                "najciensza sciana przy najslabszym zmierzonym "
                "kontrascie, z wylaczona podloga widocznosci - to jest "
                "przypadek, ktory podloga usuwa ze zbioru; poza bramka"
            ),
        ),
    )
    return gates + faint
