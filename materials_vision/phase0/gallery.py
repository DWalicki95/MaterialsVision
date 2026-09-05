"""
Choosing the fixed set of images every augmentation is judged on.

An augmentation family is admitted to the experiment by eye, not by a
metric, and what the eye is shown decides what the judgement is worth.
Pick twenty images at random from a set that is 79% one foam family at
one scale and the review answers a narrower question than it appears
to: whether the transformation is safe on the majority case. The
failures worth catching live at the edges - the thinnest walls, the
faintest contrast between wall and pore, the densest frames - and a
random draw finds them only by luck.

So the gallery is built by a rule instead. Images are split into strata
by microscope and scale bin, which is the pair the dataset actually
varies along, and within each stratum the extremes of four measured
axes are taken in a fixed order, hardest first, with one typical image
always included so the reviewer sees the ordinary case beside the
awkward ones. Nothing is drawn at random, so the gallery is a function
of the manifest and the split and can be rebuilt exactly.

**Why walls are measured rather than read from the manifest.** The one
criterion of the visual gate that no test can decide is whether thin
walls survive the transformation and the model's own downscaling. The
manifest knows pore diameters but nothing about walls, so the axis that
matters most would be missing. It is measured here on the annotation
itself, by the same code that calibrated the synthetic wall.

**Why TRAIN only.** Repeated viewing of an evaluation image is a soft
leak: it lets what the reviewer knows about those images influence the
policy that is later scored on them. TEST is never opened; VALIDATION
is allowed only after training, for comparing predictions.
"""
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL)
from materials_vision.augmentation.walls import measure_walls
from materials_vision.data.samples import PreparedSample, SampleSource

logger = logging.getLogger(__name__)

GALLERY_RULES_VERSION = "golden_gallery_rules_v1"

# Wall width, in source pixels, below which a wall counts as thin. Two
# pixels is the thinnest width the measurement can resolve at all, and
# a quarter of every wall pixel in the training set sits there; the
# share of an image's walls at or just above it says how much of that
# image is the hard case.
THIN_WALL_PX = 2.5

# How many images each stratum contributes. The minority strata are
# deliberately over-represented relative to how often they occur - the
# gallery is not an estimate of an average, it is a search for the
# places a transformation breaks, and three quarters of the strata
# would otherwise contribute one image each.
STRATUM_QUOTAS: Mapping[tuple[str, str], int] = {
    ("M1", "coarse"): 8,
    ("M1", "fine"): 4,
    ("M2", "coarse"): 4,
    ("M2", "fine"): 4,
}

# The order extremes are taken in within a stratum, hardest first, so
# that a stratum with a small quota still gets the cases most likely to
# expose a failure. The typical image comes fourth rather than last:
# a gallery made only of pathological frames would make every
# transformation look worse than it is.
SELECTION_REASONS: tuple[str, ...] = (
    "thinnest_walls",
    "lowest_wall_contrast",
    "densest",
    "most_typical",
    "smallest_pores",
    "largest_pores",
    "sparsest",
    "thickest_walls",
)

# Images on which a family gave up during the health check of the
# augmentation layer, named there so they could be looked at here.
# Their presence is not negotiable: they are the only images known in
# advance to exercise a fallback path.
FORCED_IMAGES: Mapping[str, str] = {
    "AS6_40_6": FAMILY_MASK_AWARE,
    "AS1_40_15": FAMILY_MASK_AWARE,
    "AS4_40_15": FAMILY_MASK_AWARE,
    "VAB11_prostopadly_m008": FAMILY_SEPTUM,
}

# How many images each family is reviewed on. The floors come from the
# plan's coverage table; a family whose transformation can fail in more
# ways needs more of them.
FAMILY_SIZES: Mapping[str, int] = {
    FAMILY_ORIENTATION: 6,
    FAMILY_SCALE: 12,
    FAMILY_TONAL: 8,
    FAMILY_BLUR: 8,
    FAMILY_MASK_AWARE: 12,
    FAMILY_SEPTUM: 16,
}

# Scale bins a family has nothing to show on. Magnification is frozen
# at 1.00 outside the coarse bin, so a panel would differ from its
# original by nothing at all and would only cost review time.
FAMILY_EXCLUDED_BINS: Mapping[str, frozenset[str]] = {
    FAMILY_SCALE: frozenset({"fine", "outlier"}),
}

ROLE_STRATIFIED = "stratified"
ROLE_FALLBACK = "fallback_case"
ROLE_OUTLIER = "diagnostic_outlier"


class GalleryError(ValueError):
    """Raised when no gallery meeting the coverage rules can be built."""


@dataclass(frozen=True)
class ImageAxes:
    """One candidate image, measured on the axes selection uses.

    Every field is either read from the frozen manifest or measured on
    the annotation as the training pipeline sees it, i.e. after the
    crop to the content region.

    Parameters
    ----------
    image_id : str
    formulation : str
    material : str
    microscope : str
    scale_bin : str
    pixel_size_um : float
    height_px, width_px : int
        Content geometry, which differs between the two microscopes and
        is therefore part of what the gallery has to cover.
    n_instances : int
        Annotated pores surviving the crop.
    density_per_mm2 : float
        Pores per square millimetre of imaged material. Expressed
        physically rather than per frame, so the two scales are
        comparable.
    pore_diameter_median_um : float
        Median equivalent diameter of the annotated pores.
    wall_thickness_mean_px : float
        Mean width of the walls between neighbouring pores, in source
        pixels. The model sees them at 0.8 of that.
    wall_thin_share : float
        Share of the wall network at or below ``THIN_WALL_PX``.
    wall_contrast : float
        How far a wall sits above the pore interior beside it, as a
        share of the image's own tonal range. Low values are the case
        where a blur or a downscale can erase a wall outright.
    """

    image_id: str
    formulation: str
    material: str
    microscope: str
    scale_bin: str
    pixel_size_um: float
    height_px: int
    width_px: int
    n_instances: int
    density_per_mm2: float
    pore_diameter_median_um: float
    wall_thickness_mean_px: float
    wall_thin_share: float
    wall_contrast: float

    @property
    def stratum(self) -> tuple[str, str]:
        """Microscope and scale bin, the pair strata are formed on.

        Returns
        -------
        tuple of str
        """
        return (self.microscope, self.scale_bin)


@dataclass(frozen=True)
class GalleryImage:
    """One member of the gallery, with why it is in it.

    Parameters
    ----------
    axes : ImageAxes
    role : str
        ``stratified``, ``fallback_case`` or ``diagnostic_outlier``.
    reason : str
        Which extreme of which axis put it in, or the family whose
        fallback it triggered. Kept so that a reviewer looking at a
        surprising image can see what it was chosen to represent.
    rank : int
        Position within its stratum, in selection order. Families draw
        their subsets by this rank, so a family reviewing fewer images
        still reviews the hardest ones.
    """

    axes: ImageAxes
    role: str
    reason: str
    rank: int


def measure_axes(
    source: SampleSource, indices: Optional[Sequence[int]] = None
) -> tuple[ImageAxes, ...]:
    """Measure every candidate image on the selection axes.

    Parameters
    ----------
    source : SampleSource
        Reader over the TRAIN subset of the frozen split.
    indices : Sequence of int, optional
        Positions to measure; all of them by default. Useful for a
        quick rehearsal on a handful of images.

    Returns
    -------
    tuple of ImageAxes
    """
    positions = range(len(source)) if indices is None else indices
    measured = []
    for index in positions:
        measured.append(_measure_one(source.load(index)))
    logger.info("Measured %d candidate image(s).", len(measured))
    return tuple(measured)


def select_gallery(
    axes: Iterable[ImageAxes],
    *,
    quotas: Mapping[tuple[str, str], int] = STRATUM_QUOTAS,
    forced: Mapping[str, str] = FORCED_IMAGES,
    include_outlier: bool = True,
) -> tuple[GalleryImage, ...]:
    """Choose the gallery from measured candidates.

    Parameters
    ----------
    axes : Iterable of ImageAxes
    quotas : Mapping, optional
        Images to take from each ``(microscope, scale_bin)`` stratum.
    forced : Mapping, optional
        Images that must appear whatever the strata produce, mapped to
        the family that motivated them.
    include_outlier : bool, optional
        Whether to add one close-up. The six close-ups are excluded
        from evaluation but are still trained on, so an augmentation
        that destroys them would do so unseen.

    Returns
    -------
    tuple of GalleryImage
        In stratum order, then selection rank.

    Raises
    ------
    GalleryError
        If a stratum named in the quotas holds no candidate, or a
        forced image is not among the candidates.
    """
    by_id = {entry.image_id: entry for entry in axes}
    chosen: list[GalleryImage] = []
    for stratum, quota in quotas.items():
        chosen.extend(_select_stratum(by_id.values(), stratum, quota))

    taken = {entry.axes.image_id for entry in chosen}
    for image_id, family in forced.items():
        if image_id not in by_id:
            raise GalleryError(
                f"{image_id} is required in the gallery because "
                f"{family} gave up on it, but it is not among the "
                f"candidates; it must be a TRAIN image of the frozen "
                f"split"
            )
        if image_id in taken:
            continue
        chosen.append(
            GalleryImage(
                axes=by_id[image_id],
                role=ROLE_FALLBACK,
                reason=f"fallback_of_{family}",
                rank=len(chosen),
            )
        )
        taken.add(image_id)

    if include_outlier:
        outlier = _select_outlier(by_id.values(), taken)
        if outlier is not None:
            chosen.append(outlier)

    logger.info(
        "Gallery: %d image(s) over %d stratum/strata.",
        len(chosen), len(quotas),
    )
    return tuple(chosen)


def assign_families(
    gallery: Sequence[GalleryImage],
    *,
    sizes: Mapping[str, int] = FAMILY_SIZES,
    forced: Mapping[str, str] = FORCED_IMAGES,
    excluded_bins: Mapping[str, frozenset[str]] = FAMILY_EXCLUDED_BINS,
) -> dict[str, tuple[str, ...]]:
    """Give each family the images it is reviewed on.

    A family takes its images round-robin across the strata rather than
    off the top of one list, so that a family reviewing six images
    still sees both microscopes and both scales. Within a stratum the
    order is the selection rank, i.e. hardest first.

    Close-ups are appended afterwards rather than counted against the
    quota. They are trained on, so an augmentation that ruins them
    would do so unseen, but they are excluded from evaluation and are
    1% of the images; letting one take a sixth of the orientation
    family's review slots would spend the scarce resource - a person's
    attention - on the part of the dataset no metric will report.

    Parameters
    ----------
    gallery : Sequence of GalleryImage
    sizes : Mapping, optional
        Images per family, counting the evaluated population only.
    forced : Mapping, optional
        Images a family must include whatever the round-robin yields.
    excluded_bins : Mapping, optional
        Scale bins a family has nothing to show on.

    Returns
    -------
    dict
        Family code to image ids, in review order, close-ups last.

    Raises
    ------
    GalleryError
        If a family ends up without both microscopes represented.
    """
    assignment: dict[str, tuple[str, ...]] = {}
    for family, size in sizes.items():
        excluded = excluded_bins.get(family, frozenset())
        eligible = [
            entry for entry in gallery
            if entry.axes.scale_bin not in excluded
        ]
        evaluated = [
            entry for entry in eligible
            if entry.axes.scale_bin != "outlier"
        ]
        picked = [
            image_id for image_id, owner in forced.items()
            if owner == family
            and any(e.axes.image_id == image_id for e in eligible)
        ]
        for entry in _round_robin(evaluated):
            if len(picked) >= size:
                break
            if entry.axes.image_id not in picked:
                picked.append(entry.axes.image_id)
        picked.extend(
            entry.axes.image_id for entry in eligible
            if entry.axes.scale_bin == "outlier"
            and entry.axes.image_id not in picked
        )
        assignment[family] = tuple(picked)
        _check_family_coverage(family, picked, evaluated)
    return assignment


def gallery_table(
    gallery: Sequence[GalleryImage],
    assignment: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Lay the gallery out as the rows of the frozen artifact.

    Parameters
    ----------
    gallery : Sequence of GalleryImage
    assignment : Mapping
        Output of ``assign_families``.

    Returns
    -------
    pandas.DataFrame
        One row per image, with the families reviewing it in
        ``reviewed_by`` as a semicolon-separated list.
    """
    reviewed: dict[str, list[str]] = {}
    for family, image_ids in assignment.items():
        for image_id in image_ids:
            reviewed.setdefault(image_id, []).append(family)

    rows = []
    for entry in gallery:
        axes = entry.axes
        rows.append({
            "image_id": axes.image_id,
            "formulation": axes.formulation,
            "material": axes.material,
            "microscope": axes.microscope,
            "scale_bin": axes.scale_bin,
            "role": entry.role,
            "reason": entry.reason,
            "rank_in_stratum": entry.rank,
            "pixel_size_um": axes.pixel_size_um,
            "height_px": axes.height_px,
            "width_px": axes.width_px,
            "n_instances": axes.n_instances,
            "density_per_mm2": round(axes.density_per_mm2, 3),
            "pore_diameter_median_um": round(
                axes.pore_diameter_median_um, 2
            ),
            "wall_thickness_mean_px": round(
                axes.wall_thickness_mean_px, 2
            ),
            "wall_thin_share": round(axes.wall_thin_share, 4),
            "wall_contrast": round(axes.wall_contrast, 4),
            "reviewed_by": ";".join(sorted(reviewed.get(
                axes.image_id, []
            ))),
        })
    return pd.DataFrame(rows)


def check_coverage(gallery: Sequence[GalleryImage]) -> None:
    """Verify the gallery covers what the visual gate requires.

    The conditions are the ones a reviewer cannot recover from later:
    both microscopes, and therefore both content geometries, and both
    scale bins. A gallery missing one of them would let a family be
    accepted on evidence that never included the case it fails on.

    Parameters
    ----------
    gallery : Sequence of GalleryImage

    Raises
    ------
    GalleryError
    """
    microscopes = {entry.axes.microscope for entry in gallery}
    bins = {
        entry.axes.scale_bin for entry in gallery
        if entry.axes.scale_bin != "outlier"
    }
    geometries = {
        (entry.axes.height_px, entry.axes.width_px)
        for entry in gallery
    }
    if len(microscopes) < 2:
        raise GalleryError(
            f"gallery covers only microscope(s) {sorted(microscopes)}; "
            f"both are required, since microscope, foam family and "
            f"content geometry coincide in this dataset"
        )
    if not {"coarse", "fine"}.issubset(bins):
        raise GalleryError(
            f"gallery covers only scale bin(s) {sorted(bins)}; both "
            f"coarse and fine are required"
        )
    if len(geometries) < 2:
        raise GalleryError(
            f"gallery covers only content geometry {sorted(geometries)}"
            f"; both 960x1280 and 890x1280 are required"
        )


def _measure_one(sample: PreparedSample) -> ImageAxes:
    """Measure one prepared sample on every selection axis."""
    record = sample.record
    height_px, width_px = sample.labels.shape
    pixel_size_um = float(record.pixel_size_um)
    areas_px2 = np.bincount(sample.labels.ravel())[1:]
    walls = measure_walls(sample.image, sample.labels)
    thickness_px = walls.thickness_px

    area_mm2 = height_px * width_px * pixel_size_um ** 2 / 1e6
    diameters_um = (
        2.0 * np.sqrt(areas_px2 / np.pi) * pixel_size_um
        if areas_px2.size else np.zeros(0)
    )
    return ImageAxes(
        image_id=record.image_id,
        formulation=record.formulation,
        material=record.material,
        microscope=record.microscope,
        scale_bin=record.scale_bin,
        pixel_size_um=pixel_size_um,
        height_px=int(height_px),
        width_px=int(width_px),
        n_instances=int(areas_px2.size),
        density_per_mm2=float(areas_px2.size / area_mm2),
        pore_diameter_median_um=(
            float(np.median(diameters_um)) if diameters_um.size
            else float("nan")
        ),
        wall_thickness_mean_px=(
            float(thickness_px.mean()) if thickness_px.size
            else float("nan")
        ),
        wall_thin_share=(
            float(np.mean(thickness_px <= THIN_WALL_PX))
            if thickness_px.size else float("nan")
        ),
        wall_contrast=walls.contrast,
    )


def _select_stratum(
    axes: Iterable[ImageAxes],
    stratum: tuple[str, str],
    quota: int,
) -> list[GalleryImage]:
    """Take one stratum's images, extremes first.

    An image can hold two extremes at once - the frame with the
    thinnest walls is often also the densest - in which case the reason
    recorded is the first one it won and the next candidate fills the
    slot that freed up.
    """
    candidates = [entry for entry in axes if entry.stratum == stratum]
    if not candidates:
        raise GalleryError(
            f"stratum {stratum} holds no candidate image; the quotas "
            f"describe a dataset this split does not contain"
        )

    picked: list[GalleryImage] = []
    taken: set[str] = set()
    for reason in SELECTION_REASONS:
        if len(picked) >= quota:
            break
        remaining = [
            entry for entry in candidates
            if entry.image_id not in taken
        ]
        if not remaining:
            break
        winner = _extreme(remaining, reason, candidates)
        picked.append(GalleryImage(
            axes=winner,
            role=ROLE_STRATIFIED,
            reason=reason,
            rank=len(picked),
        ))
        taken.add(winner.image_id)
    return picked


def _extreme(
    remaining: Sequence[ImageAxes],
    reason: str,
    stratum: Sequence[ImageAxes],
) -> ImageAxes:
    """Return the image holding one named extreme.

    Ties break on ``image_id`` so the gallery does not depend on the
    order the manifest happened to be written in.
    """
    keys = {
        "thinnest_walls": lambda e: (_finite(
            e.wall_thickness_mean_px, high=True
        ), e.image_id),
        "thickest_walls": lambda e: (-_finite(
            e.wall_thickness_mean_px, high=False
        ), e.image_id),
        "lowest_wall_contrast": lambda e: (_finite(
            e.wall_contrast, high=True
        ), e.image_id),
        "densest": lambda e: (-e.density_per_mm2, e.image_id),
        "sparsest": lambda e: (e.density_per_mm2, e.image_id),
        "smallest_pores": lambda e: (_finite(
            e.pore_diameter_median_um, high=True
        ), e.image_id),
        "largest_pores": lambda e: (-_finite(
            e.pore_diameter_median_um, high=False
        ), e.image_id),
        "most_typical": lambda e: (
            _atypicality(e, stratum), e.image_id
        ),
    }
    return min(remaining, key=keys[reason])


def _atypicality(
    entry: ImageAxes, stratum: Sequence[ImageAxes]
) -> float:
    """How far an image sits from its stratum's centre.

    Distance is summed over the four axes after each is divided by its
    own spread within the stratum, since a wall width in pixels and a
    density per square millimetre are not otherwise comparable. The
    spread is the interquartile range rather than the standard
    deviation: the axes are the ones extremes were just taken on, and
    a single outlying frame would otherwise set the scale.
    """
    total = 0.0
    for values, value in _axis_pairs(entry, stratum):
        finite = values[np.isfinite(values)]
        if finite.size < 2 or not np.isfinite(value):
            continue
        spread = float(np.subtract(*np.percentile(finite, [75, 25])))
        if spread <= 0.0:
            continue
        total += abs(value - float(np.median(finite))) / spread
    return total


def _axis_pairs(
    entry: ImageAxes, stratum: Sequence[ImageAxes]
) -> list[tuple[np.ndarray, float]]:
    """Pair each axis's stratum-wide values with this image's value."""
    names = (
        "wall_thickness_mean_px",
        "wall_contrast",
        "density_per_mm2",
        "pore_diameter_median_um",
    )
    return [
        (
            np.array([getattr(other, name) for other in stratum]),
            float(getattr(entry, name)),
        )
        for name in names
    ]


def _finite(value: float, *, high: bool) -> float:
    """Replace a missing measurement so it never wins an extreme.

    An image with no wall between two pores has no wall thickness. It
    is a legitimate image, but it cannot be the one chosen to show what
    a transformation does to thin walls, and a ``nan`` compares as
    neither larger nor smaller than anything.
    """
    if np.isfinite(value):
        return float(value)
    return float("inf") if high else float("-inf")


def _select_outlier(
    axes: Iterable[ImageAxes], taken: set[str]
) -> Optional[GalleryImage]:
    """Add the most extreme close-up, if the candidates hold one."""
    outliers = [
        entry for entry in axes
        if entry.scale_bin == "outlier" and entry.image_id not in taken
    ]
    if not outliers:
        return None
    winner = min(
        outliers, key=lambda e: (e.pixel_size_um, e.image_id)
    )
    return GalleryImage(
        axes=winner,
        role=ROLE_OUTLIER,
        reason="finest_close_up",
        rank=0,
    )


def _round_robin(
    gallery: Sequence[GalleryImage],
) -> list[GalleryImage]:
    """Interleave the strata, so a short subset still spans them.

    Strata are visited in the order the quotas declare them, which puts
    the majority stratum first; within a stratum images come in
    selection rank, hardest first.
    """
    order = list(STRATUM_QUOTAS) + [None]
    buckets: dict[object, list[GalleryImage]] = {
        key: [] for key in order
    }
    for entry in gallery:
        stratum = entry.axes.stratum
        buckets[stratum if stratum in buckets else None].append(entry)
    for bucket in buckets.values():
        bucket.sort(key=lambda e: (e.rank, e.axes.image_id))

    interleaved: list[GalleryImage] = []
    depth = 0
    while any(len(bucket) > depth for bucket in buckets.values()):
        for key in order:
            bucket = buckets[key]
            if len(bucket) > depth:
                interleaved.append(bucket[depth])
        depth += 1
    return interleaved


def _check_family_coverage(
    family: str,
    picked: Sequence[str],
    eligible: Sequence[GalleryImage],
) -> None:
    """Verify one family's subset spans both microscopes.

    Both are required wherever both are eligible: microscope, foam
    family and content geometry coincide in this dataset, so a family
    judged on one of them has been judged on one third of the problem.
    """
    available = {entry.axes.microscope for entry in eligible}
    covered = {
        entry.axes.microscope for entry in eligible
        if entry.axes.image_id in picked
    }
    if available.issubset(covered):
        return
    raise GalleryError(
        f"{family} would be reviewed on microscope(s) "
        f"{sorted(covered)} although {sorted(available)} are "
        f"available; raise its image count or loosen its exclusions"
    )
