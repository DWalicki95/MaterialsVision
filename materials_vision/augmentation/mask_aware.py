"""
Shading painted inside pores, with the annotation left untouched.

Two things inside a pore look like its edge without being one: a slow
change of brightness across the interior, and a dark patch where the
membrane tore and the image sees deeper into the material. A model
that reads either as a boundary reports one pore as two. Showing it
both, labelled as the single pore they belong to, is what teaches it
otherwise.

**Why these are image-only transformations.** Both need to see the
annotation - they paint inside pores and nowhere else - but neither
may change it. Written as transformations that take the mask and hand
it back, that would be a promise to keep; written as image-only
transformations that merely read the mask, it is a property of the
construction, because there is no path by which the mask could be
returned altered.

**Why the shading fades to nothing at the pore's edge.** Adding a
value to every pixel of a pore would put a step exactly on the
boundary, making the edge sharper than the photograph ever showed it
- the opposite of what these transformations are for. The strength
therefore rises from zero at the boundary towards the pore's core,
following the distance to the nearest pixel that is not part of that
pore.

**Why one effect is added and the other multiplied.** Shading should
leave the texture it passes over exactly as it was, and addition
does. A region seen deeper into the material returns less signal
everywhere beneath it, in proportion to what was there, and that is
multiplication. Swapping them would flatten the texture of a whole
pore in the first case and keep a patch at full contrast where it
should have lost some in the second.
"""
import logging
from typing import Any, Mapping, Optional

import albumentations as A
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.transform import resize

from materials_vision.augmentation.arrays import to_source_dtype
from materials_vision.augmentation.config import MaskAwareConfig

logger = logging.getLogger(__name__)

# The intensities taken to bound an image's tonal range. The extremes
# themselves are not used: a single blown-out speck or one dead pixel
# would otherwise set the scale for the whole image.
TONAL_PERCENTILES = (5.0, 95.0)

# Shape of the dark patch, as the ratio of its two axes. One would be
# a circle, which reads as drawn rather than photographed; far above
# two it becomes a streak, which reads as a scratch.
PATCH_ASPECT = (1.0, 2.0)

FIELD_RECORD_KEYS = (
    "kind",
    "strength",
    "amplitude",
    "tonal_span",
    "n_pores_eligible",
    "n_pores_shaded",
    "attempts",
    "fallback",
)

DARKENING_RECORD_KEYS = (
    "n_pores_eligible",
    "n_pores_darkened",
    "factors",
    "area_fractions",
    "attempts",
    "fallback",
)


class PoreBrightnessField(A.ImageOnlyTransform):
    """Shade the interiors of some pores, fading out at their edges.

    Parameters
    ----------
    config : MaskAwareConfig

    Notes
    -----
    One shape and one strength are drawn for the whole sample, and
    only the direction - lighter or darker - varies from pore to pore.
    Drawing the strength separately per pore would make every sample
    an average of many strengths, and there would then be no such
    thing as a weak or a strong sample to compare or to look at.
    """

    def __init__(self, config: MaskAwareConfig) -> None:
        super().__init__(p=1.0)
        self._config = config

    @property
    def targets_as_params(self) -> list[str]:
        """Inputs the drawn parameters depend on.

        Returns
        -------
        list of str
        """
        return ["image", "mask"]

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the shading and build it as one additive field.

        Parameters
        ----------
        params : dict
            Parameters drawn so far; unused.
        data : dict
            The sample being transformed.

        Returns
        -------
        dict
            The field to add under ``delta``, ``None`` when nothing
            is shaded, plus everything worth recording.
        """
        image = data["image"]
        labels = data["mask"]
        config = self._config
        eligible = _eligible_labels(
            labels, config.min_core_distance_px
        )
        if eligible.size == 0:
            return _no_shading(0, "no_pore_is_deep_enough_to_shade")

        span = _tonal_span(image)
        if span <= 0.0:
            return _no_shading(
                eligible.size, "image_has_no_tonal_range"
            )

        kind = self.py_random.choice(config.field_kinds)
        strength = self.py_random.uniform(*config.strength)
        amplitude = strength * span
        chosen = self._choose(eligible)

        boxes = find_objects(labels)
        delta = np.zeros(labels.shape, dtype=np.float32)
        shaded = 0
        for label in chosen:
            box = boxes[label - 1]
            distance = _interior_distance(labels, label, box)
            deepest = float(distance.max())
            if deepest < config.min_core_distance_px:
                continue
            ramp = _fade(distance, deepest)
            field = self._draw_field(kind, ramp.shape, amplitude)
            inside = labels[box] == label
            delta[box] += np.where(inside, ramp * field, 0.0)
            shaded += 1

        if shaded == 0:
            return _no_shading(
                eligible.size, "no_pore_is_deep_enough_to_shade"
            )
        return {
            "kind": kind,
            "strength": round(strength, 5),
            "amplitude": round(amplitude, 3),
            "tonal_span": round(span, 3),
            "n_pores_eligible": int(eligible.size),
            "n_pores_shaded": shaded,
            "attempts": 1,
            "fallback": None,
            "delta": delta,
        }

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Add the shading to the image.

        Parameters
        ----------
        img : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        delta = params["delta"]
        if delta is None:
            return img
        return to_source_dtype(img.astype(np.float32) + delta, img)

    def _choose(self, eligible: np.ndarray) -> list[int]:
        """Pick the share of eligible pores this sample shades.

        Never all of them: an image whose every pore is shaded
        presents the shading as a property of the material, which is
        the reading the family is meant to prevent.
        """
        fraction = self.py_random.uniform(*self._config.pore_fraction)
        count = max(1, int(round(eligible.size * fraction)))
        return self.py_random.sample(eligible.tolist(), count)

    def _draw_field(
        self, kind: str, shape: tuple[int, ...], amplitude: float
    ) -> np.ndarray:
        """Build one pore's shading, before the fade-out is applied.

        Every shape spans at most ``amplitude`` either side of zero,
        so the three are comparable at the same strength and a run
        that changes only the shape changes only the shape.
        """
        if kind == "constant":
            sign = self.py_random.choice((-1.0, 1.0))
            return np.full(shape, amplitude * sign, dtype=np.float32)
        if kind == "gradient":
            angle = self.py_random.uniform(0.0, 2.0 * np.pi)
            rows = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
            columns = np.linspace(
                -1.0, 1.0, shape[1], dtype=np.float32
            )
            projection = (
                columns[None, :] * float(np.cos(angle))
                + rows[:, None] * float(np.sin(angle))
            )
            peak = float(np.abs(projection).max())
            if peak == 0.0:
                return np.zeros(shape, dtype=np.float32)
            return (amplitude * projection / peak).astype(np.float32)

        side = self.py_random.choice(self._config.field_grid_sides)
        coarse = self.random_generator.uniform(-1.0, 1.0, (side, side))
        smooth = resize(
            coarse, shape, order=1, preserve_range=True,
            anti_aliasing=False,
        )
        return (amplitude * smooth).astype(np.float32)


class PoreDarkening(A.ImageOnlyTransform):
    """Darken a soft-edged patch well inside one or two pores.

    Parameters
    ----------
    config : MaskAwareConfig

    Notes
    -----
    A patch is placed only where it fits whole, clear of the pore's
    boundary by the configured margin. A placement that would touch
    the edge is re-drawn rather than trimmed: a trimmed patch follows
    the boundary, which makes it look like part of the boundary, and
    that is precisely the appearance this transformation exists to
    teach the model to disregard.
    """

    def __init__(self, config: MaskAwareConfig) -> None:
        super().__init__(p=1.0)
        self._config = config

    @property
    def targets_as_params(self) -> list[str]:
        """Inputs the drawn parameters depend on.

        Returns
        -------
        list of str
        """
        return ["image", "mask"]

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Place the patches and build them as one multiplier map.

        Parameters
        ----------
        params : dict
            Parameters drawn so far; unused.
        data : dict
            The sample being transformed.

        Returns
        -------
        dict
            The multiplier under ``attenuation``, ``None`` when
            nothing is darkened, plus everything worth recording.
        """
        labels = data["mask"]
        config = self._config
        eligible = _eligible_labels(
            labels, config.min_core_distance_px
        )
        if eligible.size == 0:
            return _no_darkening(0, "no_pore_is_deep_enough_to_hold")

        count = min(
            self.py_random.randint(*config.darkened_pores),
            int(eligible.size),
        )
        chosen = self.py_random.sample(eligible.tolist(), count)

        boxes = find_objects(labels)
        attenuation = np.ones(labels.shape, dtype=np.float32)
        factors: list[float] = []
        fractions: list[float] = []
        attempts = 0
        for label in chosen:
            box = boxes[label - 1]
            inside = labels[box] == label
            distance = _interior_distance(labels, label, box)
            patch, fraction, tries = self._place_patch(
                distance, float(np.count_nonzero(inside))
            )
            attempts += tries
            if patch is None:
                continue
            factor = self.py_random.uniform(*config.darkening_factor)
            attenuation[box] *= 1.0 - patch * (1.0 - factor)
            factors.append(round(factor, 4))
            fractions.append(round(fraction, 4))

        if not factors:
            return _no_darkening(
                eligible.size, "no_patch_fitted_clear_of_a_boundary",
                attempts=max(attempts, 1),
            )
        return {
            "n_pores_eligible": int(eligible.size),
            "n_pores_darkened": len(factors),
            "factors": tuple(factors),
            "area_fractions": tuple(fractions),
            "attempts": max(attempts, 1),
            "fallback": None,
            "attenuation": attenuation,
        }

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Multiply the image by the patch map.

        Parameters
        ----------
        img : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        attenuation = params["attenuation"]
        if attenuation is None:
            return img
        return to_source_dtype(
            img.astype(np.float32) * attenuation, img
        )

    def _place_patch(
        self, distance: np.ndarray, pore_area: float
    ) -> tuple[Optional[np.ndarray], float, int]:
        """Fit one soft-edged ellipse well inside a pore.

        The centre is not drawn from the whole interior and then
        checked. It is drawn only from the pixels deep enough that an
        ellipse of the size just chosen cannot reach the boundary from
        there: every point of the ellipse lies within its longer axis
        of the centre, so a centre at least that far plus the margin
        from anything outside the pore places the whole patch clear of
        the edge by construction. Drawing first and testing afterwards
        rejects most of the interior of a small pore and turns a
        placement that was always possible into a fallback.

        Returns the patch's weight at every pixel - one at the centre,
        falling to zero at its rim - with the share of the pore it
        covers and how many sizes were tried.
        """
        config = self._config
        grid_rows = np.arange(
            distance.shape[0], dtype=np.float32
        )[:, None]
        grid_columns = np.arange(
            distance.shape[1], dtype=np.float32
        )[None, :]

        for attempt in range(1, config.darkening_max_attempts + 1):
            fraction = self.py_random.uniform(*config.darkened_area)
            aspect = self.py_random.uniform(*PATCH_ASPECT)
            minor = float(
                np.sqrt(fraction * pore_area / (np.pi * aspect))
            )
            if minor < 1.0:
                continue
            major = aspect * minor

            rows, columns = np.nonzero(
                distance >= config.darkening_margin_px + major
            )
            if rows.size == 0:
                continue

            index = self.py_random.randrange(rows.size)
            angle = self.py_random.uniform(0.0, np.pi)
            offset_rows = grid_rows - float(rows[index])
            offset_columns = grid_columns - float(columns[index])
            along = (
                offset_columns * float(np.cos(angle))
                + offset_rows * float(np.sin(angle))
            )
            across = (
                -offset_columns * float(np.sin(angle))
                + offset_rows * float(np.cos(angle))
            )
            radius = np.sqrt(
                (along / major) ** 2 + (across / minor) ** 2
            )
            support = radius <= 1.0
            if not support.any():
                continue

            weight = np.clip(
                (1.0 - radius) / config.darkening_edge_softness,
                0.0, 1.0,
            )
            patch = np.where(support, weight, 0.0).astype(np.float32)
            covered = float(np.count_nonzero(support)) / pore_area
            return patch, covered, attempt

        return None, 0.0, config.darkening_max_attempts


def build_mask_aware(config: MaskAwareConfig) -> A.OneOf:
    """Build the mask-aware photometry.

    Parameters
    ----------
    config : MaskAwareConfig

    Returns
    -------
    A.OneOf
        Fires with the configured probability and then draws one of
        the two members with equal weight. They are alternatives: a
        pore carrying both a shading and a dark patch would show an
        interior no photograph produces.
    """
    return A.OneOf(
        [PoreBrightnessField(config), PoreDarkening(config)],
        p=config.p,
    )


def summarize_field_params(
    params: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce a shading's parameters to what belongs in a record.

    Parameters
    ----------
    params : Mapping

    Returns
    -------
    dict
    """
    return _selected(params, FIELD_RECORD_KEYS)


def summarize_darkening_params(
    params: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce a patch's parameters to what belongs in a record.

    Parameters
    ----------
    params : Mapping

    Returns
    -------
    dict
    """
    return _selected(params, DARKENING_RECORD_KEYS)


def _selected(
    params: Mapping[str, Any], keys: tuple[str, ...]
) -> dict[str, Any]:
    """Keep the named values and drop everything else.

    What is dropped is the painted field itself, which travels in the
    parameters because that is how it reaches the image, and the frame
    size the library adds to every transformation. A record is written
    to the log, where neither belongs.
    """
    return {key: params[key] for key in keys if key in params}


def _tonal_span(image: np.ndarray) -> float:
    """Width of an image's tonal range, ignoring its extremes."""
    low, high = np.percentile(image, TONAL_PERCENTILES)
    return float(high) - float(low)


def _eligible_labels(
    labels: np.ndarray, min_distance_px: float
) -> np.ndarray:
    """Instances that could hold shading clear of their boundary.

    A cheap necessary condition standing in front of an expensive
    exact one. An instance with a pixel at least ``r`` from anything
    outside it must cover at least the area of a disc of radius ``r``;
    anything smaller is discarded here rather than measured, which
    matters because the exact test costs a distance transform per
    instance and an image holds dozens.
    """
    areas = np.bincount(labels.ravel())
    if areas.size < 2:
        return np.empty(0, dtype=np.int64)
    smallest = np.pi * min_distance_px ** 2
    return np.flatnonzero(areas[1:] >= smallest) + 1


def _interior_distance(
    labels: np.ndarray, label: int, box: tuple[slice, ...]
) -> np.ndarray:
    """Distance from an instance's pixels to the nearest outside one.

    Measured one instance at a time rather than for all pores at
    once. Two annotated pores can share a boundary with no wall of
    background between them, and measured together they would read as
    a single region - leaving the shading at full strength across the
    very edge that separates them.

    The instance is padded by one pixel of background first, so an
    instance running up to the edge of its own bounding box still has
    something to be measured against.
    """
    inside = labels[box] == label
    padded = np.zeros(
        (inside.shape[0] + 2, inside.shape[1] + 2), dtype=bool
    )
    padded[1:-1, 1:-1] = inside
    return np.asarray(distance_transform_edt(padded))[1:-1, 1:-1]


def _fade(distance: np.ndarray, deepest: float) -> np.ndarray:
    """Turn distances into a weight of zero at the edge, one at the core.

    The subtraction is what makes the edge exact. A distance transform
    reports one, not zero, for a pixel lying directly against the
    outside, so weighting by distance alone would leave the outermost
    ring of every pore carrying a fraction of the shading while the
    wall beside it carried none - a step in brightness exactly on the
    boundary, which is the one thing this shading must not produce.
    Measuring from the outermost ring instead puts the zero where the
    annotation says the edge is.
    """
    span = deepest - 1.0
    if span <= 0.0:
        return np.zeros(distance.shape, dtype=np.float32)
    faded = (distance - 1.0) / span
    return np.clip(faded, 0.0, 1.0).astype(np.float32)


def _no_shading(n_eligible: int, fallback: str) -> dict[str, Any]:
    """Parameters for a sample the shading could not be applied to."""
    return {
        "kind": None,
        "strength": None,
        "amplitude": None,
        "tonal_span": None,
        "n_pores_eligible": int(n_eligible),
        "n_pores_shaded": 0,
        "attempts": 1,
        "fallback": fallback,
        "delta": None,
    }


def _no_darkening(
    n_eligible: int, fallback: str, attempts: int = 1
) -> dict[str, Any]:
    """Parameters for a sample no patch could be placed in."""
    return {
        "n_pores_eligible": int(n_eligible),
        "n_pores_darkened": 0,
        "factors": (),
        "area_fractions": (),
        "attempts": attempts,
        "fallback": fallback,
        "attenuation": None,
    }
