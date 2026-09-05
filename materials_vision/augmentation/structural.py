"""
A wall drawn across a pore, turning one instance into two.

Every other family here varies how a pore looks. This one changes what
the annotation says, on purpose and in the one direction that is safe:
a pore that was whole becomes two pores separated by a wall, and both
halves are labelled. The error being trained against is two touching
pores reported as one. Real examples of it are exactly the cases the
annotator found hard, so the training set holds fewest of them where
the model needs most.

**Why the wall's appearance is measured, not chosen.** A wall drawn as
a dark line teaches the model to look for dark lines. The width and
the brightness used here are read off the walls already present in the
training images, so a synthetic wall is a copy of a measured one; the
measurement itself lives alongside, in this package.

**Why the halves are checked rather than assumed.** Drawing a curve
between two points of an outline usually divides the shape in two, but
not always: it can clip a lobe, leave a sliver too small to be an
instance, or - on a pore with a concave outline - cut off three pieces
instead of two. Each of those would put something in the annotation
that no annotator would have drawn. The division is therefore carried
out and then inspected, and a wall that fails is discarded and
re-drawn rather than repaired.

**Why the wall is removed from the mask but only faded into the
image.** In the annotation a wall is not part of any pore, so its
pixels stop belonging to one. In the photograph a wall has edges that
blend into the pores on either side over about a pixel, and reproducing
that blend is what stops the result from looking drawn. The two do not
have to agree: the faded margin stays inside the instance, exactly as
the margin of a real wall does.
"""
import logging
from typing import Any, Mapping, Optional

import albumentations as A
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects
from skimage.draw import bezier_curve
from skimage.measure import label as connected_components

from materials_vision.augmentation.arrays import to_source_dtype
from materials_vision.augmentation.config import SeptumConfig
from materials_vision.augmentation.walls import TONAL_PERCENTILES

logger = logging.getLogger(__name__)

RECORD_KEYS = (
    "changed_mask",
    "divided_instance",
    "thickness_px",
    "sag",
    "chord_px",
    "fragment_areas_px2",
    "fragment_ratio",
    "target_intensity",
    "n_instances_before",
    "n_instances_after",
    "attempts",
    "fallback",
)


class SyntheticSeptum(A.DualTransform):
    """Divide one large pore in two with a wall of measured width.

    Parameters
    ----------
    config : SeptumConfig

    Notes
    -----
    One pore per sample. Drawing into several would change the size
    distribution of a whole image at once, and the size distribution
    is one of the things a run reports.
    """

    def __init__(self, config: SeptumConfig) -> None:
        super().__init__(p=config.p)
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
        """Draw a wall, divide the pore, and check what came out.

        Parameters
        ----------
        params : dict
            Parameters drawn so far; unused.
        data : dict
            The sample being transformed.

        Returns
        -------
        dict
            The divided pair under ``walled_image`` and
            ``divided_labels`` - both ``None`` when nothing was
            divided - plus everything worth recording.
        """
        image = data["image"]
        labels = data["mask"]
        config = self._config
        n_before = int(labels.max())
        if n_before == 0:
            return _undivided(0, "frame_holds_no_pore")

        candidates = self._candidates(labels, n_before)
        if candidates.size == 0:
            return _undivided(n_before, "no_pore_large_enough")

        boxes = find_objects(labels)
        for attempt in range(1, config.max_retries + 2):
            label = int(self.py_random.choice(candidates.tolist()))
            box = boxes[label - 1]
            division = self._divide(image, labels, label, box)
            if division is not None:
                division["attempts"] = attempt
                return division

        return _undivided(
            n_before,
            "no_wall_divided_the_pore_in_two",
            attempts=config.max_retries + 1,
        )

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Return the image with the wall painted in.

        Parameters
        ----------
        img : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        walled = params["walled_image"]
        return img if walled is None else walled

    def apply_to_mask(
        self, mask: np.ndarray, **params: Any
    ) -> np.ndarray:
        """Return the labels with the pore divided in two.

        Parameters
        ----------
        mask : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        divided = params["divided_labels"]
        return mask if divided is None else divided

    def _candidates(
        self, labels: np.ndarray, n_before: int
    ) -> np.ndarray:
        """The largest pores, the share of them drawn for this sample.

        A wall needs a pore with room on both sides of it. Dividing one
        from the small end of the distribution would produce two
        instances below anything an annotator drew, which is a worse
        error than not augmenting the sample at all.
        """
        areas = np.bincount(labels.ravel(), minlength=n_before + 1)[1:]
        share = self.py_random.uniform(*self._config.candidate_fraction)
        count = max(1, int(round(n_before * share)))
        largest = np.argsort(areas)[::-1][:count] + 1
        return largest[
            areas[largest - 1] >= 4.0 * self._config.min_fragment_area_px2
        ]

    def _divide(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label: int,
        box: tuple[slice, ...],
    ) -> Optional[dict[str, Any]]:
        """Try once to divide one pore, returning None if it failed."""
        config = self._config
        inside = labels[box] == label
        ends = self._draw_ends(inside)
        if ends is None:
            return None

        (start_row, start_col), (end_row, end_col) = ends
        chord = float(
            np.hypot(end_row - start_row, end_col - start_col)
        )
        sag = self.py_random.uniform(*config.sag) * self.py_random.choice(
            (-1.0, 1.0)
        )
        thickness = self.py_random.uniform(*config.thickness_px)
        weight = _wall_weight(
            inside.shape, ends, sag, chord, thickness,
            config.edge_softness_px,
        )
        core = (weight >= 1.0) & inside

        fragments, areas = _fragments(inside & ~core)
        if fragments is None:
            return None
        smaller = float(areas.min())
        total = float(areas.sum())
        if smaller < config.min_fragment_area_px2:
            return None
        if smaller / total < config.fragment_ratio:
            return None

        return self._build(
            image, labels, label, box, inside, weight, fragments,
            areas, thickness, sag, chord,
        )

    def _draw_ends(
        self, inside: np.ndarray
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        """Pick two points of the outline that are genuinely apart.

        The second end is drawn only from the part of the outline far
        from the first. Drawn freely, the two would often land close
        together and the wall would shave a sliver off the rim instead
        of crossing the pore - a rejection that costs an attempt and
        can be avoided by not drawing it.
        """
        outline = inside & ~binary_erosion(inside)
        rows, columns = np.nonzero(outline)
        if rows.size < 2:
            return None

        first = self.py_random.randrange(rows.size)
        spread = np.hypot(
            rows - rows[first], columns - columns[first]
        )
        far = np.flatnonzero(
            spread >= self._config.min_chord_share * spread.max()
        )
        if far.size == 0:
            return None
        second = int(self.py_random.choice(far.tolist()))
        return (
            (int(rows[first]), int(columns[first])),
            (int(rows[second]), int(columns[second])),
        )

    def _build(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label: int,
        box: tuple[slice, ...],
        inside: np.ndarray,
        weight: np.ndarray,
        fragments: np.ndarray,
        areas: np.ndarray,
        thickness: float,
        sag: float,
        chord: float,
    ) -> dict[str, Any]:
        """Assemble the divided labels and the image with the wall."""
        n_before = int(labels.max())
        divided = labels.copy()
        window = divided[box]
        window[inside] = 0
        window[fragments == 1] = label
        window[fragments == 2] = n_before + 1

        low, high = np.percentile(image, TONAL_PERCENTILES)
        interior = float(np.median(image[box][inside]))
        target = interior + self._config.contrast * (
            float(high) - float(low)
        )
        walled = image.copy()
        patch = walled[box].astype(np.float32)
        blend = np.where(inside, weight, 0.0).astype(np.float32)
        walled[box] = to_source_dtype(
            patch * (1.0 - blend) + target * blend, image
        )

        return {
            "changed_mask": True,
            "divided_instance": label,
            "thickness_px": round(thickness, 3),
            "sag": round(sag, 4),
            "chord_px": round(chord, 2),
            "fragment_areas_px2": tuple(int(area) for area in areas),
            "fragment_ratio": round(
                float(areas.min()) / float(areas.sum()), 4
            ),
            "target_intensity": round(target, 2),
            "n_instances_before": n_before,
            "n_instances_after": n_before + 1,
            "fallback": None,
            "walled_image": walled,
            "divided_labels": divided,
        }


def build_septum(config: SeptumConfig) -> SyntheticSeptum:
    """Build the synthetic wall transformation.

    Parameters
    ----------
    config : SeptumConfig

    Returns
    -------
    SyntheticSeptum
    """
    return SyntheticSeptum(config)


def summarize_septum_params(
    params: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce a division's parameters to what belongs in a record.

    Parameters
    ----------
    params : Mapping

    Returns
    -------
    dict
    """
    return {key: params[key] for key in RECORD_KEYS if key in params}


def _wall_weight(
    shape: tuple[int, ...],
    ends: tuple[tuple[int, int], tuple[int, int]],
    sag: float,
    chord: float,
    thickness: float,
    softness: float,
) -> np.ndarray:
    """Build the wall as a weight of one at its centre, zero outside.

    The curve is drawn one pixel wide first and then given its width
    by distance rather than by dilation, which is what makes both the
    width and the fade-out continuous: a wall 2.7 pixels across is
    2.7 pixels across, and its edges lose strength over a fraction of
    a pixel instead of in whole steps.

    A control point offset from the middle of the straight line bends
    the wall. An offset of zero leaves it straight, so the straight
    and the curved case are the same construction rather than two.
    """
    (start_row, start_col), (end_row, end_col) = ends
    middle_row = 0.5 * (start_row + end_row)
    middle_col = 0.5 * (start_col + end_col)
    if chord > 0.0:
        across_row = -(end_col - start_col) / chord
        across_col = (end_row - start_row) / chord
    else:
        across_row = across_col = 0.0
    control_row = middle_row + sag * chord * across_row
    control_col = middle_col + sag * chord * across_col

    rows, columns = bezier_curve(
        start_row, start_col,
        int(round(control_row)), int(round(control_col)),
        end_row, end_col,
        1.0, shape=shape,
    )
    centre = np.zeros(shape, dtype=bool)
    centre[rows, columns] = True

    distance = np.asarray(distance_transform_edt(~centre))
    half = 0.5 * thickness
    return np.clip(
        (half + softness - distance) / softness, 0.0, 1.0
    ).astype(np.float32)


def _fragments(
    remaining: np.ndarray,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Split what the wall left behind, accepting only two pieces.

    Anything other than two means the wall did not do what a wall
    does. One piece is a wall that failed to reach across; three or
    more is a wall that broke a chip off as well as dividing, and a
    chip nobody annotated is not an instance.
    """
    pieces = connected_components(
        remaining, background=0, connectivity=1
    )
    counts = np.bincount(pieces.ravel())[1:]
    if counts.size != 2:
        return None, counts
    return pieces, counts


def _undivided(
    n_before: int, fallback: str, attempts: int = 1
) -> dict[str, Any]:
    """Parameters for a sample no wall could be drawn into."""
    return {
        "changed_mask": False,
        "divided_instance": None,
        "n_instances_before": n_before,
        "n_instances_after": n_before,
        "attempts": attempts,
        "fallback": fallback,
        "walled_image": None,
        "divided_labels": None,
    }
