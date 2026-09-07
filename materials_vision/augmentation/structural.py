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
    "divided_instances",
    "n_septa_requested",
    "n_septa_drawn",
    "candidate_pool",
    "divided_area_share",
    "thickness_px",
    "contrasts",
    "sags",
    "fragment_ratios",
    "smallest_fragment_px2",
    "target_intensities",
    "n_instances_before",
    "n_instances_after",
    "attempts",
    "fallback",
)


class SyntheticSeptum(A.DualTransform):
    """Divide several large pores, each in two, with walls of measured
    width.

    Parameters
    ----------
    config : SeptumConfig

    Notes
    -----
    **How many walls a sample gets follows from the image.** The
    training images differ several-fold in how many pores they hold,
    so a fixed count would divide most of the candidates on a sparse
    image and a small fraction of them on a dense one. The count is
    therefore a share of the candidate pool, which makes it
    proportionate on both without a special case for either.

    **Each pore is divided at most once.** The pool is fixed before
    the first wall is drawn and pores are taken from it without
    replacement, so a fragment produced by one wall can never be cut
    again - which would otherwise produce chains of slivers whose
    sizes no annotator would have drawn.

    **The area a sample may lose is bounded separately from the
    count.** Walls go into the largest pores, so a count on its own
    says little about how much of the image changes; a handful of
    walls on an image of few large pores can rebuild most of its
    annotated area. The share of divided area is capped as a guard on
    that tail, and what it reached is recorded per sample, because
    dividing pores moves the size distribution and the size
    distribution is one of the things a run reports.
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

        areas = np.bincount(labels.ravel(), minlength=n_before + 1)[1:]
        candidates = self._candidates(areas, n_before)
        if candidates.size == 0:
            return _undivided(n_before, "no_pore_large_enough")

        requested = self._requested_count(candidates.size)
        # The tonal range is read once, from the image as it arrived.
        # Taken again after a wall had been painted in it would drift
        # with the walls this call is drawing, so the last wall of a
        # sample would be built to a slightly different brightness than
        # the first for no reason anyone chose.
        low, high = np.percentile(image, TONAL_PERCENTILES)
        tonal_span = float(high) - float(low)

        boxes = find_objects(labels)
        budget = config.max_divided_area_share * float(areas.sum())
        remaining = set(candidates.tolist())
        divisions: list[dict[str, Any]] = []
        attempts = 0
        divided_area = 0.0
        walled, divided = image, labels

        while len(divisions) < requested:
            affordable = sorted(
                label for label in remaining
                if areas[label - 1] <= budget
            )
            if not affordable:
                break

            label = int(self.py_random.choice(affordable))
            remaining.discard(label)
            for _ in range(config.max_retries + 1):
                attempts += 1
                division = self._divide(
                    walled, divided, label, boxes[label - 1], tonal_span
                )
                if division is not None:
                    walled = division.pop("walled_image")
                    divided = division.pop("divided_labels")
                    budget -= float(areas[label - 1])
                    divided_area += float(areas[label - 1])
                    divisions.append(division)
                    break

        if not divisions:
            return _undivided(
                n_before,
                "no_wall_divided_the_pore_in_two",
                attempts=attempts,
                requested=requested,
                pool=int(candidates.size),
            )

        return _summarize(
            divisions, walled, divided, n_before, attempts, requested,
            int(candidates.size), divided_area, float(areas.sum()),
        )

    def _requested_count(self, pool_size: int) -> int:
        """How many walls this sample asks for.

        A share of the pool rather than a fixed number, so an image of
        eighty pores and one of twenty are treated alike relative to
        what each of them offers.
        """
        rate = self.py_random.uniform(*self._config.rate)
        return int(
            min(max(round(rate * pool_size), 1), self._config.count_cap)
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
        self, areas: np.ndarray, n_before: int
    ) -> np.ndarray:
        """The largest pores, the share of them drawn for this sample.

        A wall needs a pore with room on both sides of it. Dividing one
        from the small end of the distribution would produce two
        instances below anything an annotator drew, which is a worse
        error than not augmenting the sample at all.

        The pool is settled here, once, before any wall is drawn, and
        every wall of the sample is taken from it. Recomputing it after
        each division would let a fragment re-enter as a candidate and
        be cut again.
        """
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
        tonal_span: float,
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
            areas, thickness, sag, tonal_span,
        )

    def _drawn_contrast(self, tonal_span: float) -> float:
        """How far above its pore this wall's centre is painted.

        Drawn per wall rather than per sample, because the walls in one
        micrograph are not all equally bright and a sample whose walls
        all matched would be a sample no microscope produced.

        The floor is applied here rather than to the range, because it
        is a number of grey levels and the range is a share: the same
        share is a plain wall on a contrasty image and nothing at all
        on a flat one. A wall too faint to survive the model's resize
        would leave the sample claiming two pores where the picture
        shows one, and that is not a hard example but a wrong label.

        Parameters
        ----------
        tonal_span : float
            Width of this image's tonal range, in grey levels.

        Returns
        -------
        float
            Share of the tonal range, at least the floor's worth of it.
        """
        config = self._config
        contrast = self.py_random.uniform(*config.contrast)
        if tonal_span <= 0.0:
            return contrast
        return max(contrast, config.min_contrast_grey / tonal_span)

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
        tonal_span: float,
    ) -> dict[str, Any]:
        """Assemble the divided labels and the image with the wall.

        The second fragment takes the next free id, which keeps the
        numbering dense without a renumbering pass over the frame.
        Called repeatedly for one sample, each call sees the ids the
        previous one added, so the next free id is still next.
        """
        n_before = int(labels.max())
        divided = labels.copy()
        window = divided[box]
        window[inside] = 0
        window[fragments == 1] = label
        window[fragments == 2] = n_before + 1

        interior = float(np.median(image[box][inside]))
        contrast = self._drawn_contrast(tonal_span)
        target = interior + contrast * tonal_span
        walled = image.copy()
        patch = walled[box].astype(np.float32)
        blend = np.where(inside, weight, 0.0).astype(np.float32)
        walled[box] = to_source_dtype(
            patch * (1.0 - blend) + target * blend, image
        )

        return {
            "divided_instance": label,
            "thickness_px": round(thickness, 3),
            "contrast": round(contrast, 4),
            "sag": round(sag, 4),
            "fragment_ratio": round(
                float(areas.min()) / float(areas.sum()), 4
            ),
            "smallest_fragment_px2": int(areas.min()),
            "target_intensity": round(target, 2),
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
    n_before: int,
    fallback: str,
    attempts: int = 1,
    requested: int = 0,
    pool: int = 0,
) -> dict[str, Any]:
    """Parameters for a sample no wall could be drawn into."""
    return {
        "changed_mask": False,
        "divided_instances": (),
        "n_septa_requested": requested,
        "n_septa_drawn": 0,
        "candidate_pool": pool,
        "divided_area_share": 0.0,
        "thickness_px": (),
        "contrasts": (),
        "sags": (),
        "fragment_ratios": (),
        "smallest_fragment_px2": None,
        "target_intensities": (),
        "n_instances_before": n_before,
        "n_instances_after": n_before,
        "attempts": attempts,
        "fallback": fallback,
        "walled_image": None,
        "divided_labels": None,
    }


def _summarize(
    divisions: list[dict[str, Any]],
    walled: np.ndarray,
    divided: np.ndarray,
    n_before: int,
    attempts: int,
    requested: int,
    pool: int,
    divided_area_px2: float,
    annotated_area_px2: float,
) -> dict[str, Any]:
    """Fold the walls of one sample into a single record.

    A sample now carries several walls, and the record has to describe
    the set rather than the last of them. Two of these numbers are not
    diagnostics: the walls drawn against those asked for is what says
    whether the area cap or a run of failed draws bound this sample,
    and the divided area share is what a run reports when asked how far
    the family moved the size distribution it also measures.
    """
    return {
        "changed_mask": True,
        "divided_instances": tuple(
            entry["divided_instance"] for entry in divisions
        ),
        "n_septa_requested": requested,
        "n_septa_drawn": len(divisions),
        "candidate_pool": pool,
        "divided_area_share": round(
            divided_area_px2 / annotated_area_px2, 4
        ),
        "thickness_px": tuple(
            entry["thickness_px"] for entry in divisions
        ),
        "sags": tuple(entry["sag"] for entry in divisions),
        "fragment_ratios": tuple(
            entry["fragment_ratio"] for entry in divisions
        ),
        "smallest_fragment_px2": min(
            entry["smallest_fragment_px2"] for entry in divisions
        ),
        "contrasts": tuple(
            entry["contrast"] for entry in divisions
        ),
        "target_intensities": tuple(
            entry["target_intensity"] for entry in divisions
        ),
        "n_instances_before": n_before,
        "n_instances_after": int(divided.max()),
        "attempts": attempts,
        "fallback": None,
        "walled_image": walled,
        "divided_labels": divided,
    }
