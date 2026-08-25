"""
Deterministic crop of an image/label pair to the content region, and
the instance bookkeeping it forces.

Some SEM images carry an information panel along the bottom edge - a
black strip with magnification, scale bar and acquisition settings,
written by the microscope rather than photographed. It contains no
material, so it is removed before the image reaches the model. On this
dataset the strip occupies the bottom 70 rows of every image from one
of the two microscopes; images from the other microscope had it
stripped at acquisition time and need no crop at all. Which box to cut
to is decided per image and stored in the manifest column
``load_crop_bbox``.

Removing it is not a matter of slicing an array. Annotators drew on
the full frame, so pores near the bottom edge have outlines running
into the strip: on this dataset that is true of every single image
from the affected microscope, ~400 instances in total. The crop
therefore cuts through real annotations, and the remains need a rule.

The rule implemented here:

- an annotation crossing the crop edge is **cut along that edge, not
  discarded** - the part inside the content region is still a valid,
  if truncated, pore;
- what remains survives only if its area is at least
  ``A_min_fragment``, a threshold measured once as a low percentile of
  the real annotated-instance areas. The point is to never manufacture
  a label smaller than anything an annotator actually drew: a sliver
  of a pore left by the cut is not a pore;
- **the threshold applies only to instances the crop actually
  reduced.** An instance lying wholly inside the content region is a
  genuine annotation and survives at any size. Filtering those by the
  same threshold would delete the bottom percentile of the ground
  truth on every image, because that is precisely what the threshold
  was measured as;
- instances touching any edge of the cropped frame are flagged
  ``border_instance``. Their shape is truncated by the frame, so
  morphological measurements on them are meaningless and downstream
  code excludes them - on the ground-truth and the prediction side
  alike, or the comparison would be biased;
- IDs are renumbered densely at the end, so consumers can assume
  labels ``1..n_instances`` with no gaps.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.measure import label as connected_components

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CroppedSample:
    """An image/label pair cropped to the content region.

    Parameters
    ----------
    image : np.ndarray
        Cropped image, a fresh array; the caller's data is never
        modified in place.
    labels : np.ndarray
        Cropped instance labels, densely renumbered ``1..n_instances``
        with 0 as background.
    original_ids : np.ndarray
        Label each instance carried before renumbering, indexed by
        ``new_id - 1``. Keeps a surviving fragment traceable to the
        annotation it came from.
    border_instance : np.ndarray
        Boolean per instance, indexed by ``new_id - 1``: whether it
        touches any edge of the cropped frame, including the new edge
        the crop introduced.
    n_input_instances : int
        Instances present in the labels before cropping.
    n_cut_by_crop : int
        Instances the crop reduced in area. Only these face the
        minimum-area test; an instance lying wholly inside the content
        region survives at any size.
    n_dropped_outside : int
        Instances that had no pixel inside the crop at all.
    n_dropped_below_min_area : int
        Cut instances whose surviving fragment was too small to keep.
    n_dropped_disconnected : int
        Pieces discarded because the crop left a cut instance in more
        than one fragment and only the largest is kept. Promoting the
        smaller pieces to their own IDs would invent instances no
        annotator drew.
    """

    image: np.ndarray
    labels: np.ndarray
    original_ids: np.ndarray
    border_instance: np.ndarray
    n_input_instances: int
    n_cut_by_crop: int
    n_dropped_outside: int
    n_dropped_below_min_area: int
    n_dropped_disconnected: int

    @property
    def n_instances(self) -> int:
        """Instances surviving the crop.

        Returns
        -------
        int
        """
        return int(self.original_ids.size)

    @property
    def n_border_instances(self) -> int:
        """Surviving instances touching an edge of the crop.

        Returns
        -------
        int
        """
        return int(self.border_instance.sum())


def apply_content_crop(
    image: np.ndarray,
    labels: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_fragment_area_px2: float,
) -> CroppedSample:
    """Crop an image and its instance labels to the content region.

    Image and labels are cropped with the identical box, so the two
    can never drift apart.

    Parameters
    ----------
    image : np.ndarray
        Source image, ``(H, W)`` or ``(H, W, C)``.
    labels : np.ndarray
        Instance label image, ``(H, W)``, 0 = background.
    bbox : tuple of int
        ``(x0, y0, x1, y1)`` content region, as stored in the
        manifest's ``load_crop_bbox``; ``x1``/``y1`` are exclusive.
    min_fragment_area_px2 : float
        Smallest area, in source pixels squared, a cut instance may
        keep and still count as an instance. Calibrated once as a low
        percentile of the real annotated-instance areas, so the crop
        never produces a label smaller than anything an annotator
        drew. Instances the crop did not touch ignore it.

    Returns
    -------
    CroppedSample

    Raises
    ------
    ValueError
        If the shapes disagree, the box is degenerate, or the box
        falls outside the image.
    """
    _validate(image, labels, bbox, min_fragment_area_px2)
    x0, y0, x1, y1 = bbox

    cropped_image = np.array(image[y0:y1, x0:x1], copy=True)
    cropped_labels = labels[y0:y1, x0:x1]
    full_areas = np.bincount(labels.ravel())

    return _rebuild_instances(
        cropped_image, cropped_labels, full_areas, min_fragment_area_px2
    )


def _validate(
    image: np.ndarray,
    labels: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_fragment_area_px2: float,
) -> None:
    """Reject inputs the crop cannot be applied to."""
    if labels.ndim != 2:
        raise ValueError(
            f"labels must be 2-D, got shape {labels.shape}"
        )
    if image.shape[:2] != labels.shape:
        raise ValueError(
            f"image {image.shape[:2]} and labels {labels.shape} "
            f"describe different frames"
        )
    if min_fragment_area_px2 < 0:
        raise ValueError(
            f"min_fragment_area_px2 must be >= 0, got "
            f"{min_fragment_area_px2}"
        )

    x0, y0, x1, y1 = bbox
    height, width = labels.shape
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"bbox {bbox} falls outside a {width}x{height} frame"
        )
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox {bbox} is degenerate")


def _rebuild_instances(
    cropped_image: np.ndarray,
    cropped_labels: np.ndarray,
    full_areas: np.ndarray,
    min_fragment_area_px2: float,
) -> CroppedSample:
    """Resolve fragments, drop the too-small, and renumber densely.

    An instance counts as *cut* when its area inside the crop is less
    than its area in the full frame. Only cut instances face the
    ``min_fragment_area_px2`` test: the threshold exists to suppress
    slivers the cut manufactured, and an intact instance is a real
    annotation no matter how small. Applying it to intact instances
    would delete the bottom percentile of the ground truth on every
    image, since the threshold is measured as a low percentile of
    exactly that distribution.

    The same scoping applies to connectivity. Rasterizing overlapping
    outlines resolves each pixel to a single instance, which can leave
    an instance in two pieces before any cropping happens; such an
    instance keeps all its pixels under one ID. Only when the *cut*
    breaks an instance apart is the largest piece kept and the rest
    dropped - promoting them to their own IDs would invent instances
    nobody annotated.

    Connected components are computed once over the whole cropped
    label image. ``skimage.measure.label`` joins pixels only when they
    share a value, so a single pass yields the pieces of every
    instance without re-scanning the frame per instance.
    """
    n_input_instances = int(np.count_nonzero(full_areas[1:]))
    components = connected_components(
        cropped_labels, background=0, connectivity=1
    )
    n_components = int(components.max())
    if n_components == 0:
        return _empty_sample(cropped_image, n_input_instances)

    areas = np.bincount(
        components.ravel(), minlength=n_components + 1
    )
    origin = np.zeros(n_components + 1, dtype=np.int64)
    origin[components.ravel()] = cropped_labels.ravel()

    group = np.zeros(n_components + 1, dtype=np.int32)
    surviving_ids: list[int] = []
    n_cut = 0
    n_below_min = 0
    n_fragments_discarded = 0

    for original_id in np.unique(origin[1:]):
        if original_id == 0:
            continue
        pieces = np.nonzero(origin == original_id)[0]
        pieces = pieces[pieces > 0]

        if areas[pieces].sum() == full_areas[original_id]:
            # Untouched by the crop: keep every pixel, including the
            # pieces an overlapping neighbour may have split it into.
            kept_pieces = pieces
        else:
            n_cut += 1
            largest = pieces[int(np.argmax(areas[pieces]))]
            n_fragments_discarded += int(pieces.size - 1)
            if areas[largest] < min_fragment_area_px2:
                n_below_min += 1
                continue
            kept_pieces = np.array([largest])

        surviving_ids.append(int(original_id))
        group[kept_pieces] = len(surviving_ids)

    if n_fragments_discarded:
        logger.debug(
            "Crop left %d instance(s) in more than one piece; kept the "
            "largest piece of each.", n_fragments_discarded,
        )

    relabelled = group[components]
    n_survivors = len(surviving_ids)
    return CroppedSample(
        image=cropped_image,
        labels=relabelled,
        original_ids=np.array(surviving_ids, dtype=np.int64),
        border_instance=_border_flags(relabelled, n_survivors),
        n_input_instances=n_input_instances,
        n_cut_by_crop=n_cut,
        n_dropped_outside=(
            n_input_instances - n_survivors - n_below_min
        ),
        n_dropped_below_min_area=n_below_min,
        n_dropped_disconnected=n_fragments_discarded,
    )


def _empty_sample(
    cropped_image: np.ndarray, n_input_instances: int
) -> CroppedSample:
    """Result for a crop that contains no instance at all."""
    height, width = cropped_image.shape[:2]
    return CroppedSample(
        image=cropped_image,
        labels=np.zeros((height, width), dtype=np.int32),
        original_ids=np.empty(0, dtype=np.int64),
        border_instance=np.empty(0, dtype=bool),
        n_input_instances=n_input_instances,
        n_cut_by_crop=0,
        n_dropped_outside=n_input_instances,
        n_dropped_below_min_area=0,
        n_dropped_disconnected=0,
    )


def _border_flags(
    relabelled: np.ndarray, n_instances: int
) -> np.ndarray:
    """Flag instances touching any edge of the cropped frame.

    The bottom edge introduced by the panel crop counts like any
    other: an instance that was cut there is as unsuitable for
    morphological metrics as one running off the original frame.

    Parameters
    ----------
    relabelled : np.ndarray
        Densely renumbered label image.
    n_instances : int

    Returns
    -------
    np.ndarray
        Boolean array of length ``n_instances``, indexed by
        ``new_id - 1``.
    """
    flags = np.zeros(n_instances + 1, dtype=bool)
    if n_instances == 0:
        return flags[1:]
    edge_ids = np.unique(
        np.concatenate([
            relabelled[0, :], relabelled[-1, :],
            relabelled[:, 0], relabelled[:, -1],
        ])
    )
    flags[edge_ids[edge_ids > 0]] = True
    return flags[1:]


def parse_crop_bbox(value: str) -> Optional[tuple[int, int, int, int]]:
    """Parse a manifest ``"x0,y0,x1,y1"`` crop box.

    Parameters
    ----------
    value : str

    Returns
    -------
    tuple of int or None
        ``None`` when the manifest stored no box for the row.

    Raises
    ------
    ValueError
        If the string is present but not four integers.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Malformed crop bbox {value!r}: expected 4 integers"
        )
    x0, y0, x1, y1 = (int(part) for part in parts)
    return x0, y0, x1, y1
