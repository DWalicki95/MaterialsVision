"""
Turning annotation polygons into a single instance label image.

A label image encodes every annotated pore as a region of pixels
carrying one positive integer; background is 0. Two properties have to
be decided when polygons become pixels, and both are frozen here
because every training run and every metric depends on them.

**Overlapping annotations.** Annotators occasionally draw two pore
outlines that share a sliver of pixels along a common wall. A pixel
can only carry one instance, so a rule is needed. The rule is that the
polygon drawn later in the export wins. Measured on this dataset the
contested area is tiny - about 0.025% of all annotated pixels - so any
rule would give nearly the same masks; what matters is that the rule
is fixed and reproducible, since the annotation export is frozen and
its order therefore stable.

**Connectivity.** That same overwriting can pinch an instance in two:
a sliver taken out of a narrow neck leaves the pore in two separate
pieces sharing one label. On this dataset that happens to about 2.7%
of instances. It is not harmless, because the training targets are
built per instance - a pore in two pieces produces two separate
distance basins and teaches the model to split a pore the annotation
says is whole. So after painting, every instance is reduced to its
largest connected piece. The cost was measured before adopting the
rule: 7721 pixels across the whole dataset, 0.001% of all annotated
area, with a median loss per repaired instance of 0.01%. Two instances
out of 35846 lose more than a fifth of their area; none loses half.
"""
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.measure import label as connected_components

from data_prep.inventory.annotation_stats import rasterize_annotation

logger = logging.getLogger(__name__)

OVERLAP_RULE = "last_polygon_wins"

CONNECTIVITY_RULE = "largest_connected_piece_per_instance"

MASK_DTYPE = np.uint16


@dataclass(frozen=True)
class RasterizedMask:
    """One image's instance labels plus what was lost making them.

    Parameters
    ----------
    labels : np.ndarray
        ``uint16`` label image, background 0 and instances numbered
        densely from 1.
    n_polygons : int
        Polygons the annotation contained.
    n_instances : int
        Instances in the finished label image.
    n_vanished_polygons : int
        Polygons that left no pixel at all - either degenerate, or
        completely painted over by a later polygon.
    n_repaired_instances : int
        Instances that were left in more than one piece and reduced to
        their largest piece.
    n_pieces_removed : int
        Pieces discarded by that repair.
    discarded_px : int
        Pixels those discarded pieces held.
    overlap_px : int
        Pixels claimed by more than one polygon.
    covered_px : int
        Pixels claimed by at least one polygon.
    """

    labels: np.ndarray
    n_polygons: int
    n_instances: int
    n_vanished_polygons: int
    n_repaired_instances: int
    n_pieces_removed: int
    discarded_px: int
    overlap_px: int
    covered_px: int


def rasterize_instances(
    polygons: Sequence[np.ndarray], shape: tuple[int, int]
) -> RasterizedMask:
    """Rasterize polygons into a repaired, densely numbered label image.

    Parameters
    ----------
    polygons : Sequence of np.ndarray
        Pixel-coordinate polygons, ``(x, y)`` columns, in the order
        they should be painted.
    shape : tuple of int
        ``(height, width)`` of the target image.

    Returns
    -------
    RasterizedMask

    Raises
    ------
    ValueError
        If more instances survive than ``uint16`` can label.
    """
    painted, coverage, n_degenerate = rasterize_annotation(
        polygons, shape
    )
    repaired, repair_stats = _keep_largest_piece_per_instance(painted)

    n_instances = int(repaired.max())
    if n_instances > np.iinfo(MASK_DTYPE).max:
        raise ValueError(
            f"{n_instances} instances exceed the {MASK_DTYPE.__name__} "
            f"label range"
        )

    n_painted = int(np.unique(painted[painted > 0]).size)
    return RasterizedMask(
        labels=repaired.astype(MASK_DTYPE),
        n_polygons=len(polygons),
        n_instances=n_instances,
        n_vanished_polygons=len(polygons) - n_painted,
        n_repaired_instances=repair_stats["repaired"],
        n_pieces_removed=repair_stats["pieces_removed"],
        discarded_px=repair_stats["discarded_px"],
        overlap_px=int((coverage > 1).sum()),
        covered_px=int((coverage > 0).sum()),
    )


def _keep_largest_piece_per_instance(
    painted: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reduce every instance to one connected region and renumber.

    Connected components are computed in a single pass.
    ``skimage.measure.label`` joins pixels only when they carry the
    same value, so one call yields the pieces of every instance at
    once, without scanning the frame once per instance.

    Parameters
    ----------
    painted : np.ndarray
        Label image straight from painting the polygons.

    Returns
    -------
    labels : np.ndarray
        ``int32`` labels, densely numbered from 1.
    stats : dict of str to int
        ``repaired``, ``pieces_removed`` and ``discarded_px``.
    """
    stats = {"repaired": 0, "pieces_removed": 0, "discarded_px": 0}
    components = connected_components(
        painted, background=0, connectivity=1
    )
    n_components = int(components.max())
    if n_components == 0:
        return np.zeros(painted.shape, dtype=np.int32), stats

    areas = np.bincount(
        components.ravel(), minlength=n_components + 1
    )
    origin = np.zeros(n_components + 1, dtype=np.int64)
    origin[components.ravel()] = painted.ravel()

    new_id = np.zeros(n_components + 1, dtype=np.int32)
    next_id = 0
    for instance_id in np.unique(origin[1:]):
        if instance_id == 0:
            continue
        pieces = np.nonzero(origin == instance_id)[0]
        pieces = pieces[pieces > 0]
        largest = pieces[int(np.argmax(areas[pieces]))]
        if pieces.size > 1:
            stats["repaired"] += 1
            stats["pieces_removed"] += int(pieces.size - 1)
            stats["discarded_px"] += int(
                areas[pieces].sum() - areas[largest]
            )
        next_id += 1
        new_id[largest] = next_id

    return new_id[components], stats
