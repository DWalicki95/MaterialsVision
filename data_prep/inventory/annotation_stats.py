"""
In-memory rasterization of Label Studio polygons and the per-image
instance statistics derived from them.

Reuses the repository's existing pore geometry code
(``PoreMorphologyMetrics``, ``PorousMaterialAnalyzer._filter_boundary_
pores``).
"""
import logging
from typing import Optional, Sequence

import numpy as np
from skimage.draw import polygon as sk_polygon
from skimage.measure import regionprops

from data_prep.inventory.models import InstanceStats
from materials_vision.quantitative_analysis.quantitative_analysis import (
    PoreMorphologyMetrics, PorousMaterialAnalyzer)

logger = logging.getLogger(__name__)


def rasterize_annotation(
    polygons: Sequence[np.ndarray], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Rasterize polygons (in export order) into a dense label image.

    Painting order follows the Label Studio export order, so at
    overlapping pixels the later polygon wins. That "last wins" rule
    is a local default, not a project-wide decision on how annotation
    overlaps should be resolved: it affects only the instance
    statistics computed in this module, never any training target.

    Parameters
    ----------
    polygons : Sequence[np.ndarray]
        Per-instance pixel-coordinate polygons, ``(x, y)`` columns
        (output of ``label_studio.polygon_to_pixels``).
    shape : tuple of int
        ``(height, width)`` of the target raster.

    Returns
    -------
    labels : np.ndarray
        ``int32`` array, shape ``shape``, 0 = background, dense
        positive IDs for every polygon that rasterized to at least one
        pixel.
    coverage : np.ndarray
        ``uint8`` array, shape ``shape``, count of polygons covering
        each pixel - used to measure (not resolve) overlap.
    n_degenerate_polygons : int
        Polygons with fewer than 3 points, or with zero pixels after
        rasterization (e.g. fully outside the frame after clipping).
    """
    height, width = shape
    labels: np.ndarray = np.zeros((height, width), dtype=np.int32)
    coverage: np.ndarray = np.zeros((height, width), dtype=np.uint8)

    next_id = 1
    n_degenerate = 0
    for poly in polygons:
        if poly.shape[0] < 3:
            n_degenerate += 1
            continue
        rr, cc = sk_polygon(
            poly[:, 1], poly[:, 0], shape=(height, width)
        )
        if rr.size == 0:
            n_degenerate += 1
            continue
        labels[rr, cc] = next_id
        coverage[rr, cc] += 1
        next_id += 1

    return labels, coverage, n_degenerate


def compute_instance_stats(
    labels: np.ndarray,
    coverage: np.ndarray,
    content_bbox: tuple[int, int, int, int],
    pixel_size_um: Optional[float],
    n_degenerate_polygons: int,
) -> InstanceStats:
    """Compute per-image instance statistics from a rasterized label
    image.

    Instance count and equivalent diameters always use the full,
    uncropped label image (an instance's real area must not be
    truncated just because part of it dips into the non-image band).
    Border detection reuses
    ``PorousMaterialAnalyzer._filter_boundary_pores`` directly on the
    full-image region properties with ``mask_shape`` set to the
    *content* dimensions (``tolerance=0``): since
    ``nonimage_region.detect_nonimage_region`` only ever trims rows
    from the bottom, ``content_bbox`` always originates at ``(0, 0)``,
    so an instance's true (uncropped) bounding box coordinates are
    already relative to that same origin - no separate cropped
    regionprops pass is needed, and none of an instance's real area is
    lost from the diameter statistics.

    Parameters
    ----------
    labels : np.ndarray
        Output of ``rasterize_annotation``.
    coverage : np.ndarray
        Output of ``rasterize_annotation``.
    content_bbox : tuple of int
        ``(x0, y0, x1, y1)`` of the usable image content
        (``NonImageRegion.content_bbox``); must originate at (0, 0).
    pixel_size_um : float, optional
        Unused here (diameters are always in px); accepted for API
        symmetry with the manifest's ``_um`` columns, which the caller
        derives by multiplying ``equivalent_diameter_px`` values.
    n_degenerate_polygons : int
        Passed through from ``rasterize_annotation``.

    Returns
    -------
    InstanceStats
    """
    del pixel_size_um  # px-only here; um columns derived by the caller

    x0, y0, x1, y1 = content_bbox
    if x0 != 0 or y0 != 0:
        raise ValueError(
            f"content_bbox must originate at (0, 0), got {content_bbox}"
        )
    content_shape = (y1, x1)

    props = regionprops(labels)
    n_instances = len(props)

    if n_instances == 0:
        return InstanceStats(
            n_instances=0,
            n_border_instances=0,
            n_degenerate_polygons=n_degenerate_polygons,
            overlap_px_fraction=0.0,
            equivalent_diameter_px=(0.0, 0.0, 0.0),
        )

    filtered = PorousMaterialAnalyzer._filter_boundary_pores(
        props, content_shape, tolerance=0
    )
    n_border_instances = n_instances - len(filtered)

    diameters = [
        PoreMorphologyMetrics(
            prop, pixel_size=1.0
        ).calculate_equivalent_diameter()["equivalent_diameter"]
        for prop in props
    ]
    equivalent_diameter_px = (
        float(np.min(diameters)),
        float(np.median(diameters)),
        float(np.max(diameters)),
    )

    n_covered = int((coverage > 0).sum())
    overlap_px_fraction = (
        float((coverage > 1).sum()) / n_covered if n_covered > 0 else 0.0
    )

    return InstanceStats(
        n_instances=n_instances,
        n_border_instances=n_border_instances,
        n_degenerate_polygons=n_degenerate_polygons,
        overlap_px_fraction=overlap_px_fraction,
        equivalent_diameter_px=equivalent_diameter_px,
    )
