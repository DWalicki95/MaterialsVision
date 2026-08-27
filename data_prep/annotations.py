"""
Shared access to the polygon annotations of a Label Studio export.

Two pipelines need the same thing from an export - the polygons of one
specific annotation, in pixel coordinates: the mask builder, which
rasterizes them, and the fragment-area calibration, which measures
their areas. Both go through here so they can never disagree about
which annotation belongs to an image or how a polygon maps to pixels.

Annotations are looked up by their own id rather than by re-running
the "which annotator wins" selection. The manifest already recorded
the winner for every image in ``mask_annotation_id``, and reproducing
that decision independently would risk the two answers drifting apart.
"""
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_prep.inventory.issues import IssueCollector
from data_prep.inventory.label_studio import (iter_polygon_results, load_tasks,
                                              polygon_to_pixels)

logger = logging.getLogger(__name__)


class AnnotationLookupError(RuntimeError):
    """Raised when the manifest references an annotation the export
    does not contain."""


def index_annotations_by_id(
    label_studio_json: Path,
) -> dict[int, Mapping[str, Any]]:
    """Index every annotation of an export by its annotation id.

    Parameters
    ----------
    label_studio_json : Path
        Label Studio JSON export.

    Returns
    -------
    dict of int to Mapping
        Annotation id to the annotation itself.
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    for task in load_tasks(label_studio_json):
        for annotation in task.get("annotations", []):
            annotation_id = annotation.get("id")
            if annotation_id is not None:
                by_id[int(annotation_id)] = annotation
    logger.debug(
        "Indexed %d annotation(s) from %s", len(by_id), label_studio_json
    )
    return by_id


def load_annotation_index(
    exports: Mapping[str, Path], series: Sequence[str]
) -> dict[int, Mapping[str, Any]]:
    """Index the exports of several series into one lookup table.

    Parameters
    ----------
    exports : Mapping[str, Path]
        Series name to its Label Studio export.
    series : Sequence of str
        Series to load; others are skipped.

    Returns
    -------
    dict of int to Mapping

    Raises
    ------
    AnnotationLookupError
        If a requested series has no configured export.
    """
    missing = sorted(set(series) - set(exports))
    if missing:
        raise AnnotationLookupError(
            f"No Label Studio export configured for series {missing}"
        )
    index: dict[int, Mapping[str, Any]] = {}
    for name in sorted(set(series)):
        index.update(index_annotations_by_id(exports[name]))
    return index


def polygons_in_pixels(
    annotation: Mapping[str, Any],
    width_px: int,
    height_px: int,
    *,
    collector: IssueCollector,
    image_ref: str,
) -> list[np.ndarray]:
    """Convert one annotation's polygons to pixel coordinates.

    Polygons are returned in export order, which is the order the
    rasterizer paints them in.

    Parameters
    ----------
    annotation : Mapping
        A Label Studio annotation.
    width_px, height_px : int
        Dimensions of the image the polygons belong to.
    collector : IssueCollector
        Receives warnings about skipped results and dimension
        mismatches.
    image_ref : str
        Image identifier, used in those warnings.

    Returns
    -------
    list of np.ndarray
        Each of shape ``(n_points, 2)`` with ``(x, y)`` columns.
    """
    return [
        polygon_to_pixels(
            result, width_px, height_px,
            collector=collector, image_ref=image_ref,
        )
        for result in iter_polygon_results(
            annotation, collector=collector, image_ref=image_ref
        )
    ]


def require_annotation(
    index: Mapping[int, Mapping[str, Any]],
    annotation_id: int,
    image_ref: str,
) -> Mapping[str, Any]:
    """Look up an annotation, failing loudly when it is absent.

    Parameters
    ----------
    index : Mapping[int, Mapping]
        Output of ``load_annotation_index``.
    annotation_id : int
        Value the manifest recorded in ``mask_annotation_id``.
    image_ref : str
        Image identifier, for the error message.

    Returns
    -------
    Mapping

    Raises
    ------
    AnnotationLookupError
        If the export does not contain that annotation, which means
        the manifest and the export have drifted apart.
    """
    annotation = index.get(int(annotation_id))
    if annotation is None:
        raise AnnotationLookupError(
            f"Annotation {annotation_id} (image {image_ref}) is not in "
            f"the Label Studio export: the manifest and the export are "
            f"out of sync"
        )
    return annotation
