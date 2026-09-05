"""Label Studio export adapter: task loading, annotation selection, and
polygon coordinate conversion.

The Label Studio JSON export is the authoritative image registry, not
the images directory listing: an image on disk that nobody annotated
is not part of the dataset, and the export is the only record of what
was actually annotated. This module never touches the filesystem
beyond reading the export itself.

**Two annotated classes, one of which is not a pore.** The annotators
outlined two kinds of object: ``Por``, a pore, and ``Wezel``, a node -
the solid junction where several struts of the foam meet. They are
different structures and only pores are the segmentation target, so
node polygons are dropped here, at the single point every consumer of
the export reads polygons through. Filtering by class is the only
reliable way to separate them: their sizes overlap heavily (median
equivalent diameter about 81 px for nodes against 118 px for pores),
so no threshold on geometry would divide them.

Note on collector parameters: ``select_annotation``,
``polygon_to_pixels`` and ``iter_polygon_results`` all take an
``IssueCollector`` as a required keyword argument. They have to -
each can encounter a condition that must be reported rather than
silently resolved ("multiple annotations", "ground truth present",
"empty annotation", "unexpected result type", "LS/file dimension
mismatch", "annotator fallback", "polygons excluded by class"), and
nothing may vanish from the run's record.
"""
import logging
from typing import Any, Iterator, Mapping

import numpy as np

from data_prep.inventory.issues import (AnnotationSelectionError,
                                        IssueCollector, IssueLevel,
                                        PolygonConversionError,
                                        PolygonLabelError)
from data_prep.inventory.models import AnnotationSelection

logger = logging.getLogger(__name__)

_SELECTION_RULE = "latest_updated_at_then_max_id"
_SELECTION_RULE_WITH_FALLBACK = _SELECTION_RULE + "+annotator_fallback"

PORE_LABEL = "Por"

NODE_LABEL = "Wezel"

KNOWN_POLYGON_LABELS = frozenset({PORE_LABEL, NODE_LABEL})

MASK_POLYGON_LABELS = frozenset({PORE_LABEL})

CLASS_FILTER_RULE = f"keep_{PORE_LABEL}_drop_{NODE_LABEL}"


def load_tasks(json_path) -> list[dict]:
    """Load a Label Studio export as a list of task dictionaries.

    Parameters
    ----------
    json_path : Path
        Path to the Label Studio JSON export.

    Returns
    -------
    list of dict
        One entry per Label Studio task, in file order.
    """
    import json

    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(
            f"Label Studio export must be a JSON list of tasks: "
            f"{json_path}"
        )
    return tasks


def select_annotation(
    task: Mapping[str, Any],
    *,
    collector: IssueCollector,
    avoid_annotators: tuple[int, ...] = (),
) -> AnnotationSelection:
    """Select the mask-producing annotation for one Label Studio task.

    Rule (``latest_updated_at_then_max_id``): drop cancelled
    annotations, sort the remainder by ``(updated_at, id)`` ascending,
    take the last. ``ground_truth`` never affects the choice; a
    positive value is only reported for awareness.

    If ``avoid_annotators`` is non-empty and the selected annotation's
    ``completed_by`` is in it, the selection falls back to the latest
    annotation (same rule) among the candidates whose ``completed_by``
    is not in ``avoid_annotators``. If no such candidate exists, the
    original (avoided) selection is kept.

    Parameters
    ----------
    task : Mapping
        One Label Studio task dictionary.
    collector : IssueCollector
        Records ``multiple_annotations``, ``ground_truth_present``,
        ``empty_annotation``, ``annotator_fallback_applied`` and
        ``annotator_fallback_unavailable`` (all INFO).
    avoid_annotators : tuple of int, optional
        ``completed_by`` IDs to fall back away from, per source
        (``SourceConfig.avoid_annotators``). Empty by default, which
        reproduces the unconditional latest-wins behaviour.

    Returns
    -------
    AnnotationSelection

    Raises
    ------
    AnnotationSelectionError
        If the task has no non-cancelled annotation to select from.
    """
    image_ref = _task_image_ref(task)
    candidates = [
        a for a in task.get("annotations", [])
        if not a.get("was_cancelled", False)
    ]
    if not candidates:
        raise AnnotationSelectionError(
            f"Task {task.get('id')} ({image_ref}) has no usable "
            f"(non-cancelled) annotation"
        )

    candidates_sorted = sorted(
        candidates, key=lambda a: (a["updated_at"], a["id"])
    )
    selected = candidates_sorted[-1]
    selection_rule = _SELECTION_RULE

    if avoid_annotators and selected["completed_by"] in avoid_annotators:
        fallback_candidates = [
            a for a in candidates_sorted
            if a["completed_by"] not in avoid_annotators
        ]
        if fallback_candidates:
            fallback_selected = fallback_candidates[-1]
            collector.add(
                IssueLevel.INFO,
                "annotator_fallback_applied",
                image_ref,
                f"latest annotation id={selected['id']} by annotator "
                f"{selected['completed_by']} avoided, selected "
                f"id={fallback_selected['id']} by annotator "
                f"{fallback_selected['completed_by']} instead",
            )
            selected = fallback_selected
            selection_rule = _SELECTION_RULE_WITH_FALLBACK
        else:
            collector.add(
                IssueLevel.INFO,
                "annotator_fallback_unavailable",
                image_ref,
                f"latest annotation id={selected['id']} by annotator "
                f"{selected['completed_by']} is avoided, but no "
                f"annotation from another annotator exists; kept",
            )

    if len(candidates) > 1:
        collector.add(
            IssueLevel.INFO,
            "multiple_annotations",
            image_ref,
            f"{len(candidates)} annotations, selected "
            f"id={selected['id']} (rule={selection_rule})",
        )
    if any(a.get("ground_truth") for a in candidates):
        collector.add(
            IssueLevel.INFO,
            "ground_truth_present",
            image_ref,
            "ground_truth flag present but ignored by selection rule",
        )
    if not selected.get("result"):
        collector.add(
            IssueLevel.INFO,
            "empty_annotation",
            image_ref,
            f"selected annotation id={selected['id']} has no results",
        )

    annotators = tuple(sorted({a["completed_by"] for a in candidates}))
    return AnnotationSelection(
        n_annotations=len(candidates),
        annotators=annotators,
        mask_annotator=selected["completed_by"],
        mask_annotation_id=selected["id"],
        annotation_completed_at=selected["updated_at"],
        selection_rule=selection_rule,
        annotation=selected,
    )


def polygon_label(result: Mapping[str, Any], image_ref: str) -> str:
    """Return the annotation class of one ``polygonlabels`` result.

    Parameters
    ----------
    result : Mapping
        A single ``polygonlabels`` result.
    image_ref : str
        Image identifier, for the error message.

    Returns
    -------
    str
        One of ``KNOWN_POLYGON_LABELS``.

    Raises
    ------
    PolygonLabelError
        If the result carries no class, more than one, or a class this
        pipeline has never seen. None of the three can be resolved
        here: a new class in a later export is a decision about what
        the masks contain, and must be made deliberately rather than
        by whichever branch this function happens to take.
    """
    names = result.get("value", {}).get("polygonlabels") or []
    if len(names) != 1:
        raise PolygonLabelError(
            f"Polygon {result.get('id')!r} of {image_ref} carries "
            f"{len(names)} class(es) {names}; exactly one is expected"
        )
    name = str(names[0])
    if name not in KNOWN_POLYGON_LABELS:
        raise PolygonLabelError(
            f"Polygon {result.get('id')!r} of {image_ref} carries the "
            f"unknown class {name!r}; the known classes are "
            f"{sorted(KNOWN_POLYGON_LABELS)}"
        )
    return name


def iter_polygon_results(
    annotation: Mapping[str, Any],
    *,
    collector: IssueCollector,
    image_ref: str,
) -> Iterator[dict]:
    """Return the mask-producing ``polygonlabels`` results.

    Results are filtered twice: to ``polygonlabels`` (the only result
    type these projects produce), and to the classes that become mask
    instances - pores, never nodes.

    The pass is eager rather than lazy so that the count of excluded
    polygons is recorded when this is called, not whenever a caller
    happens to exhaust the iterator.

    Parameters
    ----------
    annotation : Mapping
        Selected annotation dict (``AnnotationSelection.annotation``).
    collector : IssueCollector
        Records ``unexpected_result_type`` (WARNING) for skipped
        results and ``polygons_excluded_by_class`` (INFO) for the
        node polygons this image contributed.
    image_ref : str
        Image identifier, for the issue record.

    Returns
    -------
    Iterator of dict
        Each ``polygonlabels`` result whose class is in
        ``MASK_POLYGON_LABELS``, in export order.

    Raises
    ------
    PolygonLabelError
        Propagated from ``polygon_label``.
    """
    kept: list[dict] = []
    n_excluded = 0
    for result in annotation.get("result", []):
        if result.get("type") != "polygonlabels":
            collector.add(
                IssueLevel.WARNING,
                "unexpected_result_type",
                image_ref,
                f"type={result.get('type')!r}, id={result.get('id')!r}",
            )
            continue
        if polygon_label(result, image_ref) not in MASK_POLYGON_LABELS:
            n_excluded += 1
            continue
        kept.append(result)

    if n_excluded:
        collector.add(
            IssueLevel.INFO,
            "polygons_excluded_by_class",
            image_ref,
            f"{n_excluded} of {n_excluded + len(kept)} polygon(s) "
            f"dropped (rule={CLASS_FILTER_RULE})",
        )
    return iter(kept)


def count_excluded_polygons(annotation: Mapping[str, Any]) -> int:
    """Count the polygons the class filter drops from one annotation.

    The frozen artifacts record this per image, so that a mask holding
    fewer instances than the export holds polygons is explained by the
    artifact itself rather than only by re-reading the export.

    Parameters
    ----------
    annotation : Mapping
        A Label Studio annotation.

    Returns
    -------
    int
        ``polygonlabels`` results whose class is not in
        ``MASK_POLYGON_LABELS``. Results whose class cannot be read
        are not counted here; ``iter_polygon_results`` reports them.
    """
    n_excluded = 0
    for result in annotation.get("result", []):
        if result.get("type") != "polygonlabels":
            continue
        names = result.get("value", {}).get("polygonlabels") or []
        if len(names) == 1 and names[0] not in MASK_POLYGON_LABELS:
            n_excluded += 1
    return n_excluded


def polygon_to_pixels(
    result: Mapping[str, Any],
    width_px: int,
    height_px: int,
    *,
    collector: IssueCollector,
    image_ref: str,
) -> np.ndarray:
    """Convert one Label Studio polygon result to pixel coordinates.

    Points are stored as percentages of ``original_width``/
    ``original_height`` (per result, not per task). Conversion always
    uses those reference dimensions, never the actual file dimensions;
    a mismatch between the two is reported, not silently corrected by
    rescaling, and the result is then clipped to the real image bounds
    (points can fall slightly outside the frame in the raw export).

    Parameters
    ----------
    result : Mapping
        A single ``polygonlabels`` result (see ``iter_polygon_results``).
    width_px, height_px : int
        Actual image dimensions, used only to clip and to detect
        mismatches.
    collector : IssueCollector
        Records ``ls_dims_mismatch`` (WARNING).
    image_ref : str
        Image identifier, for the issue record.

    Returns
    -------
    np.ndarray
        Shape ``(n_points, 2)``, columns ``(x, y)``, float pixel
        coordinates clipped to ``[0, width_px - 1] x [0, height_px - 1]``.

    Raises
    ------
    PolygonConversionError
        If ``points``, ``original_width`` or ``original_height`` are
        missing.
    """
    value = result.get("value", {})
    points = value.get("points")
    orig_w = result.get("original_width")
    orig_h = result.get("original_height")
    if not points or orig_w is None or orig_h is None:
        raise PolygonConversionError(
            f"Result missing points/original_width/original_height "
            f"(image: {image_ref}, result id: {result.get('id')})"
        )

    if orig_w != width_px or orig_h != height_px:
        collector.add(
            IssueLevel.WARNING,
            "ls_dims_mismatch",
            image_ref,
            f"original_width/height={orig_w}x{orig_h} vs "
            f"file={width_px}x{height_px}",
        )

    pts = np.asarray(points, dtype=float)
    xs = np.clip(pts[:, 0] / 100.0 * orig_w, 0, width_px - 1)
    ys = np.clip(pts[:, 1] / 100.0 * orig_h, 0, height_px - 1)
    return np.stack([xs, ys], axis=1)


def _task_image_ref(task: Mapping[str, Any]) -> str:
    """Best-effort human-readable reference for a task, for logging."""
    image = task.get("data", {}).get("image")
    if image:
        return str(image).rsplit("/", 1)[-1]
    return f"task_id={task.get('id')}"
