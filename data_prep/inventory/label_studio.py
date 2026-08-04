"""Label Studio export adapter: task loading, annotation selection, and
polygon coordinate conversion.

The Label Studio JSON export is the authoritative image registry (not
the images directory listing, see the inventory plan section 4.5). This
module never touches the filesystem beyond reading the export itself.

Note on collector parameters: as with ``series_profiles.parse``, the
plan's pipeline sketch omits the ``IssueCollector`` argument from
``select_annotation``/``polygon_to_pixels``/``iter_polygon_results`` for
brevity, but reporting "multiple annotations", "ground truth present",
"empty annotation", "unexpected result type" and "LS/file dimension
mismatch" (all required by the error taxonomy) needs it, so it is a
required keyword argument here.
"""
import logging
from typing import Any, Iterator, Mapping

import numpy as np

from data_prep.inventory.issues import (AnnotationSelectionError,
                                        IssueCollector, IssueLevel,
                                        PolygonConversionError)
from data_prep.inventory.models import AnnotationSelection

logger = logging.getLogger(__name__)

_SELECTION_RULE = "latest_updated_at_then_max_id"


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
    task: Mapping[str, Any], *, collector: IssueCollector
) -> AnnotationSelection:
    """Select the mask-producing annotation for one Label Studio task.

    Rule (``latest_updated_at_then_max_id``): drop cancelled
    annotations, sort the remainder by ``(updated_at, id)`` ascending,
    take the last. ``ground_truth`` never affects the choice; a
    positive value is only reported for awareness.

    Parameters
    ----------
    task : Mapping
        One Label Studio task dictionary.
    collector : IssueCollector
        Records ``multiple_annotations``, ``ground_truth_present`` and
        ``empty_annotation`` (all INFO).

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

    if len(candidates) > 1:
        collector.add(
            IssueLevel.INFO,
            "multiple_annotations",
            image_ref,
            f"{len(candidates)} annotations, selected "
            f"id={selected['id']} (rule={_SELECTION_RULE})",
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
        selection_rule=_SELECTION_RULE,
        annotation=selected,
    )


def iter_polygon_results(
    annotation: Mapping[str, Any],
    *,
    collector: IssueCollector,
    image_ref: str,
) -> Iterator[dict]:
    """Yield only ``polygonlabels`` results from an annotation.

    Parameters
    ----------
    annotation : Mapping
        Selected annotation dict (``AnnotationSelection.annotation``).
    collector : IssueCollector
        Records ``unexpected_result_type`` (WARNING) for skipped
        results.
    image_ref : str
        Image identifier, for the issue record.

    Yields
    ------
    dict
        Each ``polygonlabels`` result.
    """
    for result in annotation.get("result", []):
        if result.get("type") != "polygonlabels":
            collector.add(
                IssueLevel.WARNING,
                "unexpected_result_type",
                image_ref,
                f"type={result.get('type')!r}, id={result.get('id')!r}",
            )
            continue
        yield result


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
