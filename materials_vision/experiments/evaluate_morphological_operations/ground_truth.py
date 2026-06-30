"""Ground-truth parsing from Label Studio JSON exports.

Each Label Studio task holds polygon annotations (one polygon per pore)
expressed in percentage coordinates. This module extracts those polygons
and rasterizes them into instance masks.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage.draw import polygon as draw_polygon

logger = logging.getLogger(__name__)

# Label Studio prefixes file_upload with an upload hash, e.g. "12fe8df0-".
_HASH_PREFIX = re.compile(r"^[0-9a-f]+-", re.IGNORECASE)


def canonical_stem_from_file_upload(file_upload: str) -> str:
    """
    Derive the canonical image stem from a Label Studio file_upload name.

    Parameters
    ----------
    file_upload : str
        Raw value, e.g. ``"12fe8df0-AS5_40_13_..._image.jpg"``.

    Returns
    -------
    str
        Stem without the upload hash prefix and without extension, e.g.
        ``"AS5_40_13_..._image"``.
    """
    without_prefix = _HASH_PREFIX.sub("", file_upload)
    return Path(without_prefix).stem


def load_label_studio_tasks(json_path: Path) -> List[dict]:
    """
    Load Label Studio tasks from a JSON export.

    Parameters
    ----------
    json_path : Path
        Path to the JSON export file.

    Returns
    -------
    List[dict]
        List of task dictionaries.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    logger.info("Loaded %d tasks from %s", len(tasks), json_path)
    return tasks


def _first_annotation_results(task: dict) -> List[dict]:
    """Return the result list of the first annotation, or an empty list."""
    annotations = task.get("annotations") or []
    if not annotations:
        return []
    return annotations[0].get("result") or []


def extract_polygons(task: dict, label_name: str) -> List[np.ndarray]:
    """
    Extract polygon point arrays for a given label from a task.

    Parameters
    ----------
    task : dict
        A single Label Studio task.
    label_name : str
        Polygon label to keep (e.g. ``"Por"``).

    Returns
    -------
    List[np.ndarray]
        One array of shape ``(n_points, 2)`` per polygon, in percentage
        coordinates ``(x, y)``.
    """
    polygons = []
    for item in _first_annotation_results(task):
        if item.get("type") != "polygonlabels":
            continue
        value = item.get("value", {})
        if label_name not in value.get("polygonlabels", []):
            continue
        points = value.get("points")
        if not points:
            continue
        polygons.append(np.asarray(points, dtype=np.float64))
    if not polygons:
        logger.warning("Task %s has no '%s' polygons", task.get("id"),
                       label_name)
    return polygons


def percent_points_to_pixels(
    points: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    Convert percentage polygon points to pixel coordinates.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n_points, 2)`` with ``(x, y)`` in percent.
    width : int
        Target image width in pixels.
    height : int
        Target image height in pixels.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 2)`` with pixel ``(x, y)``.
    """
    pixels = np.empty_like(points)
    pixels[:, 0] = points[:, 0] / 100.0 * width
    pixels[:, 1] = points[:, 1] / 100.0 * height
    return pixels


def rasterize_instances(
    polygons_px: List[np.ndarray], shape: Tuple[int, int]
) -> np.ndarray:
    """
    Rasterize polygons into a labeled instance mask.

    Parameters
    ----------
    polygons_px : List[np.ndarray]
        Polygons in pixel coordinates ``(x, y)``.
    shape : Tuple[int, int]
        Target mask shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        ``uint16`` instance mask, background is 0, each polygon gets a
        unique label starting from 1.
    """
    mask = np.zeros(shape, dtype=np.uint16)
    for label, polygon_px in enumerate(polygons_px, start=1):
        rows, cols = draw_polygon(
            polygon_px[:, 1], polygon_px[:, 0], shape=shape
        )
        mask[rows, cols] = label
    return mask


def build_ground_truth_index(
    tasks: List[dict], label_name: str
) -> Dict[str, dict]:
    """
    Index tasks by canonical image stem for later lookup.

    Parameters
    ----------
    tasks : List[dict]
        Label Studio tasks.
    label_name : str
        Polygon label to keep.

    Returns
    -------
    Dict[str, dict]
        Mapping ``stem -> {"polygons_pct", "width", "height"}``. Width and
        height are taken from the first annotation result when available.
    """
    index = {}
    for task in tasks:
        file_upload = task.get("file_upload")
        if not file_upload:
            logger.error("Task %s has no file_upload", task.get("id"))
            continue
        stem = canonical_stem_from_file_upload(file_upload)
        polygons = extract_polygons(task, label_name)
        width, height = _read_original_size(task)
        index[stem] = {
            "polygons_pct": polygons,
            "width": width,
            "height": height,
        }
    logger.info("Indexed %d ground-truth samples", len(index))
    return index


def _read_original_size(task: dict) -> Tuple[int, int]:
    """Return (width, height) from the first result, or (0, 0)."""
    results = _first_annotation_results(task)
    if not results:
        return 0, 0
    first = results[0]
    return int(first.get("original_width", 0)), int(
        first.get("original_height", 0)
    )
