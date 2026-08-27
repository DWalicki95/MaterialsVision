"""
Calibration of ``A_min_fragment`` on TRAIN.

``A_min_fragment`` is a low percentile (P1) of the ground-truth
instance area distribution. Training crops cut through pores at their
edges, and the resulting fragment is kept as a label only when it is
at least this large - the principle being that we never manufacture a
label smaller than anything an annotator actually drew. A low
percentile is what makes that threshold defensible: it sits just under
the smallest real annotations.

The manifest cannot answer this - it stores per-image minimum, median
and maximum equivalent diameters, not per-instance areas - so the
areas are recomputed here from the Label Studio polygons, reusing the
inventory's own rasterizer.

Two properties of this measurement matter:

- it runs on TRAIN only. The value feeds a training-time transform,
  and measuring it on VALIDATION or TEST would let information from
  the evaluation sets influence how the model is trained;
- it is measured *after* the deterministic crop to ``load_crop_bbox``,
  the box that removes the microscope's information panel from the
  bottom of the frame. That is the geometry instances actually have by
  the time any training crop sees them. For images from the microscope
  that leaves the panel in the file this is not a detail: every one of
  those 108 images has at least one annotated pore reaching into the
  panel area, so their areas before and after the crop differ.
"""
import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from data_prep.annotations import (AnnotationLookupError,
                                   index_annotations_by_id, polygons_in_pixels,
                                   require_annotation)
from data_prep.inventory.annotation_stats import rasterize_annotation
from data_prep.inventory.config import load_config
from data_prep.inventory.issues import IssueCollector, PolygonConversionError
from data_prep.split.models import FragmentAreaResult, MinFragmentAreaConfig

logger = logging.getLogger(__name__)


class FragmentAreaError(RuntimeError):
    """Raised when the TRAIN instance areas cannot be measured."""


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    """Parse a manifest ``"x0,y0,x1,y1"`` bbox string.

    Parameters
    ----------
    value : str

    Returns
    -------
    tuple of int
        ``(x0, y0, x1, y1)``.

    Raises
    ------
    FragmentAreaError
        If the string does not hold exactly four integers.
    """
    try:
        parts = [int(p) for p in str(value).split(",")]
    except ValueError as e:
        raise FragmentAreaError(
            f"Malformed bbox {value!r}: {e}"
        ) from e
    if len(parts) != 4:
        raise FragmentAreaError(
            f"Malformed bbox {value!r}: expected 4 integers, got "
            f"{len(parts)}"
        )
    return parts[0], parts[1], parts[2], parts[3]


def _instance_areas_px2(
    annotation: Mapping,
    width_px: int,
    height_px: int,
    crop_bbox: tuple[int, int, int, int],
    collector: IssueCollector,
    image_ref: str,
) -> tuple[np.ndarray, int]:
    """Rasterize one annotation and measure post-crop instance areas.

    Parameters
    ----------
    annotation : Mapping
        Label Studio annotation dict.
    width_px, height_px : int
        Source image dimensions.
    crop_bbox : tuple of int
        ``(x0, y0, x1, y1)`` deterministic content crop.
    collector : IssueCollector
    image_ref : str

    Returns
    -------
    areas : np.ndarray
        Post-crop area in source pixels squared, one entry per
        instance that survives the crop with at least one pixel.
    n_lost_to_crop : int
        Instances that had pixels before the crop and none after.
    """
    polygons = polygons_in_pixels(
        annotation, width_px, height_px,
        collector=collector, image_ref=image_ref,
    )
    labels, _, _ = rasterize_annotation(polygons, (height_px, width_px))
    n_labels = int(labels.max())
    if n_labels == 0:
        return np.empty(0, dtype=np.int64), 0

    x0, y0, x1, y1 = crop_bbox
    cropped = labels[y0:y1, x0:x1]

    full_counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    crop_counts = np.bincount(cropped.ravel(), minlength=n_labels + 1)

    survived = crop_counts[1:] > 0
    existed = full_counts[1:] > 0
    return (
        crop_counts[1:][survived].astype(np.int64),
        int(np.count_nonzero(existed & ~survived)),
    )


def compute_min_fragment_area(
    manifest: pd.DataFrame,
    assignment: Mapping[str, str],
    config: MinFragmentAreaConfig,
) -> FragmentAreaResult:
    """Measure the TRAIN instance-area distribution and freeze P1.

    Parameters
    ----------
    manifest : pandas.DataFrame
        Frozen inventory manifest.
    assignment : Mapping[str, str]
        Formulation to set name, from the chosen split.
    config : MinFragmentAreaConfig

    Returns
    -------
    FragmentAreaResult

    Raises
    ------
    FragmentAreaError
        If no TRAIN image yields any instance, or the manifest
        references an annotation absent from the configured Label
        Studio export.
    """
    train = manifest[
        manifest["formulation"].map(assignment) == "train"
    ].copy()
    if train.empty:
        raise FragmentAreaError("The split assigns no image to TRAIN")

    inventory = load_config(config.inventory_config)
    exports = {s.series: s.label_studio_json for s in inventory.sources}
    unknown_series = set(train["series"].unique()) - set(exports)
    if unknown_series:
        raise FragmentAreaError(
            f"No Label Studio export configured for series "
            f"{sorted(unknown_series)} (in {config.inventory_config})"
        )

    collector = IssueCollector()
    areas_by_bin: dict[str, list[np.ndarray]] = {}
    n_lost_to_crop = 0
    n_images = 0

    for series, group in train.groupby("series", sort=True):
        annotations = index_annotations_by_id(exports[str(series)])
        logger.info(
            "Series %s: %d TRAIN image(s), %d annotation(s) indexed.",
            series, len(group), len(annotations),
        )
        for row in group.itertuples(index=False):
            try:
                annotation = require_annotation(
                    annotations, int(row.mask_annotation_id),
                    str(row.image_id),
                )
            except AnnotationLookupError as e:
                raise FragmentAreaError(str(e)) from e
            try:
                areas, lost = _instance_areas_px2(
                    annotation,
                    int(row.width_px),
                    int(row.height_px),
                    parse_bbox(row.load_crop_bbox),
                    collector,
                    str(row.image_id),
                )
            except PolygonConversionError as e:
                raise FragmentAreaError(
                    f"Cannot rasterize {row.image_id}: {e}"
                ) from e
            areas_by_bin.setdefault(str(row.scale_bin), []).append(areas)
            n_lost_to_crop += lost
            n_images += 1

    return _summarize(
        areas_by_bin, n_images, n_lost_to_crop, config, len(collector.all())
    )


def _summarize(
    areas_by_bin: Mapping[str, list[np.ndarray]],
    n_images: int,
    n_lost_to_crop: int,
    config: MinFragmentAreaConfig,
    n_issues: int,
) -> FragmentAreaResult:
    """Turn the collected per-image areas into the frozen result."""
    concatenated = {
        scale_bin: np.concatenate(chunks) if chunks else np.empty(0)
        for scale_bin, chunks in areas_by_bin.items()
    }
    all_areas = (
        np.concatenate(list(concatenated.values()))
        if concatenated else np.empty(0)
    )
    if all_areas.size == 0:
        raise FragmentAreaError(
            "No instance survived the crop across the whole TRAIN set"
        )

    included = (
        np.concatenate([
            v for k, v in concatenated.items()
            if k != "outlier" and v.size
        ] or [np.empty(0)])
        if config.exclude_scale_outlier
        else all_areas
    )
    if included.size == 0:
        raise FragmentAreaError(
            "Excluding scale_outlier left no TRAIN instance to "
            "calibrate A_min_fragment on"
        )

    value = float(np.percentile(included, config.percentile))
    logger.info(
        "A_min_fragment = %.1f px^2 (P%.3g of %d TRAIN instance(s) "
        "across %d image(s); %d instance(s) lost to the crop; %d "
        "polygon issue(s)).",
        value, config.percentile, included.size, n_images,
        n_lost_to_crop, n_issues,
    )

    return FragmentAreaResult(
        a_min_fragment_px2=value,
        percentile=config.percentile,
        n_images=n_images,
        n_instances=int(included.size),
        excluded_scale_outlier=config.exclude_scale_outlier,
        value_including_outliers_px2=float(
            np.percentile(all_areas, config.percentile)
        ),
        by_scale_bin_px2={
            scale_bin: float(
                np.percentile(values, config.percentile)
            )
            for scale_bin, values in sorted(concatenated.items())
            if values.size
        },
        n_instances_lost_to_crop=n_lost_to_crop,
    )


def format_summary(result: Optional[FragmentAreaResult]) -> list[str]:
    """Render the calibration as Markdown report lines.

    Parameters
    ----------
    result : FragmentAreaResult, optional
        ``None`` renders the "step skipped" note.

    Returns
    -------
    list of str
    """
    lines = ["## A_min_fragment (kalibracja na TRAIN)", ""]
    if result is None:
        lines += ["_krok pominiety_", ""]
        return lines

    scope = (
        "bez obrazow scale_outlier"
        if result.excluded_scale_outlier
        else "z obrazami scale_outlier"
    )
    lines += [
        f"**A_min_fragment = {result.a_min_fragment_px2:.1f} px^2** "
        f"(P{result.percentile:g}, {scope})",
        "",
        f"- instancji w kalibracji: {result.n_instances}",
        f"- obrazow TRAIN: {result.n_images}",
        f"- ta sama wartosc z outlierami: "
        f"{result.value_including_outliers_px2:.1f} px^2",
        f"- instancji utraconych przez dociecie do load_crop_bbox: "
        f"{result.n_instances_lost_to_crop}",
        "",
        "Wartosci per `scale_bin` (diagnostyka: powierzchnia w px "
        "skaluje sie z kwadratem stosunku rozdzielczosci, wiec jedna "
        "wartosc globalna jest zdominowana przez `coarse`):",
        "",
        "| scale_bin | P%g [px^2] |" % result.percentile,
        "|---|---:|",
    ]
    for scale_bin, value in result.by_scale_bin_px2.items():
        lines.append(f"| {scale_bin} | {value:.1f} |")
    lines.append("")
    return lines
