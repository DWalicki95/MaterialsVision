"""
Manifest schema, per-image row construction, source precedence,
global validation and deterministic CSV writing.

This is the orchestrator: it wires together every other module in
``data_prep.inventory``.
"""
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from data_prep.inventory.annotation_stats import (compute_instance_stats,
                                                  rasterize_annotation)
from data_prep.inventory.image_properties import read_image_properties
from data_prep.inventory.issues import (AnnotationSelectionError,
                                        CrossSectionError, FilenameParseError,
                                        Issue, IssueCollector, IssueLevel,
                                        ManifestBuildAborted,
                                        ManifestSchemaError,
                                        PolygonConversionError, RejectionLog)
from data_prep.inventory.label_studio import (iter_polygon_results, load_tasks,
                                              polygon_to_pixels,
                                              select_annotation)
from data_prep.inventory.models import (InventoryConfig, ParsedName,
                                        SidecarRecord, SourceConfig)
from data_prep.inventory.nonimage_region import detect_nonimage_region
from data_prep.inventory.sem_sidecar import (
    INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM, check_pixel_size_consistency,
    find_sidecar, interpret_sidecar, parse_sidecar_file)
from data_prep.inventory.series_profiles import get_profile
from materials_vision.utils import load_pixel_sizes_by_instrument

logger = logging.getLogger(__name__)

# Which nominal instrument's pixel-size table to fall back to when a
# series has no sidecar at all
SERIES_TO_INSTRUMENT: dict[str, str] = {"AS": "TM3000", "VAB": "SU8000"}

# InstructName (SEM control software identifier) -> logical microscope
# label. TM3000 and SU8000 are the two physical microscopes used to
# acquire this dataset; each has its own field-of-view calibration, so
# knowing which one took an image matters wherever pixel size or panel
# geometry is involved.
INSTRUMENT_TO_MICROSCOPE: dict[str, str] = {"TM3000": "M1", "SU8000": "M2"}

# Fallback for rows with no matched sidecar to read InstructName from
# directly: each series was only ever scanned on one of the two
# microscopes, so the series name alone still identifies it.
_MICROSCOPE_FALLBACK_BY_SERIES: dict[str, str] = {
    series: INSTRUMENT_TO_MICROSCOPE[instrument]
    for series, instrument in SERIES_TO_INSTRUMENT.items()
}

# pixel_size_um x magnification is a hardware constant for a given
# microscope (field of view scales inversely with magnification), so
# this product should land near the same value regardless of which
# image it's measured on. Used only as a sanity check on an
# already-resolved `microscope` value, never to assign it - magnifi-
# cation itself is sometimes missing or comes from an unreliable
# source. Derived from sem_sidecar.INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM
# (the same constant used to validate a sidecar's own internal
# consistency) so the two checks cannot silently drift apart.
MICROSCOPE_PRODUCT_UM: dict[str, float] = {
    INSTRUMENT_TO_MICROSCOPE[instrument]: constant_nm / 1000.0
    for instrument, constant_nm
    in INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM.items()
}

# Height, in rows, of the bottom data panel (scale bar / acquisition
# parameters) the SEM software burns into the image. Zero for M1
# (series AS): those files already had the panel removed before
# export, so the pixel detector never sees one there. Applied to each
# row's own on-disk height in `_resolve_load_crop_bbox` rather than
# hardcoded as an absolute pixel pair, so the same fixed-row-count
# invariant holds even for a file with slightly different on-disk
# dimensions (e.g. a `dimension_outlier` row), and for the smaller
# synthetic images used in tests.
PANEL_HEIGHT_ROWS_BY_MICROSCOPE: dict[str, int] = {"M1": 0, "M2": 70}

# Pixel-size thresholds (um/px) separating coarse-scale images from
# fine-scale ones, with a gap below the fine threshold reserved for
# close-up outlier shots that aren't representative of either normal
# scale.
SCALE_BIN_COARSE_MIN_UM = 3.0
SCALE_BIN_FINE_MIN_UM = 2.4

# Reference pixel size for q_max_i = pixel_size_um / Q_REFERENCE_UM:
# the finest (most zoomed-in) working scale in the dataset, from the
# SU8000 microscope at 40x (see
# materials_vision/calibration/sem_calibration.yaml). Expressing every
# image's scale as a ratio against this reference gives one
# comparable "how zoomed in is this" number across both microscopes.
Q_REFERENCE_UM = 2.480469

# Column order is the manifest's contract
MANIFEST_COLUMNS: tuple[str, ...] = (
    # 5.1 core
    "image_id", "formulation", "magnification", "pixel_size_um",
    "source_path", "mask_path",
    # 5.2 A. identity and provenance
    "series", "material", "source_filename", "ls_project_id",
    "ls_task_id", "file_hash",
    # 5.3 B. name semantics
    "cross_section", "sample_id", "magnification_source",
    "magnification_conflict", "filename_nonstandard",
    "cross_section_redundancy_ok",
    # 5.4 C. scale
    "pixel_size_source", "pixel_size_raw_nm", "instrument",
    "microscope", "microscope_source",
    "sem_serial_number", "acquired_at", "sem_datasize_w",
    "sem_datasize_h", "geometry_rescaled", "panel_cropped_px",
    "load_crop_bbox",
    "pixel_size_consistency", "scale_bin", "scale_outlier", "q_max_i",
    # 5.5 D. image properties
    "width_px", "height_px", "file_format", "bit_depth", "n_channels",
    "channels_identical",
    # 5.6 E. non-image elements
    "scalebar_present", "nonimage_bbox", "content_bbox",
    "nonimage_detector_version",
    # 5.7 F. annotation statistics
    "n_annotations", "n_instances", "n_border_instances",
    "n_instances_below_crop_bbox",
    "n_degenerate_polygons", "overlap_px_fraction",
    "has_significant_overlap",
    "pore_equivalent_diameter_min_px",
    "pore_equivalent_diameter_median_px",
    "pore_equivalent_diameter_max_px",
    "pore_equivalent_diameter_min_um",
    "pore_equivalent_diameter_median_um",
    "pore_equivalent_diameter_max_um",
    # 5.8 G. labeling metadata
    "annotators", "mask_annotator", "mask_annotation_id",
    "annotation_completed_at", "annotation_selection_rule",
    # 5.9 H. outputs
    "mask_exists",
    "duplicate_file_hash",
)


@dataclass
class InventoryResult:
    """Everything produced by one ``build_manifest`` call.

    Parameters
    ----------
    manifest : pd.DataFrame
        One row per accepted image, columns = ``MANIFEST_COLUMNS``.
    issues : list of Issue
        Every issue recorded during the run, in recording order.
    rejected : list of Issue
        The subset of ``issues`` that caused an image to be dropped
        (ERROR or FATAL issues tied 1:1 to a skipped image).
    run_metadata : dict
        Build-time facts (row/rejection counts, sources, config
        snapshot). Write-time facts (git commit, library versions,
        manifest hash) are added by ``reporting.write_run_metadata``.
    """

    manifest: pd.DataFrame
    issues: list[Issue]
    rejected: list[Issue] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)


def build_manifest(config: InventoryConfig) -> InventoryResult:
    """Build the inventory manifest from all configured sources.

    Processes every source's Label Studio export in task-id order,
    resolving metadata precedence per image and
    recording every problem via an ``IssueCollector``. No per-image
    FATAL condition stops the run early - the full pass and the global
    validation step (duplicate hashes, scale outliers, schema check)
    always complete, so a single run surfaces every fatal problem.

    Parameters
    ----------
    config : InventoryConfig

    Returns
    -------
    InventoryResult

    Raises
    ------
    ManifestBuildAborted
        If any FATAL issue was recorded, anywhere. Callers must not
        write any artifacts when this is raised.
    """
    collector = IssueCollector()
    rejection_log = RejectionLog()
    rows: list[dict[str, Any]] = []
    seen_image_ids: dict[str, str] = {}
    pixel_sizes_by_instrument = load_pixel_sizes_by_instrument()

    for source in config.sources:
        _process_source(
            source, config, collector, rejection_log, rows,
            seen_image_ids, pixel_sizes_by_instrument,
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _apply_global_validation(df, config, collector)
        df = df.sort_values("image_id").reset_index(drop=True)
    df = _finalize_schema(df, collector)

    run_metadata = {
        "n_rows": len(df),
        "n_rejected": len(rejection_log),
        "sources": [s.series for s in config.sources],
        "manifest_version": config.manifest_version,
    }

    if collector.has_fatal():
        raise ManifestBuildAborted(collector.by_level(IssueLevel.FATAL))

    return InventoryResult(
        manifest=df,
        issues=collector.all(),
        rejected=rejection_log.entries,
        run_metadata=run_metadata,
    )


def write_artifacts(
    result: InventoryResult,
    config: InventoryConfig,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """
    Write the manifest CSV to disk, refusing to clobber a frozen one.

    Only writes the manifest itself; the validation report, dataset
    summary and run metadata are written by the corresponding
    functions in ``reporting.py`` (the CLI calls both).

    Parameters
    ----------
    result : InventoryResult
    config : InventoryConfig
    overwrite : bool
        Required to be True to replace an existing manifest file of
        the same version.

    Returns
    -------
    dict of str to Path
        ``{"manifest": <path>}``.

    Raises
    ------
    FileExistsError
        If the target manifest file already exists and
        ``overwrite`` is False.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        config.output_dir / f"manifest_{config.manifest_version}.csv"
    )
    if manifest_path.exists() and not overwrite:
        next_version = _suggest_next_version(config.manifest_version)
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Pass "
            f"--overwrite to replace it, or use a new version "
            f"(e.g. '{next_version}')."
        )

    result.manifest.to_csv(
        manifest_path, index=False, na_rep="", float_format="%.6g",
        lineterminator="\n",
    )
    logger.info(
        "Wrote manifest: %s (%d rows)", manifest_path, len(result.manifest)
    )
    return {"manifest": manifest_path}


def _suggest_next_version(current: str) -> str:
    """Suggest a plausible next version tag for the "refused to
    overwrite" error message (e.g. "v1" -> "v2")."""
    if current.startswith("v") and current[1:].isdigit():
        return f"v{int(current[1:]) + 1}"
    return f"{current}_new"


def _process_source(
    source: SourceConfig,
    config: InventoryConfig,
    collector: IssueCollector,
    rejection_log: RejectionLog,
    rows: list[dict[str, Any]],
    seen_image_ids: dict[str, str],
    pixel_sizes_by_instrument: Mapping[str, Mapping[int, float]],
) -> None:
    """Process one configured source, appending rows in place."""
    profile = get_profile(source.series)
    tasks = load_tasks(source.label_studio_json)
    files = {
        p.name: p
        for p in source.images_dir.iterdir()
        if p.is_file() and not p.name.endswith(":Zone.Identifier")
    }
    task_filenames = {_task_filename(t) for t in tasks}

    for filename in sorted(files):
        if filename not in task_filenames:
            collector.add(
                IssueLevel.WARNING, "image_without_task", filename,
                f"file present in {source.images_dir} but no Label "
                f"Studio task references it",
            )

    n_rows_before = len(rows)
    n_rejected_before = len(rejection_log)
    matched_sidecars: set[Path] = set()

    for task in sorted(tasks, key=lambda t: t["id"]):
        filename = _task_filename(task)
        if filename not in files:
            issue = collector.add(
                IssueLevel.ERROR, "task_without_image", filename,
                f"task id={task.get('id')} has no matching file in "
                f"{source.images_dir}",
            )
            rejection_log.add(issue)
            continue

        row, reject_issue = _process_one_image(
            task, filename, files[filename], source, profile, config,
            collector, seen_image_ids, pixel_sizes_by_instrument,
            matched_sidecars,
        )
        if row is None:
            assert reject_issue is not None
            rejection_log.add(reject_issue)
            continue
        rows.append(row)

    n_processed = (
        (len(rows) - n_rows_before)
        + (len(rejection_log) - n_rejected_before)
    )
    if n_processed != len(tasks):
        raise ManifestSchemaError(
            f"[{source.series}] processed {n_processed} of "
            f"{len(tasks)} tasks - rows + rejections must equal the "
            f"number of Label Studio tasks"
        )

    _report_orphan_sidecars(source, matched_sidecars, collector)


def _process_one_image(
    task: Mapping[str, Any],
    filename: str,
    image_path: Path,
    source: SourceConfig,
    profile,
    config: InventoryConfig,
    collector: IssueCollector,
    seen_image_ids: dict[str, str],
    pixel_sizes_by_instrument: Mapping[str, Mapping[int, float]],
    matched_sidecars: set[Path],
) -> tuple[Optional[dict[str, Any]], Optional[Issue]]:
    """Build one manifest row.

    Returns
    -------
    tuple
        ``(row, None)`` on success, or ``(None, issue)`` with the
        exact issue that caused the image to be rejected.
    """
    stem = Path(filename).stem

    if not profile.detect_series(stem):
        issue = collector.add(
            IssueLevel.ERROR, "series_mismatch", filename,
            f"configured series={source.series!r} but name doesn't "
            f"match its filename convention",
        )
        return None, issue

    try:
        parsed = profile.parse(
            stem, collector=collector, fuzzy_cutoff=config.fuzzy_cutoff
        )
    except FilenameParseError as e:
        issue = collector.add(
            IssueLevel.FATAL, "filename_unparsable", filename, str(e)
        )
        return None, issue
    except CrossSectionError as e:
        issue = collector.add(
            IssueLevel.FATAL, "cross_section_unknown", filename, str(e)
        )
        return None, issue

    if parsed.image_id in seen_image_ids:
        issue = collector.add(
            IssueLevel.FATAL, "image_id_collision", parsed.image_id,
            f"also produced by {seen_image_ids[parsed.image_id]!r}",
        )
        return None, issue
    seen_image_ids[parsed.image_id] = filename

    try:
        props = read_image_properties(image_path)
    except Exception as e:  # noqa: BLE001 - any decode failure is fatal
        issue = collector.add(
            IssueLevel.FATAL, "image_unreadable", parsed.image_id, str(e)
        )
        return None, issue

    region = detect_nonimage_region(
        props.gray,
        extreme_fraction=config.nonimage_extreme_fraction,
        max_band_fraction=config.nonimage_max_band_fraction,
    )

    sidecar_path = find_sidecar(
        parsed, profile, source.sem_metadata_dirs
    )
    sidecar: Optional[SidecarRecord] = None
    pixel_size_consistency: Optional[str] = None
    if sidecar_path is None:
        collector.add(
            IssueLevel.WARNING, "sidecar_missing", parsed.image_id,
            f"no sidecar found for {filename}",
        )
    else:
        matched_sidecars.add(sidecar_path)
        sidecar = interpret_sidecar(
            parse_sidecar_file(sidecar_path), sidecar_path
        )
        if sidecar.pixel_size_raw_nm is None:
            collector.add(
                IssueLevel.WARNING, "pixel_size_missing",
                parsed.image_id, f"sidecar {sidecar.path} has no "
                f"PixelSize",
            )
        if not _sidecar_name_consistent(parsed, sidecar):
            collector.add(
                IssueLevel.WARNING, "sidecar_name_mismatch",
                parsed.image_id,
                f"SampleName={sidecar.sample_name!r} vs "
                f"formulation={parsed.formulation!r}",
            )
        codes = check_pixel_size_consistency(
            sidecar, config.pixel_size_tolerance
        )
        if codes:
            pixel_size_consistency = ";".join(codes)
            collector.add(
                IssueLevel.WARNING, "pixel_size_inconsistent",
                parsed.image_id, pixel_size_consistency,
            )

    magnification, magnification_source, magnification_conflict = (
        _resolve_magnification(parsed, sidecar, collector)
    )
    (
        pixel_size_um, pixel_size_raw_nm, pixel_size_source,
        geometry_rescaled, panel_cropped_px,
    ) = _resolve_pixel_size(
        parsed, sidecar, magnification, props,
        pixel_sizes_by_instrument, collector,
    )

    microscope, microscope_source = _resolve_microscope(
        parsed, sidecar, pixel_size_um, magnification,
        config.pixel_size_tolerance, collector,
    )
    load_crop_bbox = _resolve_load_crop_bbox(
        microscope, props.width_px, props.height_px
    )
    if load_crop_bbox is not None and load_crop_bbox != region.content_bbox:
        issue = collector.add(
            IssueLevel.FATAL, "content_bbox_crop_mismatch",
            parsed.image_id,
            f"detected content_bbox={region.content_bbox} vs frozen "
            f"load_crop_bbox={load_crop_bbox} (microscope="
            f"{microscope!r}); PANEL_HEIGHT_ROWS_BY_MICROSCOPE may be "
            f"stale for this data",
        )
        return None, issue
    scale_bin = _scale_bin(pixel_size_um)
    q_max_i = _q_max_i(pixel_size_um)

    try:
        selection = select_annotation(
            task, collector=collector,
            avoid_annotators=source.avoid_annotators,
        )
    except AnnotationSelectionError as e:
        issue = collector.add(
            IssueLevel.FATAL, "annotation_none_available",
            parsed.image_id, str(e),
        )
        return None, issue

    try:
        polygons = [
            polygon_to_pixels(
                r, props.width_px, props.height_px,
                collector=collector, image_ref=parsed.image_id,
            )
            for r in iter_polygon_results(
                selection.annotation, collector=collector,
                image_ref=parsed.image_id,
            )
        ]
    except PolygonConversionError as e:
        issue = collector.add(
            IssueLevel.FATAL, "polygon_unconvertible", parsed.image_id,
            str(e),
        )
        return None, issue

    n_instances_below_crop_bbox = _count_instances_below_crop_bbox(
        polygons, region.content_bbox
    )
    if n_instances_below_crop_bbox > 0:
        collector.add(
            IssueLevel.INFO, "annotation_below_crop_bbox",
            parsed.image_id,
            f"{n_instances_below_crop_bbox} instance(s) have an "
            f"annotated point at or past y="
            f"{region.content_bbox[3]}, inside the area that gets "
            f"cropped away before training",
        )

    labels, coverage, n_degenerate = rasterize_annotation(
        polygons, (props.height_px, props.width_px)
    )
    stats = compute_instance_stats(
        labels, coverage, region.content_bbox, pixel_size_um,
        n_degenerate,
    )

    row = _build_row(
        parsed, source, filename, task, image_path, props, region,
        sidecar, magnification, magnification_source,
        magnification_conflict, pixel_size_um, pixel_size_raw_nm,
        pixel_size_source, geometry_rescaled, panel_cropped_px,
        pixel_size_consistency, microscope, microscope_source,
        load_crop_bbox, scale_bin, q_max_i,
        n_instances_below_crop_bbox, selection, stats, config,
    )
    return row, None


def _task_filename(task: Mapping[str, Any]) -> str:
    """Basename of a Label Studio task's referenced image file."""
    return str(task["data"]["image"]).rsplit("/", 1)[-1]


def _sidecar_name_consistent(
    parsed: ParsedName, sidecar: SidecarRecord
) -> bool:
    """Loose cross-check of SampleName against the parsed formulation.

    ``SampleName`` formats differ across series and is often absent
    entirely (confirmed for AS); absence is not a mismatch, only an
    explicit disagreement is.
    """
    if not sidecar.sample_name:
        return True
    return parsed.formulation.lower() in sidecar.sample_name.lower()


def _resolve_magnification(
    parsed: ParsedName,
    sidecar: Optional[SidecarRecord],
    collector: IssueCollector,
) -> tuple[Optional[int], str, bool]:
    """Resolve magnification per the plan's source precedence
    (section 6.1): sidecar wins, filename is the cross-check."""
    sidecar_mag = sidecar.magnification if sidecar is not None else None
    name_mag = parsed.magnification_from_name

    if sidecar_mag is not None:
        conflict = name_mag is not None and name_mag != sidecar_mag
        if conflict:
            collector.add(
                IssueLevel.WARNING, "magnification_conflict",
                parsed.image_id,
                f"filename={name_mag} vs sidecar={sidecar_mag}",
            )
        return sidecar_mag, "sem_sidecar", conflict
    if name_mag is not None:
        return name_mag, "filename", False
    return None, "none", False


def _resolve_pixel_size(
    parsed: ParsedName,
    sidecar: Optional[SidecarRecord],
    magnification: Optional[int],
    props,
    pixel_sizes_by_instrument: Mapping[str, Mapping[int, float]],
    collector: IssueCollector,
) -> tuple[Optional[float], Optional[float], str, bool, Optional[int]]:
    """Resolve pixel_size_um, its provenance, and the Roboflow-rescale
    correction, per the plan's source precedence (section 6.1)."""
    pixel_size_raw_nm = (
        sidecar.pixel_size_raw_nm if sidecar is not None else None
    )

    if sidecar is not None and sidecar.pixel_size_um is not None:
        pixel_size_um: Optional[float] = sidecar.pixel_size_um
        pixel_size_source = "sem_sidecar"
    elif magnification is not None:
        nominal_instrument = SERIES_TO_INSTRUMENT.get(parsed.series)
        nominal_table = (
            pixel_sizes_by_instrument.get(nominal_instrument, {})
            if nominal_instrument is not None
            else {}
        )
        pixel_size_um = nominal_table.get(magnification)
        pixel_size_source = (
            "nominal_dict" if pixel_size_um is not None else "none"
        )
    else:
        pixel_size_um = None
        pixel_size_source = "none"

    geometry_rescaled = False
    panel_cropped_px: Optional[int] = None
    if (
        sidecar is not None
        and sidecar.datasize_w is not None
        and sidecar.datasize_h is not None
    ):
        if sidecar.datasize_w == props.width_px:
            panel_cropped_px = sidecar.datasize_h - props.height_px
        else:
            geometry_rescaled = True
            if pixel_size_um is not None:
                pixel_size_um = pixel_size_um * (
                    sidecar.datasize_w / props.width_px
                )
            pixel_size_source = "rescaled"
            collector.add(
                IssueLevel.WARNING, "geometry_rescaled", parsed.image_id,
                f"sidecar DataSize width={sidecar.datasize_w} vs "
                f"file width={props.width_px}",
            )

    return (
        pixel_size_um, pixel_size_raw_nm, pixel_size_source,
        geometry_rescaled, panel_cropped_px,
    )


def _resolve_microscope(
    parsed: ParsedName,
    sidecar: Optional[SidecarRecord],
    pixel_size_um: Optional[float],
    magnification: Optional[int],
    tolerance: float,
    collector: IssueCollector,
) -> tuple[Optional[str], str]:
    """Resolve which physical microscope acquired this image.

    The sidecar's own InstructName wins whenever a sidecar was
    matched - a direct instrument reading is the most reliable
    source. Otherwise fall back to the per-series nominal instrument,
    since each series was only ever scanned on one microscope.
    pixel_size_um x magnification is never used to pick a value here
    - it only feeds an independent sanity check against whichever
    microscope was resolved above (see `_check_microscope_product`),
    because that product can look right by coincidence and
    `magnification` is not always reliable itself."""
    instrument = sidecar.instrument if sidecar is not None else None
    if instrument is not None and instrument in INSTRUMENT_TO_MICROSCOPE:
        microscope = INSTRUMENT_TO_MICROSCOPE[instrument]
        source = "sem_sidecar"
    else:
        microscope = _MICROSCOPE_FALLBACK_BY_SERIES.get(parsed.series)
        source = "series_map" if microscope is not None else "none"

    if microscope is not None:
        _check_microscope_product(
            parsed, microscope, pixel_size_um, magnification, tolerance,
            collector,
        )
    return microscope, source


def _check_microscope_product(
    parsed: ParsedName,
    microscope: str,
    pixel_size_um: Optional[float],
    magnification: Optional[int],
    tolerance: float,
    collector: IssueCollector,
) -> None:
    """Sanity-check the resolved `microscope` against an independent
    physical fact: pixel_size_um x magnification is a hardware
    constant for a given microscope (`MICROSCOPE_PRODUCT_UM`). A
    mismatch only flags the row for review - it never changes
    `microscope` itself, since the sidecar instrument or the series
    fallback are more trustworthy sources. Skipped entirely when
    either input is unknown."""
    if pixel_size_um is None or magnification is None:
        return
    expected_um = MICROSCOPE_PRODUCT_UM.get(microscope)
    if expected_um is None:
        return
    actual_um = pixel_size_um * magnification
    if abs(actual_um - expected_um) / expected_um > tolerance:
        collector.add(
            IssueLevel.WARNING, "microscope_product_conflict",
            parsed.image_id,
            f"pixel_size_um*magnification={actual_um:.6g} vs "
            f"{microscope} expected {expected_um:.6g}",
        )


def _resolve_load_crop_bbox(
    microscope: Optional[str], width_px: int, height_px: int,
) -> Optional[tuple[int, int, int, int]]:
    """Compute the bounding box of this image's actual content,
    excluding the bottom data panel if its microscope has one.
    Computed from this image's own on-disk dimensions minus
    `PANEL_HEIGHT_ROWS_BY_MICROSCOPE`, not hardcoded as an absolute
    pixel pair - see that constant's comment for why. None when
    `microscope` could not be resolved at all, since there is then no
    panel height to subtract."""
    if microscope not in PANEL_HEIGHT_ROWS_BY_MICROSCOPE:
        return None
    panel_rows = PANEL_HEIGHT_ROWS_BY_MICROSCOPE[microscope]
    return (0, 0, width_px, height_px - panel_rows)


def _scale_bin(pixel_size_um: Optional[float]) -> Optional[str]:
    """Classify pixel_size_um into a scale bin using fixed
    thresholds, not a rule relative to other images in the current
    run: a relative rule would shift its own boundaries every time
    the dataset changes, silently invalidating any split or
    comparison that assumed stable bins. See
    `_apply_global_validation` for a separate, purely informational
    check against the per-series median."""
    if pixel_size_um is None:
        return None
    if pixel_size_um >= SCALE_BIN_COARSE_MIN_UM:
        return "coarse"
    if pixel_size_um >= SCALE_BIN_FINE_MIN_UM:
        return "fine"
    return "outlier"


def _q_max_i(pixel_size_um: Optional[float]) -> Optional[float]:
    """q_max_i = pixel_size_um / Q_REFERENCE_UM: how many times more
    zoomed-out this image's pixel size is than the dataset's finest
    working scale."""
    if pixel_size_um is None:
        return None
    return pixel_size_um / Q_REFERENCE_UM


def _count_instances_below_crop_bbox(
    polygons: list, content_bbox: tuple[int, int, int, int],
) -> int:
    """Count annotated instances that reach into the area a later
    cropping step would cut away (everything at or past the bottom
    edge of `content_bbox`). An annotator draws on the full image, so
    a pore near the bottom edge can have points inside a data panel
    that gets removed before training; this only matters for images
    that actually have such a panel - for the rest, `content_bbox`
    already spans the full frame and the count is always zero."""
    _, _, _, y1 = content_bbox
    return sum(
        1 for polygon in polygons
        if polygon.size > 0 and float(polygon[:, 1].max()) >= y1
    )


def _build_row(
    parsed: ParsedName,
    source: SourceConfig,
    filename: str,
    task: Mapping[str, Any],
    image_path: Path,
    props,
    region,
    sidecar: Optional[SidecarRecord],
    magnification: Optional[int],
    magnification_source: str,
    magnification_conflict: bool,
    pixel_size_um: Optional[float],
    pixel_size_raw_nm: Optional[float],
    pixel_size_source: str,
    geometry_rescaled: bool,
    panel_cropped_px: Optional[int],
    pixel_size_consistency: Optional[str],
    microscope: Optional[str],
    microscope_source: str,
    load_crop_bbox: Optional[tuple[int, int, int, int]],
    scale_bin: Optional[str],
    q_max_i: Optional[float],
    n_instances_below_crop_bbox: int,
    selection,
    stats,
    config: InventoryConfig,
) -> dict[str, Any]:
    """Assemble one manifest row dict, columns matching
    ``MANIFEST_COLUMNS`` (order applied later by pandas)."""
    mask_path = (
        config.mask_root / parsed.series / f"{parsed.image_id}_masks.tif"
    )

    def _um(value_px: float) -> Optional[float]:
        return value_px * pixel_size_um if pixel_size_um is not None \
            else None

    diam_min_px, diam_median_px, diam_max_px = (
        stats.equivalent_diameter_px
    )

    return {
        "image_id": parsed.image_id,
        "formulation": parsed.formulation,
        "magnification": magnification,
        "pixel_size_um": pixel_size_um,
        "source_path": str(image_path),
        "mask_path": str(mask_path),
        "series": parsed.series,
        "material": parsed.material,
        "source_filename": filename,
        "ls_project_id": task.get("project"),
        "ls_task_id": task.get("id"),
        "file_hash": props.file_hash,
        "cross_section": parsed.cross_section,
        "sample_id": parsed.sample_id,
        "magnification_source": magnification_source,
        "magnification_conflict": magnification_conflict,
        "filename_nonstandard": parsed.is_nonstandard,
        "cross_section_redundancy_ok": parsed.cross_section_redundancy_ok,
        "pixel_size_source": pixel_size_source,
        "pixel_size_raw_nm": pixel_size_raw_nm,
        "instrument": sidecar.instrument if sidecar else None,
        "microscope": microscope,
        "microscope_source": microscope_source,
        "sem_serial_number": sidecar.serial_number if sidecar else None,
        "acquired_at": sidecar.acquired_at if sidecar else None,
        "sem_datasize_w": sidecar.datasize_w if sidecar else None,
        "sem_datasize_h": sidecar.datasize_h if sidecar else None,
        "geometry_rescaled": geometry_rescaled,
        "panel_cropped_px": panel_cropped_px,
        "load_crop_bbox": (
            _bbox_to_str(load_crop_bbox)
            if load_crop_bbox is not None else None
        ),
        "pixel_size_consistency": pixel_size_consistency,
        "scale_bin": scale_bin,
        "scale_outlier": scale_bin == "outlier",
        "q_max_i": q_max_i,
        "width_px": props.width_px,
        "height_px": props.height_px,
        "file_format": props.file_format,
        "bit_depth": props.bit_depth,
        "n_channels": props.n_channels,
        "channels_identical": props.channels_identical,
        "scalebar_present": region.present,
        "nonimage_bbox": (
            _bbox_to_str(region.bbox) if region.bbox else None
        ),
        "content_bbox": _bbox_to_str(region.content_bbox),
        "nonimage_detector_version": region.detector_version,
        "n_annotations": selection.n_annotations,
        "n_instances": stats.n_instances,
        "n_border_instances": stats.n_border_instances,
        "n_instances_below_crop_bbox": n_instances_below_crop_bbox,
        "n_degenerate_polygons": stats.n_degenerate_polygons,
        "overlap_px_fraction": stats.overlap_px_fraction,
        "has_significant_overlap": (
            stats.overlap_px_fraction > config.overlap_significant_fraction
        ),
        "pore_equivalent_diameter_min_px": diam_min_px,
        "pore_equivalent_diameter_median_px": diam_median_px,
        "pore_equivalent_diameter_max_px": diam_max_px,
        "pore_equivalent_diameter_min_um": _um(diam_min_px),
        "pore_equivalent_diameter_median_um": _um(diam_median_px),
        "pore_equivalent_diameter_max_um": _um(diam_max_px),
        "annotators": ";".join(str(a) for a in selection.annotators),
        "mask_annotator": selection.mask_annotator,
        "mask_annotation_id": selection.mask_annotation_id,
        "annotation_completed_at": selection.annotation_completed_at,
        "annotation_selection_rule": selection.selection_rule,
        "mask_exists": mask_path.exists(),
        "duplicate_file_hash": False,  # filled in global validation pass
    }


def _bbox_to_str(bbox: tuple[int, int, int, int]) -> str:
    """Render a ``(x0, y0, x1, y1)`` bbox as ``"x0,y0,x1,y1"``."""
    return ",".join(str(v) for v in bbox)


def _report_orphan_sidecars(
    source: SourceConfig,
    matched_sidecars: set[Path],
    collector: IssueCollector,
) -> None:
    """Report sidecar files never matched to any image, as one
    summary INFO issue (the AS archive alone has ~1700, from
    magnifications outside this dataset's 40x/50x scope)."""
    all_sidecars: set[Path] = set()
    for root in source.sem_metadata_dirs:
        all_sidecars.update(p for p in root.rglob("*.txt") if p.is_file())
    orphans = all_sidecars - matched_sidecars
    if not orphans:
        return
    sample = sorted(str(p) for p in orphans)[:20]
    collector.add(
        IssueLevel.INFO, "sidecar_orphan", source.series,
        f"{len(orphans)} orphan sidecar file(s), e.g. {sample}",
    )


def _apply_global_validation(
    df: pd.DataFrame,
    config: InventoryConfig,
    collector: IssueCollector,
) -> pd.DataFrame:
    """Apply checks that require the full, assembled manifest:
    image_id uniqueness (defense in depth), duplicate file hashes, the
    relative-scale diagnostic, and dimension outliers (all issue-only,
    no column - scale_bin/scale_outlier are set per-row in
    `_build_row` from the frozen absolute thresholds)."""
    duplicated_ids = df["image_id"][df["image_id"].duplicated()]
    for image_id in duplicated_ids.unique():
        collector.add(
            IssueLevel.FATAL, "image_id_collision", image_id,
            "duplicate image_id survived per-image detection",
        )

    hash_counts = df["file_hash"].value_counts()
    duplicate_hashes = set(hash_counts[hash_counts > 1].index)
    if duplicate_hashes:
        is_dup = df["file_hash"].isin(duplicate_hashes)
        df.loc[is_dup, "duplicate_file_hash"] = True
        for image_id in df.loc[is_dup, "image_id"]:
            collector.add(
                IssueLevel.WARNING, "duplicate_file_hash", image_id,
                "file_hash shared with at least one other row",
            )

    for series, group in df.groupby("series"):
        known = group["pixel_size_um"].dropna()
        if not known.empty and known.median() > 0:
            median = known.median()
            for idx, value in known.items():
                ratio = max(value / median, median / value)
                if ratio > config.scale_outlier_ratio:
                    # Purely informational: scale_bin/scale_outlier
                    # are already set per-row from the fixed absolute
                    # thresholds in _scale_bin. This separate check
                    # flags rows that deviate a lot from their own
                    # series' median even when the absolute rule does
                    # not - useful for spotting future data that
                    # looks inconsistent with the rest of its series
                    # for a different reason than the fixed bins
                    # catch.
                    collector.add(
                        IssueLevel.INFO,
                        "scale_outlier_relative_diagnostic",
                        df.loc[idx, "image_id"],
                        f"pixel_size_um={value:.6g} vs series median "
                        f"{median:.6g} (series={series})",
                    )

        dims = list(
            zip(group["width_px"], group["height_px"])
        )
        if not dims:
            continue
        mode_dims = Counter(dims).most_common(1)[0][0]
        for idx, row in group.iterrows():
            dims_here = (row["width_px"], row["height_px"])
            if dims_here != mode_dims:
                collector.add(
                    IssueLevel.WARNING, "dimension_outlier",
                    row["image_id"],
                    f"{dims_here[0]}x{dims_here[1]} vs series-mode "
                    f"{mode_dims[0]}x{mode_dims[1]} (series={series})",
                )

    return df


def _finalize_schema(
    df: pd.DataFrame, collector: IssueCollector
) -> pd.DataFrame:
    """Reindex to the frozen column order, raising on any mismatch."""
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(MANIFEST_COLUMNS)
    if missing or extra:
        collector.add(
            IssueLevel.FATAL, "schema_mismatch", "manifest",
            f"missing columns={sorted(missing)}, "
            f"unexpected columns={sorted(extra)}",
        )
        raise ManifestSchemaError(
            f"Manifest columns do not match MANIFEST_COLUMNS: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return df[list(MANIFEST_COLUMNS)]
