"""
Data models for the data inventory pipeline.

All values that flow between the parsing/reading stages and the manifest
builder are represented as frozen dataclasses, so that a partially built
row can never be mutated by a later processing step.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SourceConfig:
    """One entry of the ``sources`` list in the inventory configuration.

    Parameters
    ----------
    series : str
        Series identifier, e.g. ``"AS"`` or ``"VAB"``. Selects the
        ``SeriesProfile`` used to parse filenames in ``images_dir``.
    images_dir : Path
        Directory containing the Label Studio image exports.
    label_studio_json : Path
        Label Studio JSON export; the authoritative image registry.
    sem_metadata_dirs : tuple of Path
        SEM sidecar root directories, searched in order.
    avoid_annotators : tuple of int
        ``completed_by`` IDs to avoid when selecting the mask-producing
        annotation for this source. If the latest annotation
        (``latest_updated_at_then_max_id``) belongs to one of these
        annotators, ``select_annotation`` falls back to the latest
        annotation among the remaining annotators, when one exists.
        Empty by default, which reproduces the unconditional
        latest-wins behaviour.
    """

    series: str
    images_dir: Path
    label_studio_json: Path
    sem_metadata_dirs: tuple[Path, ...]
    avoid_annotators: tuple[int, ...] = ()


@dataclass(frozen=True)
class InventoryConfig:
    """Fully resolved inventory run configuration.

    Parameters
    ----------
    manifest_version : str
        Version tag used in the output filename, e.g. ``"v1"``.
    output_dir : Path
        Directory where all artifacts are written.
    mask_root : Path
        Root directory used to compute the ``mask_path`` contract column.
    sources : tuple of SourceConfig
        One entry per data series, processed in this order.
    scale_outlier_ratio : float
        Ratio above the per-series median ``pixel_size_um`` (or its
        reciprocal) beyond which a row is flagged ``scale_outlier``.
    fuzzy_cutoff : float
        ``difflib.get_close_matches`` cutoff for cross-section
        canonicalization fallback.
    nonimage_extreme_fraction : float
        Fraction of near-black/near-white pixels in a row required for it
        to be classified as part of the non-image band.
    nonimage_max_band_fraction : float
        Maximum fraction of image height the non-image band detector is
        allowed to claim, protecting against false positives eating the
        whole image.
    overlap_significant_fraction : float
        Threshold on ``overlap_px_fraction`` above which
        ``has_significant_overlap`` is set.
    pixel_size_tolerance : float
        Relative tolerance used by ``check_pixel_size_consistency``.
    """

    manifest_version: str
    output_dir: Path
    mask_root: Path
    sources: tuple[SourceConfig, ...]
    scale_outlier_ratio: float = 1.5
    fuzzy_cutoff: float = 0.85
    nonimage_extreme_fraction: float = 0.90
    nonimage_max_band_fraction: float = 0.35
    overlap_significant_fraction: float = 0.01
    pixel_size_tolerance: float = 0.01


@dataclass(frozen=True)
class ParsedName:
    """Result of parsing an image filename with a ``SeriesProfile``.

    Parameters
    ----------
    image_id : str
        Deterministic, human-readable, globally unique identifier.
    series : str
        Series this name was parsed with, e.g. ``"AS"``.
    material : str
        Foam material type, e.g. ``"AS"``, ``"VAB"``, ``"K"``.
    formulation : str
        Formulation token, the split grouping unit (e.g. ``"AS26"``).
    cross_section : str, optional
        Canonical cross-section (``"rownolegly"`` / ``"prostopadly"``),
        ``None`` for series without a cross-section concept.
    magnification_from_name : int, optional
        Magnification parsed from the filename, ``None`` when the naming
        convention does not encode it (VAB/K).
    sample_id : str, optional
        Sample/image token from the filename, kept as a string because a
        handful of real filenames use non-numeric tokens.
    source_filename : str
        The original filename (without extension) this was parsed from.
    is_nonstandard : bool
        True when the name parsed successfully but the sample token does
        not match the expected numeric pattern.
    cross_section_redundancy_ok : bool, optional
        For VAB/K names, whether the redundant second
        formulation/cross-section segment agrees with the first;
        ``None`` for series without the redundant segment (AS).
    """

    image_id: str
    series: str
    material: str
    formulation: str
    cross_section: Optional[str]
    magnification_from_name: Optional[int]
    sample_id: Optional[str]
    source_filename: str
    is_nonstandard: bool
    cross_section_redundancy_ok: Optional[bool] = None


@dataclass(frozen=True)
class SidecarRecord:
    """Interpreted contents of a SEM ``.txt`` sidecar file.

    Parameters
    ----------
    path : Path
        Sidecar file path.
    instrument : str, optional
        ``InstructName`` (e.g. ``"TM3000"``, ``"SU8000"``).
    serial_number : str, optional
        ``SerialNumber``.
    sample_name : str, optional
        ``SampleName``.
    image_name : str, optional
        ``ImageName``.
    file_format : str, optional
        ``Format`` (e.g. ``"tif"``, ``"JPG"``).
    magnification : int, optional
        ``Magnification``.
    pixel_size_raw_nm : float, optional
        Raw ``PixelSize`` value as stored in the sidecar (nm/px).
    pixel_size_um : float, optional
        ``pixel_size_raw_nm / 1000``.
    datasize_w, datasize_h : int, optional
        Parsed ``DataSize`` (acquisition-time image dimensions).
    micron_marker_nm : float, optional
        ``MicronMarker`` (scale bar physical length, nm).
    acquired_at : str, optional
        ISO 8601 timestamp built from ``Date`` + ``Time``.
    raw : Mapping[str, str]
        Full key=value dictionary, unparsed.
    """

    path: Path
    instrument: Optional[str]
    serial_number: Optional[str]
    sample_name: Optional[str]
    image_name: Optional[str]
    file_format: Optional[str]
    magnification: Optional[int]
    pixel_size_raw_nm: Optional[float]
    pixel_size_um: Optional[float]
    datasize_w: Optional[int]
    datasize_h: Optional[int]
    micron_marker_nm: Optional[float]
    acquired_at: Optional[str]
    raw: Mapping[str, str]


@dataclass(frozen=True)
class ImageProperties:
    """Properties read from the image file in a single pass.

    Parameters
    ----------
    width_px, height_px : int
        Image dimensions as stored on disk.
    file_format : str
        ``PIL.Image.format`` (e.g. ``"JPEG"``, ``"PNG"``).
    bit_depth : int
        Bits per channel.
    n_channels : int
        Number of channels (1 for grayscale, 3 for RGB).
    channels_identical : bool, optional
        Whether all channels are pixel-identical; ``None`` for
        single-channel images where the question is moot.
    file_hash : str
        SHA-256 hex digest of the raw file bytes.
    gray : np.ndarray
        Grayscale working copy, used by the non-image region detector.
        Not written to the manifest.
    """

    width_px: int
    height_px: int
    file_format: str
    bit_depth: int
    n_channels: int
    channels_identical: Optional[bool]
    file_hash: str
    gray: np.ndarray


@dataclass(frozen=True)
class NonImageRegion:
    """Result of the non-image (scale bar / data panel) band detector.

    Parameters
    ----------
    present : bool
        Whether a non-image band was detected.
    bbox : tuple of int, optional
        ``(x0, y0, x1, y1)`` of the non-image band, ``None`` if absent.
    content_bbox : tuple of int
        ``(x0, y0, x1, y1)`` of the usable image content; always set,
        equals the full frame when ``present`` is False.
    detector_version : str
        Version tag of the detector that produced this result.
    """

    present: bool
    bbox: Optional[tuple[int, int, int, int]]
    content_bbox: tuple[int, int, int, int]
    detector_version: str


@dataclass(frozen=True)
class AnnotationSelection:
    """Result of selecting the mask-producing annotation for a task.

    Parameters
    ----------
    n_annotations : int
        Number of non-cancelled annotations on the task.
    annotators : tuple of int
        ``completed_by`` of all non-cancelled annotations, sorted.
    mask_annotator : int
        ``completed_by`` of the selected annotation.
    mask_annotation_id : int
        ``id`` of the selected annotation.
    annotation_completed_at : str
        ``updated_at`` of the selected annotation (ISO 8601 UTC).
    selection_rule : str
        Name of the rule used: ``"latest_updated_at_then_max_id"``, or
        ``"latest_updated_at_then_max_id+annotator_fallback"`` when the
        source's ``avoid_annotators`` caused the selection to fall back
        away from the globally latest annotation.
    annotation : Mapping
        The selected annotation dict itself (not written to the
        manifest, consumed by ``label_studio.iter_polygon_results``).
    """

    n_annotations: int
    annotators: tuple[int, ...]
    mask_annotator: int
    mask_annotation_id: int
    annotation_completed_at: str
    selection_rule: str
    annotation: Mapping


@dataclass(frozen=True)
class InstanceStats:
    """Per-image instance statistics computed from rasterized polygons.

    Parameters
    ----------
    n_instances : int
        Number of instances in the rasterized label image.
    n_border_instances : int
        Instances touching the edge of ``content_bbox``.
    n_degenerate_polygons : int
        Polygons that produced zero rasterized pixels.
    overlap_px_fraction : float
        Fraction of covered pixels touched by more than one polygon.
    equivalent_diameter_px : tuple of float
        ``(min, median, max)`` equivalent diameter in pixels; ``(0, 0,
        0)`` when ``n_instances`` is 0.
    """

    n_instances: int
    n_border_instances: int
    n_degenerate_polygons: int
    overlap_px_fraction: float
    equivalent_diameter_px: tuple[float, float, float]
