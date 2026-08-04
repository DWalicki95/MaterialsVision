"""SEM sidecar (.txt) parsing, interpretation and consistency checks.

Sidecar files are INI-like (``[SemImageFile]`` header, ``key=value``
lines) but encoded ``iso-8859-2`` with CRLF line endings - both
confirmed against real TM3000 and SU8000 sidecars during planning.
``PixelSize`` is in nanometres per pixel; the unit is verified, not
assumed, via ``check_pixel_size_consistency``.
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

from data_prep.inventory.models import ParsedName, SidecarRecord
from data_prep.inventory.series_profiles import SeriesProfile

logger = logging.getLogger(__name__)

_SIDECAR_ENCODING = "iso-8859-2"

# PixelSize[nm] x Magnification is constant per instrument (confirmed on
# real data): TM3000 -> 129609 nm, SU8000 -> 99218.75 nm. A measurement
# deviating from its instrument's constant by more than the configured
# tolerance indicates a corrupt or misattributed sidecar.
INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM: dict[str, float] = {
    "TM3000": 129609.0,
    "SU8000": 99218.75,
}

_DATASIZE_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")


def parse_sidecar_file(path: Path) -> dict[str, str]:
    """Parse a SEM sidecar file into a raw key/value dictionary.

    Generic INI-like parsing: ``[Section]`` header lines are skipped,
    ``key=value`` lines are split on the first ``=``, values are
    returned unparsed (including empty strings for keys like
    ``SampleName=``).

    Parameters
    ----------
    path : Path
        Sidecar file path.

    Returns
    -------
    dict of str to str
        Raw key/value pairs, in file order.
    """
    raw: dict[str, str] = {}
    with open(path, encoding=_SIDECAR_ENCODING, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            raw[key.strip()] = value.strip()
    return raw


def interpret_sidecar(
    raw: Mapping[str, str], path: Path
) -> SidecarRecord:
    """Type and interpret a raw sidecar dictionary.

    Missing or unparsable numeric fields become ``None`` rather than
    raising - a sidecar with a corrupt field is still usable for
    whatever it does contain (see the taxonomy's ``pixel_size_missing``
    warning, handled by the caller).

    Parameters
    ----------
    raw : Mapping[str, str]
        Output of ``parse_sidecar_file``.
    path : Path
        Sidecar path, stored on the resulting record.

    Returns
    -------
    SidecarRecord
    """
    magnification = _parse_int(raw.get("Magnification"))
    pixel_size_raw_nm = _parse_float(raw.get("PixelSize"))
    pixel_size_um = (
        pixel_size_raw_nm / 1000.0
        if pixel_size_raw_nm is not None
        else None
    )
    datasize_w, datasize_h = _parse_datasize(raw.get("DataSize"))
    micron_marker_nm = _parse_float(raw.get("MicronMarker"))
    acquired_at = _parse_acquired_at(raw.get("Date"), raw.get("Time"))

    return SidecarRecord(
        path=path,
        instrument=_none_if_empty(raw.get("InstructName")),
        serial_number=_none_if_empty(raw.get("SerialNumber")),
        sample_name=_none_if_empty(raw.get("SampleName")),
        image_name=_none_if_empty(raw.get("ImageName")),
        file_format=_none_if_empty(raw.get("Format")),
        magnification=magnification,
        pixel_size_raw_nm=pixel_size_raw_nm,
        pixel_size_um=pixel_size_um,
        datasize_w=datasize_w,
        datasize_h=datasize_h,
        micron_marker_nm=micron_marker_nm,
        acquired_at=acquired_at,
        raw=dict(raw),
    )


def check_pixel_size_consistency(
    rec: SidecarRecord, tolerance: float
) -> list[str]:
    """Sanity-check a sidecar's pixel size against independent evidence.

    Two checks, both skipped (not failed) when the inputs needed are
    missing:

    1. ``PixelSize x Magnification`` must match the sidecar's
       instrument constant within ``tolerance`` (relative).
    2. The scale bar length implied by ``MicronMarker / PixelSize``
       must fall within 5-95% of the acquisition width - a sanity
       range, not a precise bound.

    Parameters
    ----------
    rec : SidecarRecord
    tolerance : float
        Relative tolerance for the instrument-constant check.

    Returns
    -------
    list of str
        Consistency issue codes; empty if all applicable checks pass.
    """
    codes: list[str] = []

    if (
        rec.instrument in INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM
        and rec.pixel_size_raw_nm is not None
        and rec.magnification is not None
    ):
        expected = INSTRUMENT_PIXEL_SIZE_CONSTANTS_NM[rec.instrument]
        actual = rec.pixel_size_raw_nm * rec.magnification
        if abs(actual - expected) / expected > tolerance:
            codes.append("pixel_size_magnification_product_mismatch")

    if (
        rec.micron_marker_nm is not None
        and rec.pixel_size_raw_nm is not None
        and rec.pixel_size_raw_nm > 0
        and rec.datasize_w is not None
    ):
        marker_px = rec.micron_marker_nm / rec.pixel_size_raw_nm
        lo, hi = 0.05 * rec.datasize_w, 0.95 * rec.datasize_w
        if not (lo <= marker_px <= hi):
            codes.append("micron_marker_out_of_range")

    return codes


def find_sidecar(
    parsed: ParsedName,
    profile: SeriesProfile,
    roots: Sequence[Path],
) -> Optional[Path]:
    """Locate the sidecar file for a parsed image name.

    Tries every candidate relative path from
    ``profile.sidecar_candidates`` against every configured root, in
    order, and returns the first match.

    Parameters
    ----------
    parsed : ParsedName
    profile : SeriesProfile
    roots : Sequence[Path]
        Sidecar root directories, searched in the given order.

    Returns
    -------
    Path or None
        The first matching sidecar path, or ``None`` if none exists.
    """
    for candidate in profile.sidecar_candidates(parsed):
        for root in roots:
            full = root / candidate
            if full.is_file():
                return full
    return None


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    """Return None for missing or empty-string sidecar values."""
    if value is None or value == "":
        return None
    return value


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse an integer sidecar value, returning None on failure."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Parse a float sidecar value, returning None on failure."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_datasize(
    value: Optional[str],
) -> tuple[Optional[int], Optional[int]]:
    """Parse a ``DataSize`` value like ``"1280x1040"`` into (w, h)."""
    if not value:
        return None, None
    match = _DATASIZE_RE.match(value.strip())
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _parse_acquired_at(
    date_value: Optional[str], time_value: Optional[str]
) -> Optional[str]:
    """Combine ``Date`` (MM/DD/YYYY) and ``Time`` (HH:MM:SS) to ISO 8601."""
    if not date_value or not time_value:
        return None
    try:
        dt = datetime.strptime(
            f"{date_value} {time_value}", "%m/%d/%Y %H:%M:%S"
        )
    except ValueError:
        return None
    return dt.isoformat()
