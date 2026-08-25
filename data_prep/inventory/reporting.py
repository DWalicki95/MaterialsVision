"""Human- and machine-readable reporting artifacts: the validation
report, the dataset summary, run metadata, and optional visual
verification thumbnails for the non-image region detector.

None of this module decides anything about the manifest's content -
it only renders what ``manifest.build_manifest`` already produced.
"""
import hashlib
import json
import logging
import math
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PIL import Image, ImageDraw

from data_prep.inventory.issues import Issue
from data_prep.inventory.manifest import (INSTRUMENT_TO_MICROSCOPE,
                                          PANEL_HEIGHT_ROWS_BY_MICROSCOPE,
                                          Q_REFERENCE_UM,
                                          SCALE_BIN_COARSE_MIN_UM,
                                          SCALE_BIN_FINE_MIN_UM,
                                          InventoryResult)
from data_prep.inventory.models import InventoryConfig
from materials_vision.quantitative_analysis.stats_utils import \
    calculate_statistics

logger = logging.getLogger(__name__)

_TRACKED_LIBRARIES = (
    "pandas", "numpy", "Pillow", "PyYAML", "scikit-image",
)


def write_validation_report(
    result: InventoryResult, config: InventoryConfig
) -> Path:
    """Write the validation report (CSV + Markdown).

    Every issue recorded during the run appears here, including
    INFO-level ones (fuzzy cross-section matches, multiple
    annotations) - "nothing vanishes silently" applies to the report
    just as much as to the manifest.

    Parameters
    ----------
    result : InventoryResult
    config : InventoryConfig

    Returns
    -------
    Path
        Path to the Markdown report (the CSV is written alongside it,
        same stem).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / "validation_report.csv"
    md_path = config.output_dir / "validation_report.md"

    issues_df = pd.DataFrame([
        {
            "level": i.level.value,
            "code": i.code,
            "image_ref": i.image_ref,
            "detail": i.detail,
        }
        for i in result.issues
    ])
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=["level", "code", "image_ref", "detail"]
        )
    issues_df.to_csv(csv_path, index=False, na_rep="")

    _write_validation_markdown(result.issues, md_path)
    logger.info(
        "Wrote validation report: %s (%d issues)",
        md_path, len(result.issues),
    )
    return md_path


def _write_validation_markdown(issues: list[Issue], path: Path) -> None:
    """Render issues grouped by code, most frequent first."""
    lines = ["# Raport walidacji inwentaryzacji danych", ""]
    by_level: dict[str, list[Issue]] = {}
    for issue in issues:
        by_level.setdefault(issue.level.value, []).append(issue)

    for level in ("FATAL", "ERROR", "WARNING", "INFO"):
        level_issues = by_level.get(level, [])
        lines.append(f"## {level} ({len(level_issues)})")
        lines.append("")
        if not level_issues:
            lines.append("_brak_")
            lines.append("")
            continue
        by_code = Counter(i.code for i in level_issues)
        for code, count in by_code.most_common():
            lines.append(f"### `{code}` ({count})")
            lines.append("")
            lines.append("| image_ref | detail |")
            lines.append("|---|---|")
            for issue in level_issues:
                if issue.code != code:
                    continue
                detail = issue.detail.replace("|", "\\|")
                lines.append(f"| {issue.image_ref} | {detail} |")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_summary(
    result: InventoryResult, config: InventoryConfig
) -> Path:
    """Write the dataset summary (Markdown + CSV).

    Parameters
    ----------
    result : InventoryResult
    config : InventoryConfig

    Returns
    -------
    Path
        Path to the Markdown summary (the CSV is written alongside
        it, same stem).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = config.output_dir / "dataset_summary.md"
    csv_path = config.output_dir / "dataset_summary.csv"

    df = result.manifest
    lines = ["# Podsumowanie zbioru", ""]

    if df.empty:
        lines.append("Manifest jest pusty.")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        pd.DataFrame().to_csv(csv_path, index=False)
        return md_path

    counts_rows = []
    lines.append("## Liczebności")
    lines.append("")
    for column in (
        "series", "material", "formulation", "magnification",
        "cross_section", "instrument", "microscope", "scale_bin",
    ):
        lines.append(f"### per `{column}`")
        lines.append("")
        counts = df[column].value_counts(dropna=False).sort_index()
        lines.append("| wartość | liczba |")
        lines.append("|---|---:|")
        for value, count in counts.items():
            lines.append(f"| {value} | {count} |")
            counts_rows.append(
                {"dimension": column, "value": value, "count": count}
            )
        lines.append("")

    lines.append("## Rozkład `pixel_size_um`")
    lines.append("")
    px_counts = df["pixel_size_um"].value_counts(dropna=False).sort_index()
    lines.append("| pixel_size_um | liczba |")
    lines.append("|---|---:|")
    for value, count in px_counts.items():
        lines.append(f"| {value} | {count} |")
    lines.append("")

    non_outlier = df.loc[~df["scale_outlier"], "pixel_size_um"].dropna()
    if not non_outlier.empty:
        q_max_no_outliers = non_outlier.max() / non_outlier.min()
        lines.append(
            f"`q_max` (bez `scale_outlier`) = "
            f"`{q_max_no_outliers:.4g}`"
        )
        lines.append("")
    all_px = df["pixel_size_um"].dropna()
    if not all_px.empty:
        q_max_all = all_px.max() / all_px.min()
        lines.append(f"`q_max` (wszystkie obrazy) = `{q_max_all:.4g}`")
        lines.append("")

    lines.append("## Rozkład powiększeń między formulacjami")
    lines.append("")
    mag_by_formulation = (
        df.dropna(subset=["magnification"])
        .groupby(["magnification", "formulation"])
        .size()
        .reset_index(name="n_images")
    )
    lines.append("| magnification | formulation | n_images |")
    lines.append("|---:|---|---:|")
    for _, r in mag_by_formulation.iterrows():
        lines.append(
            f"| {r['magnification']:.0f} | {r['formulation']} | "
            f"{r['n_images']} |"
        )
    lines.append("")

    lines.append("## Statystyki `n_instances` i średnic ekwiwalentnych")
    lines.append("")
    n_inst_stats = calculate_statistics(df["n_instances"].tolist())
    lines.append(f"`n_instances`: {n_inst_stats}")
    lines.append("")
    diam_stats = calculate_statistics(
        df["pore_equivalent_diameter_median_px"].dropna().tolist()
    )
    lines.append(f"`pore_equivalent_diameter_median_px`: {diam_stats}")
    lines.append("")
    p1_area_note, p1_area_value = _approximate_p1_area_px2(df)
    lines.append(
        f"Przybliżony P1 percentyl powierzchni instancji (px^2), z "
        f"median-diametrów per obraz, NIE per-instancja: "
        f"`{p1_area_value:.4g}` -- {p1_area_note}"
    )
    lines.append("")

    lines.append("## Elementy nieobrazowe")
    lines.append("")
    scalebar_counts = df.groupby("series")["scalebar_present"].sum()
    lines.append("| series | n_obrazow_z_paskiem |")
    lines.append("|---|---:|")
    for series, count in scalebar_counts.items():
        lines.append(f"| {series} | {int(count)} |")
    lines.append("")

    lines.append("## Formulacje wymagające uwagi przy splicie")
    lines.append("")
    per_formulation = df.groupby("formulation").size()
    single_image = sorted(per_formulation[per_formulation == 1].index)
    lines.append(f"Formulacje z 1 obrazem: {single_image or 'brak'}")
    lines.append("")
    formulations_with_50x = set(
        df.loc[df["magnification"] == 50, "formulation"].unique()
    )
    all_formulations = set(df["formulation"].unique())
    without_50x = sorted(all_formulations - formulations_with_50x)
    lines.append(f"Formulacje bez 50x: {without_50x}")
    lines.append("")
    outlier_formulations = sorted(
        df.loc[df["scale_outlier"], "formulation"].unique()
    )
    lines.append(
        f"Formulacje z co najmniej jednym `scale_outlier`: "
        f"{outlier_formulations or 'brak'}"
    )
    lines.append("")

    lines.append("## Stan punktów XVI.2")
    lines.append("")
    lines.append("| Punkt | Stan |")
    lines.append("|---|---|")
    lines.append(
        "| Wartości pixel_size 40x/50x | zamkniety - patrz rozkład "
        "pixel_size_um powyzej |"
    )
    lines.append(
        "| Rozkład 50x między formulacjami | zamkniety - patrz sekcja "
        "powyzej |"
    )
    lines.append(
        f"| Czy rodzaj pianki jest identyfikowalny | tak - materialy: "
        f"{sorted(df['material'].unique())} |"
    )
    dims = df[["width_px", "height_px"]].value_counts()
    lines.append(
        f"| Rozdzielczosc zrodlowa | faktyczne wymiary: "
        f"{dict(dims)} |"
    )
    lines.append("")

    pd.DataFrame(counts_rows).to_csv(csv_path, index=False)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote dataset summary: %s", md_path)
    return md_path


def _approximate_p1_area_px2(df: pd.DataFrame) -> tuple[str, float]:
    """Rough P1-percentile proxy for instance area, from per-image
    median diameters (the manifest stores no per-instance areas).
    """
    diam = df["pore_equivalent_diameter_median_px"].dropna()
    if diam.empty:
        return "brak danych", 0.0
    areas = math.pi * (diam / 2.0) ** 2
    p1 = areas.quantile(0.01)
    note = (
        "przyblizenie z median-diametrow per obraz; prawdziwy "
        "A_min_fragment (XVI.1) wymaga rozkladu powierzchni "
        "PER INSTANCJA z TRAIN, po ustaleniu splitu"
    )
    return note, float(p1)


def write_run_metadata(
    result: InventoryResult,
    config: InventoryConfig,
    manifest_path: Path,
) -> Path:
    """Write run_metadata.json: git commit, library versions, input
    paths, the configuration used, a UTC timestamp, row/rejection
    counts, and the manifest's own SHA-256 hash.

    Parameters
    ----------
    result : InventoryResult
    config : InventoryConfig
    manifest_path : Path
        The just-written manifest CSV, hashed for the record.

    Returns
    -------
    Path
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "run_metadata.json"

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "library_versions": _library_versions(),
        "manifest_version": config.manifest_version,
        "manifest_sha256": _sha256_of(manifest_path),
        "n_rows": len(result.manifest),
        "n_rejected": len(result.rejected),
        "n_issues": len(result.issues),
        "sources": [
            {
                "series": s.series,
                "images_dir": str(s.images_dir),
                "label_studio_json": str(s.label_studio_json),
                "sem_metadata_dirs": [
                    str(d) for d in s.sem_metadata_dirs
                ],
            }
            for s in config.sources
        ],
        "config": {
            "scale_outlier_ratio": config.scale_outlier_ratio,
            "fuzzy_cutoff": config.fuzzy_cutoff,
            "nonimage_extreme_fraction": config.nonimage_extreme_fraction,
            "nonimage_max_band_fraction":
                config.nonimage_max_band_fraction,
            "overlap_significant_fraction":
                config.overlap_significant_fraction,
            "pixel_size_tolerance": config.pixel_size_tolerance,
        },
        # Fixed, code-level constants (not YAML-tunable) - recorded
        # here so the manifest stays self-describing even as these
        # values evolve across manifest_version bumps.
        "derivation_rules": {
            "version": "v2",
            "scale_bin_coarse_min_um": SCALE_BIN_COARSE_MIN_UM,
            "scale_bin_fine_min_um": SCALE_BIN_FINE_MIN_UM,
            "q_reference_um": Q_REFERENCE_UM,
            "instrument_to_microscope": INSTRUMENT_TO_MICROSCOPE,
            "panel_height_rows_by_microscope":
                PANEL_HEIGHT_ROWS_BY_MICROSCOPE,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote run metadata: %s", path)
    return path


def _git_commit() -> Optional[str]:
    """Return the current git commit hash, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _library_versions() -> dict[str, Optional[str]]:
    """Return installed versions of the libraries this pipeline
    depends on, None for any that cannot be resolved."""
    versions: dict[str, Optional[str]] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_thumbnails(
    result: InventoryResult, config: InventoryConfig, n: int
) -> Path:
    """Save PNG thumbnails with the detected ``content_bbox`` drawn
    on top, for visual verification of the non-image region detector.

    Samples deterministically (every k-th row by ``image_id``, not
    randomly) so re-running with the same ``n`` reviews the same
    images.

    Parameters
    ----------
    result : InventoryResult
    config : InventoryConfig
    n : int
        Number of images to sample.

    Returns
    -------
    Path
        The thumbnails directory.
    """
    out_dir = config.output_dir / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result.manifest.sort_values("image_id").reset_index(drop=True)
    if df.empty or n <= 0:
        return out_dir
    step = max(1, len(df) // n)
    sampled = df.iloc[::step].head(n)

    for _, row in sampled.iterrows():
        try:
            with Image.open(row["source_path"]) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                x0, y0, x1, y1 = (
                    int(v) for v in row["content_bbox"].split(",")
                )
                draw.rectangle(
                    [x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=2
                )
                img.save(out_dir / f"{row['image_id']}.png")
        except Exception as e:  # noqa: BLE001 - best-effort visual aid
            logger.warning(
                "Could not render thumbnail for %s: %s",
                row["image_id"], e,
            )
    logger.info("Wrote %d thumbnail(s) to %s", len(sampled), out_dir)
    return out_dir
