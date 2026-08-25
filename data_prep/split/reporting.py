"""
Split artifacts: the per-image assignment table, the human-readable
split report, and the run metadata that makes the split reproducible.

None of this module decides anything about the split - it renders what
``search.search_split`` already chose. The report is the deliverable
required by the plan's section III.5, steps 7 and 8: the per-set
distribution of materials, microscopes and scale bins with counts, and
an explicit verification of the hard conditions of III.4.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from data_prep.run_provenance import (git_commit, library_versions,
                                      python_version, sha256_of)
from data_prep.split.constraints import check_constraints
from data_prep.split.fragment_area import format_summary
from data_prep.split.models import (SETS, FragmentAreaResult, SplitConfig,
                                    SplitResult)

logger = logging.getLogger(__name__)

MATERIALS_ORDER = ("AS", "K", "VAB")

SET_LABELS = {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}


class SplitExistsError(FileExistsError):
    """Raised when writing would overwrite an already frozen split."""


def write_split_table(
    manifest: pd.DataFrame, result: SplitResult, config: SplitConfig,
    *, overwrite: bool = False,
) -> Path:
    """Write the per-image split assignment as CSV.

    One row per manifest image. ``scale_outlier`` images outside TRAIN
    carry ``used = False``: grouping by formulation wins over the
    plan's "outliers stay in TRAIN" rule, because moving such an image
    on its own would put one formulation in two sets and reintroduce
    the leakage of III.3. The column makes that loss explicit and
    countable rather than implicit.

    Parameters
    ----------
    manifest : pandas.DataFrame
    result : SplitResult
    config : SplitConfig
    overwrite : bool, optional
        Allow replacing an existing file of the same ``split_id``.

    Returns
    -------
    Path

    Raises
    ------
    SplitExistsError
        If the target exists and ``overwrite`` is False.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.split_id}.csv"
    if path.exists() and not overwrite:
        raise SplitExistsError(
            f"Split {config.split_id} already exists at {path}. A "
            f"frozen split is never overwritten - bump split_id, or "
            f"pass --overwrite if you are deliberately replacing an "
            f"unused draft."
        )

    table = manifest[[
        "image_id", "formulation", "material", "microscope",
        "scale_bin", "pixel_size_um", "n_instances", "source_path",
    ]].copy()
    table["split"] = table["formulation"].map(result.assignment)
    table["used"] = ~(
        (table["scale_bin"] == "outlier") & (table["split"] != "train")
    )
    table = table.sort_values(["split", "formulation", "image_id"])
    table.to_csv(path, index=False, na_rep="")

    n_dropped = int((~table["used"]).sum())
    logger.info(
        "Wrote split table: %s (%d row(s), %d dropped scale_outlier "
        "image(s)).", path, len(table), n_dropped,
    )
    return path


def write_split_report(
    result: SplitResult,
    config: SplitConfig,
    fragment_area: Optional[FragmentAreaResult],
) -> Path:
    """Write the Markdown split report (plan III.5, steps 7 and 8).

    Parameters
    ----------
    result : SplitResult
    config : SplitConfig
    fragment_area : FragmentAreaResult, optional

    Returns
    -------
    Path
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.split_id}_report.md"

    lines = [
        f"# Podzial danych: `{config.split_id}`",
        "",
        f"Manifest: `{config.manifest_path}`",
        "",
        f"Seed podzialu: `{config.seed}` | kandydatow: "
        f"{result.n_generated} | dopuszczalnych: {result.n_feasible} "
        f"({100 * result.n_feasible / result.n_generated:.1f}%) | "
        f"koszt: {result.cost:.4f}",
        "",
        "Jednostka grupowania: **formulacja** (plan III.3). Zaden "
        "obraz jednej formulacji nie trafia do dwoch zbiorow.",
        "",
    ]
    lines += _liczebnosci_section(result)
    lines += _przekroje_section(result)
    lines += _warunki_section(result, config)
    lines += _formulacje_section(result)
    lines += format_summary(fragment_area)

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote split report: %s", path)
    return path


def _liczebnosci_section(result: SplitResult) -> list[str]:
    """Per-set totals, including the evaluable/outlier breakdown."""
    total_eval = sum(s.n_eval_images for s in result.stats.values())
    total_inst = sum(s.n_instances for s in result.stats.values())
    lines = [
        "## Liczebnosci",
        "",
        "| zbior | formulacji | obrazow | do oceny | udzial obrazow | "
        "coarse | fine | outlier | instancji | udzial instancji |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SETS:
        s = result.stats[name]
        lines.append(
            f"| {SET_LABELS[name]} | {len(s.formulations)} | "
            f"{s.n_images} | {s.n_eval_images} | "
            f"{100 * s.n_eval_images / total_eval:.1f}% | "
            f"{s.n_coarse} | {s.n_fine} | {s.n_outlier} | "
            f"{s.n_instances} | "
            f"{100 * s.n_instances / total_inst:.1f}% |"
        )
    lines.append("")
    lines.append(
        "Kolumna `do oceny` pomija obrazy `scale_outlier`. Poza TRAIN "
        "sa one odrzucane w calosci - patrz kolumna `used` w tabeli "
        "splitu."
    )
    lines.append("")
    return lines


def _przekroje_section(result: SplitResult) -> list[str]:
    """Per-set cross-sections by material, microscope and scale bin."""
    lines = ["## Przekroje raportowania", "", "### Obrazy per `material`",
             "", "| zbior | " + " | ".join(MATERIALS_ORDER) + " |",
             "|---|" + "---:|" * len(MATERIALS_ORDER)]
    for name in SETS:
        s = result.stats[name]
        cells = " | ".join(
            str(s.images_by_material.get(m, 0)) for m in MATERIALS_ORDER
        )
        lines.append(f"| {SET_LABELS[name]} | {cells} |")

    lines += ["", "### Obrazy per `material` x `scale_bin`", ""]
    cells_present = sorted({
        cell
        for s in result.stats.values()
        for cell in s.images_by_cell
    })
    header = " | ".join(f"{m}:{b}" for m, b in cells_present)
    lines += [
        f"| zbior | {header} |",
        "|---|" + "---:|" * len(cells_present),
    ]
    for name in SETS:
        s = result.stats[name]
        row = " | ".join(
            str(s.images_by_cell.get(cell, 0)) for cell in cells_present
        )
        lines.append(f"| {SET_LABELS[name]} | {row} |")

    lines += ["", "### Formulacje per mikroskop", "",
              "| zbior | M1 | M2 |", "|---|---:|---:|"]
    for name in SETS:
        s = result.stats[name]
        m2 = s.n_m2_formulations
        m1 = len(s.formulations) - m2
        lines.append(f"| {SET_LABELS[name]} | {m1} | {m2} |")
    lines += [
        "",
        "Rodzina materialu jest skonfundowana z mikroskopem (AS = M1, "
        "K i VAB = M2), wiec przekroj per `material` jest jednoczesnie "
        "przekrojem per mikroskop i roznic nie wolno interpretowac "
        "jako czysto materialowych (plan III.1).",
        "",
    ]
    return lines


def _warunki_section(
    result: SplitResult, config: SplitConfig
) -> list[str]:
    """Explicit verification of the hard conditions (III.5, step 8)."""
    violations = check_constraints(result.stats, config.constraints)
    lines = ["## Weryfikacja twardych warunkow (III.4)", ""]
    if violations:
        lines += [
            "**NIESPELNIONE** - splitu nie wolno zamrozic:", "",
        ]
        lines += [f"- {v}" for v in violations]
        lines.append("")
        return lines

    c = config.constraints
    checks = [
        f">= {c.min_m2_formulations_per_set} formulacja M2 w kazdym "
        f"zbiorze",
        f"oba biny skali obecne w kazdym zbiorze "
        f"(>= {c.min_scale_bin_images_per_set} obraz)",
        f">= {c.min_eval_fine_images} obrazow `fine` w VALIDATION "
        f"i w TEST",
    ]
    checks += [
        f">= {minimum} obrazow oceny {material} w VALIDATION i w TEST"
        for material, minimum in sorted(
            c.min_eval_images_by_material.items()
        )
    ]
    lines += [f"- SPELNIONY: {check}" for check in checks]
    lines.append("")
    return lines


def _formulacje_section(result: SplitResult) -> list[str]:
    """The assignment itself, one table per set."""
    by_name = {p.formulation: p for p in result.profiles}
    lines = ["## Przypisanie formulacji", ""]
    for name in SETS:
        s = result.stats[name]
        lines += [
            f"### {SET_LABELS[name]} ({len(s.formulations)} "
            f"formulacji)",
            "",
            "| formulacja | material | mikroskop | obrazow | coarse | "
            "fine | outlier | instancji |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for formulation in s.formulations:
            p = by_name[formulation]
            lines.append(
                f"| {p.formulation} | {p.material} | {p.microscope} | "
                f"{p.n_images} | {p.n_coarse} | {p.n_fine} | "
                f"{p.n_outlier} | {p.n_instances} |"
            )
        lines.append("")
    return lines


def write_split_metadata(
    result: SplitResult,
    config: SplitConfig,
    split_table_path: Path,
    fragment_area: Optional[FragmentAreaResult],
) -> Path:
    """Write ``<split_id>_metadata.json``.

    Records everything needed to regenerate the split from scratch:
    the manifest it was derived from (by hash), the seed, the
    candidate count, the quota, the constraints and the cost weights.
    Also carries ``A_min_fragment``, so the frozen value travels with
    the split that defined it.

    Parameters
    ----------
    result : SplitResult
    config : SplitConfig
    split_table_path : Path
    fragment_area : FragmentAreaResult, optional

    Returns
    -------
    Path
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.split_id}_metadata.json"

    c = config.constraints
    w = config.cost_weights
    metadata: dict[str, Any] = {
        "split_id": config.split_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": python_version(),
        "library_versions": library_versions(),
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": sha256_of(config.manifest_path),
        "split_table_sha256": sha256_of(split_table_path),
        "grouping_unit": "formulation",
        "seed": config.seed,
        "n_candidates": config.n_candidates,
        "n_feasible": result.n_feasible,
        "cost": result.cost,
        "selection_rule": "argmin_balance_cost",
        "quotas": {m: list(q) for m, q in sorted(config.quotas.items())},
        "forced_train": list(config.forced_train),
        "target_shares": dict(config.target_shares),
        "constraints": {
            "min_m2_formulations_per_set":
                c.min_m2_formulations_per_set,
            "min_scale_bin_images_per_set":
                c.min_scale_bin_images_per_set,
            "min_eval_fine_images": c.min_eval_fine_images,
            "min_eval_images_by_material":
                dict(c.min_eval_images_by_material),
        },
        "cost_weights": {
            "images": w.images,
            "instances": w.instances,
            "cell_default": w.cell_default,
            "cell_overrides": dict(w.cell_overrides),
            "lost_outlier_image": w.lost_outlier_image,
        },
        "outlier_policy": (
            "grouping wins: a scale_outlier image follows its "
            "formulation and is dropped entirely (used=False) when "
            "that formulation is not in TRAIN"
        ),
        "assignment": {
            name: list(result.stats[name].formulations) for name in SETS
        },
        "counts": {
            name: {
                "n_formulations": len(result.stats[name].formulations),
                "n_images": result.stats[name].n_images,
                "n_eval_images": result.stats[name].n_eval_images,
                "n_coarse": result.stats[name].n_coarse,
                "n_fine": result.stats[name].n_fine,
                "n_outlier": result.stats[name].n_outlier,
                "n_instances": result.stats[name].n_instances,
            }
            for name in SETS
        },
        "a_min_fragment": (
            None if fragment_area is None else {
                "value_px2": fragment_area.a_min_fragment_px2,
                "percentile": fragment_area.percentile,
                "measured_on": "train",
                "measured_after": "load_crop_bbox",
                "n_images": fragment_area.n_images,
                "n_instances": fragment_area.n_instances,
                "excluded_scale_outlier":
                    fragment_area.excluded_scale_outlier,
                "value_including_outliers_px2":
                    fragment_area.value_including_outliers_px2,
                "by_scale_bin_px2":
                    dict(fragment_area.by_scale_bin_px2),
                "n_instances_lost_to_crop":
                    fragment_area.n_instances_lost_to_crop,
            }
        ),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote split metadata: %s", path)
    return path


def format_console_summary(result: SplitResult) -> str:
    """Render a compact per-set summary for the console.

    Parameters
    ----------
    result : SplitResult

    Returns
    -------
    str
    """
    rows = []
    for name in SETS:
        s = result.stats[name]
        materials = ", ".join(
            f"{m} {s.images_by_material.get(m, 0)}"
            for m in MATERIALS_ORDER
        )
        rows.append(
            f"  {SET_LABELS[name]:<10} {len(s.formulations):>2} form."
            f"  {s.n_eval_images:>3} obr. oceny "
            f"({s.n_coarse} coarse / {s.n_fine} fine)  [{materials}]"
        )
    return "\n".join(rows)
