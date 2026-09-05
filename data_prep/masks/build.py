"""
Building every image's instance mask file from the annotation export.

Runs once per annotation export, not once per training run. The output
is a third frozen artifact alongside the manifest and the split, and
is recorded the same way: the rules that produced it, the hash of the
manifest it was built against, and a per-image tally of what the
rasterization cost.

Only pore polygons become instances. The export also holds node
polygons - the solid junctions between struts - and the per-image
tally records how many were dropped, so a mask holding fewer
instances than the export holds polygons is explained by the artifact
itself.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import tifffile

from data_prep.annotations import (load_annotation_index, polygons_in_pixels,
                                   require_annotation)
from data_prep.inventory.issues import IssueCollector
from data_prep.inventory.label_studio import (CLASS_FILTER_RULE,
                                              count_excluded_polygons)
from data_prep.masks.rasterize import (CONNECTIVITY_RULE, MASK_DTYPE,
                                       OVERLAP_RULE, rasterize_instances)
from data_prep.run_provenance import (git_commit, library_versions,
                                      python_version, sha256_of)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "image_id", "series", "mask_annotation_id", "width_px", "height_px",
    "mask_path", "n_instances",
)

MASK_SUFFIX = "_masks.tif"


class MaskBuildError(RuntimeError):
    """Raised when the masks cannot be built from the given inputs."""


@dataclass(frozen=True)
class MaskBuildResult:
    """Outcome of a mask build.

    Parameters
    ----------
    per_image : pandas.DataFrame
        One row per image: what was written and what it cost.
    output_root : Path
        Directory the mask files were written under.
    n_written : int
        Mask files written.
    """

    per_image: pd.DataFrame
    output_root: Path
    n_written: int

    def totals(self) -> dict[str, int]:
        """Sum the per-image tallies.

        Returns
        -------
        dict of str to int
        """
        columns = (
            "n_node_polygons_excluded",
            "n_polygons", "n_instances", "n_vanished_polygons",
            "n_repaired_instances", "n_pieces_removed", "discarded_px",
            "overlap_px", "covered_px",
        )
        return {
            column: int(self.per_image[column].sum())
            for column in columns
        }


def mask_path_for(
    output_root: Path, series: str, image_id: str
) -> Path:
    """Return where one image's mask file belongs.

    Parameters
    ----------
    output_root : Path
    series : str
    image_id : str

    Returns
    -------
    Path
    """
    return output_root / series / f"{image_id}{MASK_SUFFIX}"


def build_masks(
    manifest: pd.DataFrame,
    exports: Mapping[str, Path],
    output_root: Path,
    *,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> MaskBuildResult:
    """Rasterize every manifest row's annotation into a mask file.

    Parameters
    ----------
    manifest : pandas.DataFrame
        Frozen inventory manifest.
    exports : Mapping[str, Path]
        Series name to its Label Studio export.
    output_root : Path
        Directory to write mask files under, one subdirectory per
        series.
    overwrite : bool, optional
        Allow replacing mask files that already exist.
    limit : int, optional
        Stop after this many images; for smoke-testing a build.

    Returns
    -------
    MaskBuildResult

    Raises
    ------
    MaskBuildError
        If the manifest lacks a required column, or a mask file exists
        and ``overwrite`` is False.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing:
        raise MaskBuildError(
            f"Manifest is missing required column(s): {missing}"
        )

    rows = manifest if limit is None else manifest.head(limit)
    index = load_annotation_index(
        exports, rows["series"].astype(str).unique().tolist()
    )
    collector = IssueCollector()
    records = []
    n_path_drift = 0

    for row in rows.itertuples(index=False):
        target = mask_path_for(
            output_root, str(row.series), str(row.image_id)
        )
        if target.exists() and not overwrite:
            raise MaskBuildError(
                f"Mask already exists: {target}. Masks are a frozen "
                f"artifact - pass --overwrite only when deliberately "
                f"rebuilding them."
            )
        recorded = getattr(row, "mask_path", None)
        if recorded and str(recorded) != str(target):
            n_path_drift += 1

        annotation = require_annotation(
            index, int(row.mask_annotation_id), str(row.image_id)
        )
        polygons = polygons_in_pixels(
            annotation, int(row.width_px), int(row.height_px),
            collector=collector, image_ref=str(row.image_id),
        )
        mask = rasterize_instances(
            polygons, (int(row.height_px), int(row.width_px))
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, mask.labels, compression="zlib")

        records.append({
            "image_id": row.image_id,
            "series": row.series,
            "mask_path": str(target),
            "height_px": int(row.height_px),
            "width_px": int(row.width_px),
            "n_node_polygons_excluded": count_excluded_polygons(
                annotation
            ),
            "n_polygons": mask.n_polygons,
            "n_instances": mask.n_instances,
            "n_instances_manifest": int(row.n_instances),
            "n_vanished_polygons": mask.n_vanished_polygons,
            "n_repaired_instances": mask.n_repaired_instances,
            "n_pieces_removed": mask.n_pieces_removed,
            "discarded_px": mask.discarded_px,
            "overlap_px": mask.overlap_px,
            "covered_px": mask.covered_px,
        })

    per_image = pd.DataFrame(records)
    logger.info(
        "Wrote %d mask file(s) under %s.", len(per_image), output_root
    )
    if n_path_drift:
        logger.warning(
            "%d mask(s) were written somewhere other than the "
            "mask_path the manifest records, so that column does not "
            "point at them. Expected when --output-root overrides the "
            "configured location.", n_path_drift,
        )
    _log_disagreements(per_image)
    return MaskBuildResult(
        per_image=per_image,
        output_root=output_root,
        n_written=len(per_image),
    )


def _log_disagreements(per_image: pd.DataFrame) -> None:
    """Report images whose instance count differs from the manifest.

    The manifest counted instances straight from the painted polygons.
    Reducing each instance to one connected piece cannot change that
    count, so a disagreement means the export or the manifest moved.
    """
    if per_image.empty:
        return
    differing = per_image[
        per_image["n_instances"] != per_image["n_instances_manifest"]
    ]
    if differing.empty:
        logger.info(
            "Instance counts agree with the manifest for all %d image(s).",
            len(per_image),
        )
        return
    logger.warning(
        "%d image(s) have an instance count differing from the "
        "manifest, e.g. %s.",
        len(differing),
        ", ".join(
            f"{r.image_id}: {r.n_instances} vs {r.n_instances_manifest}"
            for r in differing.head(5).itertuples(index=False)
        ),
    )


def write_build_artifacts(
    result: MaskBuildResult,
    manifest_path: Path,
    exports: Mapping[str, Path],
) -> dict[str, Path]:
    """Write the per-image tally and the build metadata.

    Parameters
    ----------
    result : MaskBuildResult
    manifest_path : Path
        Manifest the masks were built against, hashed for the record.
    exports : Mapping[str, Path]
        Label Studio exports, hashed for the record: masks are a
        deterministic function of these plus the manifest.

    Returns
    -------
    dict of str to Path
        Paths of the written artifacts.
    """
    result.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = result.output_root / "instance_masks.csv"
    json_path = result.output_root / "instance_masks_metadata.json"

    result.per_image.to_csv(csv_path, index=False)

    totals = result.totals()
    covered = totals["covered_px"] or 1
    metadata = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "git_commit": git_commit(),
        "python_version": python_version(),
        "library_versions": library_versions(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_of(manifest_path),
        "label_studio_exports": {
            name: {
                "path": str(path), "sha256": sha256_of(path),
            }
            for name, path in sorted(exports.items())
        },
        "output_root": str(result.output_root),
        "n_masks": result.n_written,
        "dtype": MASK_DTYPE.__name__,
        "rules": {
            "class_filter": CLASS_FILTER_RULE,
            "overlap": OVERLAP_RULE,
            "connectivity": CONNECTIVITY_RULE,
        },
        "totals": totals,
        "overlap_fraction_of_covered": totals["overlap_px"] / covered,
        "discarded_fraction_of_covered": (
            totals["discarded_px"] / covered
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Wrote %s and %s", csv_path, json_path)
    return {"per_image": csv_path, "metadata": json_path}


def summarize(result: MaskBuildResult) -> str:
    """Render a short human-readable summary of a build.

    Parameters
    ----------
    result : MaskBuildResult

    Returns
    -------
    str
    """
    totals = result.totals()
    covered = totals["covered_px"] or 1
    instances = totals["n_instances"] or 1
    return "\n".join([
        f"  masks written        {result.n_written}",
        f"  node polygons out    {totals['n_node_polygons_excluded']}",
        f"  polygons             {totals['n_polygons']}",
        f"  instances            {totals['n_instances']}",
        f"  vanished polygons    {totals['n_vanished_polygons']}",
        f"  repaired instances   {totals['n_repaired_instances']}"
        f" ({100 * totals['n_repaired_instances'] / instances:.2f}%)",
        f"  pieces removed       {totals['n_pieces_removed']}",
        f"  discarded pixels     {totals['discarded_px']}"
        f" ({100 * totals['discarded_px'] / covered:.4f}% of covered)",
        f"  overlapping pixels   {totals['overlap_px']}"
        f" ({100 * totals['overlap_px'] / covered:.4f}% of covered)",
    ])
