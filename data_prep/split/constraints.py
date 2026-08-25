"""
Aggregation of a candidate assignment into per-set statistics, and the
hard conditions a split must satisfy to be usable at all.

The conditions encode what the evaluation has to be able to report: a
set that lacks one of the microscopes cannot measure cross-microscope
transfer, and a set that lacks one of the scale bins cannot show
whether the model generalizes across resolutions. Constraints are
checked before any scoring, so an inadmissible candidate is never
compared against an admissible one.
"""
import logging
from typing import Mapping, Sequence

from data_prep.split.models import (SETS, FormulationProfile, SetStats,
                                    SplitConstraints)

logger = logging.getLogger(__name__)

M2_MATERIALS = ("K", "VAB")

EVAL_SETS = ("val", "test")


def aggregate(
    assignment: Mapping[str, str],
    profiles: Sequence[FormulationProfile],
) -> dict[str, SetStats]:
    """Aggregate a candidate assignment into per-set statistics.

    Parameters
    ----------
    assignment : Mapping[str, str]
        Formulation to set name.
    profiles : Sequence of FormulationProfile

    Returns
    -------
    dict of str to SetStats
        Keyed by set name; every set in ``SETS`` is present, even when
        empty.
    """
    members: dict[str, list[FormulationProfile]] = {s: [] for s in SETS}
    for profile in profiles:
        members[assignment[profile.formulation]].append(profile)

    stats = {}
    for name, group in members.items():
        by_material: dict[str, int] = {}
        by_cell: dict[tuple[str, str], int] = {}
        for p in group:
            by_material[p.material] = (
                by_material.get(p.material, 0) + p.n_eval_images
            )
            for scale_bin, count in (
                ("coarse", p.n_coarse), ("fine", p.n_fine)
            ):
                key = (p.material, scale_bin)
                by_cell[key] = by_cell.get(key, 0) + count

        stats[name] = SetStats(
            name=name,
            formulations=tuple(sorted(p.formulation for p in group)),
            n_images=sum(p.n_images for p in group),
            n_eval_images=sum(p.n_eval_images for p in group),
            n_coarse=sum(p.n_coarse for p in group),
            n_fine=sum(p.n_fine for p in group),
            n_outlier=sum(p.n_outlier for p in group),
            n_instances=sum(p.n_instances for p in group),
            n_m2_formulations=sum(
                1 for p in group if p.material in M2_MATERIALS
            ),
            images_by_material=by_material,
            images_by_cell=by_cell,
        )
    return stats


def check_constraints(
    stats: Mapping[str, SetStats], constraints: SplitConstraints
) -> list[str]:
    """Check a candidate split against the hard conditions.

    Parameters
    ----------
    stats : Mapping[str, SetStats]
        Output of ``aggregate``.
    constraints : SplitConstraints

    Returns
    -------
    list of str
        Human-readable violations, empty when the candidate is
        admissible. The search only needs to know whether the list is
        empty; the report renders it for the chosen split, where it is
        expected to be empty and is printed as positive evidence.
    """
    violations = []

    for name in SETS:
        s = stats[name]
        if s.n_m2_formulations < constraints.min_m2_formulations_per_set:
            violations.append(
                f"{name}: {s.n_m2_formulations} M2 formulation(s), "
                f"needs >= {constraints.min_m2_formulations_per_set} "
                f"(cross-microscope transfer would be unmeasured)"
            )
        for scale_bin, count in (
            ("coarse", s.n_coarse), ("fine", s.n_fine)
        ):
            if count < constraints.min_scale_bin_images_per_set:
                violations.append(
                    f"{name}: {count} '{scale_bin}' image(s), needs "
                    f">= {constraints.min_scale_bin_images_per_set}"
                )

    for name in EVAL_SETS:
        s = stats[name]
        if s.n_fine < constraints.min_eval_fine_images:
            violations.append(
                f"{name}: {s.n_fine} 'fine' image(s), needs >= "
                f"{constraints.min_eval_fine_images} for a reportable "
                f"per-scale_bin cross-section"
            )
        for material, minimum in (
            constraints.min_eval_images_by_material.items()
        ):
            count = s.images_by_material.get(material, 0)
            if count < minimum:
                violations.append(
                    f"{name}: {count} evaluable {material} image(s), "
                    f"needs >= {minimum}"
                )

    return violations


def is_admissible(
    stats: Mapping[str, SetStats], constraints: SplitConstraints
) -> bool:
    """Whether a candidate satisfies every hard condition.

    Parameters
    ----------
    stats : Mapping[str, SetStats]
    constraints : SplitConstraints

    Returns
    -------
    bool
    """
    return not check_constraints(stats, constraints)
