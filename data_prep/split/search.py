"""
Deterministic search for the TRAIN/VALIDATION/TEST assignment of
formulations.

Why a search and not ``StratifiedGroupKFold``: the stratification
cell ``VAB x coarse`` contains exactly one formulation (VAB1, the only
VAB formulation with 30x images), which no three-way stratified
splitter can distribute; the plan additionally imposes hard conditions
that a splitter has no way to express (both scale bins present in
every set, at least one M2 formulation per set, minimum cross-section
sizes), and it needs the balance measured on *images and instances*
while the group quota is expressed in *formulations*.

The procedure is: fix the formulation quota per material, generate
candidates by seeded shuffling, reject those failing any hard
condition, and take the argmin of the balance cost. Seed, candidate
count and cost weights are recorded with the split, so the result is
reproducible from the artifacts alone.
"""
import logging
import random
from typing import Mapping, Sequence

from data_prep.split.constraints import aggregate, is_admissible
from data_prep.split.cost import DatasetTotals, split_cost
from data_prep.split.models import (SETS, FormulationProfile, SplitConfig,
                                    SplitResult)

logger = logging.getLogger(__name__)


class SplitSearchError(RuntimeError):
    """Raised when no admissible candidate split could be produced."""


def validate_quotas(
    profiles: Sequence[FormulationProfile], config: SplitConfig
) -> None:
    """Check the configured quota against the actual formulations.

    Parameters
    ----------
    profiles : Sequence of FormulationProfile
    config : SplitConfig

    Raises
    ------
    SplitSearchError
        If a material's quota does not sum to its formulation count,
        if a material present in the data has no quota (or vice
        versa), if a forced-TRAIN formulation does not exist, or if a
        material's TRAIN quota cannot absorb its forced formulations.
    """
    by_material: dict[str, list[str]] = {}
    for p in profiles:
        by_material.setdefault(p.material, []).append(p.formulation)

    known = set(by_material)
    configured = set(config.quotas)
    if known != configured:
        raise SplitSearchError(
            f"Quota materials {sorted(configured)} do not match the "
            f"manifest materials {sorted(known)}"
        )

    all_formulations = {p.formulation for p in profiles}
    unknown_forced = set(config.forced_train) - all_formulations
    if unknown_forced:
        raise SplitSearchError(
            f"forced_train names unknown formulation(s): "
            f"{sorted(unknown_forced)}"
        )

    for material, formulations in sorted(by_material.items()):
        quota = config.quotas[material]
        if sum(quota) != len(formulations):
            raise SplitSearchError(
                f"Quota for {material} sums to {sum(quota)} but the "
                f"manifest has {len(formulations)} formulation(s)"
            )
        n_forced = sum(
            1 for f in formulations if f in config.forced_train
        )
        if n_forced > quota[0]:
            raise SplitSearchError(
                f"Material {material} pins {n_forced} formulation(s) "
                f"to TRAIN but its TRAIN quota is {quota[0]}"
            )


def _generate_candidate(
    by_material: Mapping[str, list[str]],
    config: SplitConfig,
    rng: random.Random,
) -> dict[str, str]:
    """Draw one candidate assignment.

    Formulations pinned to TRAIN are removed from the pool first, so
    they never consume randomness and the remaining draw stays
    uniform over the formulations that are genuinely free.

    Parameters
    ----------
    by_material : Mapping[str, list[str]]
        Formulation names per material, in a fixed order.
    config : SplitConfig
    rng : random.Random

    Returns
    -------
    dict of str to str
        Formulation to set name.
    """
    assignment: dict[str, str] = {}
    for material, formulations in by_material.items():
        n_train, n_val, _ = config.quotas[material]
        forced = [f for f in formulations if f in config.forced_train]
        pool = [f for f in formulations if f not in config.forced_train]
        rng.shuffle(pool)

        for formulation in forced:
            assignment[formulation] = "train"
        free_train = n_train - len(forced)
        for formulation in pool[:free_train]:
            assignment[formulation] = "train"
        for formulation in pool[free_train:free_train + n_val]:
            assignment[formulation] = "val"
        for formulation in pool[free_train + n_val:]:
            assignment[formulation] = "test"
    return assignment


def search_split(
    profiles: Sequence[FormulationProfile], config: SplitConfig
) -> SplitResult:
    """Search for the best-balanced admissible split.

    Parameters
    ----------
    profiles : Sequence of FormulationProfile
    config : SplitConfig

    Returns
    -------
    SplitResult

    Raises
    ------
    SplitSearchError
        If the quota is inconsistent with the data, or no generated
        candidate satisfied every hard condition.
    """
    validate_quotas(profiles, config)

    by_material: dict[str, list[str]] = {}
    for p in profiles:
        by_material.setdefault(p.material, []).append(p.formulation)
    by_material = {
        material: sorted(names)
        for material, names in sorted(by_material.items())
    }

    totals = DatasetTotals(profiles)
    rng = random.Random(config.seed)

    best_cost = None
    best_assignment = None
    best_stats = None
    n_feasible = 0

    for _ in range(config.n_candidates):
        assignment = _generate_candidate(by_material, config, rng)
        stats = aggregate(assignment, profiles)
        if not is_admissible(stats, config.constraints):
            continue
        n_feasible += 1
        cost = split_cost(
            stats, totals, config.cost_weights, config.target_shares
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignment = assignment
            best_stats = stats

    if best_assignment is None or best_stats is None or best_cost is None:
        raise SplitSearchError(
            f"No admissible split among {config.n_candidates} "
            f"candidate(s) with seed {config.seed}. Relax the hard "
            f"constraints or revisit the quota - the conditions of "
            f"III.4 may be unsatisfiable for this manifest."
        )

    feasible_share = n_feasible / config.n_candidates
    logger.info(
        "Search: %d candidate(s), %d admissible (%.1f%%), best cost "
        "%.4f.",
        config.n_candidates, n_feasible, 100 * feasible_share, best_cost,
    )
    if feasible_share < 0.01:
        logger.warning(
            "Only %.2f%% of candidates were admissible: the hard "
            "constraints are close to infeasible, so the chosen split "
            "sits in a narrow corner of the space.",
            100 * feasible_share,
        )

    for name in SETS:
        s = best_stats[name]
        logger.info(
            "%-5s: %2d formulation(s), %3d image(s) (%3d evaluable: "
            "%3d coarse / %2d fine), %5d instance(s).",
            name.upper(), len(s.formulations), s.n_images,
            s.n_eval_images, s.n_coarse, s.n_fine, s.n_instances,
        )

    return SplitResult(
        assignment=best_assignment,
        cost=best_cost,
        stats=best_stats,
        profiles=tuple(profiles),
        n_generated=config.n_candidates,
        n_feasible=n_feasible,
    )
