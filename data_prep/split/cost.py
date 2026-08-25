"""
The balance cost that ranks admissible candidate splits.

Every term is an L1 deviation of an achieved share from the target
share of a set, so terms are on one scale and their weights are
directly comparable. The cost expresses only *balance*: it never sees
a metric, a model or an image, so minimizing it cannot bias the
experiment's outcome - it can only make the reported cross-sections
better proportioned.
"""
import logging
from typing import Mapping, Sequence

from data_prep.split.models import (SETS, CostWeights, FormulationProfile,
                                    SetStats)

logger = logging.getLogger(__name__)

EVAL_SETS = ("val", "test")


class DatasetTotals:
    """Dataset-level denominators the cost normalizes against.

    Computed once per run from the formulation profiles, then reused
    for every candidate.

    Parameters
    ----------
    profiles : Sequence of FormulationProfile
    """

    def __init__(self, profiles: Sequence[FormulationProfile]) -> None:
        self.eval_images = sum(p.n_eval_images for p in profiles)
        self.instances = sum(p.n_instances for p in profiles)
        self.cells: dict[tuple[str, str], int] = {}
        for p in profiles:
            for scale_bin, count in (
                ("coarse", p.n_coarse), ("fine", p.n_fine)
            ):
                if count:
                    key = (p.material, scale_bin)
                    self.cells[key] = self.cells.get(key, 0) + count


def split_cost(
    stats: Mapping[str, SetStats],
    totals: DatasetTotals,
    weights: CostWeights,
    target_shares: Mapping[str, float],
) -> float:
    """Score one admissible candidate; lower is better.

    Parameters
    ----------
    stats : Mapping[str, SetStats]
        Output of ``constraints.aggregate``.
    totals : DatasetTotals
    weights : CostWeights
    target_shares : Mapping[str, float]
        Target share per set name.

    Returns
    -------
    float
    """
    cost = 0.0
    for name in SETS:
        s = stats[name]
        target = target_shares[name]

        if totals.eval_images:
            cost += weights.images * abs(
                s.n_eval_images / totals.eval_images - target
            )
        if totals.instances:
            cost += weights.instances * abs(
                s.n_instances / totals.instances - target
            )

        for (material, scale_bin), total in totals.cells.items():
            achieved = s.images_by_cell.get((material, scale_bin), 0)
            cost += weights.cell_weight(material, scale_bin) * abs(
                achieved / total - target
            )

        if name in EVAL_SETS:
            cost += weights.lost_outlier_image * s.n_outlier

    return cost
