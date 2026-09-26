"""
The rules that choose post-processing settings from a scored grid.

Two calibrations use them - the seeding thresholds calibrated on
2026-09-14 and the grids of the optimization study - and they must
choose the same way, so the rules live here once rather than in each
script that sweeps a grid.

**F1 decides only where it decides anything.** Near the top of a grid
many settings land within a thousandth of each other, an order of
magnitude below what another sample of images would produce. Every
setting within one noise band of the best F1 is therefore treated as
tied with it, and the tied settings are told apart by how far the pore
count is off, then by how lopsided merging and splitting are.

The calibration of 2026-09-14 implemented the band differently, by
cutting F1 into fixed intervals of the band's width; two settings a
fraction of a band apart could then fall into different intervals and
never be compared on the pore count. That form is kept as
:func:`rank_key` so the old calibration can be reproduced; every choice
made since 2026-09-24 uses :func:`choose_within_band`.

**Wall agreement does not break ties.** It rises monotonically as a
setting seeds more freely, because every extra instance draws extra
wall, so as a tie-break it would push the choice towards whichever
setting splits most, regardless of whether the extra instances are
real.

**The foreground threshold is anchored, not searched.** For each
material the threshold is read at which matched pores are drawn neither
larger nor smaller than annotated - where the median signed diameter
drift crosses zero - and the materials are then compared with each
other and with the current value.
"""
import math
from typing import Mapping, Optional, Sequence

import numpy as np

from materials_vision.evaluation.aggregate import AggregateResult
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.watershed import WatershedParams


def boundary_at_decision_scale(result: AggregateResult) -> float:
    """Boundary agreement at the one tolerance decisions are made on.

    Parameters
    ----------
    result : AggregateResult

    Returns
    -------
    float
        Zero if the result was scored without that tolerance, which
        makes the last tie-break inert rather than fatal.
    """
    return float(result.boundary_f1.get(DECISION_SCALE, 0.0))


def f1_noise(result: AggregateResult) -> float:
    """Roughly how much this F1 would move on another sample this size.

    F1 treated as a proportion over the annotated instances. An
    approximation - it is a ratio of two counts that move together -
    but of the right order, which is all a tie band needs.

    Parameters
    ----------
    result : AggregateResult

    Returns
    -------
    float
    """
    if result.n_gt < 1:
        return 0.0
    share = min(max(result.f1, 0.0), 1.0)
    return math.sqrt(share * (1.0 - share) / result.n_gt)


def rank_key(
    result: AggregateResult, tolerance: float
) -> tuple[float, float, float, float]:
    """Sort key implementing the choice rule; best sorts last.

    Parameters
    ----------
    result : AggregateResult
    tolerance : float
        Width of an F1 band. Differences smaller than this are not
        differences.

    Returns
    -------
    tuple of float
        Negated where smaller is better, so sorting descending puts the
        best first.
    """
    band = result.f1 if tolerance <= 0 else round(result.f1 / tolerance)
    return (
        band,
        -abs(result.pore_count_error),
        -abs(result.merges_per_100_gt - result.splits_per_100_gt),
        boundary_at_decision_scale(result),
    )


def rank(
    results: Mapping[WatershedParams, AggregateResult], tolerance: float
) -> list[WatershedParams]:
    """Settings ordered best first by :func:`rank_key`.

    The 2026-09-14 form of the rule; see the module docstring.

    Parameters
    ----------
    results : mapping of WatershedParams to AggregateResult
    tolerance : float

    Returns
    -------
    list of WatershedParams
    """
    return sorted(
        results, key=lambda setting: rank_key(results[setting], tolerance),
        reverse=True,
    )


def choose_within_band(
    results: Mapping[WatershedParams, AggregateResult], band: float
) -> tuple[WatershedParams, list[WatershedParams]]:
    """Pick among the settings tied with the best F1.

    Parameters
    ----------
    results : mapping of WatershedParams to AggregateResult
    band : float
        Settings whose F1 is at most this far below the best are tied.

    Returns
    -------
    chosen : WatershedParams
    tied : list of WatershedParams
        Every tied setting, in the order the rule prefers them.
    """
    best_f1 = max(result.f1 for result in results.values())
    tied = [
        setting for setting, result in results.items()
        if result.f1 >= best_f1 - band
    ]
    tied.sort(key=lambda setting: (
        abs(results[setting].pore_count_error),
        abs(results[setting].merges_per_100_gt
            - results[setting].splits_per_100_gt),
        -boundary_at_decision_scale(results[setting]),
    ))
    return tied[0], tied


def seeding_decision(
    results: Mapping[WatershedParams, AggregateResult],
    reference: WatershedParams,
) -> dict:
    """Whether the seeding setting in use holds, and what replaces it.

    The setting in use holds unless some setting beats its F1 by more
    than one noise band of the reference. Only then is a replacement
    chosen, among the settings tied with the best by
    :func:`choose_within_band`.

    Parameters
    ----------
    results : mapping of WatershedParams to AggregateResult
        Must contain ``reference``.
    reference : WatershedParams

    Returns
    -------
    dict
        ``holds``, ``chosen``, ``tied`` (the settings tied with the
        best, preferred first), ``band``, ``best_f1_gain`` and
        ``on_edge`` - whether the choice lies on the border of the grid
        in any varied parameter, which calls for a wider grid.

    Raises
    ------
    KeyError
        If the reference was not scored.
    """
    reference_result = results[reference]
    band = f1_noise(reference_result)
    best_f1_gain = max(r.f1 for r in results.values()) - reference_result.f1
    holds = best_f1_gain <= band
    chosen, tied = choose_within_band(results, band)
    if holds:
        chosen = reference
    return {
        "holds": holds,
        "chosen": chosen,
        "tied": tied,
        "band": band,
        "best_f1_gain": best_f1_gain,
        "on_edge": on_grid_edge(chosen, list(results)),
    }


def on_grid_edge(
    setting: WatershedParams, grid: Sequence[WatershedParams]
) -> list[str]:
    """Varied parameters in which a setting takes an extreme grid value.

    Parameters
    ----------
    setting : WatershedParams
    grid : sequence of WatershedParams

    Returns
    -------
    list of str
        Empty when the setting lies inside the grid in every parameter
        the grid varies.
    """
    edges = []
    for name, value in setting.to_kwargs().items():
        values = {getattr(other, name) for other in grid}
        if len(values) > 1 and value in (min(values), max(values)):
            edges.append(name)
    return edges


def zero_crossing(
    thresholds: Sequence[float], values: Sequence[float]
) -> Optional[float]:
    """Where a curve sampled at increasing thresholds crosses zero.

    Linear between the two samples that bracket the first sign change.

    Parameters
    ----------
    thresholds : sequence of float
        Increasing.
    values : sequence of float
        Missing values (``nan``) are skipped.

    Returns
    -------
    float or None
        ``None`` when the curve never changes sign within the grid,
        which means the crossing lies outside it.
    """
    points = [
        (t, v) for t, v in zip(thresholds, values) if np.isfinite(v)
    ]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if v0 == 0:
            return float(t0)
        if v0 * v1 < 0:
            return float(t0 + (t1 - t0) * v0 / (v0 - v1))
    if points and points[-1][1] == 0:
        return float(points[-1][0])
    return None
