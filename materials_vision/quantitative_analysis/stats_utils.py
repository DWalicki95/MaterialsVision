from typing import Dict, List, Optional

import numpy as np


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """
    Quantile of a weighted sample with linear interpolation.

    Each sorted value sits at the midpoint of its share of the cumulative
    weight, and the midpoints are rescaled so the smallest value lands at
    0 and the largest at 1. With equal weights this is exactly
    ``np.percentile`` with its default linear method; unequal weights
    stretch the scale where the heavier observations sit. The result
    does not change when all weights are multiplied by a constant.

    Parameters
    ----------
    values : np.ndarray
        Observations; NaNs must be removed by the caller.
    weights : np.ndarray
        Non-negative weights, one per observation, with a positive sum.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    float
        The weighted quantile; ``nan`` for an empty sample.

    Raises
    ------
    ValueError
        If ``q`` is outside ``[0, 1]``, the lengths differ, or the weights
        are negative or sum to zero.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {q}.")
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same length.")
    if values.size == 0:
        return float("nan")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be non-negative with a positive sum.")
    # A zero-weight observation carries no information, and keeping it
    # would put two values at the same position.
    values = values[weights > 0]
    weights = weights[weights > 0]
    if values.size == 1:
        return float(values[0])

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoints = np.cumsum(sorted_weights) - sorted_weights / 2
    positions = (midpoints - midpoints[0]) / (midpoints[-1] - midpoints[0])
    return float(np.interp(q, positions, sorted_values))


def calculate_statistics(
    values: List, weights: Optional[List] = None
) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of values.

    Parameters
    ----------
    values : List
        Numerical values.
    weights : List, optional
        One non-negative weight per value. When given, weighted
        ``mean_corrected``, ``median_corrected`` and ``std_corrected`` are
        added next to the unweighted statistics.

    Returns
    -------
    Dict[str, float]
        Mean, median, min, max and std, plus the weighted ``*_corrected``
        statistics when weights are given.
    """
    avg = np.mean(values)
    min_ = np.min(values)
    max_ = np.max(values)
    median = np.median(values)
    std = np.std(values)
    output = {
        "mean": avg,
        "median": median,
        "min": min_,
        "max": max_,
        "std": std,
    }
    if weights is not None:
        output.update(calculate_weighted_statistics(values, weights))
    return output


def calculate_weighted_statistics(
    values: List, weights: List
) -> Dict[str, float]:
    """
    Weighted mean, median and standard deviation.

    Parameters
    ----------
    values : List
        Numerical values.
    weights : List
        One non-negative weight per value, with a positive sum.

    Returns
    -------
    Dict[str, float]
        ``mean_corrected``, ``median_corrected`` and ``std_corrected``.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(values)):
        # Mirrors the unweighted statistics, which a NaN also poisons.
        nan = float("nan")
        return {
            "mean_corrected": nan,
            "median_corrected": nan,
            "std_corrected": nan,
        }
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return {
        "mean_corrected": mean,
        "median_corrected": weighted_quantile(values, weights, 0.5),
        "std_corrected": float(np.sqrt(variance)),
    }
