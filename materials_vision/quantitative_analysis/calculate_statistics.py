import numpy as np
from typing import List, Dict


def calculate_statistics(values: List) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of values.

    Parameters
    ----------
        values: List of numerical values

    Returns:
        Dictionary containing mean, median, min, max, and std
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
    return output
