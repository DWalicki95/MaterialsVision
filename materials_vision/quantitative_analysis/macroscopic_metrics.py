"""
Macroscopic pore metrics for combined analysis reports.

Aggregates per-pore data from combined analysis reports (as produced by
the batch quantitative analysis pipeline) into per-material macroscopic
metrics: size, elongation, regularity, orientation, and topology.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, skew

from materials_vision.quantitative_analysis.quantitative_analysis import (
    EDGE_WEIGHT_KEY, ellipse_angle_to_image_direction)
from materials_vision.quantitative_analysis.stats_utils import \
    weighted_quantile

logger = logging.getLogger(__name__)


def _valid_values_and_weights(
    df: pd.DataFrame, column: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Non-missing values of a column with their edge-correction weights.

    Parameters
    ----------
    df : pd.DataFrame
        Individual pore data.
    column : str
        Column to read.

    Returns
    -------
    values : np.ndarray
        Values with missing entries dropped.
    weights : np.ndarray or None
        Matching ``edge_correction_weight`` values, or None when the
        report predates the column or any kept row lacks a weight, in
        which case the corrected metrics are reported as NaN.
    """
    valid = df[column].notna()
    values = df.loc[valid, column].to_numpy(dtype=float)
    if EDGE_WEIGHT_KEY not in df.columns:
        return values, None
    weights = df.loc[valid, EDGE_WEIGHT_KEY].to_numpy(dtype=float)
    if len(weights) == 0 or not np.all(np.isfinite(weights)):
        return values, None
    return values, weights


def _weighted_skewness(values: np.ndarray, weights: np.ndarray) -> float:
    """Population skewness of a weighted sample (``nan`` if constant)."""
    mean = np.average(values, weights=weights)
    deviations = values - mean
    variance = np.average(deviations ** 2, weights=weights)
    if variance == 0:
        return float("nan")
    third_moment = np.average(deviations ** 3, weights=weights)
    return float(third_moment / variance ** 1.5)


def find_analysis_reports(input_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all combined analysis report Excel files in the input directory.

    Searches for files matching the pattern:
    <input_dir>/<material_name>/reports/*_combined_analysis_report.xlsx

    Parameters
    ----------
    input_dir : Path
        Parent directory containing material subdirectories

    Returns
    -------
    List[Tuple[str, Path]]
        List of tuples containing (material_name, report_path)

    Raises
    ------
    FileNotFoundError
        If input_dir does not exist
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    reports = []

    subdirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    subdirs.sort()

    for subdir in subdirs:
        reports_dir = subdir / "reports"
        if not reports_dir.exists():
            logger.debug(f"No reports directory in: {subdir.name}")
            continue

        report_files = [
            f for f in reports_dir.glob("*_combined_analysis_report.xlsx")
            if not f.name.startswith('.')
        ]

        if not report_files:
            logger.debug(
                f"No analysis report found in: {reports_dir}"
            )
            continue

        report_path = report_files[0]
        material_name = subdir.name
        reports.append((material_name, report_path))
        logger.info(f"Found report for material: {material_name}")

    return reports


def read_individual_pores_sheet(
    report_path: Path
) -> Optional[pd.DataFrame]:
    """
    Read the Individual_Pores sheet from an analysis report.

    Parameters
    ----------
    report_path : Path
        Path to the Excel analysis report

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with individual pore data, or None if reading fails

    Notes
    -----
    Required columns: equivalent_diameter, aspect_ratio, circularity,
    ellipse_angle, coordination_number
    """
    required_columns = [
        'equivalent_diameter',
        'aspect_ratio',
        'circularity',
        'ellipse_angle',
        'coordination_number'
    ]

    try:
        df = pd.read_excel(
            report_path,
            sheet_name="Individual_Pores",
            engine="openpyxl"
        )
    except Exception as e:
        logger.error(f"Failed to read {report_path}: {e}")
        return None

    if df.empty:
        logger.warning(f"No pore data in {report_path}")
        return None

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"Missing columns in {report_path.name}: {missing_cols}. "
            "Skipping this material."
        )
        return None

    return df


def calculate_centered_entropy(
    angles: np.ndarray,
    num_bins: int = 18,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate normalized entropy of orientation angles.

    Centers the angle distribution around its mean direction before
    computing entropy, providing a rotation-invariant measure of
    orientation uniformity.

    Parameters
    ----------
    angles : np.ndarray
        Array of orientation angles in degrees. Any half-turn range
        works (stored ``ellipse_angle`` is -90 to 90), because the
        angles are folded modulo 180 after centering.
    num_bins : int, optional
        Number of histogram bins for entropy calculation (default: 18)
    weights : np.ndarray, optional
        One weight per angle; weights both the mean direction and the
        histogram (default: equal weights).

    Returns
    -------
    float
        Normalized entropy value between 0 and 1, where:
        - 0 indicates perfect alignment (all angles identical)
        - 1 indicates uniform distribution (isotropic)

    Notes
    -----
    The function uses circular statistics to handle the periodic nature
    of orientation data. Angles are doubled before averaging to account
    for the 180-degree symmetry of ellipse orientations.

    Formula for normalization: H_norm = H / H_max
    where H_max = ln(num_bins) for uniform distribution
    """
    angles = np.array(angles, dtype=float)
    weights = (
        np.ones_like(angles) if weights is None
        else np.asarray(weights, dtype=float)
    )

    # Handle NaN values
    valid = ~np.isnan(angles)
    angles = angles[valid]
    weights = weights[valid]
    if len(angles) == 0:
        return np.nan

    # Convert to radians and double (orientation is pi-periodic)
    angles_rad = np.deg2rad(angles) * 2

    # Calculate mean direction using circular statistics
    sin_sum = np.sum(weights * np.sin(angles_rad))
    cos_sum = np.sum(weights * np.cos(angles_rad))
    mean_angle = np.rad2deg(np.arctan2(sin_sum, cos_sum)) / 2

    # Center distribution around mean (shift so mean is at 90 degrees)
    shift = 90 - mean_angle
    angles_shifted = (angles + shift) % 180

    # Calculate histogram and entropy
    counts, _ = np.histogram(
        angles_shifted, bins=num_bins, range=(0, 180), weights=weights
    )
    if counts.sum() == 0:
        return 0.0

    probs = counts / counts.sum()
    shannon_entropy = entropy(probs)  # Uses natural log by default
    max_entropy = np.log(num_bins)

    return shannon_entropy / max_entropy


def _size_statistics(
    diameters: np.ndarray, weights: np.ndarray
) -> Dict[str, float]:
    """Mean, D10/D50/D90, span and skewness of weighted diameters."""
    d10 = weighted_quantile(diameters, weights, 0.10)
    median = weighted_quantile(diameters, weights, 0.50)
    d90 = weighted_quantile(diameters, weights, 0.90)
    return {
        'mean': float(np.average(diameters, weights=weights)),
        'median': median,
        'd10': d10,
        'd90': d90,
        'span': (d90 - d10) / median if median != 0 else np.nan,
        'skewness': _weighted_skewness(diameters, weights),
    }


def calculate_size_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate size-related metrics from equivalent diameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'equivalent_diameter' column and, optionally,
        'edge_correction_weight'

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - size_mean: Mean equivalent diameter
        - size_median: Median equivalent diameter
        - size_d10: 10th percentile
        - size_d90: 90th percentile
        - size_span: (D90 - D10) / Median
        - size_skewness: Skewness of size distribution
        and the same six with a ``_corrected`` suffix, weighted by
        'edge_correction_weight' (NaN when the column is missing).

    Notes
    -----
    Span formula: (D90 - D10) / D50
    This is a common metric in particle size analysis describing
    the width of the size distribution relative to the median.

    The uncorrected values describe the measured pores only. Pores cut by
    the image border are not measured, and large pores are cut more
    often, so these values understate pore size; the corrected values
    compensate with Miles-Lantuejoul weights.
    """
    diameters, weights = _valid_values_and_weights(
        df, 'equivalent_diameter'
    )
    names = ['mean', 'median', 'd10', 'd90', 'span', 'skewness']

    if len(diameters) == 0:
        logger.warning("No valid diameter values")
        return {
            f'size_{name}{suffix}': np.nan
            for suffix in ('', '_corrected') for name in names
        }

    raw = _size_statistics(diameters, np.ones_like(diameters))
    raw['skewness'] = float(skew(diameters))
    corrected = (
        _size_statistics(diameters, weights) if weights is not None
        else {name: np.nan for name in names}
    )
    metrics = {f'size_{name}': raw[name] for name in names}
    metrics.update(
        {f'size_{name}_corrected': corrected[name] for name in names}
    )
    return metrics


def _elongation_statistics(
    aspect_ratios: np.ndarray, weights: np.ndarray
) -> Dict[str, float]:
    """Mean, median, std and share above 2 of weighted aspect ratios."""
    mean = float(np.average(aspect_ratios, weights=weights))
    variance = float(
        np.average((aspect_ratios - mean) ** 2, weights=weights)
    )
    return {
        'mean': mean,
        'median': weighted_quantile(aspect_ratios, weights, 0.5),
        'std': float(np.sqrt(variance)),
        'fraction_gt_2': float(
            weights[aspect_ratios > 2.0].sum() / weights.sum()
        ),
    }


def calculate_elongation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate elongation-related metrics from aspect ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'aspect_ratio' column and, optionally,
        'edge_correction_weight'

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - elongation_mean: Mean aspect ratio
        - elongation_median: Median aspect ratio
        - elongation_std: Standard deviation of aspect ratio
        - elongation_fraction_gt_2: Fraction of pores with AR > 2.0
        and the same four with a ``_corrected`` suffix, weighted by
        'edge_correction_weight' (NaN when the column is missing).

    Notes
    -----
    Fraction > 2.0 identifies strongly deformed/elongated pores.
    An aspect ratio of 2.0 means the pore is twice as long as wide.
    """
    aspect_ratios, weights = _valid_values_and_weights(df, 'aspect_ratio')
    names = ['mean', 'median', 'std', 'fraction_gt_2']

    if len(aspect_ratios) == 0:
        logger.warning("No valid aspect ratio values")
        return {
            f'elongation_{name}{suffix}': np.nan
            for suffix in ('', '_corrected') for name in names
        }

    raw = _elongation_statistics(
        aspect_ratios, np.ones_like(aspect_ratios)
    )
    corrected = (
        _elongation_statistics(aspect_ratios, weights)
        if weights is not None else {name: np.nan for name in names}
    )
    metrics = {f'elongation_{name}': raw[name] for name in names}
    metrics.update(
        {f'elongation_{name}_corrected': corrected[name] for name in names}
    )
    return metrics


def calculate_regularity_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate regularity-related metrics from circularity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'circularity' column and, optionally,
        'edge_correction_weight'

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - regularity_mean: Mean circularity
        - regularity_d10: 10th percentile (identifies most irregular pores)
        and both with a ``_corrected`` suffix, weighted by
        'edge_correction_weight' (NaN when the column is missing).

    Notes
    -----
    Circularity = 4 * pi * Area / Perimeter^2
    - Value of 1.0 indicates a perfect circle
    - Lower values indicate more irregular shapes
    D10 identifies the 10% most irregular pores in the distribution.
    """
    circularity, weights = _valid_values_and_weights(df, 'circularity')

    if len(circularity) == 0:
        logger.warning("No valid circularity values")
        return {
            'regularity_mean': np.nan,
            'regularity_d10': np.nan,
            'regularity_mean_corrected': np.nan,
            'regularity_d10_corrected': np.nan,
        }

    corrected_available = weights is not None
    return {
        'regularity_mean': float(np.mean(circularity)),
        'regularity_d10': float(np.percentile(circularity, 10)),
        'regularity_mean_corrected': (
            float(np.average(circularity, weights=weights))
            if corrected_available else np.nan
        ),
        'regularity_d10_corrected': (
            weighted_quantile(circularity, weights, 0.10)
            if corrected_available else np.nan
        ),
    }


def calculate_orientation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate orientation-related metrics from ellipse angle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ellipse_angle' column and, optionally,
        'edge_correction_weight'

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - orientation_entropy: Normalized entropy (0-1 scale)
        - orientation_entropy_corrected: the same, weighted by
          'edge_correction_weight' (NaN when the column is missing)

    Notes
    -----
    Normalized entropy interpretation:
    - 0: Perfect alignment (all pores oriented the same direction)
    - 1: Uniform/isotropic distribution (random orientations)
    Higher values indicate more random/isotropic orientation patterns.
    """
    angles, weights = _valid_values_and_weights(df, 'ellipse_angle')

    if len(angles) == 0:
        logger.warning("No valid ellipse angle values")
        return {
            'orientation_entropy': np.nan,
            'orientation_entropy_corrected': np.nan,
        }

    return {
        'orientation_entropy': calculate_centered_entropy(angles),
        'orientation_entropy_corrected': (
            calculate_centered_entropy(angles, weights=weights)
            if weights is not None else np.nan
        ),
    }


def calculate_topology_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate topology-related metrics from coordination number.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'coordination_number' column and,
        optionally, 'edge_correction_weight'

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - topology_cn_mean: Mean coordination number
        - topology_cn_std: Standard deviation of coordination number
        and both with a ``_corrected`` suffix, weighted by
        'edge_correction_weight' (NaN when the column is missing). Larger
        cells tend to have more neighbours, so the size bias of the
        border filter reaches the coordination number as well.

    Notes
    -----
    Coordination number represents the number of neighboring pores
    each pore is connected to (based on Voronoi tessellation).
    Higher values indicate more densely packed pore networks.
    """
    coord_numbers, weights = _valid_values_and_weights(
        df, 'coordination_number'
    )

    if len(coord_numbers) == 0:
        logger.warning("No valid coordination number values")
        return {
            'topology_cn_mean': np.nan,
            'topology_cn_std': np.nan,
            'topology_cn_mean_corrected': np.nan,
            'topology_cn_std_corrected': np.nan,
        }

    metrics = {
        'topology_cn_mean': float(np.mean(coord_numbers)),
        'topology_cn_std': float(np.std(coord_numbers)),
        'topology_cn_mean_corrected': np.nan,
        'topology_cn_std_corrected': np.nan,
    }
    if weights is not None:
        mean = float(np.average(coord_numbers, weights=weights))
        metrics['topology_cn_mean_corrected'] = mean
        metrics['topology_cn_std_corrected'] = float(np.sqrt(
            np.average((coord_numbers - mean) ** 2, weights=weights)
        ))
    return metrics


def merge_individual_pores(
    reports: List[Tuple[str, Path]]
) -> Optional[pd.DataFrame]:
    """
    Merge Individual_Pores sheets from all materials into one DataFrame.

    Parameters
    ----------
    reports : List[Tuple[str, Path]]
        List of (material_name, report_path) tuples

    Returns
    -------
    Optional[pd.DataFrame]
        Combined DataFrame with a 'material_name' column,
        or None if no data could be read
    """
    frames = []
    for material_name, report_path in reports:
        df = read_individual_pores_sheet(report_path)
        if df is None:
            continue
        df = df.copy()
        df.insert(0, 'material_name', material_name)
        frames.append(df)
        logger.info(
            f"  Loaded {len(df)} pores from {material_name}"
        )

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Merged dataset: {len(merged)} total pores "
        f"from {len(frames)} material(s)"
    )
    return merged


def export_merged_dataset(
    df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Export merged individual pores dataset to Excel.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame with all individual pore data
    output_path : Path
        Path for the output Excel file

    Returns
    -------
    Path
        Path to the created Excel file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info(f"Merged dataset exported to: {output_path}")
    return output_path


def calculate_all_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate all macroscopic metrics for a material.

    Orchestrates calls to individual metric calculation functions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing individual pore data

    Returns
    -------
    Dict[str, float]
        Combined dictionary of all calculated metrics
    """
    metrics = {}

    # Size metrics (equivalent_diameter)
    metrics.update(calculate_size_metrics(df))

    # Elongation metrics (aspect_ratio)
    metrics.update(calculate_elongation_metrics(df))

    # Regularity metrics (circularity)
    metrics.update(calculate_regularity_metrics(df))

    # Orientation metrics (ellipse_angle)
    metrics.update(calculate_orientation_metrics(df))

    # Topology metrics (coordination_number)
    metrics.update(calculate_topology_metrics(df))

    return metrics


def plot_orientation_rose(
    ax: plt.Axes,
    angles: np.ndarray,
    num_bins: int = 36
) -> None:
    """
    Create polar rose plot with correct bin centering.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib polar axes to plot on
    angles : np.ndarray
        Stored ``ellipse_angle`` values in degrees (-90 to 90, measured
        from the vertical image axis)
    num_bins : int, optional
        Number of bins over half a turn (default: 36, i.e. 5 deg per bin)

    Notes
    -----
    The rose reads like the image: 0 degrees at 3 o'clock (East) is a
    horizontal pore and angles grow counter-clockwise, so a pore rising
    to the right appears at 45 degrees. The stored angles are measured
    from the vertical axis and are converted with
    ``ellipse_angle_to_image_direction`` first; plotting them
    unconverted turns the whole rose by 90 degrees.
    Bins are centered using (edge_left + edge_right) / 2.
    Each bin is mirrored half a turn away to create a full 360 degree
    view, because an ellipse axis has no head or tail.
    """
    angles = np.array(angles)
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        return

    directions = ellipse_angle_to_image_direction(angles)
    counts, bin_edges = np.histogram(
        directions, bins=num_bins, range=(0, 180)
    )

    # Calculate bin CENTERS: (left_edge + right_edge) / 2
    bin_centers_deg = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert to radians
    bin_centers_rad = np.deg2rad(bin_centers_deg)

    # Duplicate for 180-degree symmetry (ellipse orientation is symmetric)
    theta = np.concatenate([bin_centers_rad, bin_centers_rad + np.pi])
    radii = np.concatenate([counts, counts])

    # Bar width in radians (bins touch each other)
    width = np.deg2rad(180 / num_bins)

    # Plot bars
    ax.bar(
        theta, radii, width=width, bottom=0.0,
        color='#457b9d', alpha=0.8, edgecolor='white', linewidth=0.5
    )

    # Set 0 degrees at 3 o'clock (East), counterclockwise direction
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    ax.set_title("Orientation Rose Plot", fontsize=10, pad=15)


def generate_material_visualization(
    df: pd.DataFrame,
    material_name: str,
    output_path: Path
) -> Path:
    """
    Generate a 3x2 grid visualization for a material's pore analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing individual pore data
    material_name : str
        Name of the material for the title
    output_path : Path
        Path to save the output figure

    Returns
    -------
    Path
        Path to the saved figure

    Notes
    -----
    Layout:
    - Row 1: Network Topology (coordination_number), Size Distribution
    - Row 2: Elongation Degree (aspect_ratio), Shape Quality scatter
    - Row 3: Linear Orientation Histogram, Polar Rose Plot
    """
    sns.set_style("whitegrid")

    # Create figure with 3x2 grid, last subplot is polar
    fig = plt.figure(figsize=(14, 12))

    # Row 1, Col 1: Network Topology - coordination_number histogram
    ax1 = fig.add_subplot(3, 2, 1)
    coord_data = df['coordination_number'].dropna()
    if len(coord_data) > 0:
        sns.histplot(
            coord_data, ax=ax1, kde=False, color='teal',
            edgecolor='black', linewidth=0.5
        )
    ax1.set_xlabel("Coordination Number")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Network Topology")

    # Row 1, Col 2: Size Distribution - equivalent_diameter KDE filled
    ax2 = fig.add_subplot(3, 2, 2)
    diameter_data = df['equivalent_diameter'].dropna()
    if len(diameter_data) > 0:
        sns.kdeplot(
            diameter_data, ax=ax2, color='steelblue',
            fill=True, alpha=0.6, linewidth=2
        )
        median_val = np.median(diameter_data)
        ax2.axvline(
            median_val, color='red', linestyle='--', linewidth=2,
            label=f'Median: {median_val:.1f}'
        )
        ax2.legend()
    ax2.set_xlabel("Equivalent Diameter (um)")
    ax2.set_ylabel("Density")
    ax2.set_title("Size Distribution")

    # Row 2, Col 1: Elongation Degree - aspect_ratio KDE filled
    ax3 = fig.add_subplot(3, 2, 3)
    ar_data = df['aspect_ratio'].dropna()
    if len(ar_data) > 0:
        sns.kdeplot(
            ar_data, ax=ax3, color='darkorange',
            fill=True, alpha=0.6, linewidth=2
        )
        ax3.axvline(
            2.0, color='red', linestyle='--', linewidth=2,
            label='Threshold: 2.0'
        )
        ax3.legend()
    ax3.set_xlabel("Aspect Ratio")
    ax3.set_ylabel("Density")
    ax3.set_title("Elongation Degree")

    # Row 2, Col 2: Shape Quality - overlapped KDE for circularity & solidity
    ax4 = fig.add_subplot(3, 2, 4)
    circ_data = df['circularity'].dropna()
    solid_data = df['solidity'].dropna() if 'solidity' in df.columns else None
    if len(circ_data) > 0:
        sns.kdeplot(
            circ_data, ax=ax4, color='purple',
            fill=True, alpha=0.5, linewidth=2, label='Circularity'
        )
    if solid_data is not None and len(solid_data) > 0:
        sns.kdeplot(
            solid_data, ax=ax4, color='teal',
            fill=True, alpha=0.5, linewidth=2, label='Solidity'
        )
    ax4.set_xlabel("Value")
    ax4.set_ylabel("Density")
    ax4.set_title("Shape Quality")
    ax4.set_xlim(0, 1.1)
    ax4.legend()

    # Row 3, Col 1: Linear Orientation Histogram (-90 to 90 range)
    ax5 = fig.add_subplot(3, 2, 5)
    angle_data = df['ellipse_angle'].dropna()
    if len(angle_data) > 0:
        sns.histplot(
            angle_data, ax=ax5, bins=36, kde=False, color='forestgreen',
            edgecolor='black', linewidth=0.5
        )
    ax5.set_xlabel("Ellipse Angle (degrees)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Orientation Distribution (Linear)")
    ax5.set_xlim(-90, 90)
    ax5.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Row 3, Col 2: Polar Rose Plot
    ax6 = fig.add_subplot(3, 2, 6, projection='polar')
    if len(angle_data) > 0:
        plot_orientation_rose(ax6, angle_data.values)

    # Overall title
    fig.suptitle(
        f"Macroscopic Pore Analysis: {material_name}",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Visualization saved to: {output_path}")
    return output_path


def export_results_to_excel(
    results: List[Dict],
    output_path: Path
) -> Path:
    """
    Export calculated metrics to an Excel file.

    Parameters
    ----------
    results : List[Dict]
        List of dictionaries, each containing metrics for one material
    output_path : Path
        Path for the output Excel file

    Returns
    -------
    Path
        Path to the created Excel file
    """
    # Define column order for output
    metric_columns = [
        'size_mean',
        'size_median',
        'size_d10',
        'size_d90',
        'size_span',
        'size_skewness',
        'elongation_mean',
        'elongation_median',
        'elongation_std',
        'elongation_fraction_gt_2',
        'regularity_mean',
        'regularity_d10',
        'orientation_entropy',
        'topology_cn_mean',
        'topology_cn_std',
    ]
    column_order = ['material_name', 'n_pores'] + [
        column
        for metric in metric_columns
        for column in (metric, f'{metric}_corrected')
    ]

    df = pd.DataFrame(results)

    # Reorder columns (keep only existing columns in order)
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Excel
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info(f"Results exported to: {output_path}")

    return output_path
