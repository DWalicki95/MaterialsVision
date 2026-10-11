import numpy as np
import tifffile
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.segmentation import find_boundaries
from typing import Dict, List, Optional, Tuple
import inspect
import math
import matplotlib.pyplot as plt
from scipy import spatial
from pathlib import Path
import pandas as pd
from datetime import datetime

from materials_vision.quantitative_analysis.stats_utils import (
    calculate_statistics,
    calculate_weighted_statistics,
)
import logging

logger = logging.getLogger(__name__)

# The stored ellipse angle is measured from the vertical image axis; a
# quarter turn moves its zero to the horizontal.
VERTICAL_TO_HORIZONTAL_DEG = 90.0

# An ellipse axis has no head or tail, so directions repeat every half
# turn.
HALF_TURN_DEG = 180.0


def ellipse_angle_to_image_direction(angles_deg: np.ndarray) -> np.ndarray:
    """
    Convert stored ellipse angles to directions as seen on the image.

    ``ellipse_angle`` is measured from the vertical image axis (see
    ``PoreMorphologyMetrics._calculate_ellipse_angle``), which is not
    how a direction is read off a picture or a rose plot. This returns
    the same orientation measured counter-clockwise from the
    horizontal, as the image is displayed with row 0 at the top: 0 is a
    horizontal pore, 45 a pore rising to the right, 90 a vertical one.

    Parameters
    ----------
    angles_deg : np.ndarray
        Stored ``ellipse_angle`` values in degrees, in ``[-90, 90]``.
        Not modified.

    Returns
    -------
    np.ndarray
        Directions in degrees, folded into ``[0, 180)``. NaN stays NaN.
    """
    angles = np.asarray(angles_deg, dtype=float)
    return np.mod(angles + VERTICAL_TO_HORIZONTAL_DEG, HALF_TURN_DEG)


EDGE_WEIGHT_KEY = "edge_correction_weight"

CORRECTED_STAT_LABELS = {
    "mean_corrected": "Edge-corrected mean",
    "std_corrected": "Edge-corrected standard deviation of",
    "median_corrected": "Edge-corrected median",
}


def corrected_morphology_rows(
    metric_name: str,
    stats: Dict[str, float],
    unit: str,
    description: str,
) -> List[Dict[str, object]]:
    """
    Report rows for the edge-corrected statistics of one metric.

    Parameters
    ----------
    metric_name : str
        Morphology metric, e.g. ``"equivalent_diameter"``.
    stats : Dict[str, float]
        Its aggregated statistics; corrected keys may be absent.
    unit : str
        Unit shown in the report.
    description : str
        Human-readable description of the metric.

    Returns
    -------
    List[Dict[str, object]]
        One row per corrected statistic present in ``stats``.
    """
    return [
        {
            "Category": "Morphology",
            "Metric": f"{metric_name} ({key.replace('_', ', ')})",
            "Value": stats[key],
            "Unit": unit,
            "Description": (
                f"{label} {description} (Miles-Lantuejoul weights)"
            ),
        }
        for key, label in CORRECTED_STAT_LABELS.items()
        if key in stats
    ]


class PoreMorphologyMetrics:
    """
    Calculates morphological metrics for individual pores.

    This class computes various shape descriptors including:
    - Basic metrics: area, perimeter, circularity, solidity
    - Feret diameters: min, max, and equivalent
    - Ellipse fitting: major/minor axes, aspect ratio, orientation angle

    Parameters
    ----------
    prop : RegionProperties
        Region properties object from skimage.measure.regionprops
    pixel_size : float
        Physical size of one pixel in micrometers.

    Attributes
    ----------
    prop : RegionProperties
        Stored region properties object
    pixel_size : float
        Physical pixel size in µm
    prop_area : float
        Cached area value in µm²
    prop_perim : float
        Cached perimeter value in µm
    """

    def __init__(
        self, prop: RegionProperties, pixel_size: float
    ) -> None:
        self.prop = prop
        self.pixel_size = pixel_size
        self.prop_area = list(self._calculate_area().values())[0]
        self.prop_perim = list(self._calculate_perimeter().values())[0]

    def _calculate_area(self) -> Dict[str, float]:
        """
        Calculate pore area in µm².

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'area' and value in µm²
        """
        return {"area": self.prop.area * self.pixel_size**2}

    def _calculate_perimeter(self) -> Dict[str, float]:
        """
        Calculate pore perimeter in µm.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'perimeter' and value in µm
        """
        return {"perimeter": self.prop.perimeter * self.pixel_size}

    def calculate_min_feret(self) -> Dict[str, float]:
        """
        Calculate minimum Feret diameter using rotating calipers method on
        convex hull.

        The minimum Feret diameter (also called minimum caliper diameter) is
        the smallest distance between two parallel supporting lines tangent to
        the pore's convex hull. This metric represents the minimum width of the
        pore when measured from all possible orientations.

        Methodology
        -----------
        The algorithm implements a rotating calipers approach:

        1. **Convex Hull Construction**: First, the convex hull of the pore
           coordinates is computed. The convex hull is the smallest convex
           polygon
           that contains all pore pixels.

        2. **Edge Iteration**: For each edge of the convex hull polygon, the
           algorithm computes the width of the pore perpendicular to that edge.

        3. **Perpendicular Width Calculation**: For each edge:
           - A unit normal vector (perpendicular to the edge) is computed
           - All convex hull vertices are projected onto this normal vector
           - The width is calculated as the difference between maximum and
             minimum projections
           - This represents the distance between parallel supporting lines
             perpendicular to the current edge

        4. **Minimum Selection**: The minimum of all computed widths is the
           minimum Feret diameter.

        This approach is mathematically equivalent to rotating a pair of
        parallel
        calipers around the pore and finding the minimum separation distance.

        Edge Cases
        ----------
        - Pores with fewer than 3 pixels cannot form a convex hull and return
          0.0
        - Zero-length edges (duplicate vertices) are skipped to avoid division
          by zero
        - Computational failures return NaN with a warning logged

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'min_feret' and value in µm

        Examples
        --------
        For a rectangular pore of 10×20 pixels:
        - min_feret ≈ 10 pixels (the shorter dimension)
        - This represents the minimum "shadow width" when viewing the pore
          from different angles
        """
        coords = self.prop.coords

        # Safety check for very small pores
        # (< 3 pixels can't build convex hull)
        if len(coords) < 3:
            return {"min_feret": 0.0}

        try:
            hull = spatial.ConvexHull(coords)
            hull_points = coords[hull.vertices]

            min_width = float('inf')
            num_vertices = len(hull_points)

            for i in range(num_vertices):
                # Two consecutive vertices that form an edge of the convex hull
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % num_vertices]

                # Edge vector connecting the two vertices
                edge = p2 - p1
                norm = np.linalg.norm(edge)
                if norm == 0:
                    continue

                # Unit normal vector perpendicular to the edge
                normal = np.array([-edge[1], edge[0]]) / norm

                # Project all convex hull vertices onto the normal vector
                # The difference between max and min projections gives the
                # width of the object in the direction perpendicular to this
                # edge
                projections = np.dot(hull_points, normal)
                width = np.max(projections) - np.min(projections)

                if width < min_width:
                    min_width = width

            return {"min_feret": min_width * self.pixel_size}

        except Exception as e:
            # In case of computational error, return NaN as a placeholder value
            logger.warning(
                f"Failed to calculate MinFeret for pore {self.prop.label}: {e}"
            )
            return {"min_feret": np.nan}

    def calculate_max_feret(self) -> Dict[str, float]:
        """
        Calculate maximum Feret diameter.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'max_feret' and value in µm
        """
        return {"max_feret": self.prop.feret_diameter_max * self.pixel_size}

    def calculate_equivalent_diameter(self) -> Dict[str, float]:
        """
        Calculate equivalent diameter of the pore.

        The equivalent diameter is the diameter of a circle with the same area
        as the pore. Formula: d = 2 * sqrt(Area / π)

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'equivalent_diameter' and value in µm
        """
        equivalent_diameter = 2 * math.sqrt(self.prop_area / math.pi)
        return {"equivalent_diameter": equivalent_diameter}

    def _calculate_ellipse_major_axis(self) -> Dict[str, float]:
        """
        Calculate major axis of fitted ellipse.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'ellipse_major_axis' and value in µm
        """
        result = self.prop.axis_major_length * self.pixel_size
        return {"ellipse_major_axis": result}

    def _calculate_ellipse_minor_axis(self) -> Dict[str, float]:
        """
        Calculate minor axis of fitted ellipse.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'ellipse_minor_axis' and value in µm
        """
        result = self.prop.axis_minor_length * self.pixel_size
        return {"ellipse_minor_axis": result}

    def _calculate_aspect_ratio(
        self, ellipse_major_axis: float, ellipse_minor_axis: float
    ) -> Dict[str, float]:
        """
        Calculate aspect ratio of fitted ellipse.

        Parameters
        ----------
        ellipse_major_axis : float
            Major axis length in µm
        ellipse_minor_axis : float
            Minor axis length in µm

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'aspect_ratio' and dimensionless value
        """
        aspect_ratio = ellipse_major_axis / ellipse_minor_axis
        return {"aspect_ratio": aspect_ratio}

    def _calculate_ellipse_angle(self) -> Dict[str, float]:
        """
        Calculate ellipse orientation angle.

        The angle is scikit-image's ``regionprops`` ``orientation`` in
        degrees: measured from the vertical image axis (rows), positive
        counter-clockwise as the image is displayed with row 0 at the
        top, in ``[-90, 90]``. 0 is a vertically elongated pore, +/-90
        a horizontal one, -45 a pore rising to the right.

        The value is stored in this convention rather than converted,
        so it agrees with every report already written from it. Use
        ``ellipse_angle_to_image_direction`` for the usual direction
        measured from the horizontal, e.g. for plotting. Statistics
        that only measure alignment (anisotropy, orientation entropy)
        do not depend on the reference axis.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'ellipse_angle' and value in degrees
        """
        ellipse_angle_rad = self.prop.orientation
        ellipse_angle_degree = math.degrees(ellipse_angle_rad)
        return {"ellipse_angle": ellipse_angle_degree}

    def calculate_ellipse_metrics(self) -> Dict[str, float]:
        """
        Calculate all ellipse-related metrics.

        Computes major axis, minor axis, aspect ratio, orientation angle,
        and roundness of the fitted ellipse.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'ellipse_major_axis': Major axis length in µm
            - 'ellipse_minor_axis': Minor axis length in µm
            - 'aspect_ratio': Ratio of major to minor axis (dimensionless)
            - 'ellipse_angle': Orientation angle in degrees
            - 'roundness': Roundness metric (4*Area / (π*MajorAxis²))
        """
        final_result = {}

        ellipse_major_axis_result = self._calculate_ellipse_major_axis()
        ellipse_major_axis = list(ellipse_major_axis_result.values())[0]

        ellipse_minor_axis_result = self._calculate_ellipse_minor_axis()
        ellipse_minor_axis = list(ellipse_minor_axis_result.values())[0]

        aspect_ratio_result = self._calculate_aspect_ratio(
            ellipse_major_axis=ellipse_major_axis,
            ellipse_minor_axis=ellipse_minor_axis,
        )

        ellipse_angle_result = self._calculate_ellipse_angle()

        roundness_result = self._calculate_roundness(
            area=self.prop_area,
            ellipse_major_axis=ellipse_major_axis,
        )

        final_result = {
            **ellipse_major_axis_result,
            **ellipse_minor_axis_result,
            **aspect_ratio_result,
            **ellipse_angle_result,
            **roundness_result,
        }
        return final_result

    def _calculate_circularity(
        self, area: float, perimeter: float
    ) -> Dict[str, float]:
        """
        Calculate shape circularity.

        Circularity is defined as 4π*area / perimeter².
        A value of 1 indicates a perfect circle.

        Parameters
        ----------
        area : float
            Pore area in µm²
        ellipse_major_axis : float
            Major ellipse axis in µm

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'circularity' and dimensionless value
        """
        if perimeter == 0:
            return {"circularity": 0.0}
        result = (4 * math.pi * area) / perimeter**2
        return {"circularity": result}

    def _calculate_roundness(
            self,
            area: float,
            ellipse_major_axis: float
    ) -> Dict[str, float]:
        """
        Calculate Roundness

        Formula: 4 * Area / (π * MajorAxis²)

        - 1.0 indicates a perfect circle.
        - Unlike Circularity, Roundness is INSENSITIVE to irregular borders.
        - It mostly measures how elongated the shape is.
        """
        if ellipse_major_axis == 0:
            return {"roundness": 0.0}
        roundness = (4 * area) / (math.pi * ellipse_major_axis**2)
        return {"roundness": roundness}

    def _calculate_solidity(self) -> Dict[str, float]:
        """
        Calculate pore solidity.

        Solidity is the ratio of pore area to its convex hull area.
        Measures how convex the pore shape is (1 = perfectly convex).

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'solidity' and dimensionless value (0-1)
        """
        return {"solidity": self.prop.solidity}

    def calculate_basic_morphology(self) -> Dict[str, float]:
        """
        Calculate basic morphology metrics for a pore.

        Computes fundamental shape descriptors including area, perimeter,
        circularity, and solidity. These metrics provide a comprehensive
        characterization of pore size and shape.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'area': Pore area in µm²
            - 'perimeter': Pore perimeter in µm
            - 'circularity': Shape circularity (4πA/major_axis²), dimensionless
            - 'solidity': Ratio of area to convex hull area (0-1)
        """
        area_result = self._calculate_area()
        area = list(area_result.values())[0]

        perim_result = self._calculate_perimeter()
        perimeter = list(perim_result.values())[0]

        circ_result = self._calculate_circularity(
            area=area, perimeter=perimeter
        )

        solidity_result = self._calculate_solidity()

        final_result = {
            **area_result,
            **perim_result,
            **circ_result,
            **solidity_result,
        }
        return final_result

    @staticmethod
    def aggregate_morphology_results(
        morphology_results: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate morphology metrics across all pores.

        Computes statistical summaries (mean, median, std, min, max) for each
        morphological metric across the entire pore population.

        Parameters
        ----------
        morphology_results : List[Dict[str, float]]
            List of dictionaries, each containing morphology metrics for one
            pore.
            Each dictionary should have a 'pore_id' key and various metric keys
            (e.g., 'area', 'perimeter', 'circularity').

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary where:
            - Outer keys are metric names (e.g., 'area', 'perimeter')
            - Inner dictionaries contain statistical measures:
              - 'mean': Average value across all pores
              - 'median': Median value
              - 'std': Standard deviation
              - 'min': Minimum value
              - 'max': Maximum value
              - 'mean_corrected', 'median_corrected', 'std_corrected':
                the same statistics weighted by each pore's
                'edge_correction_weight', present only when every pore
                carries one

        Examples
        --------
        >>> morphology_results = [
        ...     {'pore_id': 1, 'area': 100.5, 'circularity': 0.95},
        ...     {'pore_id': 2, 'area': 85.3, 'circularity': 0.87},
        ...     {'pore_id': 3, 'area': 120.1, 'circularity': 0.92}
        ... ]
        >>> aggregated = PoreMorphologyMetrics.aggregate_morphology_results
        (morphology_results)
        >>> print(f"Mean area: {aggregated['area']['mean']:.2f} µm²")
        Mean area: 101.97 µm²
        """
        if not morphology_results:
            return {}

        # Get all metric names (excluding 'pore_id' and metadata fields)
        # Metadata fields added by batch analysis should not be aggregated
        excluded_fields = {
            'pore_id', 'original_pore_id', 'filename', 'magnification',
            'pixel_size', 'coordination_number', EDGE_WEIGHT_KEY
        }
        metric_names = [
            key for key in morphology_results[0].keys()
            if key not in excluded_fields
        ]
        weighted = all(EDGE_WEIGHT_KEY in pore for pore in morphology_results)

        aggregated_results = {}

        for metric_name in metric_names:
            pores = [
                pore for pore in morphology_results if metric_name in pore
            ]
            if pores:
                values = [pore[metric_name] for pore in pores]
                weights = (
                    [pore[EDGE_WEIGHT_KEY] for pore in pores]
                    if weighted else None
                )
                aggregated_results[metric_name] = calculate_statistics(
                    values, weights=weights
                )

        return aggregated_results


class GlobalMicrostructureDescriptors:
    """
    Calculates global descriptors of the porous material microstructure.

    Metrics include:
    - Porosity (global and local variance)
    - Anisotropy (preferred orientation of pores)
    """

    def __init__(
        self,
        mask: np.ndarray,
        morphology_results: List[Dict],
        pixel_size: float,
        n_rows: int = 2,
        n_cols: int = 2,
    ):
        self.morphology_results = morphology_results
        self.mask = mask
        self.pixel_size = pixel_size
        self.mask_area = self._count_mask_area()
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _count_mask_area(self, mask_: Optional[np.ndarray] = None) -> float:
        """Calculate total area of mask in µm²."""
        if mask_ is None:
            mask_ = self.mask
        mask_area = mask_.shape[0] * mask_.shape[1] * (self.pixel_size**2)
        return mask_area

    def calculate_porosity(self, mask_: Optional[np.ndarray] = None) -> float:
        """
        Calculate areal porosity (fraction of the frame covered by pores).

        Every labelled pixel counts, including pores that touch the image
        border. Border pores are left out of the per-pore morphology
        because their shape is truncated, but the part of a pore that lies
        inside the frame is still pore space, so dropping it from the
        numerator while keeping the full frame in the denominator would
        bias porosity low - most strongly for coarse-pored materials,
        where a large share of pore area touches the border.

        Parameters
        ----------
        mask_ : np.ndarray, optional
            Instance mask or a crop of it (0 = solid); defaults to the
            whole mask.

        Returns
        -------
        float
            Porosity value (0 to 1)

        Raises
        ------
        ValueError
            For an empty mask, whose porosity is undefined.
        """
        if mask_ is None:
            mask_ = self.mask
        if mask_.size == 0:
            raise ValueError("An empty mask has no porosity.")
        return float(np.count_nonzero(mask_)) / mask_.size

    def _cut_mask_into_pieces(
        self, n_rows: int = 2, n_cols: int = 2
    ) -> List[np.ndarray]:
        """
        Divide mask into equal rectangular pieces for local analysis.

        Parameters
        ----------
        n_rows : int, optional
            Number of rows to divide the mask (default: 2)
        n_cols : int, optional
            Number of columns to divide the mask (default: 2)

        Returns
        -------
        List[np.ndarray]
            List of mask pieces
        """
        if n_rows < 1 or n_cols < 1:
            raise ValueError("n_rows and n_cols must be at least 1")
        height, width = self.mask.shape[:2]
        piece_height = height // n_rows
        piece_width = width // n_cols
        if piece_height == 0 or piece_width == 0:
            raise ValueError(
                f"Image dimensions ({height}, {width}) are too small to cut "
                f"into {n_rows}x{n_cols} pieces."
            )
        usable_height = piece_height * n_rows
        usable_width = piece_width * n_cols
        cropped_image = self.mask[:usable_height, :usable_width]
        pieces = []
        for row in range(n_rows):
            for col in range(n_cols):
                start_row = row * piece_height
                end_row = start_row + piece_height
                start_col = col * piece_width
                end_col = start_col + piece_width
                piece_mask = cropped_image[
                    start_row:end_row, start_col:end_col
                ]
                pieces.append(piece_mask.copy())
        return pieces

    def calculate_local_porosity_variance(self) -> float:
        """
        Calculate variance of porosity across local regions.

        Low variance indicates homogeneous pore distribution.
        High variance indicates heterogeneous pore distribution.

        Returns:
            Variance of local porosities
        """
        pieces = self._cut_mask_into_pieces(
            n_rows=self.n_rows, n_cols=self.n_cols
        )
        local_porosities = []
        for piece in pieces:
            local_porosity = self.calculate_porosity(mask_=piece)
            local_porosities.append(local_porosity)
        porosity_variance = np.var(local_porosities)
        return porosity_variance

    def calculate_anisotropy(self, corrected: bool = False) -> float:
        """
        Calculate anisotropy of foam microstructure from ellipse orientation
        angles.

        Anisotropy measures the degree of preferred orientation:
        - 0: random orientation (isotropic)
        - 1: perfect alignment (highly anisotropic)

        Parameters
        ----------
        corrected : bool, optional
            Weight each pore by its edge-correction weight, compensating
            for large pores being excluded at the border more often than
            small ones (default: False). Returns ``nan`` when any pore
            lacks a weight.

        Returns
        -------
        float
            Anisotropy index (0 to 1)
        """
        pores = [
            pore for pore in self.morphology_results
            if "ellipse_angle" in pore
        ]
        if not pores:
            logger.warning("No ellipse angles found in morphology results")
            return 0.0

        angles = np.array([pore["ellipse_angle"] for pore in pores])
        if corrected:
            if not all(EDGE_WEIGHT_KEY in pore for pore in pores):
                return float("nan")
            weights = np.array([pore[EDGE_WEIGHT_KEY] for pore in pores])
        else:
            weights = np.ones_like(angles)

        # Convert to radians and double the angle
        # (orientation is π-periodic, not 2π-periodic)
        angles_rad = np.deg2rad(angles * 2)

        # Calculate resultant vector using circular statistics
        mean_cos = np.average(np.cos(angles_rad), weights=weights)
        mean_sin = np.average(np.sin(angles_rad), weights=weights)

        # Resultant vector length R (0 to 1)
        # R = 1 means perfect alignment, R = 0 means random
        anisotropy = np.sqrt(mean_cos**2 + mean_sin**2)

        return anisotropy

    def calculate_all_global_descriptors(self) -> Dict[str, float]:
        """
        Calculate all global descriptors.

        Returns:
            Dictionary containing porosity, local_porosity_variance,
            anisotropy and its edge-corrected counterpart
        """
        results = {
            "porosity": self.calculate_porosity(),
            "local_porosity_variance": (
                self.calculate_local_porosity_variance()),
            "anisotropy": self.calculate_anisotropy(),
            "anisotropy_corrected": self.calculate_anisotropy(corrected=True),
        }
        return results


class SpatialRelationMetrics:
    """
    Analyzes spatial relationships between pores.

    Includes nearest neighbor distance analysis.

    Parameters
    ----------
    props : List[RegionProperties]
        Pores whose nearest-neighbour distance is reported.
    pixel_size : float
        Physical size of one pixel in micrometers.
    generate_plots : bool, optional
        Whether to save the distance histogram (default: True).
    neighbour_props : List[RegionProperties], optional
        Pores that may serve as a neighbour; defaults to ``props``. Pass
        every pore in the frame, border pores included: a pore near the
        border whose true neighbour touches the border would otherwise be
        matched to a farther one.
    """

    def __init__(
        self,
        props: List[RegionProperties],
        pixel_size: float,
        generate_plots: bool = True,
        neighbour_props: Optional[List[RegionProperties]] = None,
    ):
        self.props = props
        self.neighbour_props = (
            props if neighbour_props is None else neighbour_props
        )
        self.generate_plots = generate_plots
        self.pixel_size = pixel_size

    def plot_distribution(
        self,
        values: List[float],
        output_path: Path,
        title: str = "Distribution",
        xlabel: str = "Distance (µm)",
    ):
        """Plot histogram of value distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, edgecolor="black", linewidth=1.2)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        plt.close()

    def calculate_nearest_neighbor_centroid_distance(self) -> List[float]:
        """
        Calculate nearest neighbor distances between pore centroids.

        Returns:
            List of nearest neighbor distances (in µm), one per pore in
            ``props``; empty when there are no pores or fewer than two
            candidate neighbours
        """
        if not self.props or len(self.neighbour_props) < 2:
            return []
        neighbour_labels = np.array([p.label for p in self.neighbour_props])
        neighbour_centroids_px = np.array(
            [p.centroid for p in self.neighbour_props]
        )
        query_labels = np.array([p.label for p in self.props])
        query_centroids_px = np.array([p.centroid for p in self.props])

        tree = spatial.KDTree(neighbour_centroids_px)
        distances_px, indices = tree.query(query_centroids_px, k=2)
        # A pore that is itself a candidate comes back as its own nearest
        # point; matching by label rather than by distance keeps a distinct
        # pore with a coincident centroid as a valid neighbour.
        is_self = neighbour_labels[indices[:, 0]] == query_labels
        nn_distances_px = np.where(
            is_self, distances_px[:, 1], distances_px[:, 0]
        )
        return list(nn_distances_px * self.pixel_size)

    def calculate_spatial_metrics(
        self, output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Calculate spatial relation metrics including nearest neighbor
        statistics.

        Parameters
        ----------
        output_dir : Path, optional
            Directory to save plots

        Returns
        -------
        Dict[str, float]
            Dictionary containing mean, median, min, max, and std of
            nearest neighbor distances
        """
        nearest_neighbors_distances = (
            self.calculate_nearest_neighbor_centroid_distance()
        )
        if not nearest_neighbors_distances:
            logger.warning(
                "Not enough pores for nearest neighbour distances"
            )
            nan = float("nan")
            return {
                "mean": nan, "median": nan, "min": nan, "max": nan,
                "std": nan,
            }
        stats = calculate_statistics(nearest_neighbors_distances)

        if self.generate_plots and output_dir:
            try:
                # Extract base filename from output_dir parent
                base_name = output_dir.parent.name
                output_path = (
                    output_dir / f"{base_name}_nearest_neighbor_distances.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plot_distribution(
                    nearest_neighbors_distances,
                    output_path,
                    title=(
                        f"Nearest Neighbor Distance Distribution\n{base_name}"
                    ),
                )
                logger.info(
                    f"Saved nearest neighbor distance plot: {output_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save nearest neighbor distance plot: {e}"
                )

        return stats


class TopologyConnectivityAnalysis:
    """
    Analyzes topological and connectivity properties of porous structure.

    Includes:
    - Fractal dimension of the pore boundaries (box-counting method)
    - Coordination number (Voronoi analysis)

    Parameters
    ----------
    mask : np.ndarray
        Instance mask, 0 = solid, each pore a unique positive label.
    props : List[RegionProperties]
        Pores whose coordination number is reported.
    pixel_size : float
        Physical size of one pixel in micrometers.
    neighbour_props : List[RegionProperties], optional
        Pores that take part in the Voronoi tessellation; must include
        every pore in ``props`` and defaults to it. Pass every pore in the
        frame, border pores included, so that a pore near the border is
        not short of neighbours just because they touch the border.
    """

    def __init__(
        self,
        mask: np.ndarray,
        props: List[RegionProperties],
        pixel_size: float,
        neighbour_props: Optional[List[RegionProperties]] = None,
    ):
        self.mask = mask
        self.props = props
        self.neighbour_props = (
            props if neighbour_props is None else neighbour_props
        )
        self.pixel_size = pixel_size

    @staticmethod
    def _box_counts(
        structure: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Count occupied boxes at dyadic box sizes over the whole frame.

        The frame is padded with empty pixels up to a multiple of the
        largest box, so every pixel is covered at every scale; padding
        holds no structure and therefore adds no occupied boxes.

        Parameters
        ----------
        structure : np.ndarray
            ``(H, W)`` boolean image of the structure to cover.

        Returns
        -------
        box_sizes_px : np.ndarray
            Box edge lengths 1, 2, 4, ... up to half the shorter side.
        counts : np.ndarray
            Number of boxes containing at least one structure pixel.
        """
        shorter_side_px = min(structure.shape)
        if shorter_side_px < 2:
            return np.array([], dtype=int), np.array([], dtype=int)
        largest_box_px = 2 ** int(np.floor(np.log2(shorter_side_px / 2)))
        padded_shape = [
            -(-side // largest_box_px) * largest_box_px
            for side in structure.shape
        ]
        occupied = np.zeros(padded_shape, dtype=bool)
        occupied[:structure.shape[0], :structure.shape[1]] = structure

        box_sizes_px = [1]
        counts = [int(np.count_nonzero(occupied))]
        while box_sizes_px[-1] < largest_box_px:
            h, w = occupied.shape
            occupied = occupied.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
            box_sizes_px.append(box_sizes_px[-1] * 2)
            counts.append(int(np.count_nonzero(occupied)))
        return np.array(box_sizes_px), np.array(counts)

    def calculate_fractal_dimension(
        self,
        reject_huge_boxes: bool = True,
        save_plot_path: Optional[Path] = None,
        filename: str = "fractal_dimension_plot",
    ) -> Tuple[float, float]:
        """
        Calculate the fractal dimension of the pore boundaries.

        Box-counts the pore outlines - the pixels where a pore meets the
        solid phase or a neighbouring pore - rather than the filled pore
        phase. A filled phase covering most of the frame scores close to 2
        whatever its structure, while the outline network separates fine,
        convoluted walls from coarse, smooth ones.

        The fractal dimension D is estimated from the relationship:
        log(N(ε)) = -D * log(ε) + C

        where N(ε) is the number of boxes of size ε needed to cover the
        boundaries.

        Parameters
        ----------
        reject_huge_boxes : bool, optional
            Fit only the intermediate scales, from 2 px to an eighth of the
            shorter image side (default: True). Single pixels show the
            digital grid rather than the structure, and boxes larger than
            an eighth of the frame are too few to count reliably. When
            False, every scale up to half the shorter side is fitted.
        save_plot_path : Path, optional
            Directory path to save diagnostic plot showing the log-log
            relationship and linear fit (default: None — plot not saved)
        filename : str, optional
            Filename for the diagnostic plot (
            default: 'fractal_dimension_plot')

        Returns
        -------
        fractal_dim : float
            Estimated fractal dimension, between 1 and 2 for a 2D
            boundary network; ``nan`` when the mask has no pores or the
            image is too small for two scales.
        r_squared : float
            R² coefficient of determination for the linear fit, indicating
            fit quality (closer to 1 is better); ``nan`` alongside an
            undefined dimension.

        Notes
        -----
        The box-counting method works by:
        1. Extracting the pore boundaries from the instance mask
        2. Padding the frame to a multiple of the largest box size
        3. Iteratively doubling box size and counting occupied boxes
        4. Fitting a line to log(box_size) vs log(count)
        5. Fractal dimension is the negative slope of this line

        Examples
        --------
        >>> from skimage.measure import regionprops
        >>> import tifffile
        >>> mask = tifffile.imread('porous_material.tif')
        >>> props = regionprops(mask)
        >>> analyzer = TopologyConnectivityAnalysis(mask, props,
        pixel_size=1.5)
        >>> fractal_dim, r_squared = analyzer.calculate_fractal_dimension()
        """
        logger.info(
            "Calculating fractal dimension using box-counting method..."
        )
        nan = float("nan")
        boundaries = find_boundaries(self.mask, mode="inner")
        if not boundaries.any():
            logger.warning("No pore boundaries - fractal dimension undefined")
            return nan, nan

        box_sizes_px, counts = self._box_counts(boundaries)
        if reject_huge_boxes:
            keep = (box_sizes_px >= 2) & (
                box_sizes_px <= min(self.mask.shape) / 8
            )
        else:
            keep = np.ones_like(box_sizes_px, dtype=bool)
        if np.count_nonzero(keep) < 2:
            logger.warning(
                "Image too small for a box-counting fit - fractal "
                "dimension undefined"
            )
            return nan, nan

        # Calculate fractal dimension according to formula:
        # log(N(k)) = -D * log(k) + C
        x_data = np.log(box_sizes_px[keep])
        y_data = np.log(counts[keep])
        coeffs = np.polyfit(x_data, y_data, 1)
        fractal_dim = -coeffs[0]

        # R^2 shows whether a power law describes the counts at all
        y_pred = coeffs[0] * x_data + coeffs[1]
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else nan

        logger.info(
            f"Fractal dimension: {fractal_dim:.4f}, R²: {r_squared:.4f}"
        )

        # Diagnostic plot
        if save_plot_path:
            try:
                save_plot_path_fig = save_plot_path / f"{filename}.png"
                save_plot_path_fig.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    x_data, y_data, color="blue", label="Measured data"
                )
                plt.plot(
                    x_data,
                    y_pred,
                    color="red",
                    linestyle="--",
                    label=f"Fit: D={fractal_dim:.3f}, $R^2$={r_squared:.3f}",
                )
                plt.title(
                    "Box-Counting Analysis: Fractal Dimension = "
                    f"{fractal_dim:.3f}\n"
                    f"{filename.replace('_fractal_dimension', '')}"
                )
                plt.xlabel("log(Box size $\\epsilon$)")
                plt.ylabel("log(Box count $N(\\epsilon)$)")
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.savefig(save_plot_path_fig)
                plt.close()
                logger.info(
                    f"Saved fractal dimension plot: {save_plot_path_fig}"
                )
            except Exception as e:
                logger.error(f"Failed to save fractal dimension plot: {e}")

        return fractal_dim, r_squared

    def calculate_coordination_number(
        self,
        save_plot_path: Optional[Path] = None,
        base_filename: str = "analysis",
    ) -> Tuple[Optional[Dict[int, int]], Optional[Dict[str, float]]]:
        """
        Calculate coordination number using Voronoi tessellation.

        The coordination number represents the number of nearest neighbors
        for each pore, determined through Voronoi tessellation. This metric
        describes the topological connectivity of the porous structure.

        The tessellation is built over ``neighbour_props`` and the count is
        reported for ``props`` only. Border pores therefore still separate
        their neighbours' cells, but get no coordination number of their
        own, because their cells are cut by the frame.

        Parameters
        ----------
        save_plot_path : Path, optional
            Directory path to save Voronoi diagram visualization with overlaid
            mask and coordination numbers (default: None — plot not saved)
        base_filename : str, optional
            Base filename for the output plot (default: 'analysis')

        Returns
        -------
        Tuple[Optional[Dict[int, int]], Optional[Dict[str, float]]]
            Tuple containing:
            - First element: Dictionary mapping pore_id to coordination number
              (e.g., {1: 5, 2: 6, 3: 4}). None if insufficient pores.
            - Second element: Dictionary containing statistical measures
              (mean, median, std, min, max). None if insufficient pores.
            Returns (None, None) if fewer than 4 pores take part in the
            tessellation or none is to be reported.

        Notes
        -----
        The Voronoi tessellation connects pore centroids to their natural
        neighbors, where each Voronoi cell edge represents a neighbor
        relationship.

        See Also
        --------
        scipy.spatial.Voronoi : Voronoi diagram computation
        """
        if len(self.neighbour_props) < 4 or not self.props:
            logger.warning("Not enough points to generate Voronoi Diagram")
            return None, None
        centroids = np.array([p.centroid for p in self.neighbour_props])
        centroids = centroids[:, ::-1]  # [y, x] -> x, y

        voronoi = spatial.Voronoi(points=centroids)
        neighbor_counts = np.bincount(
            voronoi.ridge_points.ravel(), minlength=len(centroids)
        )

        neighbour_index_by_label = {
            p.label: i for i, p in enumerate(self.neighbour_props)
        }
        reported = np.array(
            [neighbour_index_by_label[p.label] for p in self.props]
        )
        coordination_numbers = neighbor_counts[reported]

        stats = calculate_statistics(values=coordination_numbers)
        logger.info(
            f"Coordination number: Mean={stats['mean']:.2f},"
            f"Std={stats['std']:.2f}"
        )

        cn_by_pore_id = {
            p.label: int(cn)
            for p, cn in zip(self.props, coordination_numbers)
        }

        if save_plot_path:
            try:
                save_plot_path_fig = (
                    save_plot_path / f"{base_filename}_coordination_number.png"
                )
                save_plot_path_fig.parent.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(self.mask, cmap="gray_r", alpha=0.3)
                spatial.voronoi_plot_2d(
                    voronoi,
                    ax=ax,
                    show_vertices=False,
                    line_colors="blue",
                    line_width=1,
                    line_alpha=0.4,
                )
                is_reported = np.zeros(len(centroids), dtype=bool)
                is_reported[reported] = True
                ax.scatter(
                    centroids[~is_reported, 0], centroids[~is_reported, 1],
                    c="grey", s=20, label="Border pores (neighbours only)"
                )
                ax.scatter(
                    centroids[is_reported, 0], centroids[is_reported, 1],
                    c="green", s=20, label="Analyzed pores"
                )
                for idx, cn in zip(reported, coordination_numbers):
                    ax.text(
                        centroids[idx, 0],
                        centroids[idx, 1],
                        str(cn),
                        color="darkgreen",
                        fontsize=8,
                    )
                ax.legend()
                ax.set_title(
                    f"Topological Voronoi Analysis - {base_filename}\n"
                    f"Mean CN: {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
                plt.savefig(save_plot_path_fig)
                plt.close()
                logger.info(
                    f"Saved coordination number plot: {save_plot_path_fig}"
                )
            except Exception as e:
                logger.error(f"Failed to save coordination number plot: {e}")

        return cn_by_pore_id, stats

    def calculate_all_topology_metrics(
        self,
        output_dir: Optional[Path] = None,
        base_filename: str = "analysis",
    ) -> Dict[str, any]:
        """
        Calculate all topology and connectivity metrics.

        Parameters
        ----------
        output_dir : Path, optional
            Directory to save plots
        base_filename : str, optional
            Base filename for output files (default: 'analysis')

        Returns
        -------
        Dict[str, any]
            Dictionary containing:
            - fractal_dimension
            - fractal_r_squared
            - coordination_number_stats (aggregated statistics)
            - coordination_number_individual (dict mapping pore_id to CN)
        """
        # Update save paths if output_dir is provided
        if output_dir:
            fractal_plot_path = output_dir
            fractal_filename = f"{base_filename}_fractal_dimension"
            cn_plot_path = output_dir
        else:
            fractal_plot_path = None
            fractal_filename = "fractal_dimension_plot"
            cn_plot_path = None

        fractal_dim, r_squared = self.calculate_fractal_dimension(
            save_plot_path=fractal_plot_path, filename=fractal_filename
        )
        cn_individual, cn_stats = self.calculate_coordination_number(
            save_plot_path=cn_plot_path, base_filename=base_filename
        )

        results = {
            "fractal_dimension": fractal_dim,
            "fractal_r_squared": r_squared,
            "coordination_number_stats": cn_stats,
            "coordination_number_individual": cn_individual,
        }
        return results


class PorousMaterialAnalyzer:
    """
    Comprehensive analyzer for porous material microstructure.

    This class provides a complete workflow for quantitative analysis of porous
    materials from instance-segmented masks. It integrates multiple analysis
    methods to characterize the morphology, topology, and spatial organization
    of pores.

    Analysis categories:
    1. **Pore Morphology**: Individual pore metrics (area, perimeter, shape
    descriptors)
    2. **Global Descriptors**: Material-level properties (porosity, anisotropy)
    3. **Spatial Relations**: Nearest neighbor distances and distributions
    4. **Topology & Connectivity**: Fractal dimension, coordination number

    Parameters
    ----------
    mask_path : str
        Path to instance mask TIFF file where each pore has a unique integer
        label
        (0 = background, 1, 2, ... n = pore IDs)
    pixel_size : float
        Physical size of one pixel in micrometers.
    generate_plots : bool, optional
        Whether to generate visualization plots (default: True)
    output_base_dir : Path
        Base directory for all outputs.
    reject_boundary_pores : bool, optional
        Whether to exclude pores that touch image boundaries from analysis
        (default: True). Boundary pores may have incomplete measurements.
    boundary_tolerance : int, optional
        Tolerance in pixels for boundary detection (default: 3).
        Pores within this distance from boundaries will be rejected.
        Useful when annotators might not be precise at edges.
    plot_boundary_rejection : bool, optional
        Whether to generate a visualization showing which pores were analyzed
        vs rejected (default: True). Only generated if boundary rejection is
        enabled and pores were actually rejected.

    Attributes
    ----------
    mask : np.ndarray
        Loaded instance mask array
    props : List[RegionProperties]
        Region properties for each detected pore (filtered by boundary
        rejection if enabled)
    props_all : List[RegionProperties]
        All region properties before boundary filtering
    base_filename : str
        Base filename extracted from mask path
    output_dir : Dict[str, Path]
        Dictionary of output directory paths ('base', 'plots', 'data',
        'reports')
    reject_boundary_pores : bool
        Whether boundary pores are rejected
    boundary_tolerance : int
        Tolerance in pixels for boundary detection
    morphology_results : List[Dict[str, float]]
        Individual pore morphology metrics
    morphology_aggregated : Dict[str, Dict[str, float]]
        Aggregated morphology statistics
    global_descriptors : Dict[str, float]
        Global microstructure descriptors
    spatial_metrics : Dict[str, float]
        Spatial relation metrics
    topology_metrics : Dict[str, any]
        Topology and connectivity metrics

    Examples
    --------
    Basic usage:

    >>> analyzer = PorousMaterialAnalyzer(
    ...     mask_path='sample_foam.tif',
    ...     pixel_size=1.5,  # µm/pixel
    ...     generate_plots=True
    ... )
    >>> results = analyzer.analyze_all()
    >>> print(f"Porosity: {results['global_descriptors']['porosity']:.2%}")
    Porosity: 34.5%

    Advanced usage with custom output directory:

    >>> from pathlib import Path
    >>> analyzer = PorousMaterialAnalyzer(
    ...     mask_path='sample.tif',
    ...     output_base_dir=Path('./my_results')
    ... )
    >>> # Run individual analyses
    >>> morph = analyzer.calculate_morphology_metrics()
    >>> global_desc = analyzer.calculate_global_descriptors()
    >>> # Generate report
    >>> report_path = analyzer.generate_report()
    """

    def __init__(
        self,
        mask_path: str,
        pixel_size: float,
        output_base_dir: Path,
        generate_plots: bool = True,
        reject_boundary_pores: bool = True,
        boundary_tolerance: int = 3,
        plot_boundary_rejection: bool = True,
    ) -> None:
        self.mask_path = Path(mask_path)
        self.pixel_size = pixel_size
        self.generate_plots = generate_plots
        self.output_base_dir = output_base_dir
        self.reject_boundary_pores = reject_boundary_pores
        self.boundary_tolerance = boundary_tolerance
        self.plot_boundary_rejection = plot_boundary_rejection

        self.mask = self._load_mask()
        self.props_all = self._extract_regionprops_from_mask()

        # Filter boundary pores if requested
        if self.reject_boundary_pores:
            self.props = self._filter_boundary_pores(
                self.props_all,
                self.mask.shape,
                self.boundary_tolerance
            )
            logger.info(
                f"Filtered boundary pores: {len(self.props_all)} -> "
                f"{len(self.props)} pores "
                f"(rejected {len(self.props_all) - len(self.props)})"
            )
        else:
            self.props = self.props_all
            logger.info("Boundary pores NOT rejected - using all pores")

        # Create organized output directories
        self.base_filename = self.mask_path.stem
        self.output_dir = self._create_output_directories()

        # Storage for results
        self.morphology_results = None
        self.morphology_aggregated = None
        self.global_descriptors = None
        self.spatial_metrics = None
        self.topology_metrics = None

    def _create_output_directories(self) -> Dict[str, Path]:
        """
        Create organized output directory structure.

        Returns:
            Dictionary with paths for different output types
        """
        base_output_dir = self.output_base_dir / self.base_filename

        output_dirs = {
            "base": base_output_dir,
            "plots": base_output_dir / "plots",
            "data": base_output_dir / "data",
            "reports": base_output_dir / "reports",
        }

        # Create all directories
        try:
            for dir_path in output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Created output directory structure at: {base_output_dir}"
            )
        except Exception as e:
            logger.error(f"Failed to create output directories: {e}")
            raise

        return output_dirs

    def _load_mask(self) -> np.ndarray:
        """Load instance mask from file."""
        try:
            mask = tifffile.imread(self.mask_path)
            logger.info(f"Loaded mask from: {self.mask_path}")
            logger.info(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            return mask
        except FileNotFoundError:
            logger.error(f"Mask file not found: {self.mask_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load mask: {e}")
            raise

    def _extract_regionprops_from_mask(self) -> List[RegionProperties]:
        """Extract region properties from instance mask."""
        try:
            props: List[RegionProperties] = regionprops(self.mask)
            logger.info(f"Extracted {len(props)} pores from mask")
            if len(props) == 0:
                logger.warning("No pores detected in mask!")
            return props
        except Exception as e:
            logger.error(f"Failed to extract region properties: {e}")
            raise

    @staticmethod
    def _filter_boundary_pores(
        props: List[RegionProperties],
        mask_shape: Tuple[int, int],
        tolerance: int = 3,
    ) -> List[RegionProperties]:
        """
        Filter out pores that touch or are near image boundaries.

        This method excludes pores whose bounding boxes touch or are within
        a specified tolerance distance from any image boundary. Boundary pores
        often have incomplete measurements and can skew statistical analyses.

        Parameters
        ----------
        props : List[RegionProperties]
            List of region properties to filter
        mask_shape : Tuple[int, int]
            Shape of the mask (height, width)
        tolerance : int, optional
            Distance in pixels from boundary within which pores are rejected
            (default: 3). A tolerance of 0 means only pores directly touching
            boundaries are rejected. Higher values account for imprecise
            annotations near boundaries.

        Returns
        -------
        List[RegionProperties]
            Filtered list of region properties excluding boundary pores

        Notes
        -----
        A pore is considered a boundary pore if any part of its bounding box
        satisfies:
        - min_row <= tolerance
        - min_col <= tolerance
        - max_row >= height - tolerance
        - max_col >= width - tolerance

        Examples
        --------
        >>> props_filtered = PorousMaterialAnalyzer._filter_boundary_pores(
        ...     props, mask_shape=(512, 512), tolerance=5
        ... )
        >>> print(f"Kept {len(props_filtered)} of {len(props)} pores")
        """
        img_h, img_w = mask_shape
        filtered_props = []

        for prop in props:
            min_r, min_c, max_r, max_c = prop.bbox
            touches_border = (
                (min_r <= tolerance)
                or (min_c <= tolerance)
                or (max_r >= img_h - tolerance)
                or (max_c >= img_w - tolerance)
            )
            if not touches_border:
                filtered_props.append(prop)

        return filtered_props

    @staticmethod
    def _edge_correction_weight(
        bbox: Tuple[int, int, int, int],
        mask_shape: Tuple[int, int],
        tolerance: int,
    ) -> float:
        """
        Miles-Lantuejoul weight of a pore kept by the border filter.

        A pore survives the filter only if its bounding box fits inside the
        frame shrunk by the tolerance, so the larger the pore, the fewer
        positions it could have occupied and still be measured. Weighting
        each kept pore by the inverse of that chance - the frame area over
        the area of positions that would have kept it - removes the bias
        towards small pores that the filter otherwise puts into size,
        shape and orientation statistics.

        Parameters
        ----------
        bbox : Tuple[int, int, int, int]
            ``(min_row, min_col, max_row, max_col)`` with exclusive maxima,
            as given by ``RegionProperties.bbox``.
        mask_shape : Tuple[int, int]
            ``(height, width)`` of the frame in pixels.
        tolerance : int
            Border tolerance used by ``_filter_boundary_pores``.

        Returns
        -------
        float
            Weight of at least 1; equal to 1 only in the limit of a point
            pore in an infinitely large frame.

        Raises
        ------
        ValueError
            If the bounding box could not have passed the filter, which
            would make the weight undefined.
        """
        img_h, img_w = mask_shape
        min_r, min_c, max_r, max_c = bbox
        # Kept iff tolerance < min_row and max_row < height - tolerance
        # (likewise for columns), which leaves this many top-left positions
        # along each axis.
        rows_free = img_h - 2 * tolerance - 1 - (max_r - min_r)
        cols_free = img_w - 2 * tolerance - 1 - (max_c - min_c)
        if rows_free < 1 or cols_free < 1:
            raise ValueError(
                f"Bounding box {bbox} cannot pass a {tolerance} px border "
                f"filter in a {img_h}x{img_w} frame."
            )
        return (img_h * img_w) / (rows_free * cols_free)

    def plot_boundary_rejection_visualization(
        self, save_to_plots_dir: bool = True
    ) -> Optional[Path]:
        """
        Visualize which pores are analyzed vs rejected due to boundary
        proximity.

        Creates a plot showing the mask with color-coded pore centroids:
        - Green: Pores included in analysis
        - Red: Pores rejected due to boundary proximity

        This visualization is only generated if boundary rejection was enabled
        and at least one pore was rejected.

        Parameters
        ----------
        save_to_plots_dir : bool, optional
            Whether to save the plot to the plots directory (default: True)

        Returns
        -------
        Path or None
            Path to the saved plot, or None if no plot was generated
        """
        # Only create plot if boundary rejection was used and pores were rejec.
        if not self.reject_boundary_pores:
            logger.info("Boundary rejection disabled - skipping visualization")
            return None

        n_rejected = len(self.props_all) - len(self.props)
        if n_rejected == 0:
            logger.info("No pores rejected - skipping visualization")
            return None

        logger.info(
            f"Generating boundary rejection visualization "
            f"({n_rejected} pores rejected)..."
        )

        # Get rejected pores (difference between all and filtered)
        analyzed_labels = {prop.label for prop in self.props}
        rejected_props = [
            prop for prop in self.props_all if
            prop.label not in analyzed_labels
        ]

        # Extract centroids
        analyzed_centroids = np.array(
            [prop.centroid for prop in self.props]
        )
        rejected_centroids = np.array(
            [prop.centroid for prop in rejected_props]
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(self.mask, cmap="gray_r", alpha=0.5)

        # Plot rejected pores (red X markers)
        if len(rejected_centroids) > 0:
            ax.scatter(
                rejected_centroids[:, 1],  # x coordinate (column)
                rejected_centroids[:, 0],  # y coordinate (row)
                c="red",
                marker="x",
                s=100,
                linewidths=2,
                label=f"Rejected boundary pores (n={len(rejected_props)})",
                zorder=3,
            )

        # Plot analyzed pores (green circles)
        if len(analyzed_centroids) > 0:
            ax.scatter(
                analyzed_centroids[:, 1],
                analyzed_centroids[:, 0],
                c="green",
                marker="o",
                s=50,
                alpha=0.7,
                label=f"Analyzed pores (n={len(self.props)})",
                zorder=2,
            )

        # Draw boundary tolerance zone
        img_h, img_w = self.mask.shape
        tolerance = self.boundary_tolerance

        # Draw tolerance zone as dashed rectangles
        from matplotlib.patches import Rectangle

        # Outer boundary (image edge)
        outer_rect = Rectangle(
            (0, 0),
            img_w,
            img_h,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="Image boundary",
        )
        ax.add_patch(outer_rect)

        # Inner boundary (tolerance zone)
        if tolerance > 0:
            inner_rect = Rectangle(
                (tolerance, tolerance),
                img_w - 2 * tolerance,
                img_h - 2 * tolerance,
                linewidth=2,
                edgecolor="orange",
                facecolor="none",
                linestyle="--",
                label=f"Tolerance zone ({tolerance}px)",
            )
            ax.add_patch(inner_rect)

        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(
            f"Boundary Pore Rejection Visualization - {self.base_filename}\n"
            f"Tolerance: {tolerance} pixels | "
            f"Analyzed: {len(self.props)} | Rejected: {n_rejected}",
            fontsize=12,
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # Save plot
        if save_to_plots_dir:
            plot_path = (
                self.output_dir["plots"]
                / f"{self.base_filename}_boundary_rejection.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved boundary rejection plot: {plot_path}")
            return plot_path
        else:
            plt.show()
            plt.close()
            return None

    def calculate_morphology_metrics(self) -> List[Dict[str, float]]:
        """
        Calculate morphological metrics for each individual pore.

        Each pore also carries an ``edge_correction_weight``: the
        Miles-Lantuejoul weight when border pores are rejected, and 1.0
        when they are not, since then no pore is selected by its size.

        Returns:
            List of dictionaries, each containing metrics for one pore
        """
        logger.info(
            f"Calculating morphology metrics for {len(self.props)} pores..."
        )
        results = []
        for prop in self.props:
            pore_result = {"pore_id": prop.label}
            calculator = PoreMorphologyMetrics(
                prop, pixel_size=self.pixel_size
            )
            for name, method in inspect.getmembers(
                calculator, predicate=inspect.ismethod
            ):
                if not name.startswith("_"):
                    result_dict = method()
                    pore_result.update(result_dict)
            pore_result[EDGE_WEIGHT_KEY] = (
                self._edge_correction_weight(
                    prop.bbox, self.mask.shape, self.boundary_tolerance
                )
                if self.reject_boundary_pores else 1.0
            )
            results.append(pore_result)

        self.morphology_results = results
        logger.info(f"Morphology metrics calculated for {len(results)} pores")
        return results

    def aggregate_morphology_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate morphology metrics across all pores.

        Returns:
            Dictionary with statistical summaries for each metric
        """
        if self.morphology_results is None:
            self.calculate_morphology_metrics()

        logger.info("Aggregating morphology metrics across all pores...")
        self.morphology_aggregated = (
            PoreMorphologyMetrics.aggregate_morphology_results(
                self.morphology_results
            )
        )
        logger.info(
            f"Aggregated {len(self.morphology_aggregated)} morphology metrics"
        )
        return self.morphology_aggregated

    def calculate_global_descriptors(self) -> Dict[str, float]:
        """
        Calculate global microstructure descriptors.

        Returns:
            Dictionary containing porosity, local_porosity_variance,
            and anisotropy
        """
        if self.morphology_results is None:
            self.calculate_morphology_metrics()

        logger.info("Calculating global microstructure descriptors...")
        calculator = GlobalMicrostructureDescriptors(
            mask=self.mask,
            morphology_results=self.morphology_results,
            pixel_size=self.pixel_size,
        )
        self.global_descriptors = calculator.calculate_all_global_descriptors()
        logger.info(
            "Global descriptors calculated: "
            f"Porosity={self.global_descriptors['porosity']:.4f}, "
            f"Anisotropy={self.global_descriptors['anisotropy']:.4f}"
        )
        return self.global_descriptors

    def calculate_spatial_metrics(self) -> Dict[str, float]:
        """
        Calculate spatial relation metrics.

        Returns:
            Dictionary with nearest neighbor distance statistics
        """
        logger.info("Calculating spatial relation metrics...")
        calculator = SpatialRelationMetrics(
            props=self.props,
            generate_plots=self.generate_plots,
            pixel_size=self.pixel_size,
            neighbour_props=self.props_all,
        )
        self.spatial_metrics = calculator.calculate_spatial_metrics(
            output_dir=self.output_dir["plots"]
        )
        logger.info(
            "Spatial metrics calculated: "
            f"Mean NN distance={self.spatial_metrics['mean']:.3f} µm"
        )
        return self.spatial_metrics

    def calculate_topology_metrics(self) -> Dict[str, any]:
        """
        Calculate topology and connectivity metrics.

        Returns:
            Dictionary with fractal dimension and coordination number
        """
        logger.info("Calculating topology and connectivity metrics...")
        calculator = TopologyConnectivityAnalysis(
            mask=self.mask,
            props=self.props,
            pixel_size=self.pixel_size,
            neighbour_props=self.props_all,
        )
        self.topology_metrics = calculator.calculate_all_topology_metrics(
            output_dir=self.output_dir["plots"],
            base_filename=self.base_filename,
        )
        logger.info(
            "Topology metrics calculated: Fractal dimension="
            f"{self.topology_metrics['fractal_dimension']:.4f}"
        )
        return self.topology_metrics

    def analyze_all(
        self, generate_excel_report: bool = True
    ) -> Dict[str, any]:
        """
        Perform complete analysis of porous material microstructure.

        This is the main method that orchestrates all analyses and generates
        comprehensive results. It runs morphology, global, spatial, and
        topology analyses in sequence and optionally generates an Excel report.

        Parameters
        ----------
        generate_excel_report : bool, optional
            Whether to automatically generate and save Excel report
            (default: True)

        Returns
        -------
        Dict[str, any]
            Comprehensive results dictionary containing:

            - **morphology_individual** : List[Dict[str, float]]
                Per-pore morphology metrics
            - **morphology_aggregated** : Dict[str, Dict[str, float]]
                Statistical summaries of morphology metrics
            - **global_descriptors** : Dict[str, float]
                Porosity, anisotropy, local porosity variance
            - **spatial_metrics** : Dict[str, float]
                Nearest neighbor distance statistics
            - **topology_metrics** : Dict[str, any]
                Fractal dimension and coordination number
            - **metadata** : Dict[str, any]
                Analysis metadata (mask path, pixel size, n_pores, shape)

        Notes
        -----
        This method automatically:
        - Logs progress at each step
        - Generates and saves visualization plots
        - Creates organized output directory structure
        - Optionally generates Excel report

        All results are also stored as class attributes for later access.

        Examples
        --------
        >>> analyzer = PorousMaterialAnalyzer('foam_sample.tif')
        >>> results = analyzer.analyze_all()
        >>> # Access specific results
        >>> porosity = results['global_descriptors']['porosity']
        >>> mean_area = results['morphology_aggregated']['area']['mean']
        >>> fractal_dim = results['topology_metrics']['fractal_dimension']

        Run analysis without generating report:

        >>> results = analyzer.analyze_all(generate_excel_report=False)
        >>> # Generate report later
        >>> report_path = analyzer.generate_report()
        """
        logger.info("Starting comprehensive porous material analysis...")

        # Calculate all metrics
        morphology_individual = self.calculate_morphology_metrics()
        morphology_aggregated = self.aggregate_morphology_metrics()
        global_descriptors = self.calculate_global_descriptors()
        spatial_metrics = self.calculate_spatial_metrics()
        topology_metrics = self.calculate_topology_metrics()

        # Add individual coordination numbers to morphology results
        if (
            topology_metrics
            and topology_metrics.get('coordination_number_individual')
        ):
            cn_by_pore_id = topology_metrics['coordination_number_individual']

            # Add CN to each pore in morphology_individual
            for pore in morphology_individual:
                pore_id = pore['pore_id']
                pore['coordination_number'] = cn_by_pore_id.get(pore_id, None)

        # Generate boundary rejection visualization if enabled
        if self.plot_boundary_rejection and self.generate_plots:
            self.plot_boundary_rejection_visualization()

        # Compile all results
        results = {
            "morphology_individual": morphology_individual,
            "morphology_aggregated": morphology_aggregated,
            "global_descriptors": global_descriptors,
            "spatial_metrics": spatial_metrics,
            "topology_metrics": topology_metrics,
            "metadata": {
                "mask_path": str(self.mask_path),
                "pixel_size": self.pixel_size,
                "n_pores": len(self.props),
                "mask_shape": self.mask.shape,
            },
        }

        logger.info("Analysis complete!")

        # Generate Excel report if requested
        if generate_excel_report:
            self.generate_report()

        return results

    def generate_report(self) -> Path:
        """
        Generate a comprehensive Excel report of the analysis.

        Creates a multi-sheet Excel workbook containing all analysis results
        in a structured, filterable format suitable for further analysis or
        publication.

        Returns
        -------
        Path
            Path object pointing to the generated Excel report

        Notes
        -----
        The Excel report contains two sheets:

        **Sheet 1: Analysis_Results**
            Comprehensive metrics table with columns:
            - Category: Analysis category (Metadata, Morphology, Global
            Descriptors, etc.)
            - Metric: Specific metric name
            - Value: Numerical value
            - Unit: Measurement unit (µm, µm², degrees, or dimensionless)
            - Description: Detailed explanation of the metric

            This format allows easy filtering by category and searching for
            specific metrics.

        **Sheet 2: Individual_Pores**
            Raw data for each detected pore with columns:
            - pore_id: Unique pore identifier
            - area, perimeter, circularity, solidity
            - min_feret, max_feret
            - ellipse_major_axis, ellipse_minor_axis, aspect_ratio,
            ellipse_angle

        The report filename follows the pattern:
        `{input_filename}_analysis_report.xlsx`
        """
        logger.info("Generating Excel report...")

        # Ensure analysis has been run
        if self.morphology_aggregated is None:
            logger.warning(
                "Analysis not yet run. Running complete analysis first..."
            )
            self.analyze_all(generate_excel_report=False)

        # Create report filename
        report_filename = f"{self.base_filename}_analysis_report.xlsx"
        report_path = self.output_dir["reports"] / report_filename

        # ===== Prepare Sheet 1: Comprehensive Analysis Results =====
        logger.info("Preparing comprehensive analysis results...")
        report_data = []

        # --- METADATA ---
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Analysis Date",
                "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Unit": "-",
                "Description": "Date and time of analysis",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Mask File",
                "Value": str(self.mask_path.name),
                "Unit": "-",
                "Description": "Input mask filename",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Number of Pores",
                "Value": len(self.props),
                "Unit": "pcs",
                "Description": "Total number of detected pores",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Pixel Size",
                "Value": self.pixel_size,
                "Unit": "µm",
                "Description": "Physical size of one pixel",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Image Width",
                "Value": self.mask.shape[1],
                "Unit": "pixels",
                "Description": "Image width in pixels",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Image Height",
                "Value": self.mask.shape[0],
                "Unit": "pixels",
                "Description": "Image height in pixels",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Total Image Area",
                "Value": self.mask.shape[0]
                * self.mask.shape[1]
                * (self.pixel_size**2),
                "Unit": "µm²",
                "Description": "Total area of the analyzed region",
            }
        )
        report_data.append(
            {
                "Category": "Metadata",
                "Metric": "Boundary Pores Rejected",
                "Value": "Yes" if self.reject_boundary_pores else "No",
                "Unit": "-",
                "Description": (
                    "Whether pores touching boundaries were excluded"
                ),
            }
        )
        if self.reject_boundary_pores:
            report_data.append(
                {
                    "Category": "Metadata",
                    "Metric": "Boundary Tolerance",
                    "Value": self.boundary_tolerance,
                    "Unit": "pixels",
                    "Description": "Distance from boundary for pore rejection",
                }
            )
            report_data.append(
                {
                    "Category": "Metadata",
                    "Metric": "Pores Rejected",
                    "Value": len(self.props_all) - len(self.props),
                    "Unit": "pcs",
                    "Description": (
                        "Number of boundary pores excluded from analysis"
                    ),
                }
            )

        # --- MORPHOLOGY (AGGREGATED) ---
        metric_descriptions = {
            "area": "Pore area",
            "perimeter": "Pore perimeter",
            "circularity": "Shape circularity (4πA/P²)",
            "solidity": "Ratio of area to convex hull area",
            "roundness": "Roundness (4*Area / (π*MajorAxis²))",
            "min_feret": "Minimum Feret diameter",
            "max_feret": "Maximum Feret diameter",
            "equivalent_diameter": (
                "Equivalent diameter (diameter of circle with same area)"
            ),
            "ellipse_major_axis": "Major axis of fitted ellipse",
            "ellipse_minor_axis": "Minor axis of fitted ellipse",
            "aspect_ratio": "Ratio of major to minor axis",
            "ellipse_angle": "Orientation angle of fitted ellipse",
        }

        for metric_name, stats in self.morphology_aggregated.items():
            # Determine unit
            if "area" in metric_name:
                unit = "µm²"
            elif (
                "perimeter" in metric_name
                or "feret" in metric_name
                or "axis" in metric_name
                or "diameter" in metric_name
            ):
                unit = "µm"
            elif "angle" in metric_name:
                unit = "degrees"
            else:
                unit = "-"

            description = metric_descriptions.get(
                metric_name, f"{metric_name} of pores"
            )

            # Add mean
            report_data.append(
                {
                    "Category": "Morphology",
                    "Metric": f"{metric_name} (mean)",
                    "Value": stats["mean"],
                    "Unit": unit,
                    "Description": f"Mean {description}",
                }
            )
            # Add std
            report_data.append(
                {
                    "Category": "Morphology",
                    "Metric": f"{metric_name} (std)",
                    "Value": stats["std"],
                    "Unit": unit,
                    "Description": f"Standard deviation of {description}",
                }
            )
            # Add median
            report_data.append(
                {
                    "Category": "Morphology",
                    "Metric": f"{metric_name} (median)",
                    "Value": stats["median"],
                    "Unit": unit,
                    "Description": f"Median {description}",
                }
            )
            # Add min
            report_data.append(
                {
                    "Category": "Morphology",
                    "Metric": f"{metric_name} (min)",
                    "Value": stats["min"],
                    "Unit": unit,
                    "Description": f"Minimum {description}",
                }
            )
            # Add max
            report_data.append(
                {
                    "Category": "Morphology",
                    "Metric": f"{metric_name} (max)",
                    "Value": stats["max"],
                    "Unit": unit,
                    "Description": f"Maximum {description}",
                }
            )
            report_data.extend(
                corrected_morphology_rows(
                    metric_name, stats, unit, description
                )
            )

        # --- GLOBAL DESCRIPTORS ---
        report_data.append(
            {
                "Category": "Global Descriptors",
                "Metric": "Porosity",
                "Value": self.global_descriptors["porosity"],
                "Unit": "-",
                "Description": "Volume fraction of pores (0-1)",
            }
        )
        report_data.append(
            {
                "Category": "Global Descriptors",
                "Metric": "Local Porosity Variance",
                "Value": self.global_descriptors[
                    "local_porosity_variance"
                ],
                "Unit": "-",
                "Description": "Variance of porosity across local regions",
            }
        )
        report_data.append(
            {
                "Category": "Global Descriptors",
                "Metric": "Anisotropy",
                "Value": self.global_descriptors["anisotropy"],
                "Unit": "-",
                "Description": (
                    "Degree of preferred orientation "
                    "(0=isotropic, 1=aligned)",
                )
            }
        )
        report_data.append(
            {
                "Category": "Global Descriptors",
                "Metric": "Anisotropy (corrected)",
                "Value": self.global_descriptors["anisotropy_corrected"],
                "Unit": "-",
                "Description": (
                    "Anisotropy weighted by edge_correction_weight, "
                    "compensating for border pores excluded by size"
                ),
            }
        )

        # --- SPATIAL METRICS ---
        for key, value in self.spatial_metrics.items():
            report_data.append(
                {
                    "Category": "Spatial Metrics",
                    "Metric": f"Nearest Neighbor Distance ({key})",
                    "Value": value,
                    "Unit": "µm",
                    "Description": (
                        f"{key.capitalize()} nearest neighbor distance "
                        "between pore centroids",
                    )
                }
            )

        # --- TOPOLOGY & CONNECTIVITY ---
        report_data.append(
            {
                "Category": "Topology & Connectivity",
                "Metric": "Fractal Dimension",
                "Value": self.topology_metrics["fractal_dimension"],
                "Unit": "-",
                "Description": (
                    "Fractal dimension of the pore boundaries from "
                    "box-counting"
                ),
            }
        )
        report_data.append(
            {
                "Category": "Topology & Connectivity",
                "Metric": "Fractal R²",
                "Value": self.topology_metrics["fractal_r_squared"],
                "Unit": "-",
                "Description": "R² coefficient for fractal dimension fit",
            }
        )

        if self.topology_metrics["coordination_number_stats"]:
            cn_stats = self.topology_metrics["coordination_number_stats"]
            report_data.append(
                {
                    "Category": "Topology & Connectivity",
                    "Metric": "Coordination Number (mean)",
                    "Value": cn_stats["mean"],
                    "Unit": "-",
                    "Description": (
                        "Mean number of nearest neighbors per pore"
                    ),
                }
            )
            report_data.append(
                {
                    "Category": "Topology & Connectivity",
                    "Metric": "Coordination Number (std)",
                    "Value": cn_stats["std"],
                    "Unit": "-",
                    "Description": (
                        "Standard deviation of coordination number"
                    ),
                }
            )
            report_data.append(
                {
                    "Category": "Topology & Connectivity",
                    "Metric": "Coordination Number (median)",
                    "Value": cn_stats["median"],
                    "Unit": "-",
                    "Description": "Median coordination number",
                }
            )
            report_data.append(
                {
                    "Category": "Topology & Connectivity",
                    "Metric": "Coordination Number (min)",
                    "Value": cn_stats["min"],
                    "Unit": "-",
                    "Description": "Minimum coordination number",
                }
            )
            report_data.append(
                {
                    "Category": "Topology & Connectivity",
                    "Metric": "Coordination Number (max)",
                    "Value": cn_stats["max"],
                    "Unit": "-",
                    "Description": "Maximum coordination number",
                }
            )
            pores_with_cn = [
                pore for pore in self.morphology_results
                if pore.get("coordination_number") is not None
            ]
            if pores_with_cn:
                cn_corrected = calculate_weighted_statistics(
                    [pore["coordination_number"] for pore in pores_with_cn],
                    [pore[EDGE_WEIGHT_KEY] for pore in pores_with_cn],
                )
                for key, label in CORRECTED_STAT_LABELS.items():
                    report_data.append(
                        {
                            "Category": "Topology & Connectivity",
                            "Metric": (
                                "Coordination Number "
                                f"({key.replace('_', ', ')})"
                            ),
                            "Value": cn_corrected[key],
                            "Unit": "-",
                            "Description": (
                                f"{label} coordination number "
                                "(Miles-Lantuejoul weights)"
                            ),
                        }
                    )

        df_report = pd.DataFrame(report_data)
        df_individual = pd.DataFrame(self.morphology_results)

        logger.info(f"Prepared {len(report_data)} metrics for export")
        logger.info(
            f"Prepared {len(self.morphology_results)} individual pore records"
        )
        try:
            with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
                # Sheet 1: Analysis Results
                df_report.to_excel(
                    writer, sheet_name="Analysis_Results", index=False
                )
                logger.info("Written 'Analysis_Results' sheet")

                # Sheet 2: Individual Pores
                df_individual.to_excel(
                    writer, sheet_name="Individual_Pores", index=False
                )
                logger.info("Written 'Individual_Pores' sheet")

            logger.info(f"Excel report saved to: {report_path}")
            logger.info(f"All plots saved in: {self.output_dir['plots']}")
            logger.info("=" * 70)
            logger.info("ANALYSIS COMPLETE!")
            logger.info("=" * 70)

            return report_path

        except Exception as e:
            logger.error(f"Failed to write Excel report: {e}")
            raise
