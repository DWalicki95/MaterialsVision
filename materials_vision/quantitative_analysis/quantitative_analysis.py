import numpy as np
import tifffile
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from typing import Dict, List, Optional, Tuple
import inspect
import math
import matplotlib.pyplot as plt
from scipy import spatial
from pathlib import Path
import pandas as pd
from datetime import datetime

from config import PIXEL_SIZE, OUTPUT_PATH
from materials_vision.quantitative_analysis.calculate_statistics import (
    calculate_statistics,
)
import logging

logger = logging.getLogger(__name__)


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
    pixel_size : float, optional
        Physical size of one pixel in micrometers (default: PIXEL_SIZE from
        config)

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
        self, prop: RegionProperties, pixel_size: float = PIXEL_SIZE
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

        The angle is measured counter-clockwise from the X-axis.
        0 degrees means the object is oriented horizontally.

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
            'pore_id', 'filename', 'magnification', 'pixel_size',
            'coordination_number'
        }
        metric_names = [
            key for key in morphology_results[0].keys()
            if key not in excluded_fields
        ]

        aggregated_results = {}

        for metric_name in metric_names:
            # Extract values for this metric across all pores
            values = [
                pore[metric_name]
                for pore in morphology_results
                if metric_name in pore
            ]

            if values:
                # Calculate statistics
                stats = calculate_statistics(values)
                aggregated_results[metric_name] = stats

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
        pixel_size: float = PIXEL_SIZE,
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

    def _count_all_pores_area(self) -> float:
        """Sum of all pore areas from morphology results."""
        return np.sum(pore["area"] for pore in self.morphology_results)

    def _count_pores_area_in_mask(self, sub_mask: np.ndarray) -> float:
        """
        Count the total area of pores within a specific mask region.

        Parameters
        ----------
        sub_mask : np.ndarray
            A labeled mask array where each pore has a unique integer label

        Returns
        -------
        float
            Total area of all pores in the sub-mask (in µm²)
        """
        # Count pixels for each label (excluding background label 0)
        pore_pixel_count = np.sum(sub_mask > 0)

        # Convert to area
        pore_area = pore_pixel_count * (self.pixel_size**2)

        return pore_area

    def calculate_porosity(self, mask_: Optional[np.ndarray] = None) -> float:
        """
        Calculate porosity (volume fraction of pores).

        Parameters
        ----------
        mask_ : np.ndarray, optional
            Optional sub-mask for local porosity calculation

        Returns
        -------
        float
            Porosity value (0 to 1)
        """
        if mask_ is None:
            # Global porosity using pre-computed morphology results
            pores_area = self._count_all_pores_area()
            mask_area = self._count_mask_area()
        else:
            # Local porosity for a specific sub-mask
            pores_area = self._count_pores_area_in_mask(mask_)
            mask_area = self._count_mask_area(mask_)

        return pores_area / mask_area

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

    def calculate_anisotropy(self) -> float:
        """
        Calculate anisotropy of foam microstructure from ellipse orientation
        angles.

        Anisotropy measures the degree of preferred orientation:
        - 0: random orientation (isotropic)
        - 1: perfect alignment (highly anisotropic)

        Returns:
            Anisotropy index (0 to 1)
        """
        # Extract ellipse angles from all pores
        angles = []
        for pore in self.morphology_results:
            if "ellipse_angle" in pore:
                angles.append(pore["ellipse_angle"])

        if not angles:
            logger.warning("No ellipse angles found in morphology results")
            return 0.0

        angles = np.array(angles)

        # Convert to radians and double the angle
        # (orientation is π-periodic, not 2π-periodic)
        angles_rad = np.deg2rad(angles * 2)

        # Calculate resultant vector using circular statistics
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))

        # Resultant vector length R (0 to 1)
        # R = 1 means perfect alignment, R = 0 means random
        anisotropy = np.sqrt(mean_cos**2 + mean_sin**2)

        return anisotropy

    def calculate_all_global_descriptors(self) -> Dict[str, float]:
        """
        Calculate all global descriptors.

        Returns:
            Dictionary containing porosity, local_porosity_variance,
            and anisotropy
        """
        results = {
            "porosity": self.calculate_porosity(),
            "local_porosity_variance": (
                self.calculate_local_porosity_variance()),
            "anisotropy": self.calculate_anisotropy(),
        }
        return results


class SpatialRelationMetrics:
    """
    Analyzes spatial relationships between pores.

    Includes nearest neighbor distance analysis.
    """

    def __init__(
        self,
        props: List[RegionProperties],
        generate_plots: bool = True,
        pixel_size: float = PIXEL_SIZE,
    ):
        self.props = props
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
            List of nearest neighbor distances (in µm)
        """
        pore_centroids = np.array([prop.centroid for prop in self.props])
        tree = spatial.KDTree(pore_centroids)
        nearest_neighbors_distances = []
        for pore_centroid in pore_centroids:
            distances, indices = tree.query(pore_centroid, k=2)
            # index 0 is point itself, index 1 is nearest neighbor
            nn_distance = distances[1] * self.pixel_size
            nearest_neighbors_distances.append(nn_distance)
        return nearest_neighbors_distances

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
    - Fractal dimension (box-counting method)
    - Coordination number (Voronoi analysis)
    """

    def __init__(
        self,
        mask: np.ndarray,
        props: List[RegionProperties],
        pixel_size: float = PIXEL_SIZE,
    ):
        self.mask = mask
        self.props = props
        self.pixel_size = pixel_size

    def calculate_fractal_dimension(
        self,
        reject_huge_boxes: bool = True,
        save_plot_path: Optional[Path] = OUTPUT_PATH,
        filename: str = "fractal_dimension_plot",
    ) -> Tuple[float, float]:
        """
        Calculate fractal dimension using box-counting method.

        Implements the Minkowski-Bouligand box-counting method to estimate
        the fractal dimension of the porous structure. The method recursively
        divides the binary mask into smaller boxes and counts how many boxes
        contain part of the structure at each scale.

        The fractal dimension D is estimated from the relationship:
        log(N(ε)) = -D * log(ε) + C

        where N(ε) is the number of boxes of size ε needed to cover the
        structure.

        Parameters
        ----------
        reject_huge_boxes : bool, optional
            Whether to reject large box sizes from linear fitting to improve
            accuracy (default: True). When True, only intermediate scales are
            used for fitting.
        save_plot_path : Path, optional
            Directory path to save diagnostic plot showing the log-log
            relationship
            and linear fit (default: OUTPUT_PATH)
        filename : str, optional
            Filename for the diagnostic plot (
            default: 'fractal_dimension_plot')

        Returns
        -------
        fractal_dim : float
            Estimated fractal dimension (typically between 1 and 2 for 2D
            structures)
        r_squared : float
            R² coefficient of determination for the linear fit, indicating
            fit quality (closer to 1 is better)

        Notes
        -----
        The box-counting method works by:
        1. Converting the instance mask to binary
        2. Cropping to largest square with size as power of 2
        3. Iteratively doubling box size and counting occupied boxes
        4. Fitting a line to log(box_size) vs log(count)
        5. Fractal dimension is the negative slope of this line

        Higher fractal dimensions indicate more complex, space-filling
        structures.
        For porous materials:
        - D ≈ 1: Linear, simple structures
        - D ≈ 1.5-1.7: Typical porous foams
        - D ≈ 2: Highly complex, space-filling structures

        Examples
        --------
        >>> from skimage.measure import regionprops
        >>> import tifffile
        >>> mask = tifffile.imread('porous_material.tif')
        >>> props = regionprops(mask)
        >>> analyzer = TopologyConnectivityAnalysis(mask, props,
        pixel_size=1.5)
        >>> fractal_dim, r_squared = analyzer.calculate_fractal_dimension()
        >>> print(f"Fractal dimension: {fractal_dim:.3f}, R²: {r_squared:.3f}")
        Fractal dimension: 1.652, R²: 0.987

        """
        logger.info(
            "Calculating fractal dimension using box-counting method..."
        )

        # Instance mask -> binary mask
        Z = self.mask > 0
        # Image cut for further split to equal squares
        p = np.min(Z.shape)
        n = 2 ** int(np.floor(np.log2(p)))
        Z = Z[:n, :n]
        box_sizes = []
        counts = []

        k = 1
        curr_Z = Z.copy()

        while k <= n / 2:  # while k is smaller than half of an image
            box_sizes.append(k)
            counts.append(np.sum(curr_Z > 0))

            h, w = curr_Z.shape
            curr_Z = curr_Z.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3)) > 0
            # Box size increased 2-times
            k *= 2

        # Reject points where box sizes are huge
        if reject_huge_boxes:
            valid_idxs = slice(-4, -2) if len(box_sizes) > 4 else slice(None)
        else:
            valid_idxs = slice(None)

        # Calculate fractal dimension according to formula:
        # log(N(k)) = -D * log(k) + C
        x_data = np.log(box_sizes)[valid_idxs]
        y_data = np.log(counts)[valid_idxs]
        coeffs = np.polyfit(x_data, y_data, 1)
        fractal_dim = -coeffs[0]

        # Calculate R^2 (determination coeff.) - how well straight line fits
        # R^2 helps us interpret if we can trust calculated D or if there
        # is something wrong
        y_pred = coeffs[0] * x_data + coeffs[1]
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(
            f"Fractal dimension: {fractal_dim:.4f}, R²: {r_squared:.4f}"
        )

        # Diagnostic plot
        if save_plot_path:
            try:
                save_plot_path_fig = OUTPUT_PATH / "plots" / f"{filename}.png"
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
        save_plot_path: Optional[Path] = OUTPUT_PATH,
        base_filename: str = "analysis",
    ) -> Tuple[Optional[Dict[int, int]], Optional[Dict[str, float]]]:
        """
        Calculate coordination number using Voronoi tessellation.

        The coordination number represents the number of nearest neighbors
        for each pore, determined through Voronoi tessellation. This metric
        describes the topological connectivity of the porous structure.

        Note: If boundary pores were filtered during PorousMaterialAnalyzer
        initialization, they are automatically excluded from this analysis.

        Parameters
        ----------
        save_plot_path : Path, optional
            Directory path to save Voronoi diagram visualization with overlaid
            mask and coordination numbers (default: OUTPUT_PATH)
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
            Returns (None, None) if insufficient pores (<4).

        Notes
        -----
        The Voronoi tessellation connects pore centroids to their natural
        neighbors, where each Voronoi cell edge represents a neighbor
        relationship.

        See Also
        --------
        scipy.spatial.Voronoi : Voronoi diagram computation
        """
        centroids = np.array([p.centroid for p in self.props])
        centroids = centroids[:, ::-1]  # [y, x] -> x, y
        if len(centroids) < 4:
            logger.warning("Not enough points to generate Voronoi Diagram")
            return None, None

        voronoi = spatial.Voronoi(points=centroids)
        neighbor_counts = {i: 0 for i in range(len(centroids))}

        # Count neighbors from Voronoi ridge points
        for p1, p2 in voronoi.ridge_points:
            neighbor_counts[p1] += 1
            neighbor_counts[p2] += 1

        # Convert to array for statistics calculation
        coordination_numbers = np.array(
            [neighbor_counts[i] for i in range(len(centroids))]
        )

        stats = calculate_statistics(values=coordination_numbers)
        logger.info(
            f"Coordination number: Mean={stats['mean']:.2f},"
            f"Std={stats['std']:.2f}"
        )

        # Create mapping from pore_id to coordination number
        cn_by_pore_id = {
            self.props[i].label: neighbor_counts[i]
            for i in range(len(centroids))
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
                all_y = centroids[:, 1]
                all_x = centroids[:, 0]

                # Plot all pores (boundary pores already filtered if enabled)
                ax.scatter(
                    all_x, all_y, c="green", s=20, label="Analyzed pores"
                )

                # Annotate with coordination numbers
                for idx in range(len(centroids)):
                    ax.text(
                        all_x[idx],
                        all_y[idx],
                        str(neighbor_counts[idx]),
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
            fractal_plot_path = OUTPUT_PATH
            fractal_filename = "fractal_dimension_plot"
            cn_plot_path = OUTPUT_PATH

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
    pixel_size : float, optional
        Physical size of one pixel in micrometers (default: PIXEL_SIZE from
        config)
    generate_plots : bool, optional
        Whether to generate visualization plots (default: True)
    output_base_dir : Path, optional
        Base directory for all outputs (default: OUTPUT_PATH from config)
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
        pixel_size: float = PIXEL_SIZE,
        generate_plots: bool = True,
        output_base_dir: Optional[Path] = None,
        reject_boundary_pores: bool = True,
        boundary_tolerance: int = 3,
        plot_boundary_rejection: bool = True,
    ) -> None:
        self.mask_path = Path(mask_path)
        self.pixel_size = pixel_size
        self.generate_plots = generate_plots
        self.output_base_dir = output_base_dir or OUTPUT_PATH
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
            mask=self.mask, props=self.props, pixel_size=self.pixel_size
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
                    "Fractal dimension from box-counting method"
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
