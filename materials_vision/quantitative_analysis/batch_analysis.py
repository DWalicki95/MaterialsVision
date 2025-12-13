import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import logging

from config import PIXEL_SIZE, OUTPUT_PATH, PIXEL_SIZES
from materials_vision.quantitative_analysis.quantitative_analysis import (
    PorousMaterialAnalyzer, PoreMorphologyMetrics,
)
from materials_vision.quantitative_analysis.calculate_statistics import (
    calculate_statistics,
)
from materials_vision.utils import extract_magnification_from_filename

logger = logging.getLogger(__name__)


class BatchPorousMaterialAnalyzer:
    """
    Batch analyzer for multiple porous material images across multiple
    directories.

    This class handles the analysis of multiple images organized in a
    hierarchical directory structure:
    - Parent directory contains subdirectories
    - Each subdirectory represents one material
    - Each subdirectory contains multiple image masks of the same material

    For each material (subdirectory), all images are analyzed and combined into
    a single report where:
    - Pore IDs are continuous across images (image1: 1-N, image2: N+1-M, etc.)
    - Individual pores sheet includes 'filename', 'magnification', and
      'pixel_size' columns
    - Magnification is automatically extracted from filenames
    - Appropriate pixel size is used for each image based on its magnification
    - Analysis results are aggregated across all images in the subdirectory

    Parameters
    ----------
    parent_dir : str or Path
        Path to parent directory containing subdirectories with mask images
    pixel_size : float, optional
        Physical size of one pixel in micrometers (default: PIXEL_SIZE from
        config)
    generate_plots : bool, optional
        Whether to generate visualization plots for each image (default: True)
    output_base_dir : Path, optional
        Base directory for all outputs (default: OUTPUT_PATH from config)
    reject_boundary_pores : bool, optional
        Whether to exclude pores that touch image boundaries (default: True)
    boundary_tolerance : int, optional
        Tolerance in pixels for boundary detection (default: 3)
    plot_boundary_rejection : bool, optional
        Whether to generate boundary rejection visualizations (default: True)
    file_pattern : str, optional
        Glob pattern to match image files (default: "*.tif")
    batch_variance_mode : str, optional
        Method for calculating batch porosity variance
        (default: "across_images").
        Options:
        - "across_images": Calculate variance of porosity across complete
        images
          (measures heterogeneity between different material samples)
        - "within_images": Average of within-image variances
          (average of 2x2 sub-region variances from each image)

    Attributes
    ----------
    parent_dir : Path
        Path to parent directory
    subdirectories : List[Path]
        List of subdirectories found in parent directory
    results : Dict[str, Dict]
        Dictionary storing results for each subdirectory

    Notes
    -----
    **Magnification Extraction from Filenames**

    The class automatically extracts magnification from filenames to use the
    appropriate pixel size for each image. The expected filename pattern is:
    ``[optional_prefix]SAMPLE_MAGNIFICATION_NUMBER_jpg.rf.HASH_masks.tif``

    Examples of valid filenames:
    - ``0ab7de9d-AS2_40_10_jpg.rf.209a8405481b2434b8436c3f3acd60fd_masks.tif``
      (magnification: 40x)
    - ``AS2_100_5_jpg.rf.hash123_masks.tif`` (magnification: 100x)
    - ``sample_250_1_jpg.rf.abc_masks.tif`` (magnification: 250x)

    The magnification value is used to lookup the pixel size from the
    ``PIXEL_SIZES`` dictionary in config.py. If magnification cannot be
    extracted or is not found in the config, the default ``pixel_size``
    parameter is used.

    Examples
    --------
    Basic usage:

    >>> batch_analyzer = BatchPorousMaterialAnalyzer(
    ...     parent_dir='/path/to/materials',
    ...     pixel_size=1.5
    ... )
    >>> batch_analyzer.analyze_all_materials()

    Custom file pattern:

    >>> batch_analyzer = BatchPorousMaterialAnalyzer(
    ...     parent_dir='/path/to/materials',
    ...     file_pattern='*_masks.tif'
    ... )
    >>> batch_analyzer.analyze_all_materials()

    Expected directory structure:
    ```
    parent_dir/
    ├── material_A/
    │   ├── image_001.tif
    │   ├── image_002.tif
    │   └── image_003.tif
    ├── material_B/
    │   ├── image_001.tif
    │   └── image_002.tif
    └── material_C/
        └── image_001.tif
    ```

    Output structure:
    ```
    OUTPUT_PATH/
    ├── material_A/
    │   ├── plots/
    │   │   ├── image_001_*.png
    │   │   ├── image_002_*.png
    │   │   └── image_003_*.png
    │   └── reports/
    │       └── material_A_combined_analysis_report.xlsx
    ├── material_B/
    │   └── reports/
    │       └── material_B_combined_analysis_report.xlsx
    └── material_C/
        └── reports/
            └── material_C_combined_analysis_report.xlsx
    ```
    """

    def __init__(
        self,
        parent_dir: str | Path,
        pixel_size: float = PIXEL_SIZE,
        generate_plots: bool = True,
        output_base_dir: Optional[Path] = None,
        reject_boundary_pores: bool = True,
        boundary_tolerance: int = 3,
        plot_boundary_rejection: bool = True,
        file_pattern: str = "*.tif",
        batch_variance_mode: str = "across_images",
    ) -> None:
        self.parent_dir = Path(parent_dir)
        self.pixel_size = pixel_size
        self.generate_plots = generate_plots
        self.output_base_dir = output_base_dir or OUTPUT_PATH
        self.reject_boundary_pores = reject_boundary_pores
        self.boundary_tolerance = boundary_tolerance
        self.plot_boundary_rejection = plot_boundary_rejection
        self.file_pattern = file_pattern
        self.batch_variance_mode = batch_variance_mode

        # Validate parent directory exists
        if not self.parent_dir.exists():
            raise FileNotFoundError(
                f"Parent directory not found: {self.parent_dir}"
            )

        if not self.parent_dir.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {self.parent_dir}"
            )

        # Find all subdirectories
        self.subdirectories = self._find_subdirectories()
        logger.info(
            f"Found {len(self.subdirectories)} "
            f"subdirectories in {self.parent_dir}"
        )

        # Storage for results
        self.results: Dict[str, Dict] = {}

    def _find_subdirectories(self) -> List[Path]:
        """
        Find all subdirectories in parent directory.

        Returns
        -------
        List[Path]
            List of subdirectory paths
        """
        subdirs = [d for d in self.parent_dir.iterdir() if d.is_dir()]
        subdirs.sort()  # Sort for consistent ordering
        return subdirs

    @staticmethod
    def _get_pixel_size_for_magnification(
        magnification: Optional[int],
        default_pixel_size: float = PIXEL_SIZE
    ) -> float:
        """
        Get pixel size for a given magnification.

        Parameters
        ----------
        magnification : Optional[int]
            Magnification value
        default_pixel_size : float, optional
            Default pixel size to use if magnification not found
            (default: PIXEL_SIZE from config)

        Returns
        -------
        float
            Pixel size in micrometers
        """
        if magnification is None:
            logger.warning(
                f"No magnification provided, using default pixel size: "
                f"{default_pixel_size} µm/px"
            )
            return default_pixel_size

        if magnification not in PIXEL_SIZES:
            logger.warning(
                f"Magnification {magnification}x not found in PIXEL_SIZES "
                "config. "
                f"Available magnifications: {list(PIXEL_SIZES.keys())}. "
                f"Using default pixel size: {default_pixel_size} µm/px"
            )
            return default_pixel_size

        pixel_size = PIXEL_SIZES[magnification]
        logger.info(
            f"Using pixel size {pixel_size} µm/px for "
            f"magnification {magnification}x"
        )
        return pixel_size

    def _find_image_files(self, subdir: Path) -> List[Path]:
        """
        Find all image files in a subdirectory matching the file pattern.

        Excludes hidden files (starting with '.') like .DS_Store.

        Parameters
        ----------
        subdir : Path
            Subdirectory to search

        Returns
        -------
        List[Path]
            List of image file paths, sorted by filename
        """
        # Find all files matching pattern
        all_files = list(subdir.glob(self.file_pattern))

        # Filter out files starting with '.'
        image_files = [f for f in all_files if not f.name.startswith('.')]

        image_files.sort()  # Sort for consistent ordering
        logger.info(f"Found {len(image_files)} images in {subdir.name}")
        return image_files

    def analyze_single_material(
        self,
        subdir: Path,
        generate_excel_report: bool = True
    ) -> Dict:
        """
        Analyze all images in a single subdirectory (one material).

        All images in the subdirectory are analyzed and combined into a single
        report with continuous pore IDs across images.

        Parameters
        ----------
        subdir : Path
            Path to subdirectory containing images of one material
        generate_excel_report : bool, optional
            Whether to generate Excel report (default: True)

        Returns
        -------
        Dict
            Combined results dictionary containing:
            - 'material_name': Name of the material (subdirectory name)
            - 'n_images': Number of images processed
            - 'image_files': List of processed image filenames
            - 'morphology_individual': Combined list of all pores from all
               images
            - 'morphology_aggregated': Aggregated statistics across all pores
            - 'global_descriptors': Combined global descriptors
            - 'spatial_metrics': Combined spatial metrics
            - 'topology_metrics': Combined topology metrics
            - 'metadata': Combined metadata
        """
        material_name = subdir.name
        logger.info(f"Starting analysis for material: {material_name}")

        # Find all image files
        image_files = self._find_image_files(subdir)

        if not image_files:
            logger.warning(
                f"No images found in {subdir} matching pattern "
                f"'{self.file_pattern}'")
            return {
                'material_name': material_name,
                'n_images': 0,
                'image_files': [],
                'error': 'No images found'
            }

        # Analyze each image
        all_individual_pores = []
        all_image_results = []
        current_pore_id_offset = 0

        for idx, image_file in enumerate(image_files, 1):
            logger.info(
                "\nProcessing image "
                f"{idx}/{len(image_files)}: {image_file.name}"
            )

            try:
                # Extract magnification from filename
                magnification = extract_magnification_from_filename(
                    image_file.name
                )

                # Get appropriate pixel size for this magnification
                image_pixel_size = self._get_pixel_size_for_magnification(
                    magnification,
                    default_pixel_size=self.pixel_size
                )

                # Create analyzer for this image
                analyzer = PorousMaterialAnalyzer(
                    mask_path=str(image_file),
                    pixel_size=image_pixel_size,
                    generate_plots=self.generate_plots,
                    output_base_dir=self.output_base_dir / material_name,
                    reject_boundary_pores=self.reject_boundary_pores,
                    boundary_tolerance=self.boundary_tolerance,
                    plot_boundary_rejection=self.plot_boundary_rejection,
                )

                # Run analysis (without generating individual report)
                results = analyzer.analyze_all(generate_excel_report=False)
                all_image_results.append(results)

                # Process individual pores - add filename, magnification, and
                # adjust pore IDs
                for pore in results['morphology_individual']:
                    pore_copy = pore.copy()
                    # Adjust pore ID to be continuous across images
                    pore_copy['pore_id'] = (
                        pore['pore_id'] + current_pore_id_offset
                    )
                    pore_copy['filename'] = image_file.name
                    pore_copy['magnification'] = (
                        magnification if magnification else 'unknown'
                    )
                    pore_copy['pixel_size'] = image_pixel_size
                    all_individual_pores.append(pore_copy)

                # Update offset for next image
                # The next image's pore IDs should start after the last ID
                # from this image
                max_pore_id = max(
                    pore['pore_id'] for pore in results[
                        'morphology_individual'
                    ]
                )
                current_pore_id_offset += max_pore_id

                logger.info(
                    f"Processed {len(results['morphology_individual'])} "
                    f"pores from {image_file.name}"
                )
            except Exception as e:
                logger.error(f"Failed to analyze {image_file.name}: {e}")
                continue

        if not all_image_results:
            logger.error(
                f"No images successfully analyzed for {material_name}"
            )
            return {
                'material_name': material_name,
                'n_images': len(image_files),
                'image_files': [f.name for f in image_files],
                'error': 'All images failed to analyze'
            }

        # Combine results from all images
        combined_results = self._combine_image_results(
            material_name=material_name,
            all_individual_pores=all_individual_pores,
            all_image_results=all_image_results,
            image_files=[f.name for f in image_files]
        )

        # Generate combined Excel report if requested
        if generate_excel_report:
            report_path = self._generate_combined_report(
                material_name=material_name,
                combined_results=combined_results
            )
            combined_results['report_path'] = str(report_path)

        logger.info(f"Completed analysis for {material_name}")
        logger.info(f"Total pores analyzed: {len(all_individual_pores)}")

        return combined_results

    def _combine_image_results(
        self,
        material_name: str,
        all_individual_pores: List[Dict],
        all_image_results: List[Dict],
        image_files: List[str]
    ) -> Dict:
        """
        Combine results from multiple images into a single result dictionary.

        Parameters
        ----------
        material_name : str
            Name of the material
        all_individual_pores : List[Dict]
            Combined list of all individual pore measurements
        all_image_results : List[Dict]
            List of result dictionaries from each image
        image_files : List[str]
            List of image filenames

        Returns
        -------
        Dict
            Combined results dictionary
        """
        # Aggregate morphology across ALL pores from ALL images
        morphology_aggregated = (
            PoreMorphologyMetrics.aggregate_morphology_results(
                all_individual_pores
            )
        )

        # Combine global descriptors
        combined_global = {}

        # Extract porosity values from all images (needed for both averaging
        # and variance calculation)
        porosity_values = [
            img['global_descriptors']['porosity'] for img in all_image_results
        ]
        numeric_porosity_values = []
        for v in porosity_values:
            try:
                numeric_porosity_values.append(float(v))
            except (TypeError, ValueError):
                logger.warning(
                    f"Non-numeric porosity value '{v}' found, skipping"
                )

        # Average porosity across images
        if numeric_porosity_values:
            combined_global['porosity'] = np.mean(numeric_porosity_values)
        else:
            combined_global['porosity'] = np.nan
            logger.warning("No valid numeric values for porosity")

        # Average anisotropy across images
        anisotropy_values = [
            img[
                'global_descriptors'
            ]['anisotropy'] for img in all_image_results
        ]
        numeric_anisotropy_values = []
        for v in anisotropy_values:
            try:
                numeric_anisotropy_values.append(float(v))
            except (TypeError, ValueError):
                logger.warning(
                    f"Non-numeric anisotropy value '{v}' found, skipping"
                )

        if numeric_anisotropy_values:
            combined_global['anisotropy'] = np.mean(numeric_anisotropy_values)
        else:
            combined_global['anisotropy'] = np.nan
            logger.warning("No valid numeric values for anisotropy")

        # Calculate batch_porosity_variance based on mode
        if self.batch_variance_mode == "across_images":
            # NEW: Variance of porosity across complete images
            if len(numeric_porosity_values) > 1:
                combined_global['batch_porosity_variance'] = np.var(
                    numeric_porosity_values
                )
                logger.info(
                    "Calculated batch_porosity_variance as variance across "
                    f"{len(numeric_porosity_values)} images"
                )
            else:
                combined_global['batch_porosity_variance'] = 0.0
                logger.warning(
                    "Only one image - batch_porosity_variance set to 0.0"
                )
        elif self.batch_variance_mode == "within_images":
            # OLD: Average of within-image variances
            lpv_values = [
                img['global_descriptors']['local_porosity_variance']
                for img in all_image_results
            ]
            numeric_lpv_values = []
            for v in lpv_values:
                try:
                    numeric_lpv_values.append(float(v))
                except (TypeError, ValueError):
                    logger.warning(
                        f"Non-numeric local_porosity_variance '{v}' found, "
                        "skipping"
                    )

            if numeric_lpv_values:
                combined_global['batch_porosity_variance'] = np.mean(
                    numeric_lpv_values
                )
                logger.info(
                    "Calculated batch_porosity_variance as average of "
                    "within-image variances"
                )
            else:
                combined_global['batch_porosity_variance'] = np.nan
                logger.warning(
                    "No valid numeric values for local_porosity_variance"
                )
        else:
            raise ValueError(
                f"Invalid batch_variance_mode: '{self.batch_variance_mode}'. "
                "Must be 'across_images' or 'within_images'"
            )

        # Combine spatial metrics (averaging across images)
        combined_spatial = {}
        for key in all_image_results[0]['spatial_metrics'].keys():
            values = [img['spatial_metrics'][key] for img in all_image_results]
            # Filter out non-numeric values and convert to float
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (TypeError, ValueError):
                    logger.warning(
                        f"Non-numeric value '{v}' found for spatial "
                        f"metric '{key}', skipping"
                    )
            if numeric_values:
                combined_spatial[key] = np.mean(numeric_values)
            else:
                combined_spatial[key] = np.nan
                logger.warning(
                    f"No valid numeric values for spatial metric '{key}'"
                )

        # Combine topology metrics (averaging where appropriate)
        combined_topology = {}

        # Fractal dimension - average
        fractal_dims = [
            img['topology_metrics']['fractal_dimension']
            for img in all_image_results
        ]
        # Filter out non-numeric values
        numeric_fractal_dims = []
        for v in fractal_dims:
            try:
                numeric_fractal_dims.append(float(v))
            except (TypeError, ValueError):
                logger.warning(
                    f"Non-numeric value '{v}' found for fractal_dimension, "
                    "skipping"
                )
        if numeric_fractal_dims:
            combined_topology['fractal_dimension'] = np.mean(
                numeric_fractal_dims
            )
        else:
            combined_topology['fractal_dimension'] = np.nan
            logger.warning("No valid numeric values for fractal_dimension")

        # Fractal R² - average
        fractal_r2s = [
            img['topology_metrics']['fractal_r_squared']
            for img in all_image_results
        ]
        # Filter out non-numeric values
        numeric_fractal_r2s = []
        for v in fractal_r2s:
            try:
                numeric_fractal_r2s.append(float(v))
            except (TypeError, ValueError):
                logger.warning(
                    f"Non-numeric value '{v}' found for fractal_r_squared, "
                    "skipping"
                )
        if numeric_fractal_r2s:
            combined_topology['fractal_r_squared'] = np.mean(
                numeric_fractal_r2s
            )
        else:
            combined_topology['fractal_r_squared'] = np.nan
            logger.warning("No valid numeric values for fractal_r_squared")

        # Coordination number - recalculate from ALL individual values
        all_cn_values = []
        for pore in all_individual_pores:
            if (
                'coordination_number' in pore
                    and pore['coordination_number'] is not None
            ):
                all_cn_values.append(pore['coordination_number'])

        # Recalculate statistics from combined individual values
        if all_cn_values:
            combined_cn_stats = calculate_statistics(all_cn_values)
            combined_topology['coordination_number_stats'] = combined_cn_stats
        else:
            combined_topology['coordination_number_stats'] = None

        # Compile combined results
        combined_results = {
            'material_name': material_name,
            'n_images': len(all_image_results),
            'image_files': image_files,
            'morphology_individual': all_individual_pores,
            'morphology_aggregated': morphology_aggregated,
            'global_descriptors': combined_global,
            'spatial_metrics': combined_spatial,
            'topology_metrics': combined_topology,
            'metadata': {
                'material_name': material_name,
                'n_images': len(all_image_results),
                'total_pores': len(all_individual_pores),
                'pixel_size': self.pixel_size,
                'image_files': image_files,
            }
        }

        return combined_results

    def _generate_combined_report(
        self,
        material_name: str,
        combined_results: Dict
    ) -> Path:
        """
        Generate Excel report for combined analysis of all images in a
        material.

        Parameters
        ----------
        material_name : str
            Name of the material (subdirectory name)
        combined_results : Dict
            Combined results dictionary

        Returns
        -------
        Path
            Path to generated Excel report
        """
        logger.info(f"Generating combined Excel report for {material_name}...")

        # Create output directory
        output_dir = self.output_base_dir / material_name / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create report filename
        report_filename = f"{material_name}_combined_analysis_report.xlsx"
        report_path = output_dir / report_filename

        # ===== Prepare Sheet 1: Comprehensive Analysis Results =====
        report_data = []

        # --- METADATA ---
        report_data.append({
            "Category": "Metadata",
            "Metric": "Analysis Date",
            "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Unit": "-",
            "Description": "Date and time of analysis",
        })
        report_data.append({
            "Category": "Metadata",
            "Metric": "Material Name",
            "Value": material_name,
            "Unit": "-",
            "Description": "Name of the material (subdirectory)",
        })
        report_data.append({
            "Category": "Metadata",
            "Metric": "Number of Images",
            "Value": combined_results['n_images'],
            "Unit": "pcs",
            "Description": "Number of images analyzed for this material",
        })
        report_data.append({
            "Category": "Metadata",
            "Metric": "Total Number of Pores",
            "Value": len(combined_results['morphology_individual']),
            "Unit": "pcs",
            "Description": "Total number of pores across all images",
        })
        report_data.append({
            "Category": "Metadata",
            "Metric": "Pixel Size",
            "Value": self.pixel_size,
            "Unit": "µm",
            "Description": "Physical size of one pixel",
        })
        report_data.append({
            "Category": "Metadata",
            "Metric": "Boundary Pores Rejected",
            "Value": "Yes" if self.reject_boundary_pores else "No",
            "Unit": "-",
            "Description": "Whether pores touching boundaries were excluded",
        })

        # Add image filenames
        for idx, filename in enumerate(combined_results['image_files'], 1):
            report_data.append({
                "Category": "Metadata",
                "Metric": f"Image {idx}",
                "Value": filename,
                "Unit": "-",
                "Description": f"Filename of image {idx}",
            })

        # --- MORPHOLOGY (AGGREGATED) ---
        metric_descriptions = {
            "area": "Pore area",
            "perimeter": "Pore perimeter",
            "circularity": "Shape circularity (4πA/P²)",
            "solidity": "Ratio of area to convex hull area",
            "roundness": "Roundness (4*Area / (π*MajorAxis²))",
            "min_feret": "Minimum Feret diameter",
            "max_feret": "Maximum Feret diameter",
            "equivalent_diameter": "Equivalent diameter (diameter of circle "
            "with same area)",
            "ellipse_major_axis": "Major axis of fitted ellipse",
            "ellipse_minor_axis": "Minor axis of fitted ellipse",
            "aspect_ratio": "Ratio of major to minor axis",
            "ellipse_angle": "Orientation angle of fitted ellipse",
        }

        for metric_name, stats in combined_results[
                'morphology_aggregated'].items():
            # Determine unit
            if "area" in metric_name:
                unit = "µm²"
            elif ("perimeter" in metric_name or "feret" in metric_name
                  or "axis" in metric_name or "diameter" in metric_name):
                unit = "µm"
            elif "angle" in metric_name:
                unit = "degrees"
            else:
                unit = "-"

            description = metric_descriptions.get(
                metric_name, f"{metric_name} of pores"
            )

            # Add statistics
            for stat_name in ['mean', 'std', 'median', 'min', 'max']:
                report_data.append({
                    "Category": "Morphology",
                    "Metric": f"{metric_name} ({stat_name})",
                    "Value": stats[stat_name],
                    "Unit": unit,
                    "Description": f"{stat_name.capitalize()} {description}",
                })

        # --- GLOBAL DESCRIPTORS ---
        report_data.append({
            "Category": "Global Descriptors",
            "Metric": "Porosity (averaged)",
            "Value": combined_results['global_descriptors']['porosity'],
            "Unit": "-",
            "Description": "Average porosity across all images",
        })
        report_data.append({
            "Category": "Global Descriptors",
            "Metric": "Batch Porosity Variance",
            "Value": (
                combined_results[
                    'global_descriptors']['batch_porosity_variance']
            ),
            "Unit": "-",
            "Description": (
                "Variance of porosity across images"
                if self.batch_variance_mode == "across_images"
                else "Average within-image porosity variance"
            ),
        })
        report_data.append({
            "Category": "Global Descriptors",
            "Metric": "Anisotropy (averaged)",
            "Value": combined_results['global_descriptors']['anisotropy'],
            "Unit": "-",
            "Description": "Average anisotropy across all images",
        })

        # --- SPATIAL METRICS ---
        for key, value in combined_results['spatial_metrics'].items():
            report_data.append({
                "Category": "Spatial Metrics",
                "Metric": f"Nearest Neighbor Distance ({key}, averaged)",
                "Value": value,
                "Unit": "µm",
                "Description": (
                    f"Average {key} nearest neighbor distance "
                    "across all images"
                ),
            })

        # --- TOPOLOGY & CONNECTIVITY ---
        report_data.append({
            "Category": "Topology & Connectivity",
            "Metric": "Fractal Dimension (averaged)",
            "Value": combined_results['topology_metrics']['fractal_dimension'],
            "Unit": "-",
            "Description": "Average fractal dimension across all images",
        })
        report_data.append({
            "Category": "Topology & Connectivity",
            "Metric": "Fractal R² (averaged)",
            "Value": combined_results['topology_metrics']['fractal_r_squared'],
            "Unit": "-",
            "Description": "Average R² coefficient for fractal dimension fit",
        })

        if combined_results['topology_metrics']['coordination_number_stats']:
            cn_stats = combined_results['topology_metrics'][
                'coordination_number_stats']
            for key in ['mean', 'std', 'median', 'min', 'max']:
                report_data.append({
                    "Category": "Topology & Connectivity",
                    "Metric": f"Coordination Number ({key}, averaged)",
                    "Value": cn_stats[key],
                    "Unit": "-",
                    "Description": (
                        f"Average {key} coordination number across all images"
                    ),
                })

        # Create DataFrames
        df_report = pd.DataFrame(report_data)
        df_individual = pd.DataFrame(combined_results['morphology_individual'])

        # Reorder columns in Individual_Pores to put key identifiers first
        cols = df_individual.columns.tolist()
        # Remove key columns that we want to put first
        for col in ['pore_id', 'filename', 'magnification', 'pixel_size',
                    'coordination_number']:
            if col in cols:
                cols.remove(col)
        # Put key columns first
        key_cols = ['pore_id', 'filename', 'magnification', 'pixel_size',
                    'coordination_number']
        key_cols_present = [
            col for col in key_cols if col in df_individual.columns
        ]
        df_individual = df_individual[key_cols_present + cols]

        logger.info(f"Prepared {len(report_data)} metrics for export")
        logger.info(
            f"Prepared {len(combined_results['morphology_individual'])} "
            "individual pore records"
        )

        # Write to Excel
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
            return report_path

        except Exception as e:
            logger.error(f"Failed to write Excel report: {e}")
            raise

    def analyze_all_materials(
        self,
        generate_excel_reports: bool = True
    ) -> Dict[str, Dict]:
        """
        Analyze all materials (all subdirectories) in the parent directory.

        Parameters
        ----------
        generate_excel_reports : bool, optional
            Whether to generate Excel reports for each material (default: True)

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping material names to their combined results
        """
        logger.info(
            f"Starting batch analysis of {len(self.subdirectories)} materials"
        )
        logger.info(f"Parent directory: {self.parent_dir}")

        for subdir in self.subdirectories:
            try:
                results = self.analyze_single_material(
                    subdir=subdir,
                    generate_excel_report=generate_excel_reports
                )
                self.results[subdir.name] = results

            except Exception as e:
                logger.error(f"Failed to analyze material {subdir.name}: {e}")
                self.results[subdir.name] = {
                    'material_name': subdir.name,
                    'error': str(e)
                }
                continue

        logger.info("BATCH ANALYSIS COMPLETE!")
        logger.info(
            "Successfully analyzed: "
            f"{sum(1 for r in self.results.values() if 'error' not in r)} "
            "materials"
        )
        logger.info(
            f"Failed: {sum(1 for r in self.results.values() if 'error' in r)} "
            "materials"
        )

        return self.results

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all analyzed materials.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per material showing key metrics
        """
        summary_data = []
        for material_name, results in self.results.items():
            if 'error' in results:
                summary_data.append({
                    'Material': material_name,
                    'Status': 'Failed',
                    'Error': results.get('error', 'Unknown error'),
                })
            else:
                summary_data.append({
                    'Material': material_name,
                    'Status': 'Success',
                    'N_Images': results['n_images'],
                    'Total_Pores': len(results['morphology_individual']),
                    'Porosity': results['global_descriptors']['porosity'],
                    'Mean_Pore_Area_um2': (
                        results['morphology_aggregated']['area']['mean']
                    ),
                    'Anisotropy': results['global_descriptors']['anisotropy'],
                    'Fractal_Dimension': (
                        results['topology_metrics']['fractal_dimension']
                    ),
                })
        return pd.DataFrame(summary_data)

    def run_complete_analysis(
        self,
        generate_excel_reports: bool = True,
        save_summary: bool = True,
        summary_filename: str = "batch_analysis_summary.csv"
    ) -> tuple[Dict[str, Dict], pd.DataFrame]:
        """
        Run complete batch analysis and save summary in one call.

        This is a convenience method that combines:
        1. analyze_all_materials() - Analyze all materials
        2. get_summary() - Generate summary DataFrame
        3. Save summary to CSV (optional)

        Parameters
        ----------
        generate_excel_reports : bool, optional
            Whether to generate Excel reports for each material (default: True)
        save_summary : bool, optional
            Whether to save summary to CSV file (default: True)
        summary_filename : str, optional
            Filename for summary CSV (default: "batch_analysis_summary.csv")

        Returns
        -------
        tuple[Dict[str, Dict], pd.DataFrame]
            Tuple containing:
            - results: Dictionary mapping material names to their analysis
            results
            - summary: DataFrame with summary of all materials

        Examples
        --------
        >>> batch_analyzer = BatchPorousMaterialAnalyzer(
        ...     parent_dir='/path/to/materials'
        ... )
        >>> results, summary = batch_analyzer.run_complete_analysis()
        >>> print(summary)
        """
        # Run complete analysis
        results = self.analyze_all_materials(
            generate_excel_reports=generate_excel_reports
        )

        # Generate summary
        summary = self.get_summary()

        # Save summary to CSV if requested
        if save_summary:
            summary_path = self.output_base_dir / summary_filename
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to: {summary_path}")

        return results, summary
