"""
Side script for batch quantitative analysis with random pore removal.

This script extends the standard batch quantitative analysis to randomly remove
a specified percentage of pores from each image before calculating statistics.
This allows testing how different sampling rates affect overall material
statistics.

Key features:
- User can specify a list of percentages (e.g., [5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
- For each percentage, the script randomly removes that % of pores
- Analysis then proceeds normally with the remaining pores
- Output directories are suffixed with 'removed_{x}_pores' for organization

Note: This does NOT modify any existing modules - it only wraps/extends them.
"""

import numpy as np
import random
from pathlib import Path
from typing import Dict, Optional
import logging

from config import PIXEL_SIZE, OUTPUT_PATH
from materials_vision.quantitative_analysis.quantitative_analysis import (
    PorousMaterialAnalyzer,
)
from materials_vision.quantitative_analysis.batch_analysis import (
    BatchPorousMaterialAnalyzer,
)
from materials_vision.utils import extract_magnification_from_filename

logger = logging.getLogger(__name__)


class PorousMaterialAnalyzerRandomRemoval(PorousMaterialAnalyzer):
    """
    Extended analyzer that randomly removes a percentage of pores before calculating stats.

    This class extends PorousMaterialAnalyzer to randomly filter out a specified
    percentage of pores before running the complete analysis. This helps understand
    how sampling rate affects material statistics.

    Parameters
    ----------
    pore_removal_percentage : float
        Percentage of pores to randomly remove (0-100).
        For example, 20.0 means remove 20% of pores.
    random_seed : int, optional
        Random seed for reproducibility. If None, results will vary.
    All other parameters are inherited from PorousMaterialAnalyzer.
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
        pore_removal_percentage: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> None:
        # Initialize parent class
        super().__init__(
            mask_path=mask_path,
            pixel_size=pixel_size,
            generate_plots=generate_plots,
            output_base_dir=output_base_dir,
            reject_boundary_pores=reject_boundary_pores,
            boundary_tolerance=boundary_tolerance,
            plot_boundary_rejection=plot_boundary_rejection,
        )

        self.pore_removal_percentage = pore_removal_percentage
        self.random_seed = random_seed
        self.removed_pores_info = []  # Store info about removed pores

    def _filter_random_pores(self) -> None:
        """
        Randomly filter out a percentage of pores from self.props.

        This method:
        1. Randomly selects pores to remove based on the specified percentage
        2. Updates self.props to exclude these pores
        3. Logs information about removed pores
        """
        if len(self.props) == 0:
            logger.warning("No pores to filter - props list is empty")
            return

        if self.pore_removal_percentage <= 0:
            logger.info("Pore removal percentage is 0 - no pores will be removed")
            return

        if self.pore_removal_percentage >= 100:
            logger.warning(
                "Pore removal percentage is >= 100 - all pores would be removed. "
                "This is likely not intended. Skipping filtering."
            )
            return

        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Calculate how many pores to remove
        total_pores = len(self.props)
        n_to_remove = int(total_pores * (self.pore_removal_percentage / 100.0))

        if n_to_remove == 0:
            logger.info(
                f"Calculated 0 pores to remove from {total_pores} total pores "
                f"(removal percentage: {self.pore_removal_percentage}%)"
            )
            return

        # Randomly select pores to remove
        pores_to_remove = random.sample(list(self.props), n_to_remove)

        # Get labels of pores to remove
        labels_to_remove = {prop.label for prop in pores_to_remove}

        # Log removed pores info
        logger.info(
            f"Removing {n_to_remove} out of {total_pores} pores "
            f"({self.pore_removal_percentage}% removal rate)"
        )

        # Store info about removed pores for metadata
        for prop in pores_to_remove:
            area_pixels = prop.area
            area_um2 = area_pixels * (self.pixel_size ** 2)
            self.removed_pores_info.append({
                'label': prop.label,
                'area_pixels': area_pixels,
                'area_um2': area_um2
            })

        # Filter self.props to exclude these pores
        original_count = len(self.props)
        self.props = [
            prop for prop in self.props
            if prop.label not in labels_to_remove
        ]

        logger.info(
            f"Filtered pores: {original_count} -> {len(self.props)} "
            f"(removed {original_count - len(self.props)} pores)"
        )

    def analyze_all(
        self,
        generate_excel_report: bool = True
    ) -> Dict[str, any]:
        """
        Perform complete analysis after randomly filtering out pores.

        This method first filters out the specified percentage of pores randomly,
        then calls the parent analyze_all() method to proceed with normal analysis.

        Returns
        -------
        Dict[str, any]
            Analysis results (same as parent class)
        """
        # Filter random pores BEFORE running analysis
        logger.info("=" * 70)
        logger.info(f"RANDOMLY REMOVING {self.pore_removal_percentage}% OF PORES")
        logger.info("=" * 70)
        self._filter_random_pores()

        # Now run the normal analysis with filtered props
        results = super().analyze_all(generate_excel_report=generate_excel_report)

        # Add metadata about removed pores
        results['metadata']['pore_removal_percentage'] = self.pore_removal_percentage
        results['metadata']['pores_removed_count'] = len(self.removed_pores_info)
        results['metadata']['removed_pores_info'] = self.removed_pores_info

        return results


class BatchPorousMaterialAnalyzerRandomRemoval(BatchPorousMaterialAnalyzer):
    """
    Batch analyzer that randomly removes a percentage of pores from each image.

    This class extends BatchPorousMaterialAnalyzer to use the custom
    PorousMaterialAnalyzerRandomRemoval for individual image analysis.

    Parameters
    ----------
    pore_removal_percentage : float
        Percentage of pores to randomly remove (0-100).
    random_seed : int, optional
        Random seed for reproducibility across all images.
    All other parameters are inherited from BatchPorousMaterialAnalyzer.
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
        pore_removal_percentage: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> None:
        # Initialize parent class
        super().__init__(
            parent_dir=parent_dir,
            pixel_size=pixel_size,
            generate_plots=generate_plots,
            output_base_dir=output_base_dir,
            reject_boundary_pores=reject_boundary_pores,
            boundary_tolerance=boundary_tolerance,
            plot_boundary_rejection=plot_boundary_rejection,
            file_pattern=file_pattern,
            batch_variance_mode=batch_variance_mode,
        )

        self.pore_removal_percentage = pore_removal_percentage
        self.random_seed = random_seed

    def analyze_single_material(
        self,
        subdir: Path,
        generate_excel_report: bool = True
    ) -> Dict:
        """
        Analyze all images in a single subdirectory with random pore removal.

        This method is similar to the parent class but uses
        PorousMaterialAnalyzerRandomRemoval instead of the standard analyzer.
        """
        material_name = subdir.name
        logger.info(f"Starting analysis for material: {material_name}")

        # Find all image files
        image_files = self._find_image_files(subdir)

        if not image_files:
            logger.warning(
                f"No images found in {subdir} matching pattern "
                f"'{self.file_pattern}'"
            )
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
                f"\nProcessing image {idx}/{len(image_files)}: {image_file.name}"
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

                # Create CUSTOM analyzer for this image (with random pore removal)
                analyzer = PorousMaterialAnalyzerRandomRemoval(
                    mask_path=str(image_file),
                    pixel_size=image_pixel_size,
                    generate_plots=self.generate_plots,
                    output_base_dir=self.output_base_dir / material_name,
                    reject_boundary_pores=self.reject_boundary_pores,
                    boundary_tolerance=self.boundary_tolerance,
                    plot_boundary_rejection=self.plot_boundary_rejection,
                    pore_removal_percentage=self.pore_removal_percentage,
                    random_seed=self.random_seed,
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
                if results['morphology_individual']:
                    max_pore_id = max(
                        pore['pore_id'] for pore in results['morphology_individual']
                    )
                    current_pore_id_offset += max_pore_id

                logger.info(
                    f"Processed {len(results['morphology_individual'])} pores "
                    f"from {image_file.name}"
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


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # =========================================================================
    # USER CONFIGURATION
    # =========================================================================

    # Parent directory containing material subdirectories
    parent_directory = "/Volumes/ADATA SD620/Doktorat/semestr_4/analiza ilościowa/analiza_ilosciowa_full_dataset/podzielony_per_material"

    # List of percentages to test
    # The script will run analysis for each percentage value
    pore_removal_percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # File pattern to match mask images
    file_pattern = "*_masks.tif"

    # Random seed for reproducibility (set to None for different results each time)
    random_seed = 42

    # Base output directory (will be modified for each percentage)
    base_output_path = OUTPUT_PATH

    # =========================================================================
    # RUN ANALYSIS FOR EACH PERCENTAGE
    # =========================================================================

    print("\n" + "=" * 80)
    print("BATCH ANALYSIS WITH RANDOM PORE REMOVAL")
    print("=" * 80)
    print(f"Testing removal percentages: {pore_removal_percentages}")
    print(f"Parent directory: {parent_directory}")
    print(f"Random seed: {random_seed}")
    print("=" * 80 + "\n")

    all_results = {}

    for removal_percentage in pore_removal_percentages:
        print("\n" + "#" * 80)
        print(f"# RUNNING ANALYSIS WITH {removal_percentage}% PORES REMOVED")
        print("#" * 80 + "\n")

        # Create output directory with suffix
        output_dir = Path(base_output_path) / f"removed_{removal_percentage}_pores"

        # Create batch analyzer with current removal percentage
        batch_analyzer = BatchPorousMaterialAnalyzerRandomRemoval(
            parent_dir=parent_directory,
            pixel_size=PIXEL_SIZE,
            generate_plots=True,
            output_base_dir=output_dir,
            reject_boundary_pores=True,
            boundary_tolerance=3,
            file_pattern=file_pattern,
            pore_removal_percentage=removal_percentage,
            random_seed=random_seed,
        )

        # Run complete analysis
        try:
            results, summary = batch_analyzer.run_complete_analysis()
            all_results[removal_percentage] = {
                'results': results,
                'summary': summary,
                'output_dir': str(output_dir)
            }

            print(f"\n✓ Completed analysis for {removal_percentage}% removal")
            print(f"  Output directory: {output_dir}")
            print(f"  Summary:\n{summary}")

        except Exception as e:
            logger.error(
                f"Failed to complete analysis for {removal_percentage}% removal: {e}"
            )
            all_results[removal_percentage] = {
                'error': str(e),
                'output_dir': str(output_dir)
            }

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print(f"Total percentages tested: {len(pore_removal_percentages)}")
    print(f"Successful analyses: {sum(1 for r in all_results.values() if 'error' not in r)}")
    print(f"Failed analyses: {sum(1 for r in all_results.values() if 'error' in r)}")
    print("\nResults stored in:")
    for percentage, data in all_results.items():
        status = "✓" if 'error' not in data else "✗"
        print(f"  {status} {percentage}%: {data['output_dir']}")
    print("=" * 80 + "\n")
