"""
Script to run batch quantitative analysis on multiple materials.

This script analyzes multiple porous material samples organized in subdirectories,
generating comprehensive reports for each material.

Usage:
    python scripts/quantitative_analysis/batch_quantitative_analysis.py
"""
import logging
from pathlib import Path
from materials_vision.logging_config import setup_logging
from materials_vision.utils import load_pixel_sizes
from materials_vision.quantitative_analysis.batch_analysis import \
    BatchPorousMaterialAnalyzer


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parent directory containing material subdirectories with mask images
    parent_directory = (
        "/home/dwalicki/datasets/maski_cellpose_inferencja/organized"
    )
    output_directory = Path("/home/dwalicki/results/batch_analysis")

    # Fallback pixel size when magnification cannot be determined from filename
    pixel_sizes = load_pixel_sizes()
    fallback_pixel_size = pixel_sizes[40]

    batch_analyzer = BatchPorousMaterialAnalyzer(
        parent_dir=parent_directory,
        pixel_size=fallback_pixel_size,
        output_base_dir=output_directory,
        generate_plots=True,
        reject_boundary_pores=True,
        boundary_tolerance=3,
        file_pattern="*_masks.tif",
    )

    results, summary = batch_analyzer.run_complete_analysis()
