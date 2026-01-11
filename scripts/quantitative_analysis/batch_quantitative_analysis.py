"""
Script to run batch quantitative analysis on multiple materials.

This script analyzes multiple porous material samples organized in subdirectories,
generating comprehensive reports for each material.

Usage:
    python scripts/quantitative_analysis/batch_quantitative_analysis.py
"""
import logging
from materials_vision.logging_config import setup_logging
from config import PIXEL_SIZE
from materials_vision.quantitative_analysis.batch_analysis import \
    BatchPorousMaterialAnalyzer


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Example: Batch analyze multiple materials
    parent_directory = "/home/dwalicki/datasets/maski_cellpose_inferencja/organized"

    # Create batch analyzer
    batch_analyzer = BatchPorousMaterialAnalyzer(
        parent_dir=parent_directory,
        pixel_size=PIXEL_SIZE,
        generate_plots=True,
        reject_boundary_pores=True,
        boundary_tolerance=3,
        file_pattern="*_masks.tif",  # or "*_masks.tif" for specific naming
    )

    results, summary = batch_analyzer.run_complete_analysis()
