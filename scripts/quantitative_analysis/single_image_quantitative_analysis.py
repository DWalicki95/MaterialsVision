"""
Script to run quantitative analysis on a single porous material image.

This script analyzes a single mask image and generates comprehensive
morphological metrics and visualizations.

Usage:
    python scripts/quantitative_analysis/single_image_quantitative_analysis.py
"""
import logging
from pathlib import Path
from materials_vision.logging_config import setup_logging
from materials_vision.utils import load_pixel_sizes
from materials_vision.quantitative_analysis.quantitative_analysis import \
    PorousMaterialAnalyzer


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mask_path = (
        '/Volumes/ADATA SD620/Doktorat/semestr_4/analiza ilościowa/DG/'
        'do_oceny/train/'
        '401_jpg.rf.208cbf760d266850b64cc7d9347856c1_train_0000_masks.tif'
    )
    output_directory = Path("/home/dwalicki/results/single_analysis")

    # Set magnification matching the SEM image (40, 50, 100, 250, 500, 1000)
    magnification = 40
    pixel_size = load_pixel_sizes()[magnification]

    analyzer = PorousMaterialAnalyzer(
        mask_path=mask_path,
        pixel_size=pixel_size,
        output_base_dir=output_directory,
        generate_plots=True,
        reject_boundary_pores=True,
        boundary_tolerance=3,
        plot_boundary_rejection=True
    )

    # Run complete analysis (automatically generates Excel report)
    results = analyzer.analyze_all()

    # Output structure:
    # output_directory/
    # └── {mask_filename}/
    #     ├── plots/
    #     ├── data/
    #     └── reports/
    #         └── {mask_filename}_analysis_report.xlsx
