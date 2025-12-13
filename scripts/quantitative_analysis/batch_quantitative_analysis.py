from config import PIXEL_SIZE
from materials_vision.quantitative_analysis.batch_analysis import \
    BatchPorousMaterialAnalyzer


# Example usage
if __name__ == "__main__":
    # Example: Batch analyze multiple materials
    parent_directory = "/Volumes/ADATA SD620/Doktorat/semestr_4/analiza ilościowa/example_batch_analysis"

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
