from config import PIXEL_SIZE
from materials_vision.quantitative_analysis.quantitative_analysis import \
    PorousMaterialAnalyzer


# Example usage
if __name__ == "__main__":
    # Example: Analyze a single porous material sample
    mask_path = '/Volumes/ADATA SD620/Doktorat/semestr_4/analiza ilościowa/DG/do_oceny/train/401_jpg.rf.208cbf760d266850b64cc7d9347856c1_train_0000_masks.tif'

    # Create analyzer instance with boundary pore rejection (default behavior)
    analyzer = PorousMaterialAnalyzer(
        mask_path=mask_path,
        pixel_size=PIXEL_SIZE,
        generate_plots=True,
        reject_boundary_pores=True,
        boundary_tolerance=3,
        plot_boundary_rejection=True
    )

    # To include boundary pores in analysis, set reject_boundary_pores=False:
    # analyzer = PorousMaterialAnalyzer(
    #     mask_path=mask_path,
    #     reject_boundary_pores=False
    # )

    # Run complete analysis (automatically generates Excel report)
    results = analyzer.analyze_all()

    # Alternatively, generate report separately
    # report_path = analyzer.generate_report()
    # print(f"Report saved to: {report_path}")

    # Output structure:
    # OUTPUT_PATH/
    # └── {mask_filename}/
    #     ├── plots/
    #     │   ├── nearest_neighbor_distances.png
    #     │   ├── {mask_filename}_fractal_dimension.png
    #     │   └── {mask_filename}_coordination_number.png
    #     ├── data/
    #     └── reports/
    #         └── {mask_filename}_analysis_report.xlsx
    #             ├── Sheet 1: Analysis_Results (comprehensive metrics table)
    #             └── Sheet 2: Individual_Pores (per-pore data)
