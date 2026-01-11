"""
Script to run Cellpose inference pipeline.

This script executes the complete Cellpose inference workflow including
loading images, running model inference, and saving results (masks, flows,
and style vectors).

Usage:
    python scripts/run_cellpose_inference.py
    python scripts/run_cellpose_inference.py --no-flows --no-styles
"""
import argparse
import logging
from pathlib import Path
from materials_vision.logging_config import setup_logging
from materials_vision.cellpose.inference import CellposeInferenceEvaluation
from config import MODEL_PATH_INFERENCE, OUTPUT_PATH_INFERENCE, \
    PATH_TO_FILES_INFERENCE


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Cellpose inference on a directory of images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--path-to-files",
        type=str,
        default=PATH_TO_FILES_INFERENCE,
        help=(
            "Path to directory containing input images "
            f"(default: {PATH_TO_FILES_INFERENCE})"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH_INFERENCE,
        help=(
            "Path to pretrained Cellpose model "
            f"(default: {MODEL_PATH_INFERENCE})"
        )
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH_INFERENCE,
        help=(
            "Directory to save inference outputs "
            f"(default: {OUTPUT_PATH_INFERENCE})"
        )
    )

    parser.add_argument(
        "--no-flows",
        action="store_true",
        help="Do not save flow field outputs"
    )

    parser.add_argument(
        "--no-styles",
        action="store_true",
        help="Do not save style vector outputs"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to execute Cellpose inference pipeline.

    Parses command line arguments, initializes the inference evaluator,
    and runs the complete pipeline.
    """
    args = parse_args()

    # Setup logging with appropriate level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)
    logger.info("Cellpose Inference Pipeline")
    logger.info(f"Input directory: {args.path_to_files}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_path}")
    logger.info(f"Save flows: {not args.no_flows}")
    logger.info(f"Save styles: {not args.no_styles}")

    # Validate paths
    if not Path(args.path_to_files).exists():
        logger.error(f"Input directory does not exist: {args.path_to_files}")
        return

    if not Path(args.model_path).exists():
        logger.error(f"Model file does not exist: {args.model_path}")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_dir}")

    # Initialize evaluator
    evaluator = CellposeInferenceEvaluation(
        path_to_files=args.path_to_files,
        model_path=args.model_path,
        output_path=args.output_path
    )

    # Run pipeline
    try:
        evaluator.run_pipeline(
            save_flows=not args.no_flows,
            save_styles=not args.no_styles
        )
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
