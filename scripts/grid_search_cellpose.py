"""
Grid search pipeline for Cellpose-SAM (CPSAM) fine-tuning.

Systematically explores hyperparameter combinations defined
in a YAML config file, trains CPSAM models, evaluates them,
and logs everything to MLflow.

Usage:
    python scripts/grid_search_cellpose.py \
        --config materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml \
        --mode training \
        --gpu-device 0 \
        --dry-run
"""
import argparse
import logging
import os

from materials_vision.experiments.cellpose.cellpose_grid_search.grid_search import (
    dry_run,
    generate_eval_combinations,
    generate_training_combinations,
    load_config,
    run_eval_grid,
    run_full_grid,
    run_training_grid,
)
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    "materials_vision/experiments/cellpose"
    "/cellpose_grid_search/grid_search_config.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Cellpose-SAM grid search pipeline"
    )
    p.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG,
        help="Path to grid search YAML config",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["training", "evaluation", "full"],
        default="training",
        help="Grid search mode",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path (required for mode=evaluation)",
    )
    p.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU device index",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without running",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run already completed experiments",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the grid search pipeline."""
    setup_logging(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)

    mode = args.mode or config.get("mode", "training")
    logger.info("Grid search mode: %s", mode)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_device
    )
    gpu = True

    if args.dry_run:
        if mode in ("training", "full"):
            combos = generate_training_combinations(
                config["training_grid"]
            )
            dry_run(combos, "training", config)
        if mode in ("evaluation", "full"):
            combos = generate_eval_combinations(
                config["eval_grid"]
            )
            dry_run(combos, "evaluation", config)
        return

    if mode == "training":
        run_training_grid(
            config, gpu, args.force_rerun
        )
    elif mode == "evaluation":
        if not args.model_path:
            raise ValueError(
                "--model-path is required "
                "when mode=evaluation"
            )
        run_eval_grid(
            config,
            args.model_path,
            gpu,
            args.force_rerun,
        )
    elif mode == "full":
        run_full_grid(
            config, gpu, args.force_rerun
        )


if __name__ == "__main__":
    main()
