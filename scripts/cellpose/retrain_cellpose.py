"""
Script to retrain Cellpose model on custom dataset.

This script loads configuration from a YAML file and retrains the Cellpose
model with the specified parameters.

Usage:
    python scripts/cellpose/retrain_cellpose.py
"""
import logging
from pathlib import Path

from materials_vision.experiments.cellpose.retraining.training import \
    retrain_cyto
from materials_vision.logging_config import setup_logging

if __name__ == '__main__':
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config_path = Path(
        "materials_vision/experiments/cellpose/retraining/"
        "cellpose_retraining_config.yaml"
    )

    model_path, train_losses, test_losses, run_id = retrain_cyto(
        config_path=config_path,
    )

    logger.info("Training completed!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"MLflow Run ID: {run_id}")
