from pathlib import Path
from materials_vision.cellpose.training import retrain_cyto


if __name__ == '__main__':
    config_path = Path("materials_vision/experiments/experiment_config.yaml")

    model_path, train_losses, test_losses, run_id = retrain_cyto(
        config_path=config_path,
    )

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"MLflow Run ID: {run_id}")
