from pathlib import Path
from typing import Dict
import mlflow
import yaml


def load_experiment_config(config_path: Path) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def log_dict_as_params(config: Dict, parent_key: str = ''):
    """
    Recursively logs a nested dictionary to MLflow as parameters.

    Parameters
    ----------
    config : Dict
        The dictionary to log.
    parent_key : str, optional
        The base key to use for nested keys, by default ''.
    """
    for key, value in config.items():
        param_name = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            log_dict_as_params(value, param_name)
        else:
            mlflow.log_param(param_name, value)
