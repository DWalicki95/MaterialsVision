from datetime import datetime
from pathlib import Path
import yaml


def load_pixel_sizes() -> dict:
    """
    Load SEM pixel size calibration from shared YAML.

    Returns
    -------
    dict
        Mapping of magnification (int) to pixel size in µm/px (float).
    """
    path = Path(__file__).parent / "config" / "sem_calibration.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["pixel_sizes"]


def create_current_time_output_directory(dir_base_path: Path):
    """
    Create timestamped output directory.

    Parameters
    ----------
    dir_base_path : Path
        Parent directory where the output directory will be created.

    Returns
    -------
    Path
        Path to the created directory.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(dir_base_path) / f"output_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
