"""Loading and validation of the inventory run configuration."""
import logging
from pathlib import Path

import yaml

from data_prep.inventory.models import InventoryConfig, SourceConfig

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or points to
    missing paths."""


def load_config(path: Path) -> InventoryConfig:
    """Load and validate an inventory configuration YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML.

    Returns
    -------
    InventoryConfig

    Raises
    ------
    ConfigError
        If the file is missing, malformed, or any referenced path
        (images directory, Label Studio export, sidecar directory) does
        not exist.
    """
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration must be a mapping: {path}")

    try:
        raw_sources = raw["sources"]
        manifest_version = raw["manifest_version"]
        output_dir = Path(raw["output_dir"])
        mask_root = Path(raw["mask_root"])
    except KeyError as e:
        raise ConfigError(f"Missing required config key: {e}") from e

    if not raw_sources:
        raise ConfigError("Configuration must declare at least one source")

    sources = tuple(_load_source(entry) for entry in raw_sources)

    config = InventoryConfig(
        manifest_version=manifest_version,
        output_dir=output_dir,
        mask_root=mask_root,
        sources=sources,
        scale_outlier_ratio=raw.get("scale_outlier_ratio", 1.5),
        fuzzy_cutoff=raw.get("fuzzy_cutoff", 0.85),
        nonimage_extreme_fraction=raw.get(
            "nonimage_extreme_fraction", 0.90
        ),
        nonimage_max_band_fraction=raw.get(
            "nonimage_max_band_fraction", 0.35
        ),
        overlap_significant_fraction=raw.get(
            "overlap_significant_fraction", 0.01
        ),
        pixel_size_tolerance=raw.get("pixel_size_tolerance", 0.01),
    )
    _validate_paths(config)
    return config


def _load_source(entry: dict) -> SourceConfig:
    """Build a ``SourceConfig`` from one ``sources`` list entry.

    Parameters
    ----------
    entry : dict
        Raw YAML mapping for a single source.

    Returns
    -------
    SourceConfig

    Raises
    ------
    ConfigError
        If a required key is missing.
    """
    try:
        series = entry["series"]
        images_dir = Path(entry["images_dir"])
        label_studio_json = Path(entry["label_studio_json"])
        sem_metadata_dirs = tuple(
            Path(p) for p in entry["sem_metadata_dirs"]
        )
    except KeyError as e:
        raise ConfigError(
            f"Missing required source key: {e} in {entry}"
        ) from e
    return SourceConfig(
        series=series,
        images_dir=images_dir,
        label_studio_json=label_studio_json,
        sem_metadata_dirs=sem_metadata_dirs,
    )


def _validate_paths(config: InventoryConfig) -> None:
    """Raise ``ConfigError`` if any referenced input path is missing.

    Output paths (``output_dir``, ``mask_root``) are not required to
    exist yet; they are created on write.

    Parameters
    ----------
    config : InventoryConfig
    """
    for source in config.sources:
        if not source.images_dir.is_dir():
            raise ConfigError(
                f"[{source.series}] images_dir does not exist: "
                f"{source.images_dir}"
            )
        if not source.label_studio_json.is_file():
            raise ConfigError(
                f"[{source.series}] label_studio_json does not exist: "
                f"{source.label_studio_json}"
            )
        for sem_dir in source.sem_metadata_dirs:
            if not sem_dir.is_dir():
                raise ConfigError(
                    f"[{source.series}] sem_metadata_dir does not "
                    f"exist: {sem_dir}"
                )
