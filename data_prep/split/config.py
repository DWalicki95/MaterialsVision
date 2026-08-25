"""Loading and validation of the dataset split run configuration."""
import logging
from pathlib import Path

import yaml

from data_prep.split.models import (SETS, CostWeights, MinFragmentAreaConfig,
                                    SplitConfig, SplitConstraints)

logger = logging.getLogger(__name__)

_TARGET_SHARE_TOLERANCE = 1e-6


class SplitConfigError(ValueError):
    """Raised when the split configuration is malformed or points to
    missing paths."""


def load_split_config(path: Path) -> SplitConfig:
    """Load and validate a split configuration YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML.

    Returns
    -------
    SplitConfig

    Raises
    ------
    SplitConfigError
        If the file is missing or malformed, if the manifest it points
        to does not exist, if the target shares do not sum to one, or
        if a quota is not a triple of non-negative integers.
    """
    if not path.exists():
        raise SplitConfigError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise SplitConfigError(f"Configuration must be a mapping: {path}")

    try:
        split_id = str(raw["split_id"])
        manifest_path = Path(raw["manifest_path"])
        output_dir = Path(raw["output_dir"])
        seed = int(raw["seed"])
        n_candidates = int(raw["n_candidates"])
        raw_quotas = raw["quotas"]
        raw_targets = raw["target_shares"]
    except KeyError as e:
        raise SplitConfigError(
            f"Missing required configuration key: {e} (in {path})"
        ) from e

    if not manifest_path.exists():
        raise SplitConfigError(
            f"manifest_path does not exist: {manifest_path}"
        )
    if n_candidates < 1:
        raise SplitConfigError(
            f"n_candidates must be positive, got {n_candidates}"
        )

    quotas = _parse_quotas(raw_quotas)
    target_shares = _parse_target_shares(raw_targets)

    config = SplitConfig(
        split_id=split_id,
        manifest_path=manifest_path,
        output_dir=output_dir,
        seed=seed,
        n_candidates=n_candidates,
        quotas=quotas,
        forced_train=tuple(raw.get("forced_train") or ()),
        target_shares=target_shares,
        constraints=_parse_constraints(raw.get("constraints") or {}),
        cost_weights=_parse_cost_weights(raw.get("cost_weights") or {}),
        min_fragment_area=_parse_min_fragment_area(
            raw.get("min_fragment_area"), path
        ),
    )
    logger.info("Loaded split configuration: %s", path)
    return config


def _parse_quotas(raw) -> dict[str, tuple[int, int, int]]:
    """Parse the per-material ``(train, val, test)`` quota mapping."""
    if not isinstance(raw, dict) or not raw:
        raise SplitConfigError(
            "quotas must be a non-empty mapping of material to "
            "[n_train, n_val, n_test]"
        )
    quotas = {}
    for material, values in raw.items():
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(not isinstance(v, int) or v < 0 for v in values)
        ):
            raise SplitConfigError(
                f"Quota for {material!r} must be three non-negative "
                f"integers [train, val, test], got {values!r}"
            )
        quotas[str(material)] = (
            int(values[0]), int(values[1]), int(values[2])
        )
    return quotas


def _parse_target_shares(raw) -> dict[str, float]:
    """Parse and check the per-set target shares."""
    if not isinstance(raw, dict) or set(raw) != set(SETS):
        raise SplitConfigError(
            f"target_shares must be a mapping over exactly {list(SETS)}"
        )
    shares = {str(k): float(v) for k, v in raw.items()}
    total = sum(shares.values())
    if abs(total - 1.0) > _TARGET_SHARE_TOLERANCE:
        raise SplitConfigError(
            f"target_shares must sum to 1.0, got {total}"
        )
    return shares


def _parse_constraints(raw) -> SplitConstraints:
    """Parse the hard-constraint block, falling back to defaults."""
    if not isinstance(raw, dict):
        raise SplitConfigError("constraints must be a mapping")
    by_material = raw.get("min_eval_images_by_material") or {}
    if not isinstance(by_material, dict):
        raise SplitConfigError(
            "constraints.min_eval_images_by_material must be a mapping"
        )
    defaults = SplitConstraints()
    return SplitConstraints(
        min_m2_formulations_per_set=int(
            raw.get("min_m2_formulations_per_set",
                    defaults.min_m2_formulations_per_set)
        ),
        min_scale_bin_images_per_set=int(
            raw.get("min_scale_bin_images_per_set",
                    defaults.min_scale_bin_images_per_set)
        ),
        min_eval_fine_images=int(
            raw.get("min_eval_fine_images",
                    defaults.min_eval_fine_images)
        ),
        min_eval_images_by_material={
            str(k): int(v) for k, v in by_material.items()
        },
    )


def _parse_cost_weights(raw) -> CostWeights:
    """Parse the cost weight block, falling back to defaults."""
    if not isinstance(raw, dict):
        raise SplitConfigError("cost_weights must be a mapping")
    overrides = raw.get("cell_overrides") or {}
    if not isinstance(overrides, dict):
        raise SplitConfigError(
            "cost_weights.cell_overrides must be a mapping"
        )
    defaults = CostWeights()
    return CostWeights(
        images=float(raw.get("images", defaults.images)),
        instances=float(raw.get("instances", defaults.instances)),
        cell_default=float(
            raw.get("cell_default", defaults.cell_default)
        ),
        cell_overrides={
            str(k): float(v) for k, v in overrides.items()
        },
        lost_outlier_image=float(
            raw.get("lost_outlier_image", defaults.lost_outlier_image)
        ),
    )


def _parse_min_fragment_area(raw, config_path: Path):
    """Parse the optional ``A_min_fragment`` calibration block.

    Parameters
    ----------
    raw : dict or None
        The ``min_fragment_area`` block; ``None`` disables the step.
    config_path : Path
        Split configuration path, used to resolve a relative
        ``inventory_config`` against the repository root.

    Returns
    -------
    MinFragmentAreaConfig or None
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SplitConfigError("min_fragment_area must be a mapping")
    try:
        inventory_config = Path(raw["inventory_config"])
    except KeyError as e:
        raise SplitConfigError(
            "min_fragment_area.inventory_config is required (it "
            "supplies the Label Studio export paths; the manifest "
            "stores no per-instance areas)"
        ) from e

    if not inventory_config.is_absolute():
        inventory_config = (
            config_path.resolve().parent.parent.parent / inventory_config
        )
    if not inventory_config.exists():
        raise SplitConfigError(
            f"min_fragment_area.inventory_config does not exist: "
            f"{inventory_config}"
        )

    percentile = float(raw.get("percentile", 1.0))
    if not 0.0 < percentile < 100.0:
        raise SplitConfigError(
            f"min_fragment_area.percentile must lie in (0, 100), got "
            f"{percentile}"
        )
    return MinFragmentAreaConfig(
        inventory_config=inventory_config,
        percentile=percentile,
        exclude_scale_outlier=bool(
            raw.get("exclude_scale_outlier", True)
        ),
    )
