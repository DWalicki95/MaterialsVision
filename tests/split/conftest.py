"""Shared fixtures for the dataset split test suite.

The synthetic dataset below mirrors the real one's awkward shape at a
smaller scale: a dominant material carrying both scale bins, a
coarse-only material, a fine-only material with a single
coarse-carrying exception, and a tiny formulation that is only fit for
TRAIN.
"""
import pandas as pd
import pytest

from data_prep.split.models import (CostWeights, FormulationProfile,
                                    SplitConfig, SplitConstraints)


def make_profile(
    formulation: str,
    material: str,
    microscope: str = "M1",
    n_coarse: int = 0,
    n_fine: int = 0,
    n_outlier: int = 0,
    n_instances: int = 100,
) -> FormulationProfile:
    """Build a FormulationProfile with image counts kept consistent.

    Parameters
    ----------
    formulation, material, microscope : str
    n_coarse, n_fine, n_outlier : int
    n_instances : int

    Returns
    -------
    FormulationProfile
    """
    return FormulationProfile(
        formulation=formulation,
        material=material,
        microscope=microscope,
        n_images=n_coarse + n_fine + n_outlier,
        n_coarse=n_coarse,
        n_fine=n_fine,
        n_outlier=n_outlier,
        n_instances=n_instances,
    )


@pytest.fixture
def profile_factory():
    """Expose ``make_profile`` to tests that build their own cases."""
    return make_profile


@pytest.fixture
def profiles() -> tuple[FormulationProfile, ...]:
    """Six AS, three K and three VAB formulations."""
    built = [
        make_profile(f"AS{i}", "AS", "M1", n_coarse=20, n_fine=2)
        for i in range(1, 7)
    ]
    built += [
        make_profile(f"K{i}", "K", "M2", n_coarse=12) for i in (1, 2, 3)
    ]
    built += [
        make_profile("VAB1", "VAB", "M2", n_coarse=3, n_fine=6),
        make_profile("VAB2", "VAB", "M2", n_fine=6),
        make_profile("VAB3", "VAB", "M2", n_fine=2, n_outlier=2),
    ]
    return tuple(sorted(built, key=lambda p: p.formulation))


@pytest.fixture
def config(tmp_path) -> SplitConfig:
    """A split configuration matching the ``profiles`` fixture."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("placeholder", encoding="utf-8")
    return SplitConfig(
        split_id="split_test",
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        seed=1234,
        n_candidates=2000,
        quotas={"AS": (4, 1, 1), "K": (1, 1, 1), "VAB": (1, 1, 1)},
        forced_train=("VAB3",),
        target_shares={"train": 0.70, "val": 0.15, "test": 0.15},
        constraints=SplitConstraints(
            min_m2_formulations_per_set=1,
            min_scale_bin_images_per_set=1,
            min_eval_fine_images=2,
            min_eval_images_by_material={"VAB": 5},
        ),
        cost_weights=CostWeights(),
    )


@pytest.fixture
def manifest() -> pd.DataFrame:
    """A small, internally consistent manifest DataFrame."""
    rows = []
    spec = [
        ("AS1", "AS", "M1", ["coarse"] * 3 + ["fine"]),
        ("AS2", "AS", "M1", ["coarse"] * 2 + ["fine"]),
        ("K1", "K", "M2", ["coarse"] * 2),
        ("VAB1", "VAB", "M2", ["fine", "outlier"]),
    ]
    for formulation, material, microscope, bins in spec:
        for index, scale_bin in enumerate(bins):
            rows.append({
                "image_id": f"{formulation}_{index}",
                "formulation": formulation,
                "material": material,
                "microscope": microscope,
                "scale_bin": scale_bin,
                "pixel_size_um": 3.24 if scale_bin == "coarse" else 2.48,
                "n_instances": 10,
                "source_path": f"/data/{formulation}_{index}.jpg",
                "file_hash": f"hash_{formulation}_{index}",
            })
    return pd.DataFrame(rows)
