"""Shared fixtures for the training-time split access tests."""
import hashlib
import json

import pandas as pd
import pytest

SPLIT_ID = "split_test"


@pytest.fixture
def split_rows() -> pd.DataFrame:
    """A miniature split table with the shape of the real one.

    Includes the case that matters: a ``scale_outlier`` image whose
    formulation sits outside TRAIN, hence ``used = False``.
    """
    spec = [
        ("AS1_0", "AS1", "AS", "M1", "coarse", "train", True),
        ("AS1_1", "AS1", "AS", "M1", "coarse", "train", True),
        ("AS1_2", "AS1", "AS", "M1", "fine", "train", True),
        ("K1_0", "K1", "K", "M2", "coarse", "train", True),
        ("VAB1_0", "VAB1", "VAB", "M2", "fine", "train", True),
        ("AS2_0", "AS2", "AS", "M1", "coarse", "val", True),
        ("AS2_1", "AS2", "AS", "M1", "fine", "val", True),
        ("K2_0", "K2", "K", "M2", "coarse", "val", True),
        ("AS3_0", "AS3", "AS", "M1", "coarse", "test", True),
        ("AS3_1", "AS3", "AS", "M1", "outlier", "test", False),
        ("VAB2_0", "VAB2", "VAB", "M2", "fine", "test", True),
    ]
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "formulation": formulation,
                "material": material,
                "microscope": microscope,
                "scale_bin": scale_bin,
                "pixel_size_um": 3.24,
                "n_instances": 40,
                "source_path": f"/data/{image_id}.jpg",
                "split": split,
                "used": used,
            }
            for (image_id, formulation, material, microscope,
                 scale_bin, split, used) in spec
        ]
    )


@pytest.fixture
def manifest_file(tmp_path):
    """A stand-in manifest whose hash the split metadata records."""
    path = tmp_path / "manifest_v2.csv"
    path.write_text("image_id\nAS1_0\n", encoding="utf-8")
    return path


@pytest.fixture
def split_csv(tmp_path, split_rows, manifest_file):
    """A split table with its metadata sidecar written alongside."""
    path = tmp_path / f"{SPLIT_ID}.csv"
    split_rows.to_csv(path, index=False)

    metadata = {
        "split_id": SPLIT_ID,
        "manifest_sha256": hashlib.sha256(
            manifest_file.read_bytes()
        ).hexdigest(),
    }
    meta_path = tmp_path / f"{SPLIT_ID}_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return path
