"""Tests for data_prep.inventory.reporting."""
import json

from data_prep.inventory.manifest import build_manifest, write_artifacts
from data_prep.inventory.reporting import (write_dataset_summary,
                                           write_run_metadata,
                                           write_thumbnails,
                                           write_validation_report)


def test_write_validation_report(mini_dataset):
    result = build_manifest(mini_dataset.config)
    md_path = write_validation_report(result, mini_dataset.config)
    assert md_path.exists()
    csv_path = mini_dataset.config.output_dir / "validation_report.csv"
    assert csv_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "sidecar_missing" in content
    assert "task_without_image" in content


def test_write_dataset_summary(mini_dataset):
    result = build_manifest(mini_dataset.config)
    md_path = write_dataset_summary(result, mini_dataset.config)
    assert md_path.exists()
    csv_path = mini_dataset.config.output_dir / "dataset_summary.csv"
    assert csv_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "q_max" in content
    assert "XVI.2" in content


def test_write_dataset_summary_empty_manifest(mini_dataset, tmp_path):
    import dataclasses

    import pandas as pd

    from data_prep.inventory.manifest import InventoryResult

    empty_result = InventoryResult(
        manifest=pd.DataFrame(), issues=[], rejected=[],
    )
    config = dataclasses.replace(
        mini_dataset.config, output_dir=tmp_path / "empty_out"
    )
    md_path = write_dataset_summary(empty_result, config)
    assert md_path.exists()


def test_write_run_metadata(mini_dataset):
    result = build_manifest(mini_dataset.config)
    paths = write_artifacts(result, mini_dataset.config, overwrite=False)
    meta_path = write_run_metadata(
        result, mini_dataset.config, paths["manifest"]
    )
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["n_rows"] == mini_dataset.expected_rows
    assert data["n_rejected"] == mini_dataset.expected_rejections
    assert "manifest_sha256" in data
    assert len(data["manifest_sha256"]) == 64
    assert "library_versions" in data
    assert data["manifest_version"] == "v1"


def test_write_thumbnails(mini_dataset):
    result = build_manifest(mini_dataset.config)
    thumb_dir = write_thumbnails(result, mini_dataset.config, n=3)
    assert thumb_dir.exists()
    pngs = list(thumb_dir.glob("*.png"))
    assert 0 < len(pngs) <= 3
