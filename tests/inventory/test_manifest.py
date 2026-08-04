"""Tests for data_prep.inventory.manifest."""
import dataclasses

import pandas as pd
import pytest

from data_prep.inventory.issues import ManifestBuildAborted
from data_prep.inventory.manifest import (MANIFEST_COLUMNS, build_manifest,
                                          write_artifacts)


class TestBuildManifestMiniDataset:
    def test_row_and_rejection_counts(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        assert len(result.manifest) == mini_dataset.expected_rows
        assert len(result.rejected) == mini_dataset.expected_rejections
        assert (
            len(result.manifest) + len(result.rejected)
            == mini_dataset.total_tasks
        )

    def test_columns_match_contract_order(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        assert list(result.manifest.columns) == list(MANIFEST_COLUMNS)

    def test_sorted_by_image_id(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        ids = list(result.manifest["image_id"])
        assert ids == sorted(ids)

    def test_sidecar_missing_flagged(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS1_40_2"]
        assert pd.isna(row["instrument"])
        assert row["pixel_size_source"] == "nominal_dict"
        codes = [i.code for i in result.issues]
        assert "sidecar_missing" in codes

    def test_geometry_rescaled_corrects_pixel_size(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS2_40_1"]
        assert bool(row["geometry_rescaled"]) is True
        assert row["pixel_size_source"] == "rescaled"
        # sidecar DataSize width is double the file width -> pixel
        # size doubles from the raw sidecar value.
        assert abs(row["pixel_size_um"] - 3.24023 * 2) < 1e-6
        codes = [i.code for i in result.issues]
        assert "geometry_rescaled" in codes

    def test_panel_crop_without_rescale(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS3_40_1"]
        assert bool(row["geometry_rescaled"]) is False
        assert row["panel_cropped_px"] == 14
        assert abs(row["pixel_size_um"] - 3.24023) < 1e-6

    def test_normal_as_row_fields(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS1_40_1"]
        assert row["series"] == "AS"
        assert row["material"] == "AS"
        assert row["magnification"] == 40
        assert row["magnification_source"] == "sem_sidecar"
        assert row["pixel_size_source"] == "sem_sidecar"
        assert row["n_instances"] == 1
        assert row["mask_path"].endswith("AS1_40_1_masks.tif")

    def test_vab_row_fields(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        vab_id = "VAB1_prostopadly_m001"
        row = result.manifest.set_index("image_id").loc[vab_id]
        assert row["series"] == "VAB"
        assert row["material"] == "VAB"
        assert row["cross_section"] == "prostopadly"
        assert row["magnification"] == 40
        assert row["magnification_source"] == "sem_sidecar"

    def test_task_without_image_rejected(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        codes = [i.code for i in result.rejected]
        assert codes == ["task_without_image"]

    def test_image_without_task_reported_not_rejected(
        self, mini_dataset
    ):
        result = build_manifest(mini_dataset.config)
        codes = [i.code for i in result.issues]
        assert "image_without_task" in codes
        # The orphan image must not appear in either rows or rejections.
        assert "AS4_40_1" not in set(result.manifest["image_id"])
        rejected_refs = {i.image_ref for i in result.rejected}
        assert not any("AS4_40_1" in ref for ref in rejected_refs)


class TestDeterminism:
    def test_two_builds_produce_identical_csv_bytes(
        self, mini_dataset, tmp_path
    ):
        result1 = build_manifest(mini_dataset.config)
        config1 = dataclasses.replace(
            mini_dataset.config, output_dir=tmp_path / "out1"
        )
        write_artifacts(result1, config1, overwrite=False)

        result2 = build_manifest(mini_dataset.config)
        config2 = dataclasses.replace(
            mini_dataset.config, output_dir=tmp_path / "out2"
        )
        write_artifacts(result2, config2, overwrite=False)

        bytes1 = (config1.output_dir / "manifest_v1.csv").read_bytes()
        bytes2 = (config2.output_dir / "manifest_v1.csv").read_bytes()
        assert bytes1 == bytes2

    def test_null_rendered_as_empty_string(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        write_artifacts(result, mini_dataset.config, overwrite=False)
        csv_path = (
            mini_dataset.config.output_dir / "manifest_v1.csv"
        )
        content = csv_path.read_text(encoding="utf-8")
        header = content.splitlines()[0].split(",")
        instrument_col = header.index("instrument")
        for line in content.splitlines()[1:]:
            fields = line.split(",")
            row_id = fields[header.index("image_id")]
            if row_id == "AS1_40_2":  # the no-sidecar row
                assert fields[instrument_col] == ""


class TestOverwriteProtection:
    def test_refuses_to_overwrite_without_flag(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        write_artifacts(result, mini_dataset.config, overwrite=False)
        with pytest.raises(FileExistsError):
            write_artifacts(
                result, mini_dataset.config, overwrite=False
            )

    def test_overwrite_flag_allows_replace(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        write_artifacts(result, mini_dataset.config, overwrite=False)
        # Should not raise.
        write_artifacts(result, mini_dataset.config, overwrite=True)


class TestImageIdCollision:
    def test_duplicate_image_id_aborts_build(self, tmp_path):
        import json as jsonlib

        import numpy as np
        from PIL import Image

        from data_prep.inventory.models import InventoryConfig, SourceConfig

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        w, h = 64, 48
        base = np.full((h, w), 128, dtype=np.uint8)
        rgb = np.stack([base, base, base], axis=-1)

        # Two different files whose LS-hash/Roboflow-hash noise
        # differs but whose core name ("AS1_40_1") is identical.
        name_a = "aaaaaaaa-AS1_40_1_jpg.rf.111111111111_image"
        name_b = "bbbbbbbb-AS1_40_1_jpg.rf.222222222222_image"
        Image.fromarray(rgb, mode="RGB").save(images_dir / f"{name_a}.jpg")
        Image.fromarray(rgb, mode="RGB").save(images_dir / f"{name_b}.jpg")

        def _task(task_id, url, ann_id):
            return {
                "id": task_id,
                "data": {"image": url},
                "annotations": [{
                    "id": ann_id, "completed_by": 1,
                    "was_cancelled": False, "ground_truth": False,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "result": [],
                }],
            }

        tasks = [
            _task(1, f"/data/upload/1/{name_a}.jpg", 1),
            _task(2, f"/data/upload/1/{name_b}.jpg", 2),
        ]
        ls_json = tmp_path / "export.json"
        ls_json.write_text(jsonlib.dumps(tasks), encoding="utf-8")

        config = InventoryConfig(
            manifest_version="v1",
            output_dir=tmp_path / "out",
            mask_root=tmp_path / "masks",
            sources=(
                SourceConfig(
                    series="AS", images_dir=images_dir,
                    label_studio_json=ls_json,
                    sem_metadata_dirs=(tmp_path / "sem",),
                ),
            ),
        )
        (tmp_path / "sem").mkdir()

        with pytest.raises(ManifestBuildAborted):
            build_manifest(config)
