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


class TestMicroscopeAndScale:
    def test_microscope_from_sidecar(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS1_40_1"]
        assert row["microscope"] == "M1"
        assert row["microscope_source"] == "sem_sidecar"

    def test_microscope_series_fallback_without_sidecar(
        self, mini_dataset
    ):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS1_40_2"]
        assert row["microscope"] == "M1"
        assert row["microscope_source"] == "series_map"

    def test_vab_microscope_and_scale_bin(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        vab_id = "VAB1_prostopadly_m001"
        row = result.manifest.set_index("image_id").loc[vab_id]
        assert row["microscope"] == "M2"
        assert row["microscope_source"] == "sem_sidecar"
        assert row["scale_bin"] == "fine"
        assert bool(row["scale_outlier"]) is False
        assert abs(row["q_max_i"] - 1.0) < 1e-6
        assert row["load_crop_bbox"] == "0,0,128,190"

    def test_as_scale_bin_coarse_and_load_crop_bbox(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        row = result.manifest.set_index("image_id").loc["AS1_40_1"]
        assert row["scale_bin"] == "coarse"
        assert bool(row["scale_outlier"]) is False
        assert row["load_crop_bbox"] == "0,0,128,96"


class TestScaleOutlierRelativeDiagnostic:
    def test_relative_deviation_is_info_not_column(self, tmp_path):
        import json as jsonlib

        import numpy as np
        from PIL import Image

        from data_prep.inventory.models import InventoryConfig, SourceConfig

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        sem_dir = tmp_path / "sem"
        w, h = 64, 48

        def _image(name):
            rng = np.random.default_rng(0)
            base = rng.integers(60, 200, size=(h, w), dtype=np.uint8)
            rgb = np.stack([base, base, base], axis=-1)
            Image.fromarray(rgb, mode="RGB").save(
                images_dir / f"{name}.jpg"
            )

        def _sidecar(formulation, image_id, pixel_size_nm):
            # find_sidecar looks up <formulation>/<image_id>.txt, where
            # image_id is the parsed core name, not the raw LS/Roboflow
            # filename (see ASProfile.sidecar_candidates).
            path = sem_dir / formulation / f"{image_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "[SemImageFile]", "InstructName=TM3000",
                "Magnification=40", f"PixelSize={pixel_size_nm}",
                f"DataSize={w}x{h}", "Format=JPG",
            ]
            path.write_text(
                "\r\n".join(lines) + "\r\n", encoding="iso-8859-2"
            )

        # Both stay in the "coarse" absolute bin (>= 3.0 um/px), but
        # their ratio (2.0x) exceeds the default scale_outlier_ratio
        # (1.5x) around their shared series median (6.0 um/px) - the
        # relative rule must flag the low one as a diagnostic without
        # touching scale_bin/scale_outlier on either row.
        name_low = "AS1_40_1_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_image"
        name_high = "AS2_40_1_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb_image"
        _image(name_low)
        _image(name_high)
        _sidecar("AS1", "AS1_40_1", 3000.0)
        _sidecar("AS2", "AS2_40_1", 9000.0)

        def _task(task_id, name, ann_id):
            return {
                "id": task_id,
                "data": {"image": f"/data/upload/1/{name}.jpg"},
                "annotations": [{
                    "id": ann_id, "completed_by": 1,
                    "was_cancelled": False, "ground_truth": False,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "result": [],
                }],
            }

        tasks = [_task(1, name_low, 1), _task(2, name_high, 2)]
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
                    sem_metadata_dirs=(sem_dir,),
                ),
            ),
        )

        result = build_manifest(config)

        assert bool(
            result.manifest.set_index("image_id")
            .loc["AS1_40_1", "scale_outlier"]
        ) is False
        assert bool(
            result.manifest.set_index("image_id")
            .loc["AS2_40_1", "scale_outlier"]
        ) is False
        codes = [i.code for i in result.issues]
        assert "scale_outlier_relative_diagnostic" in codes
        assert "scale_outlier" not in codes


class TestMicroscopeProductConflict:
    def test_product_mismatch_is_warning_not_reassignment(
        self, tmp_path
    ):
        import json as jsonlib

        import numpy as np
        from PIL import Image

        from data_prep.inventory.models import InventoryConfig, SourceConfig

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        sem_dir = tmp_path / "sem"
        w, h = 64, 48

        rng = np.random.default_rng(0)
        base = rng.integers(60, 200, size=(h, w), dtype=np.uint8)
        rgb = np.stack([base, base, base], axis=-1)
        name = "AS1_40_1_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_image"
        Image.fromarray(rgb, mode="RGB").save(images_dir / f"{name}.jpg")

        # InstructName=TM3000 (-> microscope M1) but PixelSize is the
        # SU8000 40x nominal value: pixel_size_um * magnification is
        # nowhere near MICROSCOPE_PRODUCT_UM["M1"] (~129.6). The
        # conflict must be flagged, not silently reassign microscope.
        sidecar_path = sem_dir / "AS1" / f"{name.split('_jpg')[0]}.txt"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "[SemImageFile]", "InstructName=TM3000", "Magnification=40",
            "PixelSize=2480.469", f"DataSize={w}x{h}", "Format=JPG",
        ]
        sidecar_path.write_text(
            "\r\n".join(lines) + "\r\n", encoding="iso-8859-2"
        )

        task = {
            "id": 1,
            "data": {"image": f"/data/upload/1/{name}.jpg"},
            "annotations": [{
                "id": 1, "completed_by": 1,
                "was_cancelled": False, "ground_truth": False,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "result": [],
            }],
        }
        ls_json = tmp_path / "export.json"
        ls_json.write_text(jsonlib.dumps([task]), encoding="utf-8")

        config = InventoryConfig(
            manifest_version="v1",
            output_dir=tmp_path / "out",
            mask_root=tmp_path / "masks",
            sources=(
                SourceConfig(
                    series="AS", images_dir=images_dir,
                    label_studio_json=ls_json,
                    sem_metadata_dirs=(sem_dir,),
                ),
            ),
        )

        result = build_manifest(config)

        row = result.manifest.set_index("image_id").loc["AS1_40_1"]
        assert row["microscope"] == "M1"
        assert row["microscope_source"] == "sem_sidecar"
        codes = [i.code for i in result.issues]
        assert "microscope_product_conflict" in codes


class TestContentBboxCropMismatch:
    def test_mismatch_aborts_build(self, tmp_path):
        import json as jsonlib

        import numpy as np
        from PIL import Image

        from data_prep.inventory.models import InventoryConfig, SourceConfig

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        sem_dir = tmp_path / "sem"
        w, h = 128, 96

        # No panel drawn at all, so detect_nonimage_region reports the
        # full frame as content_bbox - but the sidecar resolves this
        # row to microscope M2 (SU8000), whose frozen
        # PANEL_HEIGHT_ROWS_BY_MICROSCOPE expects the bottom 70 rows
        # cropped away. content_bbox and load_crop_bbox must disagree.
        rng = np.random.default_rng(0)
        base = rng.integers(60, 200, size=(h, w), dtype=np.uint8)
        vab_name = "VAB1_prostopadla_VAB1_prostopadla_m001"
        Image.fromarray(base, mode="L").save(
            images_dir / f"{vab_name}.png"
        )
        sidecar_path = (
            sem_dir / "VAB1 prostopadla" / "VAB1 prostopadla_m001.txt"
        )
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "[SemImageFile]", "InstructName=SU8000", "Magnification=40",
            "PixelSize=2480.469", f"DataSize={w}x{h}", "Format=tif",
        ]
        sidecar_path.write_text(
            "\r\n".join(lines) + "\r\n", encoding="iso-8859-2"
        )

        task = {
            "id": 1,
            "data": {"image": f"/data/upload/2/{vab_name}.png"},
            "annotations": [{
                "id": 1, "completed_by": 1,
                "was_cancelled": False, "ground_truth": False,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "result": [],
            }],
        }
        ls_json = tmp_path / "export.json"
        ls_json.write_text(jsonlib.dumps([task]), encoding="utf-8")

        config = InventoryConfig(
            manifest_version="v1",
            output_dir=tmp_path / "out",
            mask_root=tmp_path / "masks",
            sources=(
                SourceConfig(
                    series="VAB", images_dir=images_dir,
                    label_studio_json=ls_json,
                    sem_metadata_dirs=(sem_dir,),
                ),
            ),
        )

        with pytest.raises(ManifestBuildAborted) as excinfo:
            build_manifest(config)
        codes = [i.code for i in excinfo.value.fatal_issues]
        assert "content_bbox_crop_mismatch" in codes


class TestInstancesBelowCropBbox:
    def test_instance_reaching_into_panel_is_counted(self, tmp_path):
        import json as jsonlib

        import numpy as np
        from PIL import Image

        from data_prep.inventory.manifest import (
            PANEL_HEIGHT_ROWS_BY_MICROSCOPE)
        from data_prep.inventory.models import InventoryConfig, SourceConfig

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        sem_dir = tmp_path / "sem"
        w, h = 128, 260
        panel_rows = PANEL_HEIGHT_ROWS_BY_MICROSCOPE["M2"]
        content_h = h - panel_rows

        # A real panel, matching PANEL_HEIGHT_ROWS_BY_MICROSCOPE, so
        # this row passes the content_bbox/load_crop_bbox check and
        # reaches annotation processing.
        rng = np.random.default_rng(0)
        content = rng.integers(
            60, 200, size=(content_h, w), dtype=np.uint8
        )
        panel = np.zeros((panel_rows, w), dtype=np.uint8)
        full = np.concatenate([content, panel], axis=0)
        vab_name = "VAB1_prostopadla_VAB1_prostopadla_m001"
        Image.fromarray(full, mode="L").save(
            images_dir / f"{vab_name}.png"
        )
        sidecar_path = (
            sem_dir / "VAB1 prostopadla" / "VAB1 prostopadla_m001.txt"
        )
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "[SemImageFile]", "InstructName=SU8000", "Magnification=40",
            "PixelSize=2480.469", f"DataSize={w}x{h}", "Format=tif",
        ]
        sidecar_path.write_text(
            "\r\n".join(lines) + "\r\n", encoding="iso-8859-2"
        )

        # Square straddling the content/panel boundary: its top edge
        # is in the content area, its bottom edge is inside the panel.
        x0, y0, x1, y1 = 20, content_h - 20, 40, content_h + 20
        points = [
            [x0 / w * 100, y0 / h * 100],
            [x1 / w * 100, y0 / h * 100],
            [x1 / w * 100, y1 / h * 100],
            [x0 / w * 100, y1 / h * 100],
        ]
        task = {
            "id": 1,
            "data": {"image": f"/data/upload/2/{vab_name}.png"},
            "annotations": [{
                "id": 1, "completed_by": 1,
                "was_cancelled": False, "ground_truth": False,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "result": [{
                    "original_width": w, "original_height": h,
                    "image_rotation": 0, "id": "res1",
                    "from_name": "poly_tool", "to_name": "image",
                    "type": "polygonlabels",
                    "value": {
                        "points": points, "closed": True,
                        "polygonlabels": ["Por"],
                    },
                }],
            }],
        }
        ls_json = tmp_path / "export.json"
        ls_json.write_text(jsonlib.dumps([task]), encoding="utf-8")

        config = InventoryConfig(
            manifest_version="v1",
            output_dir=tmp_path / "out",
            mask_root=tmp_path / "masks",
            sources=(
                SourceConfig(
                    series="VAB", images_dir=images_dir,
                    label_studio_json=ls_json,
                    sem_metadata_dirs=(sem_dir,),
                ),
            ),
        )

        result = build_manifest(config)

        row = result.manifest.iloc[0]
        assert row["n_instances_below_crop_bbox"] == 1
        codes = [i.code for i in result.issues]
        assert "annotation_below_crop_bbox" in codes

    def test_instance_inside_content_is_not_counted(self, mini_dataset):
        result = build_manifest(mini_dataset.config)
        vab_id = "VAB1_prostopadly_m001"
        row = result.manifest.set_index("image_id").loc[vab_id]
        assert row["n_instances_below_crop_bbox"] == 0


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
