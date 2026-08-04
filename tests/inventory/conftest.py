"""Shared fixtures for data_prep.inventory tests: a small, fully valid
synthetic dataset covering the main manifest-building code paths
(normal image, missing sidecar, Roboflow-style rescale, panel-only
crop, a VAB image, an orphan task and an orphan file) without needing
the real, multi-gigabyte data on disk.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_prep.inventory.models import InventoryConfig, SourceConfig

_AS_WIDTH, _AS_HEIGHT = 128, 96


def _make_rgb_image(
    path: Path, width: int, height: int, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(height, width), dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def _make_gray_image(
    path: Path, width: int, height: int, seed: int = 1
) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(height, width), dtype=np.uint8)
    Image.fromarray(base, mode="L").save(path)


def _write_sidecar(path: Path, fields: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[SemImageFile]"] + [f"{k}={v}" for k, v in fields.items()]
    path.write_text(
        "\r\n".join(lines) + "\r\n", encoding="iso-8859-2"
    )


def _polygon_task(
    task_id, image_url, width, height, ann_id, square=(20, 20, 40, 40)
):
    x0, y0, x1, y1 = square
    points = [
        [x0 / width * 100, y0 / height * 100],
        [x1 / width * 100, y0 / height * 100],
        [x1 / width * 100, y1 / height * 100],
        [x0 / width * 100, y1 / height * 100],
    ]
    return {
        "id": task_id,
        "data": {"image": image_url},
        "annotations": [{
            "id": ann_id,
            "completed_by": 1,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": "2026-01-01T00:00:00.000000Z",
            "updated_at": "2026-01-01T00:00:00.000000Z",
            "result": [{
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "id": f"res{ann_id}",
                "from_name": "poly_tool",
                "to_name": "image",
                "type": "polygonlabels",
                "value": {
                    "points": points,
                    "closed": True,
                    "polygonlabels": ["Por"],
                },
            }],
        }],
    }


@dataclass
class MiniDataset:
    config: InventoryConfig
    expected_rows: int
    expected_rejections: int
    total_tasks: int


@pytest.fixture
def mini_dataset(tmp_path) -> MiniDataset:
    as_images = tmp_path / "as_images"
    as_images.mkdir()
    vab_images = tmp_path / "vab_images"
    vab_images.mkdir()
    as_sem = tmp_path / "as_sem"
    vab_sem = tmp_path / "vab_sem"

    w, h = _AS_WIDTH, _AS_HEIGHT

    # 1. Normal AS image with a consistent sidecar.
    name1 = "AS1_40_1_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_image"
    _make_rgb_image(as_images / f"{name1}.jpg", w, h, seed=10)
    _write_sidecar(as_sem / "AS1" / "AS1_40_1.txt", {
        "InstructName": "TM3000", "Magnification": 40,
        "PixelSize": 3240.23, "DataSize": f"{w}x{h}",
        "MicronMarker": 200000, "Format": "JPG",
        "ImageName": "AS1_40_1.jpg", "Date": "1/1/2026",
        "Time": "12:00:00",
    })

    # 2. AS image with NO sidecar at all.
    name2 = "AS1_40_2_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb_image"
    _make_rgb_image(as_images / f"{name2}.jpg", w, h, seed=20)

    # 3. AS image whose sidecar DataSize width differs (Roboflow-style
    #    rescale): geometry_rescaled=True, pixel_size_um corrected.
    name3 = "AS2_40_1_jpg.rf.cccccccccccccccccccccccccccccccc_image"
    _make_rgb_image(as_images / f"{name3}.jpg", w, h, seed=30)
    _write_sidecar(as_sem / "AS2" / "AS2_40_1.txt", {
        "InstructName": "TM3000", "Magnification": 40,
        "PixelSize": 3240.23, "DataSize": f"{w * 2}x{h * 2}",
        "MicronMarker": 200000, "Format": "JPG",
        "ImageName": "AS2_40_1.jpg", "Date": "1/1/2026",
        "Time": "12:00:00",
    })

    # 4. AS image whose sidecar DataSize has the same width but a
    #    taller height (panel crop only, no rescale).
    name4 = "AS3_40_1_jpg.rf.dddddddddddddddddddddddddddddddd_image"
    _make_rgb_image(as_images / f"{name4}.jpg", w, h, seed=40)
    _write_sidecar(as_sem / "AS3" / "AS3_40_1.txt", {
        "InstructName": "TM3000", "Magnification": 40,
        "PixelSize": 3240.23, "DataSize": f"{w}x{h + 14}",
        "MicronMarker": 200000, "Format": "JPG",
        "ImageName": "AS3_40_1.jpg", "Date": "1/1/2026",
        "Time": "12:00:00",
    })

    # 5. Orphan AS image file with no Label Studio task.
    name_orphan = (
        "AS4_40_1_jpg.rf.eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_image"
    )
    _make_rgb_image(as_images / f"{name_orphan}.jpg", w, h, seed=50)

    as_tasks = [
        _polygon_task(1, f"/data/upload/1/{name1}.jpg", w, h, 101),
        _polygon_task(2, f"/data/upload/1/{name2}.jpg", w, h, 102),
        _polygon_task(3, f"/data/upload/1/{name3}.jpg", w, h, 103),
        _polygon_task(4, f"/data/upload/1/{name4}.jpg", w, h, 104),
        # 6. Task referencing an image file that was never created.
        _polygon_task(
            5, "/data/upload/1/missing_file_jpg.rf.ffff_image.jpg",
            w, h, 105,
        ),
    ]
    as_json = tmp_path / "as_export.json"
    as_json.write_text(json.dumps(as_tasks), encoding="utf-8")

    # 7. Normal VAB image with a consistent sidecar.
    vab_name = "VAB1_prostopadla_VAB1_prostopadla_m001"
    _make_gray_image(vab_images / f"{vab_name}.png", w, h)
    _write_sidecar(
        vab_sem / "VAB1 prostopadla" / "VAB1 prostopadla_m001.txt",
        {
            "InstructName": "SU8000", "Magnification": 40,
            "PixelSize": 2480.469, "DataSize": f"{w}x{h}",
            "MicronMarker": 150000, "Format": "tif",
            "ImageName": "VAB1 prostopadla_m001.tif",
            "Date": "1/1/2026", "Time": "12:00:00",
        },
    )
    vab_tasks = [
        _polygon_task(
            201, f"/data/upload/2/{vab_name}.png", w, h, 201,
        ),
    ]
    vab_json = tmp_path / "vab_export.json"
    vab_json.write_text(json.dumps(vab_tasks), encoding="utf-8")

    config = InventoryConfig(
        manifest_version="v1",
        output_dir=tmp_path / "out",
        mask_root=tmp_path / "masks",
        sources=(
            SourceConfig(
                series="AS", images_dir=as_images,
                label_studio_json=as_json,
                sem_metadata_dirs=(as_sem,),
            ),
            SourceConfig(
                series="VAB", images_dir=vab_images,
                label_studio_json=vab_json,
                sem_metadata_dirs=(vab_sem,),
            ),
        ),
    )

    return MiniDataset(
        config=config,
        expected_rows=5,  # 4 AS (name1-4) + 1 VAB
        expected_rejections=1,  # task 5, missing file
        total_tasks=6,  # 5 AS tasks + 1 VAB task
    )
