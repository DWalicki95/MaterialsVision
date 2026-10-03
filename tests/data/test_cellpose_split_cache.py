"""Tests for exporting the frozen split to Cellpose's file cache.

What these pin is the seam: the files Cellpose trains from must be the
samples the PEFT-SAM runs were trained and scored on, as the real
split reader and sample source prepare them, and nothing else.
"""
import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from materials_vision.data.samples import SampleSource
from materials_vision.data.split_io import load_split
from materials_vision.experiments.cellpose.split_cache import (
    CacheExportError, export_source, flows_path, image_path, mask_path)

FRAME = (20, 16)                      # (height, width)
PANEL_ROWS = 6


def _labels() -> np.ndarray:
    labels = np.zeros(FRAME, dtype=np.uint16)
    labels[2:6, 2:6] = 1
    # Crosses into the panel of the microscope that carries one, so the
    # crop has an instance to rebuild rather than only whole ones.
    labels[10:17, 8:14] = 2
    return labels


@pytest.fixture
def split_on_disk(tmp_path, split_rows, split_csv):
    """Images with distinct random content, masks, and a manifest.

    Returns
    -------
    tuple
        The split table path, the manifest, and the mask directory.
    """
    rng = np.random.default_rng(7)
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    rows = []
    for row in split_rows.itertuples(index=False):
        is_m2 = row.microscope == "M2"
        pixels = rng.integers(0, 256, FRAME, dtype=np.uint8)
        if is_m2:
            Image.fromarray(pixels, mode="L").save(
                images / f"{row.image_id}.png"
            )
        else:
            Image.fromarray(
                np.stack([pixels] * 3, axis=-1), mode="RGB"
            ).save(images / f"{row.image_id}.png")
        tifffile.imwrite(masks / f"{row.image_id}_masks.tif", _labels())
        crop_h = FRAME[0] - PANEL_ROWS if is_m2 else FRAME[0]
        rows.append({
            "image_id": row.image_id,
            "load_crop_bbox": f"0,0,{FRAME[1]},{crop_h}",
            "mask_path": str(masks / f"{row.image_id}_masks.tif"),
            "width_px": FRAME[1],
            "height_px": FRAME[0],
            "n_channels": 1 if is_m2 else 3,
            "channels_identical": None if is_m2 else True,
            "n_instances": 2,
            "q_max_i": 1.3,
        })

    updated = split_rows.copy()
    updated["source_path"] = [
        str(images / f"{image_id}.png") for image_id in updated["image_id"]
    ]
    updated.to_csv(split_csv, index=False)
    return split_csv, pd.DataFrame(rows), masks


def _source(split_on_disk, subset="train") -> SampleSource:
    split_csv, manifest, _ = split_on_disk
    return SampleSource(
        load_split(split_csv, subset), manifest, min_fragment_area_px2=1
    )


def test_cache_holds_exactly_what_the_source_prepares(
    tmp_path, split_on_disk
):
    source = _source(split_on_disk)

    cached = export_source(source, tmp_path / "cache")

    for position, (image_file, mask_file) in enumerate(
        zip(cached.image_paths, cached.mask_paths)
    ):
        prepared = source.load(position)
        image = tifffile.imread(image_file)
        labels = tifffile.imread(mask_file)
        assert image.dtype == prepared.image.dtype
        assert labels.dtype == prepared.labels.dtype
        np.testing.assert_array_equal(image, prepared.image)
        np.testing.assert_array_equal(labels, prepared.labels)


def test_panel_is_cropped_before_cellpose_sees_the_image(
    tmp_path, split_on_disk
):
    cached = export_source(_source(split_on_disk), tmp_path / "cache")

    shapes = {
        image_id: tifffile.imread(path).shape
        for image_id, path in zip(cached.image_ids, cached.image_paths)
    }
    assert shapes["K1_0"] == (FRAME[0] - PANEL_ROWS, FRAME[1])
    assert shapes["AS1_0"] == FRAME


def test_cache_lists_the_subsets_rows_in_source_order(
    tmp_path, split_on_disk
):
    split_csv, _, _ = split_on_disk
    table = pd.read_csv(split_csv)
    train_ids = table.loc[
        (table["split"] == "train") & table["used"], "image_id"
    ]
    source = _source(split_on_disk)

    cached = export_source(source, tmp_path / "cache")

    assert set(cached.image_ids) == set(train_ids)
    assert cached.image_ids == tuple(r.image_id for r in source.records)
    assert cached.image_paths == tuple(
        image_path(tmp_path / "cache", i) for i in cached.image_ids
    )
    assert cached.mask_paths == tuple(
        mask_path(tmp_path / "cache", i) for i in cached.image_ids
    )


def test_unchanged_samples_are_not_rewritten_and_keep_their_flows(
    tmp_path, split_on_disk
):
    cache = tmp_path / "cache"
    first = export_source(_source(split_on_disk), cache)
    flows = flows_path(first.image_paths[0])
    flows.write_bytes(b"flows")

    second = export_source(_source(split_on_disk), cache)

    assert first.n_written == len(first)
    assert second.n_written == 0
    assert second.n_flows_invalidated == 0
    assert flows.read_bytes() == b"flows"


def test_a_changed_mask_is_rewritten_and_its_flows_invalidated(
    tmp_path, split_on_disk
):
    _, _, masks = split_on_disk
    cache = tmp_path / "cache"
    first = export_source(_source(split_on_disk), cache)
    for image_file in first.image_paths:
        flows_path(image_file).write_bytes(b"flows")
    changed_id = first.image_ids[0]
    labels = _labels()
    labels[2:6, 2:6] = 0
    labels[3:7, 3:7] = 1
    tifffile.imwrite(masks / f"{changed_id}_masks.tif", labels)
    source = _source(split_on_disk)

    second = export_source(source, cache)

    assert second.n_written == 1
    assert second.n_flows_invalidated == 1
    assert not flows_path(image_path(cache, changed_id)).exists()
    assert all(
        flows_path(path).exists() for path in second.image_paths[1:]
    )
    np.testing.assert_array_equal(
        tifffile.imread(mask_path(cache, changed_id)), source.load(0).labels
    )


def test_flow_files_are_named_as_cellpose_names_them(tmp_path):
    from materials_vision.experiments.cellpose.utils import \
        precompute_flows_batched

    image_file = image_path(tmp_path, "AS1_0")
    flows_path(image_file).write_bytes(b"flows")

    # Every flow field already exists, so Cellpose only resolves the
    # names and computes nothing.
    resolved = precompute_flows_batched(
        [str(mask_path(tmp_path, "AS1_0"))], [str(image_file)]
    )

    assert resolved == [str(flows_path(image_file))]


class _DuplicatedSource:
    """Stands in for a source that names one image twice."""

    def __init__(self, source: SampleSource) -> None:
        self.records = source.records[:1] * 2

    def load(self, position):  # pragma: no cover - never reached
        raise AssertionError("A refused export must read nothing.")


def test_a_source_naming_an_image_twice_is_refused(tmp_path, split_on_disk):
    with pytest.raises(CacheExportError, match="more than once"):
        export_source(
            _DuplicatedSource(_source(split_on_disk)), tmp_path / "cache"
        )
    assert not (tmp_path / "cache").exists()
