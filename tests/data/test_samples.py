"""Tests for turning split rows into prepared samples."""
import numpy as np
import pytest
import tifffile
from PIL import Image

from materials_vision.data.samples import SampleSource, SampleSourceError
from materials_vision.data.split_io import load_split

PANEL_ROWS = 6
FRAME = (20, 16)                      # (height, width)
CONTENT_HEIGHT = FRAME[0] - PANEL_ROWS


def _labels_with(instances):
    labels = np.zeros(FRAME, dtype=np.uint16)
    for instance_id, (rows, cols) in instances.items():
        labels[rows, cols] = instance_id
    return labels


@pytest.fixture
def dataset_on_disk(tmp_path, split_rows):
    """Write images and masks for the fixture split, plus a manifest.

    ``AS1_*`` come from a microscope whose panel is already stripped,
    ``K1_*``/``VAB1_*`` from one that leaves a 6-row panel in place.
    """
    import pandas as pd

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    rows = []
    for row in split_rows.itertuples(index=False):
        is_m2 = row.microscope == "M2"
        mode = "L" if is_m2 else "RGB"
        crop_h = CONTENT_HEIGHT if is_m2 else FRAME[0]

        pixels = np.full(FRAME, 120, dtype=np.uint8)
        if mode == "RGB":
            pixels = np.stack([pixels] * 3, axis=-1)
        Image.fromarray(pixels, mode=mode).save(
            images / f"{row.image_id}.png"
        )

        labels = _labels_with({
            1: (slice(2, 6), slice(2, 6)),
            2: (slice(8, 12), slice(8, 12)),
        })
        tifffile.imwrite(masks / f"{row.image_id}_masks.tif", labels)

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

    manifest = pd.DataFrame(rows)
    updated = split_rows.copy()
    updated["source_path"] = [
        str(images / f"{image_id}.png")
        for image_id in updated["image_id"]
    ]
    return manifest, updated


@pytest.fixture
def source(tmp_path, dataset_on_disk, split_csv):
    import pandas as pd

    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)
    subset = load_split(split_csv, "train")
    del pd
    return SampleSource(
        subset, manifest, min_fragment_area_px2=1
    )


def test_records_carry_split_and_manifest_facts(source):
    record = source.record(0)

    assert record.index == 0
    assert record.image_id == source.records[0].image_id
    assert record.crop_bbox[2] == FRAME[1]
    assert record.n_instances_expected == 2
    assert record.q_max_i == pytest.approx(1.3)
    assert len(source) == 5


def test_rgb_image_is_reduced_to_one_channel(source):
    index = next(
        i for i, r in enumerate(source.records)
        if r.microscope == "M1"
    )

    sample = source.load(index)

    assert sample.image.ndim == 2
    assert sample.image.dtype == np.uint8
    assert sample.image.shape == FRAME


def test_panel_is_cropped_only_for_the_affected_microscope(source):
    m1 = next(
        i for i, r in enumerate(source.records) if r.microscope == "M1"
    )
    m2 = next(
        i for i, r in enumerate(source.records) if r.microscope == "M2"
    )

    assert source.load(m1).image.shape == FRAME
    assert source.load(m2).image.shape == (CONTENT_HEIGHT, FRAME[1])


def test_image_and_labels_keep_the_same_shape(source):
    for index in range(len(source)):
        sample = source.load(index)

        assert sample.image.shape == sample.labels.shape


def test_instances_survive_and_are_renumbered(source):
    sample = source.load(0)

    assert sample.n_instances == 2
    assert np.unique(sample.labels).tolist() == [0, 1, 2]
    assert sample.border_instance.shape == (2,)


def test_mask_disagreeing_with_the_manifest_is_refused(
    tmp_path, dataset_on_disk, split_csv
):
    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)
    manifest = manifest.copy()
    manifest["n_instances"] = 99
    source = SampleSource(
        load_split(split_csv, "train"), manifest,
        min_fragment_area_px2=1,
    )

    with pytest.raises(SampleSourceError, match="out of sync"):
        source.load(0)


def test_image_absent_from_the_manifest_is_refused(
    dataset_on_disk, split_csv
):
    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)
    trimmed = manifest.iloc[1:]

    with pytest.raises(SampleSourceError, match="absent from the manif"):
        SampleSource(
            load_split(split_csv, "train"), trimmed,
            min_fragment_area_px2=1,
        )


def test_multichannel_image_without_identical_channels_is_refused(
    dataset_on_disk, split_csv
):
    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)
    manifest = manifest.copy()
    manifest.loc[manifest["n_channels"] == 3, "channels_identical"] = False

    with pytest.raises(SampleSourceError, match="identical channel"):
        SampleSource(
            load_split(split_csv, "train"), manifest,
            min_fragment_area_px2=1,
        )


def test_missing_manifest_column_is_refused(dataset_on_disk, split_csv):
    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)

    with pytest.raises(SampleSourceError, match="missing column"):
        SampleSource(
            load_split(split_csv, "train"),
            manifest.drop(columns=["q_max_i"]),
            min_fragment_area_px2=1,
        )


def test_missing_image_file_is_refused(
    tmp_path, dataset_on_disk, split_csv
):
    manifest, updated_split = dataset_on_disk
    updated_split.to_csv(split_csv, index=False)
    source = SampleSource(
        load_split(split_csv, "train"), manifest,
        min_fragment_area_px2=1,
    )
    source.record(0).source_path.unlink()

    with pytest.raises(SampleSourceError, match="image not found"):
        source.load(0)


def test_loading_does_not_depend_on_call_order(source):
    first_pass = [source.load(i).labels.copy() for i in range(3)]
    second_pass = [source.load(i).labels for i in (2, 0, 1)]

    assert np.array_equal(first_pass[2], second_pass[0])
    assert np.array_equal(first_pass[0], second_pass[1])
