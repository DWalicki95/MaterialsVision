"""Tests for reading instance mask files at training time."""
import numpy as np
import pytest
import tifffile

from materials_vision.data.masks import MaskLoadError, load_instance_mask


def _write(tmp_path, labels, name="mask.tif"):
    path = tmp_path / name
    tifffile.imwrite(path, labels)
    return path


def test_valid_mask_is_returned_unchanged(tmp_path):
    labels = np.zeros((10, 12), dtype=np.uint16)
    labels[2:4, 2:4] = 1
    labels[6:8, 6:8] = 2
    path = _write(tmp_path, labels)

    loaded = load_instance_mask(path)

    assert np.array_equal(loaded, labels)


def test_empty_mask_is_allowed(tmp_path):
    path = _write(tmp_path, np.zeros((8, 8), dtype=np.uint16))

    loaded = load_instance_mask(path)

    assert loaded.max() == 0


def test_missing_file_names_the_builder(tmp_path):
    with pytest.raises(MaskLoadError, match="build_instance_masks"):
        load_instance_mask(tmp_path / "absent.tif")


def test_shape_mismatch_with_its_image_is_refused(tmp_path):
    path = _write(tmp_path, np.zeros((10, 12), dtype=np.uint16))

    with pytest.raises(MaskLoadError, match="but its image is"):
        load_instance_mask(path, expected_shape=(12, 10))


def test_matching_shape_passes(tmp_path):
    path = _write(tmp_path, np.zeros((10, 12), dtype=np.uint16))

    loaded = load_instance_mask(path, expected_shape=(10, 12))

    assert loaded.shape == (10, 12)


def test_gap_in_the_numbering_is_refused(tmp_path):
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 3          # 2 is missing
    path = _write(tmp_path, labels)

    with pytest.raises(MaskLoadError, match="gaps at"):
        load_instance_mask(path)


def test_gap_check_can_be_switched_off(tmp_path):
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[1:3, 1:3] = 7
    path = _write(tmp_path, labels)

    loaded = load_instance_mask(path, check_dense_ids=False)

    assert loaded.max() == 7


def test_non_two_dimensional_file_is_refused(tmp_path):
    path = _write(tmp_path, np.zeros((4, 5, 3), dtype=np.uint16))

    with pytest.raises(MaskLoadError, match="2-D"):
        load_instance_mask(path)


def test_negative_labels_are_refused(tmp_path):
    labels = np.zeros((6, 6), dtype=np.int16)
    labels[1:3, 1:3] = -1
    path = _write(tmp_path, labels)

    with pytest.raises(MaskLoadError, match="negative"):
        load_instance_mask(path)
