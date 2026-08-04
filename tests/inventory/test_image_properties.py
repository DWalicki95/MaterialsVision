"""Tests for data_prep.inventory.image_properties.

Fixture images are generated into tmp_path at test time rather than
committed to the repo, since the repository's .gitignore blanket-
excludes *.png (they would silently never be tracked otherwise).
"""
import hashlib

import numpy as np
import pytest
from PIL import Image

from data_prep.inventory.image_properties import read_image_properties

_WIDTH, _HEIGHT = 64, 48


@pytest.fixture
def rgb_identical_channels(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(
        60, 200, size=(_HEIGHT, _WIDTH), dtype=np.uint8
    )
    rgb = np.stack([base, base, base], axis=-1)
    path = tmp_path / "rgb_identical.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


@pytest.fixture
def rgb_different_channels(tmp_path):
    rng = np.random.default_rng(1)
    r = rng.integers(0, 255, size=(_HEIGHT, _WIDTH), dtype=np.uint8)
    g = rng.integers(0, 255, size=(_HEIGHT, _WIDTH), dtype=np.uint8)
    b = rng.integers(0, 255, size=(_HEIGHT, _WIDTH), dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    path = tmp_path / "rgb_different.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


@pytest.fixture
def single_channel(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.integers(
        60, 200, size=(_HEIGHT, _WIDTH), dtype=np.uint8
    )
    path = tmp_path / "gray.png"
    Image.fromarray(base, mode="L").save(path)
    return path


def test_rgb_channels_identical(rgb_identical_channels):
    props = read_image_properties(rgb_identical_channels)
    assert props.width_px == _WIDTH
    assert props.height_px == _HEIGHT
    assert props.file_format == "PNG"
    assert props.n_channels == 3
    assert props.channels_identical is True
    assert props.bit_depth == 8
    assert props.gray.shape == (_HEIGHT, _WIDTH)


def test_rgb_channels_differ(rgb_different_channels):
    props = read_image_properties(rgb_different_channels)
    assert props.n_channels == 3
    assert props.channels_identical is False


def test_single_channel(single_channel):
    props = read_image_properties(single_channel)
    assert props.n_channels == 1
    assert props.channels_identical is None
    assert props.gray.shape == (_HEIGHT, _WIDTH)


def test_file_hash_matches_sha256(rgb_identical_channels):
    props = read_image_properties(rgb_identical_channels)
    expected = hashlib.sha256(
        rgb_identical_channels.read_bytes()
    ).hexdigest()
    assert props.file_hash == expected


def test_single_read_hash_and_pixels_consistent(rgb_identical_channels):
    # Same file read twice must give identical hash and grayscale data
    # (i.e. no state leaks between reads).
    a = read_image_properties(rgb_identical_channels)
    b = read_image_properties(rgb_identical_channels)
    assert a.file_hash == b.file_hash
    assert (a.gray == b.gray).all()
