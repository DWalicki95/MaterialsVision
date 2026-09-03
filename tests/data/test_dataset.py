"""Tests for the torch dataset wrapper."""
import numpy as np
import pytest
import torch

from materials_vision.data.dataset import (LABEL_TRANSFORM_KWARGS,
                                           InstanceSegmentationDataset)


class _FakeRecord:
    def __init__(self, index):
        self.index = index
        self.image_id = f"img_{index}"


class _FakeSample:
    def __init__(self, image, labels, record):
        self.image = image
        self.labels = labels
        self.record = record


class _FakeSource:
    """Stands in for SampleSource; keeps the dataset tests torch-only."""

    def __init__(self, n_images=3, shape=(6, 8)):
        self._n = n_images
        self._shape = shape

    def __len__(self):
        return self._n

    def load(self, index):
        image = np.full(self._shape, 100 + index, dtype=np.uint8)
        labels = np.zeros(self._shape, dtype=np.int32)
        labels[1:3, 1:4] = 1
        labels[4:6, 5:8] = 2
        return _FakeSample(image, labels, _FakeRecord(index))


def _identity_label_transform(labels):
    return np.stack([labels, labels])


@pytest.fixture
def dataset():
    return InstanceSegmentationDataset(
        _FakeSource(), label_transform=_identity_label_transform,
        run_seed=99,
    )


def test_image_becomes_three_identical_channels(dataset):
    x, _ = dataset[0]

    assert x.shape == (3, 6, 8)
    assert x.dtype == torch.float32
    assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])


def test_values_are_not_normalized(dataset):
    """SAM normalizes with its own statistics inside the forward pass."""
    x, _ = dataset[0]

    assert float(x.max()) == 100.0


def test_target_comes_from_the_label_transform(dataset):
    _, y = dataset[0]

    assert y.shape == (2, 6, 8)
    assert y.dtype == torch.float32
    assert set(np.unique(y.numpy()).tolist()) == {0.0, 1.0, 2.0}


def test_length_follows_the_source(dataset):
    assert len(dataset) == 3


def test_empty_source_is_refused():
    with pytest.raises(ValueError, match="empty source"):
        InstanceSegmentationDataset(
            _FakeSource(n_images=0),
            label_transform=_identity_label_transform,
        )


def test_without_a_transform_the_sample_passes_through(dataset):
    x, _ = dataset[1]

    assert float(x.min()) == float(x.max()) == 101.0


def test_augmentation_receives_a_seed_and_its_output_is_used(dataset):
    seen = {}

    def transform(image, labels, *, record, seed):
        seen["seed"] = seed
        return image + 1, labels

    dataset._transform = transform
    x, _ = dataset[2]

    assert seen["seed"] == dataset.sample_seed(2)
    assert float(x.max()) == 103.0


def test_augmentation_receives_the_sample_record(dataset):
    """Scale augmentation is conditioned on the sample's calibration.

    How far an image may be magnified depends on how many micrometres
    one of its pixels covers, which only the record knows. Without it a
    policy would have to magnify every image by the same amount and
    would push the finely sampled ones past the resolution that was
    actually photographed.
    """
    seen = {}

    def transform(image, labels, *, record, seed):
        seen["image_id"] = record.image_id
        return image, labels

    dataset._transform = transform
    dataset[1]

    assert seen["image_id"] == "img_1"


def test_augmentation_seed_depends_on_index_and_epoch(dataset):
    first = dataset.sample_seed(0)
    second = dataset.sample_seed(1)
    dataset.set_epoch(1)
    later = dataset.sample_seed(0)

    assert first != second
    assert first != later
    assert 0 <= first < 2 ** 64


def test_augmentation_seed_is_reproducible(dataset):
    dataset.set_epoch(4)
    first = dataset.sample_seed(2)
    dataset.set_epoch(0)
    dataset.set_epoch(4)

    assert dataset.sample_seed(2) == first


def test_negative_epoch_is_refused(dataset):
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        dataset.set_epoch(-1)


def test_source_is_reachable_for_evaluation(dataset):
    assert len(dataset.source) == 3
    assert dataset.source.load(0).record.image_id == "img_0"


def test_label_transform_configuration_is_frozen():
    """Both values differ from the library defaults on purpose."""
    assert LABEL_TRANSFORM_KWARGS["apply_label"] is False
    assert LABEL_TRANSFORM_KWARGS["min_size"] == 0
    assert LABEL_TRANSFORM_KWARGS["instances"] is True


def test_instances_are_the_first_target_channel():
    """Pins a library contract that is easy to get backwards.

    ``PerObjectDistanceTransform`` prepends the foreground mask and
    then the instance labels, so the order ends up
    ``[instances, foreground, distances, boundary_distances]`` - the
    instance channel is first, not last. Reading the wrong end gives a
    plausible-looking normalized distance map instead of the labels.
    """
    from materials_vision.data.dataset import build_label_transform

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:5, 1:5] = 1
    labels[7:11, 7:11] = 2

    target = build_label_transform()(labels)

    assert target.shape == (4, 12, 12)
    assert np.array_equal(target[0].astype(np.int32), labels)
    assert set(np.unique(target[1]).tolist()) == {0.0, 1.0}


def test_non_2d_image_is_refused(dataset):
    class _RgbSource(_FakeSource):
        def load(self, index):
            sample = super().load(index)
            sample.image = np.stack([sample.image] * 3, axis=-1)
            return sample

    dataset._source = _RgbSource()

    with pytest.raises(ValueError, match="one working channel"):
        dataset[0]
