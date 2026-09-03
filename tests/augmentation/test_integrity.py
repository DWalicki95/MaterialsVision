"""Tests for the checks an augmented sample has to pass."""
import numpy as np
import pytest

from materials_vision.augmentation.integrity import (IntegrityError,
                                                     check_connectivity,
                                                     check_labels_preserved,
                                                     check_mask_untouched,
                                                     check_sample)

CONTEXT = "test sample"


def _labels(instances):
    """A frame holding one square instance per side length given."""
    width = sum(size + 1 for size in instances) + 1
    height = max(instances) + 2
    labels = np.zeros((height, width), dtype=np.int32)
    column = 1
    for index, size in enumerate(instances, start=1):
        labels[1:1 + size, column:column + size] = index
        column += size + 1
    return labels


def _image_like(labels):
    return np.full(labels.shape, 120, dtype=np.uint8)


def test_a_valid_sample_passes():
    labels = _labels([3, 4])

    check_sample(_image_like(labels), labels, context=CONTEXT)


def test_a_three_channel_image_is_refused():
    """The pipeline carries one working channel until the very end."""
    labels = _labels([3])
    image = np.stack([_image_like(labels)] * 3, axis=-1)

    with pytest.raises(IntegrityError, match="one channel"):
        check_sample(image, labels, context=CONTEXT)


def test_an_image_and_mask_of_different_sizes_are_refused():
    labels = _labels([3])
    image = np.full((labels.shape[0] + 1, labels.shape[1]), 7,
                    dtype=np.uint8)

    with pytest.raises(IntegrityError, match="different frames"):
        check_sample(image, labels, context=CONTEXT)


def test_a_float_label_image_is_refused():
    """A floating-point mask is the signature of an interpolated one."""
    labels = _labels([3, 4]).astype(np.float32)

    with pytest.raises(IntegrityError, match="must be integers"):
        check_sample(_image_like(labels), labels, context=CONTEXT)


def test_gaps_in_the_numbering_are_refused():
    labels = _labels([3, 4])
    labels[labels == 1] = 3

    with pytest.raises(IntegrityError, match="leave gaps"):
        check_sample(_image_like(labels), labels, context=CONTEXT)


def test_an_emptied_mask_is_refused():
    labels = _labels([3, 4])
    emptied = np.zeros_like(labels)

    with pytest.raises(IntegrityError, match="came out empty"):
        check_sample(_image_like(labels), emptied, context=CONTEXT)


def test_an_empty_mask_is_allowed_when_it_was_always_empty():
    """A frame the annotator left blank is not a failure."""
    labels = np.zeros((6, 6), dtype=np.int32)

    check_sample(
        _image_like(labels), labels, context=CONTEXT,
        expect_instances=False,
    )


def test_a_float_image_outside_the_intensity_range_is_refused():
    labels = _labels([3])
    image = _image_like(labels).astype(np.float32)
    image[0, 0] = 300.0

    with pytest.raises(IntegrityError, match="outside the"):
        check_sample(image, labels, context=CONTEXT)


def test_a_non_finite_image_is_refused():
    labels = _labels([3])
    image = _image_like(labels).astype(np.float32)
    image[0, 0] = np.nan

    with pytest.raises(IntegrityError, match="NaN or infinity"):
        check_sample(image, labels, context=CONTEXT)


def test_a_rearranged_mask_preserves_its_instance_areas():
    labels = _labels([3, 4])

    check_labels_preserved(labels, np.rot90(labels), context=CONTEXT)


def test_an_interpolated_mask_does_not():
    """The failure this check exists to catch.

    Resampling a label image with anything but nearest neighbour
    averages neighbouring ids, which both invents values nobody
    annotated and moves every instance's area.
    """
    labels = _labels([4, 6])
    blended = labels.copy()
    blended[1, 1] = 0

    with pytest.raises(IntegrityError, match="instance areas changed"):
        check_labels_preserved(labels, blended, context=CONTEXT)


def test_a_lost_instance_is_caught_as_a_changed_area():
    labels = _labels([3, 4])
    without_second = np.where(labels == 2, 0, labels)

    with pytest.raises(IntegrityError, match="instance areas changed"):
        check_labels_preserved(labels, without_second, context=CONTEXT)


def test_an_untouched_mask_passes_the_strict_check():
    labels = _labels([3, 4])

    check_mask_untouched(labels, labels.copy(), context=CONTEXT)


def test_a_shifted_mask_passes_areas_but_fails_the_strict_check():
    """Why brightness changes get the stronger of the two checks.

    Rolling the mask by one pixel leaves every area untouched, so the
    area check cannot see it. A transformation that only changes
    brightness has no business moving the mask at all, so it is held to
    bitwise equality instead.
    """
    labels = _labels([3, 4])
    shifted = np.roll(labels, 1, axis=1)

    check_labels_preserved(labels, shifted, context=CONTEXT)
    with pytest.raises(IntegrityError, match="the mask changed"):
        check_mask_untouched(labels, shifted, context=CONTEXT)


def test_a_connected_instance_passes():
    labels = _labels([3, 4])

    check_connectivity(labels, context=CONTEXT)


def test_an_instance_in_two_pieces_is_refused():
    """Two pieces under one id become two basins in the targets."""
    labels = np.zeros((6, 9), dtype=np.int32)
    labels[1:4, 1:3] = 1
    labels[1:4, 6:8] = 1

    with pytest.raises(IntegrityError, match="more than"):
        check_connectivity(labels, context=CONTEXT)


def test_connectivity_of_an_empty_frame_is_not_a_failure():
    check_connectivity(np.zeros((5, 5), dtype=np.int32), context=CONTEXT)


def test_the_context_names_the_culprit_in_the_message():
    """A failure has to say which transformation produced it."""
    labels = _labels([3, 4])

    with pytest.raises(IntegrityError, match="AS1_40 after F1"):
        check_mask_untouched(
            labels, np.rot90(labels).copy(),
            context="AS1_40 after F1_orientation",
        )
