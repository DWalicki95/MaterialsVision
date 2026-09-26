"""Tests for the physical-area filter applied after the watershed.

The threshold is set in square micrometres and has to become a
different number of pixels on images of different resolution; the
tests pin that conversion, the boundary case, and that the caller's
label image is left untouched.
"""
import numpy as np
import pytest

from materials_vision.evaluation.inference import drop_small_instances
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   MIN_INSTANCE_AREA_UM2,
                                                   WatershedParams)
from materials_vision.tracking import postprocessing_id


def _labels():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0, 0] = 1              # 1 px
    labels[2:4, 2:4] = 3          # 4 px
    labels[6:10, 6:10] = 5        # 16 px
    return labels


def test_threshold_is_converted_with_the_pixel_size():
    # 8 um2 at 2 um per pixel is 2 px: only the 1-px instance goes.
    kept = drop_small_instances(_labels(), 8.0, 2.0)
    assert sorted(np.bincount(kept.ravel())[1:].tolist()) == [4, 16]
    # The same area at 1 um per pixel is 8 px: the 4-px one goes too.
    kept = drop_small_instances(_labels(), 8.0, 1.0)
    assert np.bincount(kept.ravel())[1:].tolist() == [16]


def test_an_instance_exactly_at_the_threshold_is_kept():
    kept = drop_small_instances(_labels(), 4.0, 1.0)
    assert sorted(np.bincount(kept.ravel())[1:].tolist()) == [4, 16]


def test_survivors_are_renumbered_in_order_and_input_untouched():
    labels = _labels()
    before = labels.copy()
    kept = drop_small_instances(labels, 2.0, 1.0)
    assert np.unique(kept).tolist() == [0, 1, 2]
    assert kept[2, 2] == 1 and kept[6, 6] == 2
    np.testing.assert_array_equal(labels, before)


def test_non_positive_pixel_size_is_refused():
    with pytest.raises(ValueError):
        drop_small_instances(_labels(), 8.0, 0.0)


def test_the_filter_gets_its_own_name_and_old_names_stay():
    filtered = WatershedParams(
        0.30, 0.40, min_instance_area_um2=MIN_INSTANCE_AREA_UM2
    )
    assert postprocessing_id(FROZEN_WATERSHED) == (
        "c0.3_b0.4_f0.5_fs1_d1.6_m0"
    )
    assert postprocessing_id(filtered) == (
        f"c0.3_b0.4_f0.5_fs1_d1.6_m0_a{MIN_INSTANCE_AREA_UM2:g}"
    )
