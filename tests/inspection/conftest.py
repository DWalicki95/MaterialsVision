"""Synthetic images measured by the real evaluation, for inspection tests.

Every evaluation here comes out of ``evaluate_image`` itself, so that a
figure or a relative drift is tested against the numbers the evaluation
actually produces rather than against a stand-in built to match them.
"""
from pathlib import Path

import numpy as np
import pytest

from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.aggregate import evaluate_image
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.size_bins import SizeBins

FRAME = (60, 60)

PIXEL_SIZE_UM = 1.0

# Four pores inside the frame, none touching an edge.
CORNERS = ((5, 5), (5, 35), (35, 5), (35, 35))

SIZE_BINS = SizeBins(edges_um2=(50.0, 120.0, 200.0),
                     n_calibration_instances=1)


def square_pores(sides_px, corners=CORNERS):
    """Label image of square pores, ids from 1.

    ``sides_px`` is one side for every pore, or one per corner.
    """
    if np.isscalar(sides_px):
        sides_px = [sides_px] * len(corners)
    labels = np.zeros(FRAME, dtype=np.int32)
    for instance_id, ((row, col), side) in enumerate(
        zip(corners, sides_px), start=1
    ):
        labels[row:row + side, col:col + side] = instance_id
    return labels


def record(index=0, material="AS", scale_bin="coarse",
           pixel_size_um=PIXEL_SIZE_UM):
    return SampleRecord(
        index=index, image_id=f"{material}1_40_{index}",
        formulation=f"{material}1", material=material, microscope="M1",
        scale_bin=scale_bin, pixel_size_um=pixel_size_um, q_max_i=1.0,
        source_path=Path("image.tif"), mask_path=Path("mask.tif"),
        crop_bbox=(0, 0, FRAME[1], FRAME[0]), n_instances_expected=4,
    )


@pytest.fixture
def pores():
    """``square_pores``, handed to tests as a fixture rather than an
    import, because a second module named ``conftest`` would collide
    with the one at the root of the test suite."""
    return square_pores


@pytest.fixture
def size_bins():
    """The size classes every synthetic image is scored with."""
    return SIZE_BINS


@pytest.fixture
def evaluate():
    """Measure a prediction against an annotation with the real code."""
    def measure(gt, pred, bins=SIZE_BINS, **record_fields):
        return evaluate_image(
            record(**record_fields), gt, pred, size_bins=bins,
            boundary_scales=(DECISION_SCALE,),
        )
    return measure
