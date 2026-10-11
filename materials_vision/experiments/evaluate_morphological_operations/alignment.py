"""Bringing a classical prediction onto the frame the annotation is on.

The classical methods ran on the source image as stored, before the
microscope's information panel is cropped away. The annotation is
evaluated after that crop, so the prediction is cropped with the
identical box and the identical fragment rule: an instance the cut
leaves smaller than ``A_MIN_FRAGMENT_PX2`` is dropped on both sides
alike, rather than surviving as a sliver on one side only.

A prediction on a different pixel grid from its source image is
refused, not resized. Resizing a label image moves outlines and
changes the areas the metrics are computed from, so the score would
measure the resampling as much as the segmentation.
"""
import logging

import numpy as np
from PIL import Image

from materials_vision.data.instances import apply_content_crop
from materials_vision.data.samples import SampleRecord
from materials_vision.training import A_MIN_FRAGMENT_PX2

from .data_loading import PredictionLoadError

logger = logging.getLogger(__name__)


def source_shape(record: SampleRecord) -> tuple[int, int]:
    """The ``(height, width)`` of the stored source image.

    Parameters
    ----------
    record : SampleRecord

    Returns
    -------
    tuple of int
    """
    with Image.open(record.source_path) as handle:
        width, height = handle.size
    return height, width


def prediction_on_content_frame(
    pred_labels: np.ndarray, record: SampleRecord
) -> np.ndarray:
    """Crop a prediction made on the source image to the content region.

    Parameters
    ----------
    pred_labels : np.ndarray
        Instance labels on the source image's grid.
    record : SampleRecord

    Returns
    -------
    np.ndarray
        Labels on the cropped frame, numbered ``1..n``. A new array.

    Raises
    ------
    PredictionLoadError
        If the prediction is not on the source image's grid.
    """
    expected = source_shape(record)
    if pred_labels.shape != expected:
        raise PredictionLoadError(
            f"{record.image_id}: prediction is {pred_labels.shape}, the "
            f"source image {expected}; a label image is not resized"
        )
    # The crop takes an image alongside the labels and cuts both with
    # one box; only the labels are wanted here.
    placeholder = np.zeros(expected, dtype=np.uint8)
    cropped = apply_content_crop(
        placeholder, pred_labels, record.crop_bbox,
        min_fragment_area_px2=A_MIN_FRAGMENT_PX2,
    )
    return cropped.labels
