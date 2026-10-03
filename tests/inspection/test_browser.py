"""Tests for prediction, figures and the browser, with the model stubbed.

The stub replaces the segmenter only: everything after it - the
post-processing call, the physical-area filter, the matching and the
figure - is the real code, so that a browser which drifted from the
scoring path would fail here rather than show something never scored.
"""
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.evaluation.watershed import DEPLOYED_POSTPROCESSING
from materials_vision.inspection.browser import (ALL_MATERIALS,
                                                 PredictionBrowser,
                                                 image_choices)
from materials_vision.inspection.figures import comparison_figure
from materials_vision.inspection.model import predict

FRAME = (40, 40)

PIXEL_SIZE_UM = 2.0


def _labels():
    """Two 10 x 10 pores (400 um2 each) and one 2 x 2 fragment (16 um2)."""
    labels = np.zeros(FRAME, dtype=np.int32)
    labels[5:15, 5:15] = 1
    labels[20:30, 20:30] = 2
    labels[35:37, 35:37] = 3
    return labels


def _sample(index, material, labels):
    record = SampleRecord(
        index=index, image_id=f"{material}1_40_{index}",
        formulation=f"{material}1", material=material, microscope="M1",
        scale_bin="coarse", pixel_size_um=PIXEL_SIZE_UM, q_max_i=1.0,
        source_path=Path("image.tif"), mask_path=Path("mask.tif"),
        crop_bbox=(0, 0, FRAME[1], FRAME[0]),
        n_instances_expected=int(labels.max()),
    )
    return PreparedSample(
        record=record, image=np.full(FRAME, 100, dtype=np.uint8),
        labels=labels,
        border_instance=np.zeros(int(labels.max()), dtype=bool),
        n_cut_by_crop=0, n_dropped_below_min_area=0,
        n_dropped_disconnected=0,
    )


class StubSegmenter:
    """Hands back fixed instances, as the decoder and watershed would."""

    def __init__(self, labels):
        self._labels = labels
        self.n_initialized = 0

    def initialize(self, image):
        assert image.shape == (*FRAME, 3) and image.dtype == np.uint8
        self.n_initialized += 1

    def generate(self, output_mode, **kwargs):
        assert output_mode == "instance_segmentation"
        return self._labels.astype(np.float32)


class StubSource:
    """Two AS images and one K image, counting how often it is read."""

    def __init__(self):
        self._samples = [
            _sample(index, material, _labels())
            for index, material in enumerate(("AS", "AS", "K"))
        ]
        self.n_loaded = 0

    @property
    def records(self):
        return tuple(sample.record for sample in self._samples)

    def load(self, index):
        self.n_loaded += 1
        return self._samples[index]


def test_predict_applies_the_deployed_fragment_filter():
    sample = _sample(0, "AS", _labels())

    labels = predict(
        StubSegmenter(_labels()), sample, DEPLOYED_POSTPROCESSING
    )

    assert labels.max() == 2
    assert not labels[35:37, 35:37].any()


def test_comparison_figure_puts_prediction_left_annotation_right():
    figure = comparison_figure(_sample(0, "AS", _labels()), _labels())

    assert isinstance(figure, Figure)
    left, right = figure.axes
    assert left.get_title() == "prediction: 3 pores"
    assert right.get_title() == "annotation: 3 pores"


def test_image_choices_narrow_to_one_material():
    records = StubSource().records

    assert [index for _, index in image_choices(records)] == [0, 1, 2]
    assert [index for _, index in image_choices(records, "K")] == [2]
    assert image_choices(records, ALL_MATERIALS) == image_choices(records)


def test_browser_segments_each_image_once():
    source = StubSource()
    segmenter = StubSegmenter(_labels())
    browser = PredictionBrowser(source, segmenter)

    first = browser.prediction(1)
    again = browser.prediction(1)

    assert first is again
    assert (source.n_loaded, segmenter.n_initialized) == (1, 1)


def test_browser_figure_shows_what_the_filter_left():
    browser = PredictionBrowser(StubSource(), StubSegmenter(_labels()))

    left, right = browser.figure(0).axes

    assert left.get_title() == "prediction: 2 pores"
    assert right.get_title() == "annotation: 3 pores"


def test_widget_draws_the_first_image_on_creation():
    pytest.importorskip("ipywidgets")
    source = StubSource()
    browser = PredictionBrowser(source, StubSegmenter(_labels()))

    browser.widget()

    assert source.n_loaded == 1
