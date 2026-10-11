"""Tests for the demo data builder, with the model replaced by a stub.

Everything but the forward pass runs for real: record selection,
timing, the quantitative analysis of prediction and annotation, the
asset encoding the viewer decodes, and the JSON the browser parses.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scripts.build_demo_data as demo
from materials_vision.data.samples import PreparedSample, SampleRecord


def _labels(seed: int) -> np.ndarray:
    """Grid of square pores, some cut by the border."""
    labels = np.zeros((96, 128), dtype=np.int32)
    label = 0
    for row in range(-6, 96, 20):
        for col in range(-6, 128, 20):
            label += 1
            labels[max(row, 0):row + 16 + seed, max(col, 0):col + 16] = label
    return labels


def _sample(index: int, formulation: str) -> PreparedSample:
    labels = _labels(index % 2)
    record = SampleRecord(
        index=index, image_id=f"{formulation}_40_{index}",
        formulation=formulation, material=formulation[:2],
        microscope="M1", scale_bin="coarse", pixel_size_um=3.24023,
        q_max_i=1.0, source_path=Path(f"{formulation}_40_{index}.tif"),
        mask_path=Path("unused.tif"), crop_bbox=(0, 0, 128, 96),
        n_instances_expected=int(labels.max()),
    )
    image = np.where(labels > 0, 40, 200).astype(np.uint8)
    return PreparedSample(
        record=record, image=image, labels=labels,
        border_instance=np.zeros(labels.max(), dtype=bool),
        n_cut_by_crop=0, n_dropped_below_min_area=0,
        n_dropped_disconnected=0,
    )


class _FakeSource:
    def __init__(self):
        self.samples = [
            _sample(0, "AS19"), _sample(1, "AS19"), _sample(2, "K3"),
            _sample(3, "AS3"),
        ]

    @property
    def records(self):
        return tuple(s.record for s in self.samples)

    def load(self, index):
        return self.samples[index]


@pytest.fixture
def demo_output(tmp_path, monkeypatch):
    calls = []

    def fake_predict(segmenter, sample):
        calls.append(sample.record.image_id)
        return sample.labels.copy()

    monkeypatch.setattr(demo, "build_source", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(demo, "load_segmenter", lambda *a, **k: object())
    monkeypatch.setattr(demo, "predict", fake_predict)
    monkeypatch.setattr(sys, "argv", [
        "build_demo_data.py", "--checkpoint", "epoch-12.pt",
        "--split", "split.csv", "--manifest", "manifest.csv",
        "--formulations", "AS19", "AS3", "--output", str(tmp_path / "out"),
        "--device", "cpu",
    ])
    demo.main()
    return tmp_path / "out", calls


def test_data_is_strict_json_for_the_browser(demo_output):
    out, _ = demo_output
    text = (out / "data.json").read_text()

    data = json.loads(text, parse_constant=pytest.fail)
    assert data["run"]["device"] == "CPU"
    assert data["run"]["subset"] == "test"


def test_only_requested_formulations_in_requested_order(demo_output):
    out, calls = demo_output
    data = json.loads((out / "data.json").read_text())

    assert [s["name"] for s in data["samples"]] == ["AS19", "AS3"]
    assert [len(s["images"]) for s in data["samples"]] == [2, 1]
    assert calls[0] == calls[1] == "AS19_40_0", "warm-up, then timed run"
    assert "K3_40_2" not in calls


def test_every_image_carries_times_metrics_and_expert(demo_output):
    out, _ = demo_output
    data = json.loads((out / "data.json").read_text())
    image = data["samples"][0]["images"][0]

    assert image["t_infer_s"] >= 0 and image["t_analysis_s"] > 0
    assert image["mag"] == 40
    assert data["samples"][0]["instrument"] == "TM3000"
    assert image["n_pores"] > 0 and image["n_excluded"] > 0
    assert image["expert"]["n_pores"] == image["n_pores"]
    assert all("w" in pore and "dir" in pore for pore in image["pores"])
    assert "size_mean_corrected" in data["samples"][0]["macro"]


def test_perfect_prediction_agrees_fully(demo_output):
    out, _ = demo_output
    data = json.loads((out / "data.json").read_text())

    for sample in data["samples"]:
        assert sample["agreement"]["f1"] == 1.0
        assert sample["agreement"]["fp"] == sample["agreement"]["fn"] == 0


def test_label_png_decodes_to_the_predicted_labels(demo_output):
    out, _ = demo_output
    image = json.loads((out / "data.json").read_text())["samples"][0][
        "images"][0]
    packed = np.array(Image.open(out / "lbl" / f"{image['id']}.png"))
    decoded = packed[..., 0].astype(int) | (packed[..., 1].astype(int) << 8)

    assert decoded.shape == (image["h"], image["w"])
    labels_in_png = set(np.unique(decoded)) - {0}
    measured = {pore["l"] for pore in image["pores"]}
    assert measured <= labels_in_png
    assert (out / "img" / f"{image['id']}.jpg").exists()


def test_unknown_formulation_is_rejected():
    with pytest.raises(ValueError, match="VAB99"):
        demo.select_records(_FakeSource().records, ["AS19", "VAB99"])


def test_labels_beyond_16_bit_are_rejected(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "lbl").mkdir()
    labels = np.full((4, 4), 70_000, dtype=np.int32)

    with pytest.raises(ValueError):
        demo.save_assets(
            np.zeros((4, 4), np.uint8), labels, tmp_path, "too_many"
        )


@pytest.mark.parametrize(
    "microscope, pixel_size_um, expected",
    [("M1", 2.59219, 50), ("M2", 2.480469, 40),
     ("M1", 1.0, None), ("unknown", 3.24023, None)],
)
def test_magnification_comes_from_the_calibration(
    microscope, pixel_size_um, expected
):
    assert demo.magnification_from_calibration(
        microscope, pixel_size_um
    ) == expected
