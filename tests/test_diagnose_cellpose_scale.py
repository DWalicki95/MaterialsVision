"""Tests for choosing the variants of the Cellpose scale diagnosis.

The model, the data and the scoring are replaced; what these pin is
which predictions a run asks for, so that ``--skip-native`` really
skips the native pass and ``--diameters`` reaches Cellpose unchanged.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.cellpose.diagnose_cellpose_scale as cli


class FakeSource:
    """Two blank images with no record of interest."""

    def __len__(self) -> int:
        return 2

    def load(self, position: int) -> SimpleNamespace:
        return SimpleNamespace(record=None, labels=np.zeros((4, 4)),
                               image=np.zeros((4, 4)))


@pytest.fixture
def calls(monkeypatch):
    """Replace everything that needs weights or data; record the calls."""
    seen = {"flows": 0, "diameters": []}

    def fake_flows(model, image):
        seen["flows"] += 1
        return (None, None)

    def fake_diameter(model, image, diameter):
        seen["diameters"].append(diameter)
        return np.zeros((4, 4), dtype=np.int32)

    monkeypatch.setattr(cli, "build_source", lambda *a: FakeSource())
    monkeypatch.setattr(cli, "load_size_bins", lambda path: None)
    monkeypatch.setattr(cli, "load_snapshot", lambda path: None)
    monkeypatch.setattr(cli, "predict_flows", fake_flows)
    monkeypatch.setattr(cli, "niter_masks",
                        lambda m, f, n: np.zeros((4, 4), dtype=np.int32))
    monkeypatch.setattr(cli, "diameter_masks", fake_diameter)
    monkeypatch.setattr(cli, "evaluate_image", lambda *a, **k: None)
    monkeypatch.setattr(cli, "summary", lambda evaluations: {
        "f1": 0.0, "precision": 0.0, "recall": 0.0,
        "pore_count_error": 0.0, "size_bin_recall": [],
    })
    return seen


def test_skip_native_runs_only_the_requested_diameters(tmp_path, calls):
    out = tmp_path / "report.json"
    assert cli.main([
        "--checkpoint", "unused", "--skip-native",
        "--diameters", "20", "15", "10", "--out", str(out),
    ]) == cli.EXIT_OK
    assert calls["flows"] == 0
    assert calls["diameters"] == [20.0, 15.0, 10.0] * 2
    assert list(json.loads(out.read_text())["variants"]) == [
        "diameter20", "diameter15", "diameter10",
    ]


def test_default_run_keeps_the_september_variants(tmp_path, calls):
    out = tmp_path / "report.json"
    assert cli.main(["--checkpoint", "unused", "--out", str(out)]) == 0
    assert calls["flows"] == 2
    assert list(json.loads(out.read_text())["variants"]) == [
        "native_niter200", "native_niter500", "native_niter1000",
        "native_niter2000", "diameter60", "diameter90",
    ]


@pytest.mark.parametrize("argv", [
    ["--diameters", "0"],
    ["--diameters", "-15"],
    ["--skip-native", "--diameters"],
])
def test_rejects_a_run_that_cannot_measure(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(["--checkpoint", "unused", *argv])
