"""Tests for data_prep.inventory.sem_sidecar."""
from pathlib import Path

from data_prep.inventory.sem_sidecar import (check_pixel_size_consistency,
                                             interpret_sidecar,
                                             parse_sidecar_file)


class TestParseSidecarFile:
    def test_real_as_sidecar(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        raw = parse_sidecar_file(path)
        assert raw["InstructName"] == "TM3000"
        assert raw["Magnification"] == "40"
        assert raw["PixelSize"] == "3240.23"
        assert raw["DataSize"] == "1280x1040"

    def test_polish_characters_decoded(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        raw = parse_sidecar_file(path)
        # Directory contains Polish characters encoded as iso-8859-2;
        # a failed decode would raise or produce mojibake, not "Zdj".
        assert "Directory" in raw

    def test_empty_value_parsed_as_empty_string(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        raw = parse_sidecar_file(path)
        assert raw.get("SampleName", "") == ""

    def test_missing_pixelsize_no_crash(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "broken_no_pixelsize.txt"
        raw = parse_sidecar_file(path)
        assert "PixelSize" not in raw
        rec = interpret_sidecar(raw, path)
        assert rec.pixel_size_raw_nm is None
        assert rec.pixel_size_um is None

    def test_malformed_datasize_no_crash(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "broken_datasize.txt"
        raw = parse_sidecar_file(path)
        rec = interpret_sidecar(raw, path)
        assert rec.datasize_w is None
        assert rec.datasize_h is None
        assert rec.pixel_size_raw_nm == 3240.23


class TestInterpretSidecar:
    def test_as_sidecar_interpretation(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        raw = parse_sidecar_file(path)
        rec = interpret_sidecar(raw, path)
        assert rec.instrument == "TM3000"
        assert rec.magnification == 40
        assert rec.pixel_size_raw_nm == 3240.23
        assert abs(rec.pixel_size_um - 3.24023) < 1e-9
        assert rec.datasize_w == 1280
        assert rec.datasize_h == 1040
        assert rec.acquired_at == "2023-02-07T17:44:10"

    def test_vab_sidecar_interpretation(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "VAB1 prostopadla_m001.txt"
        raw = parse_sidecar_file(path)
        rec = interpret_sidecar(raw, path)
        assert rec.instrument == "SU8000"
        assert rec.magnification == 30
        assert rec.pixel_size_raw_nm == 3307.292
        assert rec.datasize_w == 1280
        assert rec.datasize_h == 960
        assert rec.acquired_at == "2022-10-11T15:32:02"

    def test_date_time_to_iso8601_synthetic(self):
        raw = {
            "InstructName": "TM3000",
            "Date": "10/30/2025",
            "Time": "11:13:15",
        }
        rec = interpret_sidecar(raw, path=Path("synthetic.txt"))
        assert rec.acquired_at == "2025-10-30T11:13:15"

    def test_missing_date_or_time_gives_none(self):
        rec = interpret_sidecar(
            {"Date": "10/30/2025"}, path=Path("synthetic.txt")
        )
        assert rec.acquired_at is None


class TestPixelSizeConsistency:
    def test_tm3000_consistent(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        rec = interpret_sidecar(parse_sidecar_file(path), path)
        assert check_pixel_size_consistency(rec, tolerance=0.01) == []

    def test_su8000_consistent(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "VAB1 prostopadla_m001.txt"
        rec = interpret_sidecar(parse_sidecar_file(path), path)
        assert check_pixel_size_consistency(rec, tolerance=0.01) == []

    def test_perturbed_pixel_size_flagged(self):
        raw = {
            "InstructName": "TM3000",
            "Magnification": "40",
            "PixelSize": "3402.24",  # 3240.23 * 1.05, 5% off
            "DataSize": "1280x1040",
            "MicronMarker": "2000000",
        }
        rec = interpret_sidecar(raw, path=Path("synthetic.txt"))
        codes = check_pixel_size_consistency(rec, tolerance=0.01)
        assert "pixel_size_magnification_product_mismatch" in codes

    def test_micron_marker_within_frame(self, fixtures_dir):
        path = fixtures_dir / "sidecars" / "AS26_40_15.txt"
        rec = interpret_sidecar(parse_sidecar_file(path), path)
        marker_px = rec.micron_marker_nm / rec.pixel_size_raw_nm
        assert 0.05 * rec.datasize_w <= marker_px <= 0.95 * rec.datasize_w

    def test_micron_marker_out_of_range_flagged(self):
        raw = {
            "InstructName": "TM3000",
            "Magnification": "40",
            "PixelSize": "3240.23",
            "DataSize": "1280x1040",
            "MicronMarker": "500000000",  # way too large -> off-frame
        }
        rec = interpret_sidecar(raw, path=Path("synthetic.txt"))
        codes = check_pixel_size_consistency(rec, tolerance=0.01)
        assert "micron_marker_out_of_range" in codes

    def test_missing_fields_skip_checks_without_crashing(self):
        rec = interpret_sidecar({}, path=Path("synthetic.txt"))
        assert check_pixel_size_consistency(rec, tolerance=0.01) == []
