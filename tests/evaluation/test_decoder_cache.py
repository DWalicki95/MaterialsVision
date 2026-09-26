"""Tests for the stored decoder output and the calibration sample.

The replay has to grow the same instances the live segmenter would, so
it is checked against the library's watershed called directly on the
same maps, under the settings the calibration will sweep. The sample has
to be fixed by the split alone and reproducible, so it is checked for
its composition and for returning the same images twice.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.calibration_sample import (
    allocate, select_calibration_sample)
from materials_vision.evaluation.decoder_cache import (MAP_NAMES, load_maps,
                                                       replay_segmenter,
                                                       save_maps)
from materials_vision.evaluation.inference import segment
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   WatershedParams)


def _maps(seed=0, shape=(48, 64)):
    """Two blob-shaped pores, the way the decoder describes them."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    centres = [(16, 18), (30, 44)]
    distance = np.min(
        [np.hypot(rows - r, cols - c) for r, c in centres], axis=0
    )
    foreground = np.clip(1.2 - distance / 14, 0, 1)
    noise = rng.normal(0, 0.02, shape)
    # Both distance maps are small deep inside a pore, which is where
    # the watershed takes its seeds.
    return {
        "foreground": (foreground + noise).astype(np.float32),
        "center_distances": np.clip(distance / 14, 0, 1).astype(np.float32),
        "boundary_distances": np.clip(distance / 10, 0, 1).astype(
            np.float32
        ),
    }


class TestReplay:
    @pytest.mark.parametrize("settings", [
        FROZEN_WATERSHED,
        WatershedParams(0.3, 0.4, foreground_threshold=0.4,
                        foreground_smoothing=0.0),
        WatershedParams(0.3, 0.4, foreground_threshold=0.6,
                        foreground_smoothing=2.0, distance_smoothing=1.0),
    ])
    def test_matches_the_library_watershed(self, settings):
        from micro_sam.instance_segmentation import _apply_smoothing
        from torch_em.util.segmentation import \
            watershed_from_center_and_boundary_distances

        maps = _maps()
        foreground = maps["foreground"]
        if settings.foreground_smoothing > 0:
            foreground = _apply_smoothing(
                foreground, settings.foreground_smoothing, None, None
            )
        expected = watershed_from_center_and_boundary_distances(
            center_distances=maps["center_distances"],
            boundary_distances=maps["boundary_distances"],
            foreground_map=foreground,
            center_distance_threshold=settings.center_distance_threshold,
            boundary_distance_threshold=(
                settings.boundary_distance_threshold
            ),
            foreground_threshold=settings.foreground_threshold,
            distance_smoothing=settings.distance_smoothing,
            min_size=settings.min_size,
        )
        replayed = segment(replay_segmenter(maps), settings)
        np.testing.assert_array_equal(replayed, expected.astype(np.int32))
        # Both pores found, or the comparison above proves nothing.
        assert len(np.unique(replayed[replayed > 0])) == 2

    def test_the_area_filter_runs_through_the_same_path(self):
        maps = _maps()
        plain = segment(replay_segmenter(maps), FROZEN_WATERSHED)
        areas_px = np.bincount(plain.ravel())[1:]
        # A threshold between the two pores' areas drops the smaller.
        cut_px = (areas_px.min() + areas_px.max()) / 2
        pixel_size_um = 2.0
        setting = WatershedParams(
            0.30, 0.40, min_instance_area_um2=cut_px * pixel_size_um ** 2
        )
        filtered = segment(
            replay_segmenter(maps), setting, pixel_size_um=pixel_size_um
        )
        assert np.unique(filtered[filtered > 0]).tolist() == [1]
        assert np.count_nonzero(filtered) == areas_px.max()

    def test_the_area_filter_needs_a_pixel_size(self):
        setting = WatershedParams(0.30, 0.40, min_instance_area_um2=10.0)
        with pytest.raises(ValueError):
            segment(replay_segmenter(_maps()), setting)

    def test_survives_a_round_trip_through_disk(self, tmp_path):
        maps = _maps(seed=3)
        path = tmp_path / "img.npz"
        save_maps(path, maps)
        loaded = load_maps(path)
        for name in MAP_NAMES:
            np.testing.assert_array_equal(loaded[name], maps[name])
            assert loaded[name].dtype == np.float32

    def test_incomplete_file_is_refused(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, foreground=np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(ValueError):
            load_maps(path)


def _record(index, material, formulation, scale_bin="coarse"):
    return SampleRecord(
        index=index, image_id=f"img_{index}", formulation=formulation,
        material=material, microscope="M1", scale_bin=scale_bin,
        pixel_size_um=1.0, q_max_i=1.0, source_path=Path("x"),
        mask_path=Path("y"), crop_bbox=(0, 0, 1, 1),
        n_instances_expected=1,
    )


def _records():
    records = []
    for index in range(60):
        formulation = "AS1" if index < 40 else "AS2"
        scale_bin = "fine" if index % 10 == 0 else "coarse"
        records.append(_record(index, "AS", formulation, scale_bin))
    records += [_record(60 + i, "K", "K1") for i in range(5)]
    records += [_record(65 + i, "VAB", "V1") for i in range(3)]
    records.append(_record(68, "K", "K1", scale_bin="outlier"))
    return records


class TestCalibrationSample:
    def test_small_materials_whole_close_ups_out(self):
        chosen = select_calibration_sample(_records(), n_subsampled=12)
        materials = [record.material for record in chosen]
        assert materials.count("K") == 5
        assert materials.count("VAB") == 3
        assert materials.count("AS") == 12
        assert all(record.scale_bin != "outlier" for record in chosen)

    def test_strata_keep_their_proportions(self):
        chosen = select_calibration_sample(_records(), n_subsampled=12)
        strata = [
            (record.formulation, record.scale_bin) for record in chosen
            if record.material == "AS"
        ]
        # 36 + 4 + 18 + 2 images of AS; 12 of 60 is a fifth of each.
        assert strata.count(("AS1", "coarse")) == 7
        assert strata.count(("AS2", "coarse")) == 4
        assert strata.count(("AS1", "fine")) + strata.count(
            ("AS2", "fine")
        ) == 1

    def test_is_reproducible_and_seed_dependent(self):
        first = select_calibration_sample(_records(), n_subsampled=12)
        again = select_calibration_sample(_records(), n_subsampled=12)
        other = select_calibration_sample(
            _records(), n_subsampled=12, seed=1
        )
        assert [r.index for r in first] == [r.index for r in again]
        assert [r.index for r in first] != [r.index for r in other]

    def test_keeps_the_split_order(self):
        chosen = select_calibration_sample(_records(), n_subsampled=12)
        indices = [record.index for record in chosen]
        assert indices == sorted(indices)

    def test_order_of_input_does_not_change_the_draw(self):
        records = _records()
        shuffled = [replace(r) for r in reversed(records)]
        chosen = {r.index for r in select_calibration_sample(
            records, n_subsampled=12
        )}
        chosen_shuffled = {r.index for r in select_calibration_sample(
            shuffled, n_subsampled=12
        )}
        assert chosen == chosen_shuffled


def test_a_thinly_spread_bin_keeps_its_share():
    # 390 coarse images in 14 formulations and 30 fine ones spread over
    # 10: the split's shape, where a single-step allocation of 45 gave
    # the fine bin nothing.
    records, index = [], 0
    for formulation in range(14):
        for _ in range(390 // 14 + (formulation < 390 % 14)):
            records.append(_record(index, "AS", f"AS{formulation}"))
            index += 1
    for formulation in range(10):
        for _ in range(3):
            records.append(
                _record(index, "AS", f"AS{formulation}", scale_bin="fine")
            )
            index += 1
    chosen = select_calibration_sample(records, n_subsampled=45)
    fine = [record for record in chosen if record.scale_bin == "fine"]
    # 30 of 420 is 3.2 of 45.
    assert len(fine) == 3
    assert len(chosen) == 45


class TestAllocate:
    def test_adds_up_exactly(self):
        shares = allocate({("a",): 7, ("b",): 7, ("c",): 7}, 10)
        assert sum(shares.values()) == 10

    def test_refuses_more_than_exists(self):
        with pytest.raises(ValueError):
            allocate({("a",): 2}, 3)
