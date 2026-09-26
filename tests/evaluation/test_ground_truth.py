"""Tests for the quantities read from the annotation alone.

The wall thickness sets the upper end of the blur the post-processing
calibration may consider, so it is checked against walls of known width,
including the two cases that are easy to get wrong: pores annotated as
touching, which must read zero rather than be skipped, and an oblique
wall, which must be measured across rather than along an image axis.
"""
import numpy as np
import pytest

from materials_vision.evaluation.ground_truth import (
    areal_porosity, encoder_grid_period_px, interior_instance_areas_px2,
    nearest_pore, wall_crossings)


def _two_pores_with_wall(width_px, height=12, pore_px=6):
    labels = np.zeros((height, 2 * pore_px + width_px), dtype=np.int32)
    labels[:, :pore_px] = 1
    labels[:, pore_px + width_px:] = 2
    return labels


class TestWallCrossings:
    @pytest.mark.parametrize("width_px", [0, 1, 2, 3, 4])
    def test_vertical_wall_reads_its_width(self, width_px):
        crossings = wall_crossings(_two_pores_with_wall(width_px))
        assert crossings.per_wall_min_px.tolist() == [width_px]
        assert np.allclose(crossings.thickness_px, width_px)

    def test_touching_pores_are_counted_not_skipped(self):
        crossings = wall_crossings(_two_pores_with_wall(0))
        assert crossings.n_touching_walls == 1

    def test_oblique_wall_is_measured_across(self):
        size = 60
        rows, cols = np.mgrid[:size, :size]
        # A band of perpendicular width 4 px along the diagonal.
        offset = (cols - rows) / np.sqrt(2)
        labels = np.zeros((size, size), dtype=np.int32)
        labels[offset < -2] = 1
        labels[offset > 2] = 2
        crossings = wall_crossings(labels)
        # Along an image axis the band would read 4 * sqrt(2) ~ 5.7.
        assert np.median(crossings.thickness_px) == pytest.approx(4, abs=1)

    def test_one_minimum_per_facing_pair(self):
        labels = np.zeros((10, 30), dtype=np.int32)
        labels[:, :8] = 1
        labels[:, 10:18] = 2
        labels[:, 21:] = 3
        crossings = wall_crossings(labels)
        assert sorted(crossings.per_wall_min_px.tolist()) == [2, 3]

    def test_single_pore_has_no_walls(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1:3, 1:3] = 1
        crossings = wall_crossings(labels)
        assert crossings.thickness_px.size == 0
        assert crossings.n_touching_walls == 0


def test_nearest_pore_refuses_an_empty_frame():
    with pytest.raises(ValueError):
        nearest_pore(np.zeros((4, 4), dtype=np.int32))


def test_areal_porosity_counts_every_pore_pixel():
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[0, :] = 1
    labels[3, 0] = 2
    assert areal_porosity(labels) == pytest.approx(6 / 20)


def test_areal_porosity_refuses_an_empty_frame():
    with pytest.raises(ValueError):
        areal_porosity(np.zeros((0, 0), dtype=np.int32))


class TestInteriorAreas:
    def test_border_instances_are_left_out(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[0, 0:2] = 1
        labels[2:4, 2:5] = 2
        areas = interior_instance_areas_px2(
            labels, np.array([True, False])
        )
        assert areas.tolist() == [6.0]

    def test_flags_must_cover_the_labels(self):
        labels = np.zeros((3, 3), dtype=np.int32)
        labels[1, 1] = 1
        with pytest.raises(ValueError):
            interior_instance_areas_px2(labels, np.array([False, False]))


def test_grid_period_follows_the_longer_side():
    assert encoder_grid_period_px((960, 1280)) == pytest.approx(20.0)
    assert encoder_grid_period_px((1024, 768)) == pytest.approx(16.0)
