"""Tests for neighbour-based metrics and the fractal dimension.

A pore near the border keeps its real neighbours even when those touch
the border and are left out of the per-pore measurements; otherwise its
nearest-neighbour distance is overstated and its coordination number
understated. The fractal dimension is taken from the pore boundaries over
the whole frame, so a filled pore phase no longer pins it near 2.
"""
import numpy as np
import pytest
import tifffile
from scipy.spatial import Delaunay
from skimage.measure import regionprops

from materials_vision.quantitative_analysis.quantitative_analysis import (
    SpatialRelationMetrics, TopologyConnectivityAnalysis)
from scripts.quantitative_analysis import \
    batch_quantitative_analysis_random_pore_removal as removal


def _interior_pore_beside_border_pore() -> np.ndarray:
    """Interior pores 1 and 2 far apart; border pore 3 right next to 1."""
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[20:26, 10:16] = 1
    mask[70:76, 80:86] = 2
    mask[20:26, 0:6] = 3
    return mask


def test_nearest_neighbour_may_be_a_border_pore(make_analyzer):
    analyzer = make_analyzer(
        _interior_pore_beside_border_pore(), pixel_size=1.0
    )
    stats = analyzer.calculate_spatial_metrics()

    assert {p.label for p in analyzer.props} == {1, 2}
    assert stats["min"] == pytest.approx(10.0)


def test_neighbour_distance_without_candidates_is_nan():
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[10:15, 10:15] = 1
    props = regionprops(mask)
    stats = SpatialRelationMetrics(
        props, pixel_size=1.0, generate_plots=False
    ).calculate_spatial_metrics()

    assert np.isnan(stats["mean"])


def test_coincident_centroids_of_distinct_pores_are_neighbours():
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[10:30, 10:30] = 1
    mask[15:25, 15:25] = 2
    props = regionprops(mask)
    distances = SpatialRelationMetrics(
        props, pixel_size=1.0, generate_plots=False
    ).calculate_nearest_neighbor_centroid_distance()

    assert distances == pytest.approx([0.0, 0.0], abs=1e-9)


def _ring_of_border_pores() -> np.ndarray:
    """One interior pore surrounded by six pores touching the border."""
    mask = np.zeros((60, 60), dtype=np.uint16)
    mask[25:35, 25:35] = 1
    for label, (row, col) in enumerate(
        [(0, 0), (0, 25), (0, 52), (52, 0), (52, 30), (25, 52)], start=2
    ):
        mask[row:row + 8, col:col + 8] = label
    return mask


def _delaunay_neighbour_count(mask: np.ndarray, label: int) -> int:
    """Neighbours of one pore in the Delaunay graph, Voronoi's dual."""
    props = regionprops(mask)
    centroids = np.array([p.centroid for p in props])
    index = [p.label for p in props].index(label)
    indptr, indices = Delaunay(centroids).vertex_neighbor_vertices
    return len(indices[indptr[index]:indptr[index + 1]])


def test_coordination_counts_border_neighbours(make_analyzer):
    mask = _ring_of_border_pores()
    analyzer = make_analyzer(mask)
    analyzer.analyze_all(generate_excel_report=False)
    cn = analyzer.topology_metrics["coordination_number_individual"]

    assert set(cn) == {1}
    assert cn[1] == _delaunay_neighbour_count(mask, 1)
    assert cn[1] >= 5
    border = [p for p in analyzer.morphology_results if p["pore_id"] != 1]
    assert border == []


def test_coordination_without_border_pores_has_too_few_points():
    mask = _ring_of_border_pores()
    interior = [p for p in regionprops(mask) if p.label == 1]
    topology = TopologyConnectivityAnalysis(mask, interior, pixel_size=1.0)

    assert topology.calculate_coordination_number() == (None, None)


def test_fractal_dimension_of_a_smooth_outline_is_one():
    mask = np.zeros((512, 512), dtype=np.uint16)
    mask[50:460, 40:470] = 1
    topology = TopologyConnectivityAnalysis(mask, [], pixel_size=1.0)
    dimension, r_squared = topology.calculate_fractal_dimension()

    assert dimension == pytest.approx(1.0, abs=0.05)
    assert r_squared > 0.99


def test_fractal_dimension_uses_the_whole_frame():
    """Pores only outside the top-left square used to give NaN."""
    mask = np.zeros((300, 700), dtype=np.uint16)
    rng = np.random.default_rng(0)
    for label in range(1, 40):
        row = rng.integers(10, 280)
        col = rng.integers(400, 680)
        mask[row:row + 12, col:col + 12] = label
    topology = TopologyConnectivityAnalysis(mask, [], pixel_size=1.0)
    dimension, _ = topology.calculate_fractal_dimension()

    assert np.isfinite(dimension)


def test_fractal_dimension_of_dense_foam_is_not_pinned_at_two():
    """88% of the frame is pore, yet the walls are a line network."""
    mask = np.zeros((512, 512), dtype=np.uint16)
    label = 0
    for row in range(0, 512, 32):
        for col in range(0, 512, 32):
            label += 1
            mask[row + 2:row + 32, col + 2:col + 32] = label
    topology = TopologyConnectivityAnalysis(mask, [], pixel_size=1.0)
    dimension, _ = topology.calculate_fractal_dimension()

    assert np.count_nonzero(mask) / mask.size > 0.85
    assert dimension < 1.8


def test_fractal_dimension_of_empty_mask_is_nan():
    mask = np.zeros((128, 128), dtype=np.uint16)
    topology = TopologyConnectivityAnalysis(mask, [], pixel_size=1.0)

    assert all(np.isnan(topology.calculate_fractal_dimension()))


def test_random_removal_takes_pores_out_of_the_mask(tmp_path):
    mask = _ring_of_border_pores()
    mask_path = tmp_path / "AS1_40_1_masks.tif"
    tifffile.imwrite(mask_path, mask)
    analyzer = removal.PorousMaterialAnalyzerRandomRemoval(
        mask_path=str(mask_path),
        pixel_size=1.0,
        output_base_dir=tmp_path / "out",
        generate_plots=False,
        plot_boundary_rejection=False,
        pores_to_keep=0,
        random_seed=0,
    )
    loaded_mask = analyzer.mask
    analyzer._filter_random_pores()

    assert analyzer.props == []
    assert not np.isin(analyzer.mask, [1]).any()
    assert len(analyzer.props_all) == 6
    assert np.array_equal(loaded_mask, mask)
