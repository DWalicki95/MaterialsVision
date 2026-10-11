"""Tests for the Miles-Lantuejoul correction of the border-filter bias.

Rejecting pores that touch the border keeps a large pore at fewer
positions than a small one, so the measured pores under-represent large
ones. The weight has to be exactly the inverse of that chance under the
filter the analyzer really applies - which is what the exhaustive test
below checks - and the corrected statistics have to reach the reports
next to the uncorrected ones.
"""
import numpy as np
import pandas as pd
import pytest
from skimage.measure import regionprops

from materials_vision.quantitative_analysis.macroscopic_metrics import (
    calculate_all_metrics, calculate_size_metrics, export_results_to_excel)
from materials_vision.quantitative_analysis.quantitative_analysis import (
    EDGE_WEIGHT_KEY, PoreMorphologyMetrics, PorousMaterialAnalyzer)

FRAME = (20, 24)
TOLERANCE = 2


def _accepted_positions(height, width):
    """Run the real filter over every placement of a rectangular pore."""
    accepted = []
    for top in range(FRAME[0] - height + 1):
        for left in range(FRAME[1] - width + 1):
            mask = np.zeros(FRAME, dtype=np.uint16)
            mask[top:top + height, left:left + width] = 1
            kept = PorousMaterialAnalyzer._filter_boundary_pores(
                regionprops(mask), FRAME, TOLERANCE
            )
            accepted.extend(kept)
    return accepted


@pytest.mark.parametrize("height, width", [(3, 5), (7, 2), (1, 1)])
def test_weight_is_inverse_of_the_filter_acceptance(height, width):
    """Accepted placements times the weight must cover the whole frame."""
    accepted = _accepted_positions(height, width)
    weight = PorousMaterialAnalyzer._edge_correction_weight(
        accepted[0].bbox, FRAME, TOLERANCE
    )

    assert len(accepted) * weight == pytest.approx(FRAME[0] * FRAME[1])


def test_larger_pore_gets_larger_weight():
    small = PorousMaterialAnalyzer._edge_correction_weight(
        (5, 5, 8, 8), FRAME, TOLERANCE
    )
    large = PorousMaterialAnalyzer._edge_correction_weight(
        (5, 5, 14, 14), FRAME, TOLERANCE
    )

    assert large > small > 1.0


def test_weight_of_pore_that_cannot_pass_the_filter_raises():
    with pytest.raises(ValueError):
        PorousMaterialAnalyzer._edge_correction_weight(
            (0, 0, 20, 5), FRAME, TOLERANCE
        )


def _two_size_mask() -> np.ndarray:
    """Interior pores: four 4x4 squares and one 20x20 square."""
    mask = np.zeros((60, 60), dtype=np.uint16)
    for label, (row, col) in enumerate(
        [(5, 5), (5, 50), (50, 5), (50, 50)], start=1
    ):
        mask[row:row + 4, col:col + 4] = label
    mask[20:40, 20:40] = 5
    return mask


def test_analyzer_attaches_weights_to_every_measured_pore(make_analyzer):
    analyzer = make_analyzer(_two_size_mask())
    pores = analyzer.calculate_morphology_metrics()

    weights = {pore["pore_id"]: pore[EDGE_WEIGHT_KEY] for pore in pores}
    assert set(weights) == {1, 2, 3, 4, 5}
    assert weights[5] > weights[1]


def test_weights_are_one_without_border_rejection(make_analyzer):
    analyzer = make_analyzer(_two_size_mask(), reject_boundary_pores=False)

    assert all(
        pore[EDGE_WEIGHT_KEY] == 1.0
        for pore in analyzer.calculate_morphology_metrics()
    )


def test_aggregated_morphology_reports_raw_and_corrected(make_analyzer):
    analyzer = make_analyzer(_two_size_mask())
    diameter = analyzer.aggregate_morphology_metrics()["equivalent_diameter"]

    assert diameter["mean_corrected"] > diameter["mean"]
    assert EDGE_WEIGHT_KEY not in analyzer.morphology_aggregated


def test_old_results_without_weights_aggregate_as_before():
    pores = [{"pore_id": 1, "area": 2.0}, {"pore_id": 2, "area": 4.0}]
    stats = PoreMorphologyMetrics.aggregate_morphology_results(pores)

    assert stats["area"]["mean"] == 3.0
    assert "mean_corrected" not in stats["area"]


def test_global_descriptors_include_corrected_anisotropy(make_analyzer):
    analyzer = make_analyzer(_two_size_mask())
    descriptors = analyzer.calculate_global_descriptors()

    assert 0.0 <= descriptors["anisotropy_corrected"] <= 1.0


def test_report_contains_raw_and_corrected_rows(make_analyzer):
    analyzer = make_analyzer(_two_size_mask())
    analyzer.analyze_all(generate_excel_report=False)
    report = pd.read_excel(analyzer.generate_report(), "Analysis_Results")

    metrics = set(report["Metric"])
    assert "equivalent_diameter (median)" in metrics
    assert "equivalent_diameter (median, corrected)" in metrics
    assert "Anisotropy (corrected)" in metrics
    pores = pd.read_excel(
        analyzer.output_dir["reports"]
        / f"{analyzer.base_filename}_analysis_report.xlsx",
        "Individual_Pores",
    )
    assert EDGE_WEIGHT_KEY in pores.columns


def _pores_frame(diameters, weights=None):
    data = {
        "equivalent_diameter": diameters,
        "aspect_ratio": [1.5] * len(diameters),
        "circularity": [0.8] * len(diameters),
        "ellipse_angle": [10.0] * len(diameters),
        "coordination_number": [6] * len(diameters),
    }
    if weights is not None:
        data[EDGE_WEIGHT_KEY] = weights
    return pd.DataFrame(data)


def test_macroscopic_raw_size_metrics_are_unchanged():
    diameters = [10.0, 20.0, 30.0, 40.0, 200.0]
    metrics = calculate_size_metrics(_pores_frame(diameters, [1.0] * 5))

    assert metrics["size_d10"] == pytest.approx(np.percentile(diameters, 10))
    assert metrics["size_median"] == pytest.approx(np.median(diameters))
    assert metrics["size_d10_corrected"] == pytest.approx(
        metrics["size_d10"]
    )
    assert metrics["size_skewness_corrected"] == pytest.approx(
        metrics["size_skewness"]
    )


def test_macroscopic_corrected_size_follows_weights():
    metrics = calculate_size_metrics(
        _pores_frame([10.0, 20.0, 30.0, 40.0, 200.0], [1, 1, 1, 1, 50])
    )

    assert metrics["size_median_corrected"] > metrics["size_median"]


def test_macroscopic_metrics_from_old_report_have_nan_corrected():
    metrics = calculate_all_metrics(_pores_frame([10.0, 20.0, 30.0]))

    corrected = {k: v for k, v in metrics.items() if k.endswith("_corrected")}
    assert corrected
    assert all(np.isnan(value) for value in corrected.values())
    assert not np.isnan(metrics["size_median"])


def test_export_keeps_corrected_columns_next_to_raw_ones(tmp_path):
    metrics = calculate_all_metrics(_pores_frame([10.0, 20.0], [1.0, 2.0]))
    metrics.update(material_name="AS1", n_pores=2)
    path = export_results_to_excel([metrics], tmp_path / "metrics.xlsx")

    columns = list(pd.read_excel(path).columns)
    index = columns.index("size_median")
    assert columns[index + 1] == "size_median_corrected"
