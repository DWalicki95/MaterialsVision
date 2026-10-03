"""Tests for the global microstructure descriptors.

Porosity is an area fraction, so it must count every pore pixel in the
frame - including pores the per-pore morphology leaves out because the
border truncates them. The analyzer filters those pores by default,
and porosity used to be summed from the filtered list, which made it
depend on a setting that has nothing to do with it.
"""
import numpy as np
import pytest
import tifffile

from materials_vision.quantitative_analysis.quantitative_analysis import (
    GlobalMicrostructureDescriptors, PorousMaterialAnalyzer)


def _mask_with_border_pore() -> np.ndarray:
    """60x60 frame: one large pore on the border, one small inside."""
    mask = np.zeros((60, 60), dtype=np.uint16)
    mask[0:30, 0:30] = 1
    mask[40:50, 40:50] = 2
    return mask


def _analyzer_porosity(tmp_path, mask, reject_boundary_pores):
    mask_path = tmp_path / "AS1_40_1_masks.tif"
    tifffile.imwrite(mask_path, mask)
    analyzer = PorousMaterialAnalyzer(
        mask_path=str(mask_path),
        pixel_size=2.0,
        output_base_dir=tmp_path / f"out_{reject_boundary_pores}",
        generate_plots=False,
        reject_boundary_pores=reject_boundary_pores,
        plot_boundary_rejection=False,
    )
    return analyzer, analyzer.calculate_global_descriptors()["porosity"]


def test_porosity_counts_border_pores(tmp_path):
    mask = _mask_with_border_pore()
    analyzer, porosity = _analyzer_porosity(tmp_path, mask, True)

    assert len(analyzer.props) == 1
    assert porosity == pytest.approx((900 + 100) / 3600)


def test_porosity_does_not_depend_on_border_rejection(tmp_path):
    mask = _mask_with_border_pore()
    _, with_rejection = _analyzer_porosity(tmp_path, mask, True)
    _, without_rejection = _analyzer_porosity(tmp_path, mask, False)

    assert with_rejection == pytest.approx(without_rejection)


def test_porosity_is_independent_of_pixel_size():
    mask = _mask_with_border_pore()
    porosities = [
        GlobalMicrostructureDescriptors(mask, [], pixel_size=px)
        .calculate_porosity()
        for px in (0.5, 3.24)
    ]

    assert porosities[0] == pytest.approx(porosities[1])


def test_porosity_of_crop():
    mask = _mask_with_border_pore()
    descriptors = GlobalMicrostructureDescriptors(mask, [], pixel_size=1.0)

    assert descriptors.calculate_porosity(mask[0:30, 0:30]) == 1.0
    assert descriptors.calculate_porosity(mask[30:, 0:30]) == 0.0


def test_porosity_of_empty_mask_raises():
    descriptors = GlobalMicrostructureDescriptors(
        np.zeros((0, 0), dtype=np.uint16), [], pixel_size=1.0
    )

    with pytest.raises(ValueError):
        descriptors.calculate_porosity()


def test_local_porosity_variance_of_half_filled_frame():
    """Two full quadrants and two empty ones: porosities 1, 1, 0, 0."""
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[:50, :] = 1
    descriptors = GlobalMicrostructureDescriptors(mask, [], pixel_size=3.0)

    assert descriptors.calculate_local_porosity_variance() == (
        pytest.approx(0.25)
    )


def test_local_porosity_variance_of_uniform_frame_is_zero():
    mask = np.ones((40, 40), dtype=np.uint16)
    descriptors = GlobalMicrostructureDescriptors(mask, [], pixel_size=1.0)

    assert descriptors.calculate_local_porosity_variance() == 0.0
