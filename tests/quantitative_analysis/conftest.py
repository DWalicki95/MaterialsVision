"""Shared fixtures for the quantitative analysis tests."""
import numpy as np
import pytest
import tifffile

from materials_vision.quantitative_analysis.quantitative_analysis import \
    PorousMaterialAnalyzer


@pytest.fixture
def make_analyzer(tmp_path):
    """Build a ``PorousMaterialAnalyzer`` over an in-memory mask.

    The mask goes through a TIFF file because that is the only way the
    analyzer accepts one, so the tests exercise the real loading path.
    """
    counter = iter(range(1_000_000))

    def _make(mask: np.ndarray, **kwargs) -> PorousMaterialAnalyzer:
        run = next(counter)
        mask_path = tmp_path / f"AS1_40_{run}_masks.tif"
        tifffile.imwrite(mask_path, mask)
        options = dict(
            pixel_size=2.0,
            output_base_dir=tmp_path / f"out_{run}",
            generate_plots=False,
            plot_boundary_rejection=False,
        )
        options.update(kwargs)
        return PorousMaterialAnalyzer(mask_path=str(mask_path), **options)

    return _make
