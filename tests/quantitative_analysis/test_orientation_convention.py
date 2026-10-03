"""Tests for the reference axis of the stored ellipse angle.

A pore that looks horizontal on the image has to come out as a
horizontal direction and land at 3 o'clock on the rose plot. The tests
start from a drawn mask and go through the same per-pore measurement
the analysis uses, so they fail if either end of the chain changes its
convention.
"""
import numpy as np
import pytest
from matplotlib.figure import Figure
from skimage.measure import regionprops

from materials_vision.quantitative_analysis.macroscopic_metrics import \
    plot_orientation_rose
from materials_vision.quantitative_analysis.quantitative_analysis import (
    PoreMorphologyMetrics, ellipse_angle_to_image_direction)

SIZE_PX = 200
SEMI_MAJOR_PX = 60
SEMI_MINOR_PX = 15
ROSE_BIN_DEG = 5.0


def stored_angle(direction_deg: float) -> float:
    """Draw one elongated pore and measure it as the analysis does.

    ``direction_deg`` is counter-clockwise from the horizontal as the
    image is displayed, row 0 at the top - how a person would describe
    the pore by looking at it.
    """
    rows, cols = np.mgrid[0:SIZE_PX, 0:SIZE_PX] - SIZE_PX / 2
    phi = np.deg2rad(direction_deg)
    # Rows grow downwards on screen, so rising to the right means the
    # row index falls as the column index grows.
    along = cols * np.cos(phi) - rows * np.sin(phi)
    across = cols * np.sin(phi) + rows * np.cos(phi)
    labels = (
        (along / SEMI_MAJOR_PX) ** 2 + (across / SEMI_MINOR_PX) ** 2 <= 1
    ).astype(np.int32)
    region = regionprops(labels)[0]
    metrics = PoreMorphologyMetrics(region, pixel_size=1.0)
    return metrics.calculate_ellipse_metrics()["ellipse_angle"]


def axis_gap_deg(a: float, b: float) -> float:
    """Distance between two axis directions, which repeat every 180."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def circle_gap_deg(a: float, b: float) -> float:
    """Distance between two positions on a full circle."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("direction_deg", [0.0, 30.0, 45.0, 90.0, 135.0])
def test_direction_matches_how_the_pore_looks(direction_deg):
    measured = float(
        ellipse_angle_to_image_direction(stored_angle(direction_deg))
    )

    assert axis_gap_deg(measured, direction_deg) < 1.0


def test_stored_angle_is_measured_from_the_vertical():
    assert axis_gap_deg(stored_angle(90.0), 0.0) < 1.0
    assert axis_gap_deg(stored_angle(0.0), 90.0) < 1.0


def test_direction_is_folded_into_half_a_turn_and_keeps_nan():
    stored = np.array([-90.0, -45.0, 0.0, 45.0, 90.0, np.nan])

    directions = ellipse_angle_to_image_direction(stored)

    assert np.all((directions[:-1] >= 0.0) & (directions[:-1] < 180.0))
    assert np.isnan(directions[-1])
    assert np.isnan(stored[-1]) and stored[0] == -90.0


@pytest.mark.parametrize("direction_deg", [0.0, 45.0, 90.0, 135.0])
def test_rose_draws_the_pore_where_it_points(direction_deg):
    axes = Figure().add_subplot(projection="polar")

    plot_orientation_rose(axes, np.full(10, stored_angle(direction_deg)))

    tallest = max(bar.get_height() for bar in axes.patches)
    centers_deg = [
        np.rad2deg(bar.get_x() + bar.get_width() / 2.0) % 360.0
        for bar in axes.patches if bar.get_height() == tallest
    ]
    assert len(centers_deg) == 2
    for opposite in (direction_deg, direction_deg + 180.0):
        assert min(
            circle_gap_deg(center, opposite) for center in centers_deg
        ) <= ROSE_BIN_DEG
