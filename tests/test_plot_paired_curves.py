"""Tests for the figures behind the Stage 1 loss plot.

What the plot reports in its legend is the window mean of the paired
differences, so that is what is pinned: differences only over shared
epochs, and no mean at all when the window is incomplete.
"""
import pytest

import scripts.peft_sam.plot_paired_curves as cli


def test_differences_cover_only_epochs_both_runs_logged():
    reference = {1: 0.07, 2: 0.06, 3: 0.05}
    candidate = {2: 0.065, 3: 0.045, 4: 0.04}

    assert cli.differences(reference, candidate) == pytest.approx(
        {2: 0.005, 3: -0.005}
    )


def test_window_mean_averages_the_whole_window():
    values = {epoch: float(epoch) for epoch in range(1, 13)}

    assert cli.window_mean(values, (7, 12)) == pytest.approx(9.5)


def test_window_mean_refuses_an_incomplete_window():
    values = {epoch: 0.0 for epoch in range(1, 12)}

    assert cli.window_mean(values, (7, 12)) is None


def test_every_variant_has_its_own_colour():
    assert len(cli.SERIES) >= len(cli.VARIANTS)
