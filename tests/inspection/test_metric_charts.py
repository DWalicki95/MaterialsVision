"""Tests for the metric bar charts and the table behind them."""
import pytest
from matplotlib.figure import Figure

from materials_vision.inspection.metric_charts import (METRICS, metric_figures,
                                                       metrics_table)
from materials_vision.inspection.scoring import scores_by_material


@pytest.fixture
def two_models(evaluate, pores, size_bins):
    """A model predicting the annotation, and one enlarging every pore."""
    def scores(side_px):
        return scores_by_material([
            evaluate(pores(10), pores(side_px), index=0, material="AS"),
            evaluate(pores(10), pores(side_px), index=1, material="K"),
        ], size_bins)
    return {"exact": scores(10), "enlarged": scores(12)}


def test_one_figure_per_section(two_models):
    figures = metric_figures(two_models)

    assert list(figures) == list(dict.fromkeys(m.section for m in METRICS))
    assert all(isinstance(figure, Figure) for figure in figures.values())


def test_every_metric_gets_a_visible_panel(two_models):
    figures = metric_figures(two_models)

    visible = sum(
        axis.get_visible()
        for figure in figures.values() for axis in figure.axes
    )
    assert visible == len(METRICS)


def test_undefined_value_is_marked_not_drawn_as_zero(two_models):
    material = metric_figures(two_models)["material"]

    texts = [text.get_text() for axis in material.axes
             for text in axis.texts]
    assert "n/a" in texts


def test_size_bias_prints_the_pairs_behind_each_value(two_models):
    size_bias = metric_figures(two_models)["size_bias"]

    texts = [text.get_text() for axis in size_bias.axes
             for text in axis.texts]
    # Every pore is 100 um2, so Q2 holds all four pairs of an image and
    # Q1 none.
    assert "+20.0\nn=4" in texts
    assert "n/a\nn=0" in texts


def test_table_holds_every_metric_model_and_subset(two_models):
    table = metrics_table(two_models)

    n_counted = sum(spec.count is not None for spec in METRICS)
    assert len(table) == len(METRICS) + n_counted
    assert set(table.columns) == {
        (subset, model)
        for subset in ("overall", "AS", "K")
        for model in ("exact", "enlarged")
    }
    row = table.xs("Median pore diameter", level="metric")
    assert row[("overall", "enlarged")].iloc[0] == pytest.approx(20.0)


def test_models_scored_on_different_pores_are_refused(
    two_models, evaluate, pores, size_bins
):
    fewer = scores_by_material([
        evaluate(pores(10), pores(10), index=0, material="AS"),
        evaluate(pores(10, corners=((5, 5),)), pores(10), index=1,
                 material="K"),
    ], size_bins)

    with pytest.raises(ValueError, match="same images"):
        metric_figures({"exact": two_models["exact"], "other": fewer})


def test_more_models_than_colours_are_refused(two_models):
    many = {f"model {i}": two_models["exact"] for i in range(4)}

    with pytest.raises(ValueError):
        metric_figures(many)
