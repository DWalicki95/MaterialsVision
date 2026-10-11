"""Tests for setting the Cellpose folds beside the PEFT-SAM folds."""
import json

import pytest

import scripts.cellpose.compare_cellpose_cv as cli
import scripts.cellpose.evaluate_cellpose as evaluate_cli
from materials_vision.experiments.cellpose.split_evaluation import \
    CellposeSettings
from materials_vision.experiments.cellpose.split_training import snapshot_path
from scripts.cellpose.run_cellpose_fold import run_name
from scripts.peft_sam.summarize_cv import FoldScores, SummaryError

RECIPE = {
    "snapshot": {"arm": "lr1e-5", "learning_rate": 1e-5,
                 "completed_epochs": 101},
    "postprocessing": {"flow_threshold": 0.6, "cellprob_threshold": -1.0,
                       "min_size_px": 15, "niter": 2000},
}


def _section(label, tp, fp, fn, n_images=10):
    return {
        "label": label, "n_images": n_images, "n_gt": tp + fn,
        "n_pred": tp + fp, "n_true_positives": tp,
        "n_false_positives": fp, "n_false_negatives": fn,
        "n_merges": 1, "n_splits": 1,
    }


def _scores(fold, tp, fp, fn, material="AS"):
    return FoldScores(
        fold=fold,
        overall=_section("overall", tp, fp, fn),
        per_material=[_section(f"material={material}", tp, fp, fn)],
        per_formulation=None,
        held_out={material: [f"{material}{fold}"]},
    )


def test_pooled_difference_comes_from_summed_counts():
    cellpose = [_scores(1, 90, 10, 10), _scores(2, 900, 300, 100)]
    peft_sam = [_scores(1, 80, 20, 20), _scores(2, 950, 200, 50)]

    result = cli.paired(cellpose, peft_sam)

    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)  # noqa: E731
    assert result["pooled"]["delta_f1"] == pytest.approx(
        f1(990, 310, 110) - f1(1030, 220, 70)
    )
    assert [row["fold"] for row in result["per_fold"]] == [1, 2]
    assert result["delta_across_folds"]["n_cellpose_ahead"] == 1
    assert result["pooled_by_material"]["AS"]["delta_f1"] == pytest.approx(
        result["pooled"]["delta_f1"]
    )


def test_each_recipes_own_f1_is_reported_before_the_difference():
    cellpose = [_scores(1, 90, 10, 10), _scores(2, 80, 20, 20)]
    peft_sam = [_scores(1, 70, 30, 30), _scores(2, 60, 40, 40)]

    result = cli.paired(cellpose, peft_sam)
    report = cli.format_markdown({
        "cellpose_setting": "ft0.4_cp0_m15_n2000", "paired": result,
        "missing": {"cellpose": [], "peft_sam": []},
    })

    across = result["f1_across_folds"]
    assert across["cellpose"]["mean"] == pytest.approx(0.85)
    assert across["cellpose"]["min"] == pytest.approx(0.8)
    assert across["peft_sam"]["max"] == pytest.approx(0.7)
    assert across["peft_sam"]["sd"] == pytest.approx(0.0707, abs=1e-4)
    assert report.index("Instance F1 of each recipe") < report.index(
        "Per-fold difference"
    )
    assert "| mean | 0.8500 | 0.6500 |" in report
    assert "**0.8500**" in report


def test_folds_scored_on_different_pores_are_refused():
    with pytest.raises(SummaryError, match="not the same images"):
        cli.paired([_scores(1, 90, 10, 10)], [_scores(1, 90, 10, 11)])


def test_only_folds_scored_for_both_are_paired():
    result = cli.paired(
        [_scores(1, 90, 10, 10), _scores(3, 90, 10, 10)],
        [_scores(1, 80, 20, 20), _scores(2, 80, 20, 20)],
    )

    assert result["folds_included"] == [1]
    with pytest.raises(SummaryError, match="No fold"):
        cli.paired([_scores(3, 1, 1, 1)], [_scores(2, 1, 1, 1)])


def test_fold_reports_are_found_under_the_names_evaluation_writes(
    tmp_path
):
    """The key a fold's evaluation writes is the key this reads."""
    setting = cli.cellpose_settings(RECIPE)
    eval_dir = tmp_path / "eval"
    folds_dir = tmp_path / "folds"
    eval_dir.mkdir()
    folds_dir.mkdir()
    for fold in (1, 2):
        snapshot = snapshot_path(tmp_path / "ck", run_name(fold), 101)
        name = f"{evaluate_cli.snapshot_name(snapshot)}@{setting.label()}"
        entry = _scores(fold, 90, 10, 10)
        (eval_dir / f"fold{fold}.json").write_text(json.dumps({
            name: {"overall": entry.overall,
                   "per_material": entry.per_material},
        }))
        (folds_dir / f"fold{fold}_metadata.json").write_text(json.dumps({
            "counts": {"n_test_images": 10},
            "held_out": entry.held_out,
        }))
    for fold in (3, 4, 5):
        (folds_dir / f"fold{fold}_metadata.json").write_text(json.dumps({
            "counts": {"n_test_images": 10}, "held_out": {},
        }))

    found, missing = cli.load_cellpose_folds(
        RECIPE, setting, eval_dir, folds_dir
    )
    filtered, _ = cli.load_cellpose_folds(
        RECIPE, setting.with_area_filter(), eval_dir, folds_dir
    )

    assert [scores.fold for scores in found] == [1, 2]
    assert missing == [3, 4, 5]
    assert filtered == []


def test_the_recipe_setting_is_read_back_whole():
    assert cli.cellpose_settings(RECIPE) == CellposeSettings(
        flow_threshold=0.6, cellprob_threshold=-1.0, min_size_px=15,
        niter=2000,
    )
