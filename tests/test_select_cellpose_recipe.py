"""Tests for turning VALIDATION readings into the Cellpose recipe."""
import json

import pytest

import scripts.cellpose.select_cellpose_recipe as cli
from materials_vision.experiments.cellpose.split_evaluation import (
    CellposeSettings, calibration_grid)
from materials_vision.experiments.cellpose.split_training import saved_epochs
from scripts.cellpose.train_cellpose_split import run_name

N_VAL = 107

UNDER = CellposeSettings(niter=1000)

NITERS = (1000, 2000)


def _entry(name, setting, f1, n_images=N_VAL):
    return {"overall": {
        "f1": f1, "n_images": n_images,
        "label": f"{name}@{setting.label()}",
    }}


def _selection_payload(best=("lr1e-5", 101), best_f1=0.9, under=UNDER):
    payload = {
        name: _entry(name, under, 0.5 + 0.001 * i)
        for i, name in enumerate(sorted(cli.expected_snapshots()))
    }
    arm, epoch = best
    name = f"{run_name(arm)}/epoch-{epoch}"
    payload[name] = _entry(name, under, best_f1)
    return payload


def _grid_payload(stem, best, best_f1=0.95, niters=NITERS):
    payload = {}
    for setting in calibration_grid(niters):
        name = f"{stem}@{setting.label()}"
        payload[name] = _entry(stem, setting, 0.8)
    name = f"{stem}@{best.label()}"
    payload[name] = _entry(stem, best, best_f1)
    return payload


def test_the_snapshot_with_the_highest_f1_is_chosen():
    chosen = cli.select_snapshot(_selection_payload(), UNDER)["snapshot"]

    assert chosen["arm"] == "lr1e-5"
    assert chosen["learning_rate"] == 1e-5
    assert chosen["completed_epochs"] == 101
    assert chosen["val_f1"] == 0.9
    assert chosen["selected_under"] == "ft0.4_cp0_m15_n1000"
    assert chosen["selection_niter"] == 1000
    assert chosen["ranking"][0][1] == 0.9
    assert len(chosen["ranking"]) == 2 * len(saved_epochs())


def test_a_report_scored_under_other_steps_is_refused():
    payload = _selection_payload(under=CellposeSettings())

    with pytest.raises(cli.SelectionError, match="not scored under"):
        cli.select_snapshot(payload, UNDER)


def test_an_incomplete_table_is_refused():
    payload = _selection_payload()
    del payload[f"{run_name('lr1e-3')}/epoch-11"]

    with pytest.raises(cli.SelectionError, match="Missing"):
        cli.select_snapshot(payload, UNDER)


def test_readings_on_differing_populations_are_refused():
    payload = _selection_payload()
    name = f"{run_name('lr1e-3')}/epoch-11"
    payload[name] = _entry(name, UNDER, 0.4, n_images=8)

    with pytest.raises(cli.SelectionError, match="differing"):
        cli.select_snapshot(payload, UNDER)


def test_an_exact_tie_is_refused_rather_than_broken():
    payload = _selection_payload()
    name = f"{run_name('lr1e-3')}/epoch-151"
    payload[name] = _entry(name, UNDER, 0.9)

    with pytest.raises(cli.SelectionError, match="shared"):
        cli.select_snapshot(payload, UNDER)


def test_a_snapshot_is_not_chosen_under_several_settings():
    payload = _selection_payload()
    payload[f"{run_name('lr1e-3')}/epoch-11@ft0.2_cp0_m15"] = _entry(
        "x", UNDER, 0.3
    )

    with pytest.raises(cli.SelectionError, match="one setting"):
        cli.select_snapshot(payload, UNDER)


def test_thresholds_and_steps_are_calibrated_on_the_chosen_snapshot():
    recipe = cli.select_snapshot(_selection_payload(), UNDER)
    stem = f"{run_name('lr1e-5')}/epoch-101"
    best = CellposeSettings(
        flow_threshold=0.6, cellprob_threshold=-1.0, niter=2000
    )

    chosen = cli.select_postprocessing(
        _grid_payload(stem, best), recipe, NITERS
    )["postprocessing"]

    assert chosen["flow_threshold"] == 0.6
    assert chosen["cellprob_threshold"] == -1.0
    assert chosen["niter"] == 2000
    assert chosen["grid_niters"] == [1000, 2000]
    assert chosen["val_f1"] == 0.95
    assert chosen["val_f1_selected_under"] == 0.8
    assert len(chosen["ranking"]) == 40

    other = f"{run_name('lr1e-3')}/epoch-101"
    with pytest.raises(cli.SelectionError, match="grid settings on"):
        cli.select_postprocessing(
            _grid_payload(other, best), recipe, NITERS
        )


def test_a_grid_over_other_steps_is_refused():
    recipe = cli.select_snapshot(_selection_payload(), UNDER)
    stem = f"{run_name('lr1e-5')}/epoch-101"

    with pytest.raises(cli.SelectionError, match="grid settings on"):
        cli.select_postprocessing(
            _grid_payload(stem, CellposeSettings(), niters=(200,)),
            recipe, NITERS,
        )


def test_thresholds_wait_for_the_snapshot():
    with pytest.raises(cli.SelectionError, match="snapshot first"):
        cli.select_postprocessing({}, {}, NITERS)


def test_each_choice_is_recorded_once(tmp_path):
    readings = tmp_path / "selection.json"
    readings.write_text(json.dumps(_selection_payload()))
    recipe = tmp_path / "recipe.json"
    argv = [
        "snapshot", "--readings", str(readings), "--recipe", str(recipe),
        "--niter", "1000",
    ]

    assert cli.main(argv) == cli.EXIT_OK
    recorded = json.loads(recipe.read_text())
    assert recorded["snapshot"]["completed_epochs"] == 101
    assert recorded["snapshot"]["readings"] == str(readings)

    assert cli.main(argv) == cli.EXIT_FAILED
    assert json.loads(recipe.read_text()) == recorded


def test_the_second_round_is_recorded_apart_from_the_first():
    assert cli.DEFAULT_RECIPE.name == "recipe_v2.json"
