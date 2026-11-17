"""Tests for reading the cross-validation folds together."""
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.create_cv_folds as create_cli
import scripts.peft_sam.summarize_cv as cli
from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.aggregate import (aggregate, cross_sections,
                                                   evaluate_image)
from materials_vision.evaluation.size_bins import SizeBins

BINS = SizeBins(edges_um2=(38505.0, 98891.0, 243941.0),
                n_calibration_instances=25247)


def _record(formulation, material):
    return SampleRecord(
        index=0, image_id=f"{formulation}_0", formulation=formulation,
        material=material, microscope="M1", scale_bin="coarse",
        pixel_size_um=3.24023, q_max_i=1.3,
        source_path=Path("image.tif"), mask_path=Path("mask.tif"),
        crop_bbox=(0, 0, 128, 128), n_instances_expected=0,
    )


def _grid(n_pores):
    labels = np.zeros((128, 128), dtype=np.int32)
    for index in range(n_pores):
        row, col = 4 + 32 * (index // 4), 4 + 32 * (index % 4)
        labels[row:row + 20, col:col + 20] = index + 1
    return labels


def _merged(labels, first, second):
    """Join two pores of a grid into one prediction."""
    out = labels.copy()
    out[out == second] = first
    return out


@pytest.fixture
def evaluations():
    gt = _grid(8)
    return [
        evaluate_image(_record("AS1", "AS"), gt, gt.copy(), size_bins=BINS),
        evaluate_image(_record("AS2", "AS"), gt, _merged(gt, 1, 2),
                       size_bins=BINS),
        evaluate_image(_record("K1", "K"), gt, _grid(5), size_bins=BINS),
    ]


def test_pooling_sections_equals_scoring_them_in_one_pass(evaluations):
    whole = asdict(aggregate(evaluations))
    parts = [asdict(s) for s in cross_sections(evaluations, "formulation")]

    pooled = cli.pooled(parts)

    for field in ("f1", "precision", "recall", "pore_count_error",
                  "merges_per_100_gt", "splits_per_100_gt"):
        assert pooled[field] == pytest.approx(whole[field]), field
    for field in cli.COUNT_FIELDS:
        assert pooled[field] == whole[field], field


def test_pooling_nothing_is_refused():
    with pytest.raises(cli.SummaryError):
        cli.pooled([])


def test_a_bare_entry_is_taken_only_when_its_setting_is_known():
    payload = {"run/epoch-12": {"x": 1}, "other/epoch-12@frozen": {"x": 2}}

    assert cli.find_entry(payload, "run/epoch-12", "frozen", None) is None
    assert cli.find_entry(
        payload, "run/epoch-12", "frozen", "frozen"
    ) == {"x": 1}
    assert cli.find_entry(
        payload, "other/epoch-12", "frozen", None
    ) == {"x": 2}


def _section(label, n_images, n_gt, tp, fp, n_merges=0, n_splits=0):
    return {
        "label": label, "n_images": n_images, "n_gt": n_gt,
        "n_pred": tp + fp, "n_true_positives": tp,
        "n_false_positives": fp, "n_false_negatives": n_gt - tp,
        "n_merges": n_merges, "n_splits": n_splits,
    }


def test_a_single_held_out_formulation_takes_its_materials_figure():
    scores = cli.FoldScores(
        fold=1,
        overall=_section("overall", 4, 40, 30, 5),
        per_material=[
            _section("material=AS", 3, 30, 25, 3),
            _section("material=K", 1, 10, 5, 2),
        ],
        per_formulation=None,
        held_out={"AS": ["AS3", "AS19"], "K": ["K3"]},
    )

    rows = cli.formulation_rows(scores)

    assert [r["formulation"] for r in rows] == ["K3"]
    assert rows[0]["material"] == "K"
    assert rows[0]["f1"] == pytest.approx(10 / 17)


@pytest.fixture
def cv_dirs(tmp_path, frozen_split, monkeypatch):
    """Folds of the miniature split and a report for every one of them.

    Each report's per-formulation figures add up to its per-material
    ones and those to the overall ones, as they do in a real report.
    """
    split_csv, _ = frozen_split
    folds_dir, eval_dir = tmp_path / "cv5", tmp_path / "eval"
    assert create_cli.main([
        "--split", str(split_csv), "--out-dir", str(folds_dir),
        "--seed", "11",
    ]) == create_cli.EXIT_OK
    eval_dir.mkdir()
    final_report = tmp_path / "test_scores.json"
    monkeypatch.setattr(cli, "FINAL_MODEL_REPORTS", {12: final_report})

    frozen, deployed = "frozen", "min_instance_area_um2=200"
    for fold in range(1, 6):
        table = pd.read_csv(folds_dir / f"fold{fold}.csv")
        held = table[(table["split"] == "test") & table["used"]]
        per_formulation, per_material = [], []
        for material, rows in held.groupby("material"):
            parts = [
                _section(f"formulation={name}", len(images), 10 * len(images),
                         8 * len(images) - fold % 2, 1)
                for name, images in rows.groupby("formulation")
            ]
            per_formulation += parts
            per_material.append(
                {**cli.pooled(parts), "label": f"material={material}"}
            )
        entry = {
            "overall": {**cli.pooled(per_material), "label": "overall"},
            "per_material": per_material,
        }
        if fold == 1:
            final_report.write_text(json.dumps(
                {cli.final_model_entry(12): entry, "b0/epoch-3": {}}
            ))
            continue
        stem = f"{cli.run_name(fold)}/epoch-12"
        entry["per_formulation"] = per_formulation
        (eval_dir / f"fold{fold}.json").write_text(json.dumps({
            f"{stem}@{frozen}": entry, f"{stem}@{deployed}": entry,
        }))
    return folds_dir, eval_dir


def _main(cv_dirs, *extra):
    folds_dir, eval_dir = cv_dirs
    return cli.main([
        "--folds-dir", str(folds_dir), "--eval-dir", str(eval_dir),
        "--no-mlflow", *extra,
    ])


def test_the_summary_pools_every_fold(cv_dirs):
    assert _main(cv_dirs) == cli.EXIT_OK

    summary = json.loads(
        (cv_dirs[1] / "cv5_v1_summary_frozen.json").read_text()
    )
    assert summary["folds_included"] == [1, 2, 3, 4, 5]
    assert summary["pooled"]["n_gt"] == sum(
        row["n_gt"] for row in summary["per_fold"]
    )
    # Fold one of the miniature split holds out one formulation of each
    # material, so all of them resolve from its material figures.
    formulations = {row["formulation"] for row in summary["per_formulation"]}
    assert formulations == {
        name for name in pd.read_csv(cv_dirs[0] / "fold1.csv")["formulation"]
        if name != "VAB11"
    }
    assert (cv_dirs[1] / "cv5_v1_summary_frozen.png").exists()
    assert (cv_dirs[1] / "cv5_v1_summary_frozen.md").exists()


def test_fold_one_is_left_out_only_where_test_was_not_scored(cv_dirs):
    assert _main(cv_dirs, "--postprocessing", "deployed") == cli.EXIT_OK

    summary = json.loads(
        (cv_dirs[1] / "cv5_v1_summary_deployed.json").read_text()
    )
    assert summary["folds_included"] == [2, 3, 4, 5]
    assert summary["folds_missing"] == [1]


def test_a_fold_not_yet_scored_stops_the_summary(cv_dirs):
    (cv_dirs[1] / "fold4.json").unlink()

    assert _main(cv_dirs) == cli.EXIT_FAILED
    assert _main(cv_dirs, "--allow-missing") == cli.EXIT_OK


def test_a_partial_score_is_not_taken_for_the_fold(cv_dirs):
    path = cv_dirs[1] / "fold3.json"
    payload = json.loads(path.read_text())
    for entry in payload.values():
        entry["overall"]["n_images"] -= 1
    path.write_text(json.dumps(payload))

    assert _main(cv_dirs) == cli.EXIT_FAILED


def test_nan_figures_print_as_a_dash():
    assert cli._num(math.nan) == "-"
    assert cli._pct(0.051) == "+5.1%"


def _entry(folds_dir, fold, missed=0):
    """A report entry for a fold, consistent across its cross-sections.

    ``missed`` removes true positives from every formulation, so that
    entries scored under different settings can be told apart.
    """
    table = pd.read_csv(folds_dir / f"fold{fold}.csv")
    held = table[(table["split"] == "test") & table["used"]]
    per_formulation, per_material = [], []
    for material, rows in held.groupby("material"):
        parts = [
            _section(f"formulation={name}", len(images), 10 * len(images),
                     8 * len(images) - missed, 1)
            for name, images in rows.groupby("formulation")
        ]
        per_formulation += parts
        per_material.append(
            {**cli.pooled(parts), "label": f"material={material}"}
        )
    return {
        "overall": {**cli.pooled(per_material), "label": "overall"},
        "per_material": per_material,
        "per_formulation": per_formulation,
    }


@pytest.fixture
def em_dirs(tmp_path, cv_dirs):
    """Every fold of another base model at 24 epochs, fold one included,
    each scored under the frozen setting and its own calibration."""
    import scripts.peft_sam.calibrate_watershed as calibrate
    from materials_vision.evaluation import WatershedParams

    folds_dir = cv_dirs[0]
    eval_dir = tmp_path / "eval_em"
    calibration_dir = tmp_path / "calibration_em"
    eval_dir.mkdir()
    figures = type("Figures", (), {
        "f1": 0.9, "boundary_f1": {}, "splits_per_100_gt": 0.0,
        "merges_per_100_gt": 0.0, "pore_count_error": 0.0,
    })()
    for fold in range(1, 6):
        chosen = WatershedParams(
            center_distance_threshold=0.30 + 0.05 * (fold % 2),
            boundary_distance_threshold=0.5,
        )
        calibrate.write_report(
            calibration_dir / f"fold{fold}.json", tmp_path / "e.pt",
            folds_dir / f"fold{fold}.csv", [0], 0.004, [(chosen, figures)],
        )
        stem = cli.run_name(fold, "vit_l_em_organelles", 24) + "/epoch-24"
        (eval_dir / f"fold{fold}.json").write_text(json.dumps({
            f"{stem}@frozen": _entry(folds_dir, fold, missed=1),
            f"{stem}@{chosen.label()}": _entry(folds_dir, fold, missed=0),
        }))
    return folds_dir, eval_dir, calibration_dir


def _em_main(em_dirs, *extra):
    folds_dir, eval_dir, calibration_dir = em_dirs
    return cli.main([
        "--folds-dir", str(folds_dir), "--eval-dir", str(eval_dir),
        "--base-model", "vit_l_em_organelles", "--epochs", "24",
        "--calibration-dir", str(calibration_dir), "--no-mlflow", *extra,
    ])


def _summary(directory, name):
    return json.loads((directory / name).read_text())


def test_another_base_model_reads_all_five_folds_into_its_own_files(em_dirs):
    assert _em_main(em_dirs) == cli.EXIT_OK

    summary = _summary(
        em_dirs[1], "cv5_v1_vit_l_em_organelles_e24_summary_frozen.json"
    )
    assert summary["folds_included"] == [1, 2, 3, 4, 5]
    assert summary["model"]["base_model"] == "vit_l_em_organelles"
    assert summary["model"]["epochs"] == 24


def test_each_fold_is_read_under_its_own_calibration(em_dirs):
    assert _em_main(em_dirs) == cli.EXIT_OK
    assert _em_main(em_dirs, "--postprocessing", "calibrated") == cli.EXIT_OK

    frozen = _summary(
        em_dirs[1], "cv5_v1_vit_l_em_organelles_e24_summary_frozen.json"
    )
    calibrated = _summary(
        em_dirs[1], "cv5_v1_vit_l_em_organelles_e24_summary_calibrated.json"
    )
    assert calibrated["folds_included"] == [1, 2, 3, 4, 5]
    # The calibrated entries miss nothing; the frozen ones miss pores.
    assert calibrated["pooled"]["f1"] > frozen["pooled"]["f1"]
    for row in calibrated["per_fold"]:
        assert row["n_true_positives"] == 8 * row["n_images"]


def test_a_fold_not_yet_calibrated_is_missing(em_dirs):
    (em_dirs[2] / "fold4.json").unlink()

    assert _em_main(em_dirs, "--postprocessing", "calibrated") == (
        cli.EXIT_FAILED
    )
    assert _em_main(
        em_dirs, "--postprocessing", "calibrated", "--allow-missing"
    ) == cli.EXIT_OK


def test_a_broken_calibration_report_stops_the_summary(em_dirs):
    (em_dirs[2] / "fold2.json").write_text("{}")

    assert _em_main(em_dirs, "--postprocessing", "calibrated") == (
        cli.EXIT_FAILED
    )


def test_calibrated_needs_its_reports_and_a_trained_model(em_dirs):
    folds_dir, eval_dir, calibration_dir = em_dirs
    base = ["--folds-dir", str(folds_dir), "--eval-dir", str(eval_dir),
            "--no-mlflow", "--postprocessing", "calibrated"]

    assert cli.main(base) == cli.EXIT_FAILED
    assert cli.main(
        base + ["--calibration-dir", str(calibration_dir),
                "--pretrained", "vit_l_lm"]
    ) == cli.EXIT_FAILED


def test_the_mean_without_the_validation_fold_is_reported(em_dirs):
    assert _em_main(em_dirs) == cli.EXIT_OK

    summary = _summary(
        em_dirs[1], "cv5_v1_vit_l_em_organelles_e24_summary_frozen.json"
    )
    across = summary["across_folds"]
    others = [row["f1"] for row in summary["per_fold"]
              if row["fold"] != cli.VALIDATION_FOLD]
    assert across["validation_fold"] == 2
    assert across["f1_mean_without_validation_fold"] == pytest.approx(
        sum(others) / len(others)
    )


def test_a_final_model_scored_again_is_read_from_the_eval_dir(
    tmp_path, cv_dirs, monkeypatch
):
    folds_dir, _ = cv_dirs
    eval_dir = tmp_path / "eval_lm24"
    eval_dir.mkdir()
    once = tmp_path / "final_24ep_test_scores.json"
    once.write_text(json.dumps(
        {cli.final_model_entry(24): _entry(folds_dir, 1, missed=1)}
    ))
    monkeypatch.setattr(cli, "FINAL_MODEL_REPORTS", {24: once})
    deployed = cli.POSTPROCESSING_CONFIGS["deployed"].label()
    for fold in range(2, 6):
        stem = cli.run_name(fold, epochs=24) + "/epoch-24"
        (eval_dir / f"fold{fold}.json").write_text(json.dumps({
            f"{stem}@frozen": _entry(folds_dir, fold),
            f"{stem}@{deployed}": _entry(folds_dir, fold),
        }))
    argv = ["--folds-dir", str(folds_dir), "--eval-dir", str(eval_dir),
            "--epochs", "24", "--no-mlflow"]

    # Only the single TEST score: fold one is there under frozen alone,
    # and its absence elsewhere is expected rather than an error.
    assert cli.main(argv) == cli.EXIT_OK
    assert cli.main(argv + ["--postprocessing", "deployed"]) == cli.EXIT_OK
    assert _summary(
        eval_dir, "cv5_v1_e24_summary_deployed.json"
    )["folds_missing"] == [1]
    summary = _summary(eval_dir, "cv5_v1_e24_summary_frozen.json")
    assert summary["per_fold"][0]["n_true_positives"] < (
        8 * summary["per_fold"][0]["n_images"]
    )

    # Scored again into the eval directory: that report is read instead.
    (eval_dir / "fold1.json").write_text(json.dumps({
        f"{cli.final_model_entry(24)}@frozen": _entry(folds_dir, 1),
    }))
    assert cli.main(argv) == cli.EXIT_OK
    summary = _summary(eval_dir, "cv5_v1_e24_summary_frozen.json")
    assert summary["per_fold"][0]["n_true_positives"] == (
        8 * summary["per_fold"][0]["n_images"]
    )


def test_a_zero_shot_generalist_is_read_from_its_own_entries(
    tmp_path, cv_dirs
):
    folds_dir, _ = cv_dirs
    eval_dir = tmp_path / "zero_shot"
    eval_dir.mkdir()
    for fold in range(1, 6):
        (eval_dir / f"fold{fold}.json").write_text(json.dumps({
            f"pretrained/{name}@{label}": _entry(folds_dir, fold)
            for name in ("vit_l_lm", "vit_l_em_organelles")
            for label in ("frozen", "boundary_distance_threshold=0.5,"
                          "center_distance_threshold=0.5")
        }))
    argv = ["--folds-dir", str(folds_dir), "--eval-dir", str(eval_dir),
            "--no-mlflow"]

    for name in ("vit_l_lm", "vit_l_em_organelles"):
        for setting in ("frozen", "library_default"):
            assert cli.main(
                argv + ["--pretrained", name, "--postprocessing", setting]
            ) == cli.EXIT_OK
            summary = _summary(
                eval_dir, f"cv5_v1_pretrained_{name}_summary_{setting}.json"
            )
            assert summary["folds_included"] == [1, 2, 3, 4, 5]
    assert cli.main(argv + ["--pretrained", "vit_l"]) == cli.EXIT_FAILED


@pytest.mark.parametrize("fold, extra", [
    (3, []),
    (3, ["--epochs", "24"]),
    (1, ["--base-model", "vit_l_em_organelles", "--epochs", "24"]),
    (4, ["--base-model", "vit_l", "--decoder-lr", "3e-4", "--epochs", "24"]),
])
def test_the_summary_looks_for_the_name_the_evaluator_files_under(
    tmp_path, fold, extra
):
    """From the snapshot a fold run fixes in advance to the report key."""
    import scripts.peft_sam.evaluate_checkpoint as evaluate
    import scripts.peft_sam.train_cv_fold as fold_cli
    from scripts.peft_sam.train_final_model import scored_snapshot

    fold_args = fold_cli.parse_args(
        ["--fold", str(fold), "--save-root", str(tmp_path), *extra]
    )
    snapshot = scored_snapshot(
        tmp_path, fold_args.epochs, fold_cli.fold_run_name(fold_args)
    )
    evaluator_args = evaluate.parse_args(["--checkpoint", str(snapshot)])
    filed_under = evaluate.result_stem(
        f"{snapshot.parent.name}/{snapshot.stem}", evaluator_args
    )
    summary_args = cli.parse_args(["--eval-dir", str(tmp_path), *extra])

    assert cli.fold_source(fold, summary_args).stem == filed_under


def test_the_final_models_entry_is_the_name_of_its_snapshot():
    from scripts.peft_sam.train_final_model import scored_snapshot

    for epochs, root in cli.FINAL_MODEL_SAVE_ROOTS.items():
        snapshot = scored_snapshot(root, epochs)
        assert cli.final_model_entry(epochs) == (
            f"{snapshot.parent.name}/{snapshot.stem}"
        )
