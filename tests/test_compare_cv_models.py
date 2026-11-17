"""Tests for setting base models beside each other on the same folds.

The fold reports are written under the names ``train_cv_fold.py`` gives
each model's folds and read through the summary's own reader, so a
change to either naming breaks these tests rather than a comparison.
"""
import json

import pandas as pd
import pytest

import scripts.create_cv_folds as create_cli
import scripts.peft_sam.calibrate_watershed as calibrate
import scripts.peft_sam.compare_cv_models as cli
from materials_vision.evaluation import WatershedParams
from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from scripts.peft_sam.summarize_cv import pooled
from scripts.peft_sam.train_cv_fold import run_name

EPOCHS = 24

CALIBRATED_SETTING = WatershedParams(
    center_distance_threshold=0.35, boundary_distance_threshold=0.5,
)


def _section(label, n_images, n_gt, tp, fp=1):
    return {
        "label": label, "n_images": n_images, "n_gt": n_gt,
        "n_pred": tp + fp, "n_true_positives": tp,
        "n_false_positives": fp, "n_false_negatives": n_gt - tp,
        "n_merges": 0, "n_splits": 0,
    }


def _entry(folds_dir, fold, missed, n_gt_per_image=10):
    """A fold's report entry; ``missed`` pores lost per formulation."""
    table = pd.read_csv(folds_dir / f"fold{fold}.csv")
    held = table[(table["split"] == "test") & table["used"]]
    per_formulation, per_material = [], []
    for material, rows in held.groupby("material"):
        parts = [
            _section(
                f"formulation={name}", len(images),
                n_gt_per_image * len(images),
                8 * len(images) - missed,
            )
            for name, images in rows.groupby("formulation")
        ]
        per_formulation += parts
        per_material.append(
            {**pooled(parts), "label": f"material={material}"}
        )
    overall = {**pooled(per_material), "label": "overall"}
    overall["size_bin_recall"] = [
        {"label": "Q1", "n_gt": 10, "n_matched": 10 - missed},
    ]
    return {
        "overall": overall,
        "per_material": per_material,
        "per_formulation": per_formulation,
    }


@pytest.fixture
def folds_dir(tmp_path, frozen_split):
    split_csv, _ = frozen_split
    out_dir = tmp_path / "cv5"
    assert create_cli.main([
        "--split", str(split_csv), "--out-dir", str(out_dir),
        "--seed", "11",
    ]) == create_cli.EXIT_OK
    return out_dir


@pytest.fixture
def roots(tmp_path):
    return tmp_path / "eval", tmp_path / "calibration"


def _write_model(roots, folds_dir, base_model, missed, decoder_lr=None,
                 folds=range(1, 6)):
    """Score a model's folds under frozen and under its calibration."""
    eval_root, calibration_root = roots
    figures = type("Figures", (), {
        "f1": 0.9, "boundary_f1": {}, "splits_per_100_gt": 0.0,
        "merges_per_100_gt": 0.0, "pore_count_error": 0.0,
    })()
    names = {} if decoder_lr is None else {"decoder_learning_rate": decoder_lr}
    for fold in folds:
        calibrate.write_report(
            calibration_root / base_model / f"fold{fold}.json",
            eval_root / "e.pt", folds_dir / f"fold{fold}.csv", [0], 0.004,
            [(CALIBRATED_SETTING, figures)],
        )
        stem = run_name(fold, base_model, EPOCHS, **names) + "/epoch-24"
        missed_here = missed[fold - 1] if isinstance(missed, list) else missed
        directory = eval_root / base_model
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"fold{fold}.json").write_text(json.dumps({
            f"{stem}@frozen": _entry(folds_dir, fold, missed_here + 1),
            f"{stem}@{CALIBRATED_SETTING.label()}": _entry(
                folds_dir, fold, missed_here
            ),
        }))


def _argv(roots, folds_dir, *extra):
    eval_root, calibration_root = roots
    return [
        "--eval-root", str(eval_root),
        "--calibration-root", str(calibration_root),
        "--folds-dir", str(folds_dir), "--epochs", str(EPOCHS), *extra,
    ]


def _compare(roots, folds_dir, *extra):
    args = cli.parse_args(_argv(roots, folds_dir, *extra))
    assert cli.refusal(args) is None
    return cli.compare(args)


@pytest.fixture
def lm_reference(roots, folds_dir, monkeypatch):
    """The reference at 24 epochs, its fold one scored again."""
    import scripts.peft_sam.summarize_cv as summary_cli

    monkeypatch.setattr(
        summary_cli, "FINAL_MODEL_REPORTS",
        {EPOCHS: roots[0] / "no_single_test_report.json"},
    )
    eval_root, calibration_root = roots
    _write_model(roots, folds_dir, "vit_l_lm", 2, folds=range(2, 6))
    calibrate.write_report(
        calibration_root / "vit_l_lm" / "fold1.json", eval_root / "e.pt",
        folds_dir / "fold1.csv", [0], 0.004,
        [(CALIBRATED_SETTING, type("F", (), {
            "f1": 0.9, "boundary_f1": {}, "splits_per_100_gt": 0.0,
            "merges_per_100_gt": 0.0, "pore_count_error": 0.0,
        })())],
    )
    stem = summary_cli.final_model_entry(EPOCHS)
    (eval_root / "vit_l_lm" / "fold1.json").write_text(json.dumps({
        f"{stem}@frozen": _entry(folds_dir, 1, 3),
        f"{stem}@{CALIBRATED_SETTING.label()}": _entry(folds_dir, 1, 2),
    }))


def test_a_model_ahead_everywhere_wins(roots, folds_dir, lm_reference):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)

    comparison = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
    )

    against = comparison["against_reference"]["vit_l_em_organelles"]
    assert against["folds_paired"] == [1, 2, 3, 4, 5]
    assert against["n_folds_model_ahead"] == 5
    assert against["mean_delta_f1"] > cli.MIN_MEAN_GAIN_F1
    assert against["per_formulation"]["p_value"] < cli.ALPHA
    assert against["verdict"] == cli.VERDICT_MODEL_WINS


def test_a_model_behind_leaves_the_reference_standing(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 4)

    comparison = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
    )

    against = comparison["against_reference"]["vit_l_em_organelles"]
    assert against["mean_delta_f1"] < 0
    assert against["verdict"] == cli.VERDICT_REFERENCE_HOLDS


def test_ahead_in_too_few_folds_does_not_win(roots, folds_dir, lm_reference):
    # Far ahead in three folds, behind in two: the mean alone would win.
    _write_model(
        roots, folds_dir, "vit_l_em_organelles", [0, 0, 0, 3, 3],
    )

    against = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
    )["against_reference"]["vit_l_em_organelles"]

    assert against["n_folds_model_ahead"] == 3
    assert against["verdict"] == cli.VERDICT_REFERENCE_HOLDS


def test_the_calibrated_entries_decide_by_default(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)

    calibrated = _compare(roots, folds_dir, "--model", "vit_l_em_organelles")
    frozen = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
        "--postprocessing", "frozen",
    )

    assert calibrated["postprocessing"] == cli.CALIBRATED
    assert (calibrated["figures"]["vit_l_em_organelles"]["pooled_f1"]
            > frozen["figures"]["vit_l_em_organelles"]["pooled_f1"])


def test_a_decoder_trained_from_scratch_is_found_under_its_rate(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l", 1, decoder_lr=1e-4)

    comparison = _compare(
        roots, folds_dir, "--model", "vit_l", "--decoder-lr", "vit_l=1e-4",
    )

    assert comparison["figures"]["vit_l"]["folds_included"] == [1, 2, 3, 4, 5]
    assert comparison["decoder_learning_rates"]["vit_l"] == 1e-4


def test_diagnostics_are_reported_but_do_not_decide(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)

    comparison = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
    )

    em = comparison["figures"]["vit_l_em_organelles"]
    lm = comparison["figures"]["vit_l_lm"]
    assert em["q1_recall"] == pytest.approx(1.0)
    assert lm["q1_recall"] == pytest.approx(0.8)
    assert set(em["pooled_f1_by_material"]) == {"AS", "K", "VAB"}
    assert "f1_mean_without_validation_fold" in em["across_folds"]


def test_a_missing_fold_is_refused_or_gets_no_verdict(
    roots, folds_dir, lm_reference
):
    _write_model(
        roots, folds_dir, "vit_l_em_organelles", 0, folds=range(1, 5),
    )
    argv = _argv(roots, folds_dir, "--model", "vit_l_em_organelles")

    assert cli.main(argv) == cli.EXIT_FAILED
    comparison = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
        "--allow-missing",
    )
    against = comparison["against_reference"]["vit_l_em_organelles"]
    assert against["folds_paired"] == [1, 2, 3, 4]
    assert against["verdict"] == cli.VERDICT_INCOMPLETE


def test_folds_scored_on_different_images_are_refused(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)
    path = roots[0] / "vit_l_em_organelles" / "fold3.json"
    payload = json.loads(path.read_text())
    for entry in payload.values():
        entry["overall"]["n_gt"] += 5
    path.write_text(json.dumps(payload))

    assert cli.main(
        _argv(roots, folds_dir, "--model", "vit_l_em_organelles")
    ) == cli.EXIT_FAILED


def test_zero_shot_generalists_are_read_alongside(
    roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)
    directory = roots[0] / cli.ZERO_SHOT_DIR_NAME
    directory.mkdir()
    labels = [
        POSTPROCESSING_CONFIGS[name].label()
        for name in cli.ZERO_SHOT_SETTINGS
    ]
    for fold in range(1, 6):
        (directory / f"fold{fold}.json").write_text(json.dumps({
            f"pretrained/vit_l_em_organelles@{label}": _entry(
                folds_dir, fold, 5
            )
            for label in labels
        }))

    comparison = _compare(
        roots, folds_dir, "--model", "vit_l_em_organelles",
        "--zero-shot", "vit_l_em_organelles", "--zero-shot", "vit_l_lm",
    )

    zero_shot = comparison["zero_shot"]
    assert set(zero_shot) == {"vit_l_em_organelles"}
    assert set(zero_shot["vit_l_em_organelles"]) == set(cli.ZERO_SHOT_SETTINGS)


def test_the_report_is_written_as_json_and_markdown(
    tmp_path, roots, folds_dir, lm_reference
):
    _write_model(roots, folds_dir, "vit_l_em_organelles", 0)
    out = tmp_path / "out" / "paired.json"

    code = cli.main(_argv(
        roots, folds_dir, "--model", "vit_l_em_organelles", "--out", str(out),
    ))

    assert code == cli.EXIT_OK
    assert json.loads(out.read_text())["reference"] == "vit_l_lm"
    assert "model lepszy od referencji" in out.with_suffix(".md").read_text()


@pytest.mark.parametrize("extra, message", [
    ([], "--model"),
    (["--model", "vit_l_lm"], "reference"),
    (["--model", "vit_l"], "--decoder-lr vit_l"),
    (["--model", "vit_l_em_organelles", "--zero-shot", "vit_l"],
     "no instance decoder"),
])
def test_a_request_that_cannot_be_read_is_refused(extra, message):
    assert message in cli.refusal(cli.parse_args(extra))


def test_a_decoder_rate_must_name_a_model_and_a_rate():
    with pytest.raises(SystemExit):
        cli.parse_args(["--decoder-lr", "vit_l"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--decoder-lr", "nomodel=1e-4"])
