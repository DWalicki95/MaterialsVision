"""End-to-end test: full CLI run (config file -> exit code ->
artifacts on disk) against a miniature, fully synthetic fixture
dataset (see conftest.py's ``mini_dataset``).
"""
import sys

import yaml

import scripts.build_inventory_manifest as cli


def _write_config_yaml(config, path):
    data = {
        "manifest_version": config.manifest_version,
        "output_dir": str(config.output_dir),
        "mask_root": str(config.mask_root),
        "sources": [
            {
                "series": s.series,
                "images_dir": str(s.images_dir),
                "label_studio_json": str(s.label_studio_json),
                "sem_metadata_dirs": [
                    str(d) for d in s.sem_metadata_dirs
                ],
            }
            for s in config.sources
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_full_cli_run(mini_dataset, tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(mini_dataset.config, config_path)

    monkeypatch.setattr(
        sys, "argv",
        ["build_inventory_manifest.py", "--config", str(config_path)],
    )
    exit_code = cli.main()

    # 1 ERROR-level issue (task_without_image) -> EXIT_ERRORS.
    assert exit_code == cli.EXIT_ERRORS

    output_dir = mini_dataset.config.output_dir
    expected_artifacts = [
        "manifest_v1.csv",
        "validation_report.csv",
        "validation_report.md",
        "dataset_summary.md",
        "dataset_summary.csv",
        "run_metadata.json",
    ]
    for name in expected_artifacts:
        assert (output_dir / name).exists(), f"missing artifact: {name}"

    import pandas as pd
    manifest = pd.read_csv(output_dir / "manifest_v1.csv")
    assert len(manifest) == mini_dataset.expected_rows


def test_dry_run_writes_nothing(mini_dataset, tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(mini_dataset.config, config_path)

    monkeypatch.setattr(
        sys, "argv",
        [
            "build_inventory_manifest.py", "--config",
            str(config_path), "--dry-run",
        ],
    )
    exit_code = cli.main()
    assert exit_code == cli.EXIT_ERRORS

    output_dir = mini_dataset.config.output_dir
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_series_filter_restricts_sources(
    mini_dataset, tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(mini_dataset.config, config_path)

    monkeypatch.setattr(
        sys, "argv",
        [
            "build_inventory_manifest.py", "--config",
            str(config_path), "--series", "VAB", "--dry-run",
        ],
    )
    exit_code = cli.main()
    assert exit_code == cli.EXIT_OK  # VAB alone has no errors


def test_unknown_series_filter_is_fatal(
    mini_dataset, tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(mini_dataset.config, config_path)

    monkeypatch.setattr(
        sys, "argv",
        [
            "build_inventory_manifest.py", "--config",
            str(config_path), "--series", "NOPE",
        ],
    )
    exit_code = cli.main()
    assert exit_code == cli.EXIT_FATAL


def test_refuses_overwrite_via_cli(mini_dataset, tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(mini_dataset.config, config_path)

    monkeypatch.setattr(
        sys, "argv",
        ["build_inventory_manifest.py", "--config", str(config_path)],
    )
    assert cli.main() == cli.EXIT_ERRORS
    # Second run without --overwrite must refuse and exit fatal.
    assert cli.main() == cli.EXIT_FATAL
