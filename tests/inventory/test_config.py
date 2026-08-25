"""Tests for data_prep.inventory.config."""
import pytest

from data_prep.inventory.config import ConfigError, load_config


def _write_yaml(tmp_path, images_dir, ls_json, sem_dir, **overrides):
    output_dir = tmp_path / "out"
    mask_root = tmp_path / "masks"
    content = f"""
manifest_version: v1
output_dir: {output_dir}
mask_root: {mask_root}
sources:
  - series: AS
    images_dir: {images_dir}
    label_studio_json: {ls_json}
    sem_metadata_dirs:
      - {sem_dir}
"""
    for key, value in overrides.items():
        content += f"{key}: {value}\n"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


@pytest.fixture
def valid_layout(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ls_json = tmp_path / "export.json"
    ls_json.write_text("[]", encoding="utf-8")
    sem_dir = tmp_path / "sem"
    sem_dir.mkdir()
    return images_dir, ls_json, sem_dir


def test_load_config_valid(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(tmp_path, images_dir, ls_json, sem_dir)

    config = load_config(config_path)

    assert config.manifest_version == "v1"
    assert len(config.sources) == 1
    assert config.sources[0].series == "AS"
    assert config.sources[0].images_dir == images_dir
    assert config.fuzzy_cutoff == 0.85


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_missing_images_dir(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(
        tmp_path, tmp_path / "missing_images", ls_json, sem_dir
    )
    with pytest.raises(ConfigError, match="images_dir"):
        load_config(config_path)


def test_load_config_missing_label_studio_json(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(
        tmp_path, images_dir, tmp_path / "missing.json", sem_dir
    )
    with pytest.raises(ConfigError, match="label_studio_json"):
        load_config(config_path)


def test_load_config_missing_sem_dir(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(
        tmp_path, images_dir, ls_json, tmp_path / "missing_sem"
    )
    with pytest.raises(ConfigError, match="sem_metadata_dir"):
        load_config(config_path)


def test_load_config_overrides(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(
        tmp_path, images_dir, ls_json, sem_dir,
        scale_outlier_ratio=2.0, fuzzy_cutoff=0.9,
    )
    config = load_config(config_path)
    assert config.scale_outlier_ratio == 2.0
    assert config.fuzzy_cutoff == 0.9


def test_load_config_avoid_annotators_defaults_empty(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    config_path = _write_yaml(tmp_path, images_dir, ls_json, sem_dir)

    config = load_config(config_path)

    assert config.sources[0].avoid_annotators == ()


def test_load_config_avoid_annotators_parsed(tmp_path, valid_layout):
    images_dir, ls_json, sem_dir = valid_layout
    output_dir = tmp_path / "out"
    mask_root = tmp_path / "masks"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
manifest_version: v1
output_dir: {output_dir}
mask_root: {mask_root}
sources:
  - series: AS
    images_dir: {images_dir}
    label_studio_json: {ls_json}
    sem_metadata_dirs:
      - {sem_dir}
    avoid_annotators: [1, 3]
""", encoding="utf-8")

    config = load_config(config_path)

    assert config.sources[0].avoid_annotators == (1, 3)


def test_load_config_avoid_annotators_invalid_type_raises(
    tmp_path, valid_layout
):
    images_dir, ls_json, sem_dir = valid_layout
    output_dir = tmp_path / "out"
    mask_root = tmp_path / "masks"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
manifest_version: v1
output_dir: {output_dir}
mask_root: {mask_root}
sources:
  - series: AS
    images_dir: {images_dir}
    label_studio_json: {ls_json}
    sem_metadata_dirs:
      - {sem_dir}
    avoid_annotators: ["not_an_int"]
""", encoding="utf-8")

    with pytest.raises(ConfigError, match="avoid_annotators"):
        load_config(config_path)
