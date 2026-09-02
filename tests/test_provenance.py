"""Tests for the run provenance record.

The property that matters most is that nothing here can stop a run: a
version that cannot be read must become a recorded gap, never an
exception thrown hours into training.
"""
import json
from importlib import metadata

import pytest

from materials_vision import provenance


def test_a_known_package_reports_its_version():
    assert provenance.package_version("numpy") is not None


def test_an_absent_package_is_recorded_as_a_gap(caplog):
    with caplog.at_level("WARNING"):
        assert provenance.package_version("no_such_package_at_all") is None

    assert "not installed" in caplog.text


def test_a_git_installed_package_reports_its_commit():
    """``peft_sam`` has no release to identify it, only a commit."""
    commit = provenance.package_commit("peft_sam")

    assert commit is not None
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)


def test_a_pypi_package_has_no_commit_to_report(caplog):
    with caplog.at_level("WARNING"):
        assert provenance.package_commit("numpy") is None


def test_an_absent_package_has_no_commit():
    assert provenance.package_commit("no_such_package_at_all") is None


def test_an_unreadable_installation_record_is_a_gap(monkeypatch, caplog):
    class _Broken:
        def read_text(self, name):
            return "{not json"

    monkeypatch.setattr(metadata, "distribution", lambda name: _Broken())

    with caplog.at_level("WARNING"):
        assert provenance.package_commit("peft_sam") is None

    assert "readable JSON" in caplog.text


def test_the_record_covers_the_libraries_that_decide_behaviour():
    record = provenance.run_provenance()

    for name in ("micro_sam", "peft_sam", "torch_em", "torch", "numpy"):
        assert name in record["packages"]
    assert "peft_sam" in record["package_commits"]


def test_the_record_names_the_repository_state():
    record = provenance.run_provenance()

    assert set(record["git"]) == {"commit", "branch", "dirty"}


def test_a_dirty_tree_is_recorded_rather_than_hidden(monkeypatch):
    """A commit alone does not describe code that was edited."""
    monkeypatch.setattr(
        provenance, "_git",
        lambda *arguments: "M some/file.py"
        if arguments[0] == "status" else "abc123",
    )

    assert provenance._repository_state()["dirty"] is True


def test_the_record_is_json_serialisable():
    """It has to survive being written beside the run's results."""
    json.dumps(provenance.run_provenance())


def test_nothing_in_the_record_raises_when_git_is_unavailable(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *arguments: None)

    record = provenance.run_provenance()

    assert record["git"] == {"commit": None, "branch": None, "dirty": None}


def test_hardware_is_described_even_without_an_accelerator(monkeypatch):
    record = provenance.run_provenance()["hardware"]

    assert "platform" in record
    assert "cuda_available" in record


def test_the_torch_build_tag_survives_the_metadata_layer():
    """``metadata.version`` drops the local tag that names the build.

    It reports ``2.9.1`` where the library reports ``2.9.1+cu128``,
    and the tag is the part that says which CUDA kernels ran.
    """
    import torch

    record = provenance.run_provenance()

    assert record["hardware"]["torch_build"] == torch.__version__
    assert "+" in torch.__version__
    assert record["packages"]["torch"] != torch.__version__


@pytest.mark.parametrize("name", provenance.TRACKED_PACKAGES)
def test_every_tracked_package_is_actually_installed(name):
    """A tracked package that is absent means the list has drifted."""
    assert provenance.package_version(name) is not None
