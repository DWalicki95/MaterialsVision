"""Tests for data_prep.inventory.sem_sidecar.find_sidecar."""
from pathlib import Path

from data_prep.inventory.issues import IssueCollector
from data_prep.inventory.sem_sidecar import find_sidecar
from data_prep.inventory.series_profiles import ASProfile, VABProfile


def test_find_sidecar_as(fixtures_dir):
    profile = ASProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "00ae5c27-AS26_40_15_jpg.rf."
        "4d3e1176b573efd51a1741cc835fb1c4_image",
        collector=collector,
    )
    root = fixtures_dir / "sem_roots" / "AS"
    found = find_sidecar(parsed, profile, roots=[root])
    assert found == root / "AS26" / "AS26_40_15.txt"


def test_find_sidecar_k_abbreviated_redundant_segment(fixtures_dir):
    profile = VABProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "5ad2e566-K3_rownolegle_K3_r_m004", collector=collector
    )
    root = fixtures_dir / "sem_roots" / "vab_and_k"
    found = find_sidecar(parsed, profile, roots=[root])
    assert found == root / "K3 rownolegle" / "K3 r_m004.txt"


def test_find_sidecar_vab_full_tokens(fixtures_dir):
    profile = VABProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "5efaa3ca-VAB2_prostopadla_VAB2_prostopadla_m003",
        collector=collector,
    )
    root = fixtures_dir / "sem_roots" / "vab_and_k"
    found = find_sidecar(parsed, profile, roots=[root])
    assert found == root / "VAB2 prostopadla" / "VAB2 prostopadla_m003.txt"


def test_find_sidecar_searches_multiple_roots_in_order(fixtures_dir):
    profile = ASProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "00ae5c27-AS26_40_15_jpg.rf."
        "4d3e1176b573efd51a1741cc835fb1c4_image",
        collector=collector,
    )
    empty_root = fixtures_dir / "sem_roots" / "vab_and_k"
    real_root = fixtures_dir / "sem_roots" / "AS"
    found = find_sidecar(parsed, profile, roots=[empty_root, real_root])
    assert found == real_root / "AS26" / "AS26_40_15.txt"


def test_find_sidecar_missing_returns_none(fixtures_dir):
    profile = ASProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "00ae5c27-AS999_40_1_jpg.rf.deadbeef00000000000000000000_image",
        collector=collector,
    )
    root = fixtures_dir / "sem_roots" / "AS"
    found = find_sidecar(parsed, profile, roots=[root])
    assert found is None


def test_find_sidecar_nonexistent_root_no_crash():
    profile = ASProfile()
    collector = IssueCollector()
    parsed = profile.parse(
        "00ae5c27-AS26_40_15_jpg.rf."
        "4d3e1176b573efd51a1741cc835fb1c4_image",
        collector=collector,
    )
    found = find_sidecar(
        parsed, profile, roots=[Path("/nonexistent/root")]
    )
    assert found is None
