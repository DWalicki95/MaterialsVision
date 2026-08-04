"""Tests for data_prep.inventory.cross_section."""
import pytest

from data_prep.inventory.cross_section import (canonicalize_cross_section,
                                               lookup_cross_section)
from data_prep.inventory.issues import (CrossSectionError, IssueCollector,
                                        IssueLevel)


@pytest.mark.parametrize("token,expected", [
    ("rownolegle", "rownolegly"),
    ("rownolegla", "rownolegly"),
    ("rownolegly", "rownolegly"),
    ("r", "rownolegly"),
    ("prostopadle", "prostopadly"),
    ("prostopadla", "prostopadly"),
    ("prostopadly", "prostopadly"),
    ("p", "prostopadly"),
])
def test_known_variants_exact(token, expected):
    collector = IssueCollector()
    result = canonicalize_cross_section(
        token, fuzzy_cutoff=0.85, collector=collector, image_ref="test"
    )
    assert result == expected
    assert collector.all() == []


def test_case_insensitive():
    collector = IssueCollector()
    result = canonicalize_cross_section(
        "ROWNOLEGLE", fuzzy_cutoff=0.85, collector=collector,
        image_ref="test",
    )
    assert result == "rownolegly"
    assert collector.all() == []


def test_fuzzy_typo_reported():
    collector = IssueCollector()
    result = canonicalize_cross_section(
        "prostopdaly", fuzzy_cutoff=0.85, collector=collector,
        image_ref="K9_prostopdaly_m001",
    )
    assert result == "prostopadly"
    issues = collector.all()
    assert len(issues) == 1
    assert issues[0].level == IssueLevel.INFO
    assert issues[0].code == "fuzzy_cross_section"
    assert issues[0].image_ref == "K9_prostopdaly_m001"


def test_unknown_token_raises():
    collector = IssueCollector()
    with pytest.raises(CrossSectionError):
        canonicalize_cross_section(
            "xyz", fuzzy_cutoff=0.85, collector=collector, image_ref="test"
        )
    assert collector.all() == []


def test_high_cutoff_rejects_fuzzy():
    collector = IssueCollector()
    with pytest.raises(CrossSectionError):
        canonicalize_cross_section(
            "xyzpad", fuzzy_cutoff=0.99, collector=collector,
            image_ref="test",
        )


def test_lookup_cross_section_known():
    assert lookup_cross_section("prostopadle") == "prostopadly"
    assert lookup_cross_section("r") == "rownolegly"


def test_lookup_cross_section_unknown_returns_none():
    assert lookup_cross_section("xyz") is None
