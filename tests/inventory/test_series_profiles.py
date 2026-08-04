"""Tests for data_prep.inventory.series_profiles."""
import pytest

from data_prep.inventory.issues import (FilenameParseError, IssueCollector,
                                        IssueLevel)
from data_prep.inventory.series_profiles import (ASProfile, VABProfile,
                                                 get_profile)


@pytest.fixture
def collector():
    return IssueCollector()


class TestASProfile:
    def test_standard_name(self, collector):
        profile = ASProfile()
        parsed = profile.parse(
            "00ae5c27-AS26_40_15_jpg.rf."
            "4d3e1176b573efd51a1741cc835fb1c4_image",
            collector=collector,
        )
        assert parsed.image_id == "AS26_40_15"
        assert parsed.series == "AS"
        assert parsed.material == "AS"
        assert parsed.formulation == "AS26"
        assert parsed.cross_section is None
        assert parsed.magnification_from_name == 40
        assert parsed.sample_id == "15"
        assert parsed.is_nonstandard is False
        assert collector.all() == []

    def test_formulation_with_letter_suffix(self, collector):
        profile = ASProfile()
        parsed = profile.parse(
            "134f5637-AS1A_40_21a_jpg.rf."
            "cb805e9287f756e41085bc8f29ea4e53_image",
            collector=collector,
        )
        assert parsed.formulation == "AS1A"
        assert parsed.magnification_from_name == 40
        assert parsed.sample_id == "21a"
        assert parsed.is_nonstandard is True

    def test_no_ls_hash_prefix(self, collector):
        profile = ASProfile()
        parsed = profile.parse(
            "AS21_40_3_jpg.rf.deadbeef00000000000000000000000_image",
            collector=collector,
        )
        assert parsed.image_id == "AS21_40_3"
        assert parsed.formulation == "AS21"
        assert parsed.sample_id == "3"
        assert parsed.is_nonstandard is False

    def test_missing_sample_token(self, collector):
        profile = ASProfile()
        parsed = profile.parse(
            "fd0e5a1a-AS19_50_jpg.rf."
            "f25e40d498f1f34e1215ac4124dfb075_image",
            collector=collector,
        )
        assert parsed.formulation == "AS19"
        assert parsed.magnification_from_name == 50
        assert parsed.sample_id is None
        assert parsed.is_nonstandard is True

    @pytest.mark.parametrize("filename,expected_sample", [
        (
            "48b38033-AS18_40_-_jpg.rf."
            "2908c191d291ecab71a248bc561ee127_image",
            "-",
        ),
        (
            "de87c821-AS25_40_1-5_jpg.rf."
            "f30c07c94d07c9aa7b2b4ee36c8ce6f7_image",
            "1-5",
        ),
        (
            "70fd3bb2-AS4_40_500_1_jpg.rf."
            "5bf208e9d131361c11615cbcd074a6a0_image",
            "500_1",
        ),
        (
            "6ba19865-AS4_40_500_2_jpg.rf."
            "9c4f71c9bd2bfd8b8a0cfc7fcc1a293e_image",
            "500_2",
        ),
    ])
    def test_nonstandard_sample_tokens(
        self, collector, filename, expected_sample
    ):
        profile = ASProfile()
        parsed = profile.parse(filename, collector=collector)
        assert parsed.sample_id == expected_sample
        assert parsed.is_nonstandard is True
        assert parsed.formulation.startswith("AS")

    def test_unparsable_name_raises(self, collector):
        profile = ASProfile()
        with pytest.raises(FilenameParseError):
            profile.parse("not_an_as_filename", collector=collector)

    def test_sidecar_candidates(self, collector):
        profile = ASProfile()
        parsed = profile.parse(
            "00ae5c27-AS26_40_15_jpg.rf."
            "4d3e1176b573efd51a1741cc835fb1c4_image",
            collector=collector,
        )
        candidates = profile.sidecar_candidates(parsed)
        assert candidates == (
            profile.sidecar_candidates(parsed)[0],
        )
        from pathlib import Path
        assert candidates[0] == Path("AS26") / "AS26_40_15.txt"

    def test_detect_series(self):
        profile = ASProfile()
        assert profile.detect_series(
            "00ae5c27-AS26_40_15_jpg.rf.hash_image"
        )
        assert not profile.detect_series(
            "5ad2e566-K3_rownolegle_K3_r_m004"
        )


class TestVABProfile:
    def test_k_formulation(self, collector):
        profile = VABProfile()
        parsed = profile.parse(
            "5ad2e566-K3_rownolegle_K3_r_m004", collector=collector
        )
        assert parsed.formulation == "K3"
        assert parsed.material == "K"
        assert parsed.cross_section == "rownolegly"
        assert parsed.image_id == "K3_rownolegly_m004"
        assert parsed.magnification_from_name is None
        assert parsed.sample_id == "004"
        assert parsed.cross_section_redundancy_ok is True
        assert collector.all() == []

    def test_vab_formulation_full_redundant_token(self, collector):
        profile = VABProfile()
        parsed = profile.parse(
            "5efaa3ca-VAB2_prostopadla_VAB2_prostopadla_m003",
            collector=collector,
        )
        assert parsed.formulation == "VAB2"
        assert parsed.material == "VAB"
        assert parsed.cross_section == "prostopadly"
        assert parsed.image_id == "VAB2_prostopadly_m003"
        assert parsed.cross_section_redundancy_ok is True

    def test_vab_prostopadly_variant(self, collector):
        profile = VABProfile()
        parsed = profile.parse(
            "0c7a7d00-VAB3_prostopadly_VAB3_prostopadly_m007",
            collector=collector,
        )
        assert parsed.formulation == "VAB3"
        assert parsed.cross_section == "prostopadly"
        assert parsed.image_id == "VAB3_prostopadly_m007"

    def test_redundancy_mismatch_is_warning_not_fatal(self, collector):
        profile = VABProfile()
        parsed = profile.parse(
            "aaaaaaaa-K3_rownolegle_K3_p_m001", collector=collector
        )
        assert parsed.cross_section == "rownolegly"
        assert parsed.cross_section_redundancy_ok is False
        issues = collector.all()
        assert len(issues) == 1
        assert issues[0].level == IssueLevel.WARNING
        assert issues[0].code == "vab_redundancy_mismatch"

    def test_unparsable_name_raises(self, collector):
        profile = VABProfile()
        with pytest.raises(FilenameParseError):
            profile.parse("not_a_vab_filename", collector=collector)

    def test_sidecar_candidates_use_raw_tokens(self, collector):
        from pathlib import Path
        profile = VABProfile()
        parsed = profile.parse(
            "5ad2e566-K3_rownolegle_K3_r_m004", collector=collector
        )
        candidates = profile.sidecar_candidates(parsed)
        assert candidates == (
            Path("K3 rownolegle") / "K3 r_m004.txt",
        )

    def test_sidecar_candidates_vab_full_tokens(self, collector):
        from pathlib import Path
        profile = VABProfile()
        parsed = profile.parse(
            "5efaa3ca-VAB2_prostopadla_VAB2_prostopadla_m003",
            collector=collector,
        )
        candidates = profile.sidecar_candidates(parsed)
        assert candidates == (
            Path("VAB2 prostopadla") / "VAB2 prostopadla_m003.txt",
        )

    def test_detect_series(self):
        profile = VABProfile()
        assert profile.detect_series(
            "5ad2e566-K3_rownolegle_K3_r_m004"
        )
        assert profile.detect_series(
            "5efaa3ca-VAB2_prostopadla_VAB2_prostopadla_m003"
        )
        assert not profile.detect_series(
            "00ae5c27-AS26_40_15_jpg.rf.hash_image"
        )


def test_get_profile_registry():
    assert isinstance(get_profile("AS"), ASProfile)
    assert isinstance(get_profile("VAB"), VABProfile)
    with pytest.raises(KeyError):
        get_profile("unknown")
