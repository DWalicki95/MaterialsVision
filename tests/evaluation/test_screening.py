"""Tests for the registered screening rule of the fine-tuning arms.

Each condition is pinned on both sides, and the branch table is pinned
row by row, because the rule was written down before any arm was scored
and the code must say exactly what the plan says.
"""
import pytest

from materials_vision.evaluation.screening import (ArmVerdict, branch,
                                                   late_mean, noise_band,
                                                   screen_arm)


def _figures(f1=0.88, merges=3.0, splits=2.0, count=0.01, boundary=0.80,
             k_f1=0.86):
    return {
        "f1": f1, "merges_per_100_gt": merges, "splits_per_100_gt": splits,
        "abs_pore_count_error": count, "boundary_f1": boundary,
        "K/f1": k_f1,
    }


REFERENCES = [
    _figures(f1=0.880, merges=3.0, splits=2.0, count=0.010, boundary=0.80,
             k_f1=0.860),
    _figures(f1=0.884, merges=3.4, splits=2.2, count=0.020, boundary=0.79,
             k_f1=0.855),
    _figures(f1=0.878, merges=2.8, splits=1.8, count=0.015, boundary=0.81,
             k_f1=0.865),
]


class TestNoiseBand:
    def test_range_when_it_is_wide_enough(self):
        band, how = noise_band([0.880, 0.884, 0.878])
        assert band == pytest.approx(0.006)
        assert how == "range"

    def test_two_sigma_below_the_floor(self):
        band, how = noise_band([0.8800, 0.8810, 0.8805])
        assert how == "2sigma"
        assert band == pytest.approx(2 * 0.0005)

    def test_needs_two_seeds(self):
        with pytest.raises(ValueError):
            noise_band([0.88])


def test_late_mean_refuses_a_missing_epoch():
    assert late_mean({10: 0.8, 11: 0.9, 12: 1.0}) == pytest.approx(0.9)
    with pytest.raises(KeyError):
        late_mean({10: 0.8, 11: 0.9})


class TestScreenArm:
    band = 0.006

    def test_passes_when_every_condition_holds(self):
        verdict = screen_arm(_figures(f1=0.890), REFERENCES, self.band)
        assert verdict.gain == pytest.approx(0.890 - 0.880667, abs=1e-5)
        assert verdict.passes

    def test_gain_equal_to_the_band_does_not_clear_it(self):
        mean = sum(r["f1"] for r in REFERENCES) / 3
        arm_f1 = mean + self.band
        # The band as the same arithmetic produces it, so the two are
        # equal to the last bit and "more than" is what is tested.
        band = arm_f1 - mean
        verdict = screen_arm(_figures(f1=arm_f1), REFERENCES, band)
        assert not verdict.clears_band

    @pytest.mark.parametrize("name, value", [
        ("merges", 3.5), ("splits", 2.3), ("count", 0.021),
        ("boundary", 0.78),
    ])
    def test_each_guard_blocks_on_its_worse_side(self, name, value):
        verdict = screen_arm(
            _figures(f1=0.890, **{name: value}), REFERENCES, self.band
        )
        assert not verdict.passes
        assert len(verdict.guards_failed) == 1

    def test_matching_the_worst_seed_is_allowed(self):
        verdict = screen_arm(
            _figures(f1=0.890, merges=3.4, boundary=0.79), REFERENCES,
            self.band,
        )
        assert verdict.guards_failed == ()

    def test_material_below_the_reference_range_blocks(self):
        verdict = screen_arm(_figures(f1=0.890, k_f1=0.854), REFERENCES,
                             self.band)
        assert not verdict.material_ok
        assert not verdict.passes


def _verdict(passes, gain=0.01):
    return ArmVerdict(gain=gain, clears_band=passes, guards_failed=(),
                      material_ok=True, passes=passes)


@pytest.mark.parametrize("a1, a2, expected", [
    (True, False, "capacity"),
    (False, True, "rate"),
    (False, False, None),
])
def test_branch_table(a1, a2, expected):
    assert branch(_verdict(a1), _verdict(a2), band=0.006)[1] == expected


def test_both_passing_prefer_the_step_unless_capacity_leads_by_the_band():
    assert branch(_verdict(True, 0.012), _verdict(True, 0.010), 0.006)[1] \
        == "rate"
    assert branch(_verdict(True, 0.020), _verdict(True, 0.010), 0.006)[1] \
        == "capacity"
