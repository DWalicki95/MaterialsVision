"""Tests for candidate aggregation and the hard conditions of III.4."""
from data_prep.split.constraints import (aggregate, check_constraints,
                                         is_admissible)
from data_prep.split.models import SplitConstraints


def _assign(profiles, val, test):
    """Assign everything to TRAIN except the named formulations."""
    return {
        p.formulation: (
            "val" if p.formulation in val
            else "test" if p.formulation in test
            else "train"
        )
        for p in profiles
    }


def test_aggregate_counts_images_instances_and_cells(profiles):
    assignment = _assign(profiles, val={"AS1", "VAB2"}, test={"AS2", "K1"})

    stats = aggregate(assignment, profiles)

    val = stats["val"]
    assert val.formulations == ("AS1", "VAB2")
    assert val.n_images == 22 + 6
    assert val.n_eval_images == 22 + 6
    assert val.images_by_material == {"AS": 22, "VAB": 6}
    assert val.images_by_cell[("AS", "coarse")] == 20
    assert val.images_by_cell[("VAB", "fine")] == 6
    assert val.n_m2_formulations == 1


def test_aggregate_excludes_outliers_from_evaluable_images(profiles):
    assignment = _assign(profiles, val={"VAB3"}, test={"K1"})

    stats = aggregate(assignment, profiles)

    assert stats["val"].n_images == 4
    assert stats["val"].n_outlier == 2
    assert stats["val"].n_eval_images == 2


def test_set_without_m2_formulation_is_rejected(profiles):
    assignment = _assign(profiles, val={"AS1"}, test={"AS2"})

    violations = check_constraints(
        aggregate(assignment, profiles), SplitConstraints()
    )

    assert any("M2 formulation" in v for v in violations)


def test_set_without_fine_images_is_rejected(profile_factory):
    profiles = (
        profile_factory("AS1", "AS", "M1", n_coarse=20, n_fine=2),
        profile_factory("K1", "K", "M2", n_coarse=12),
        profile_factory("K2", "K", "M2", n_coarse=12),
    )
    assignment = {"AS1": "train", "K1": "val", "K2": "test"}

    violations = check_constraints(
        aggregate(assignment, profiles),
        SplitConstraints(min_eval_fine_images=1),
    )

    assert any("'fine' image" in v for v in violations)


def test_thin_material_cross_section_is_rejected(profiles):
    assignment = _assign(profiles, val={"AS1", "VAB3"}, test={"AS2", "K1"})

    constraints = SplitConstraints(
        min_eval_fine_images=1, min_eval_images_by_material={"VAB": 5}
    )
    violations = check_constraints(
        aggregate(assignment, profiles), constraints
    )

    assert any("evaluable VAB image" in v for v in violations)


def test_admissible_candidate_has_no_violations(profiles):
    assignment = _assign(
        profiles, val={"AS1", "K1", "VAB2"}, test={"AS2", "K2", "VAB1"}
    )

    stats = aggregate(assignment, profiles)
    constraints = SplitConstraints(
        min_eval_fine_images=2, min_eval_images_by_material={"VAB": 5}
    )

    assert check_constraints(stats, constraints) == []
    assert is_admissible(stats, constraints)
