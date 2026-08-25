"""Tests for the split search: validity, determinism and balance."""
import dataclasses

import pytest

from data_prep.split.constraints import aggregate
from data_prep.split.cost import DatasetTotals, split_cost
from data_prep.split.models import SETS, CostWeights
from data_prep.split.search import (SplitSearchError, search_split,
                                    validate_quotas)


def test_every_formulation_lands_in_exactly_one_set(profiles, config):
    result = search_split(profiles, config)

    assert set(result.assignment) == {p.formulation for p in profiles}
    assert set(result.assignment.values()) <= set(SETS)
    members = [
        f for name in SETS for f in result.stats[name].formulations
    ]
    assert sorted(members) == sorted(result.assignment)
    assert len(members) == len(set(members))


def test_quota_is_respected_per_material(profiles, config):
    result = search_split(profiles, config)

    material_of = {p.formulation: p.material for p in profiles}
    for material, (n_train, n_val, n_test) in config.quotas.items():
        counts = {name: 0 for name in SETS}
        for formulation, name in result.assignment.items():
            if material_of[formulation] == material:
                counts[name] += 1
        assert counts == {
            "train": n_train, "val": n_val, "test": n_test
        }


def test_forced_train_formulations_are_pinned(profiles, config):
    result = search_split(profiles, config)

    for formulation in config.forced_train:
        assert result.assignment[formulation] == "train"


def test_same_seed_gives_the_same_split(profiles, config):
    first = search_split(profiles, config)
    second = search_split(profiles, config)

    assert first.assignment == second.assignment
    assert first.cost == second.cost


def test_different_seed_can_give_a_different_split(profiles, config):
    other = dataclasses.replace(config, seed=config.seed + 1)

    first = search_split(profiles, config)
    second = search_split(profiles, other)

    assert first.assignment != second.assignment


def test_chosen_split_is_the_cheapest_seen(profiles, config):
    result = search_split(profiles, config)
    totals = DatasetTotals(profiles)

    recomputed = split_cost(
        aggregate(result.assignment, profiles), totals,
        config.cost_weights, config.target_shares,
    )

    assert recomputed == pytest.approx(result.cost)


def test_searching_is_never_worse_than_taking_the_first_candidate(
    profiles, config
):
    first_only = dataclasses.replace(config, n_candidates=1)

    searched = search_split(profiles, config)
    first = search_split(profiles, first_only)

    assert searched.cost <= first.cost


def test_outliers_outside_train_are_penalized(profiles, config):
    totals = DatasetTotals(profiles)
    weights = CostWeights(images=0.0, instances=0.0, cell_default=0.0)
    base = {p.formulation: "train" for p in profiles}
    with_outlier_in_val = dict(base, VAB3="val", VAB2="train")

    cost_train = split_cost(
        aggregate(base, profiles), totals, weights, config.target_shares
    )
    cost_val = split_cost(
        aggregate(with_outlier_in_val, profiles), totals, weights,
        config.target_shares,
    )

    assert cost_val > cost_train


def test_quota_not_matching_the_data_is_rejected(profiles, config):
    broken = dataclasses.replace(
        config, quotas=dict(config.quotas, AS=(3, 1, 1))
    )

    with pytest.raises(SplitSearchError, match="sums to 5"):
        validate_quotas(profiles, broken)


def test_unknown_material_in_quota_is_rejected(profiles, config):
    broken = dataclasses.replace(
        config, quotas=dict(config.quotas, ZZZ=(1, 0, 0))
    )

    with pytest.raises(SplitSearchError, match="do not match"):
        validate_quotas(profiles, broken)


def test_unknown_forced_train_formulation_is_rejected(profiles, config):
    broken = dataclasses.replace(config, forced_train=("NOPE",))

    with pytest.raises(SplitSearchError, match="unknown formulation"):
        validate_quotas(profiles, broken)


def test_forced_train_exceeding_train_quota_is_rejected(
    profiles, config
):
    broken = dataclasses.replace(
        config, forced_train=("VAB1", "VAB2", "VAB3")
    )

    with pytest.raises(SplitSearchError, match="pins 3"):
        validate_quotas(profiles, broken)


def test_unsatisfiable_constraints_raise(profiles, config):
    impossible = dataclasses.replace(
        config,
        constraints=dataclasses.replace(
            config.constraints, min_eval_fine_images=10_000
        ),
    )

    with pytest.raises(SplitSearchError, match="No admissible split"):
        search_split(profiles, impossible)
