"""Tests for the frozen image sampler.

The load-bearing test here is
``test_image_order_is_immune_to_augmentation_randomness``. The
experiment attributes a metric difference between two runs to the
augmentation policy that differed between them; that only holds if the
runs saw the same images in the same order, and this is the one test
that checks it.
"""
import pytest
import torch

from materials_vision.data.sampling import (ProportionalImageSampler,
                                            derive_seed, sampler_run_metadata)
from materials_vision.data.split_io import load_split

N_IMAGES = 50
RUN_SEED = 1234


def _order_after_consuming(n_draws: int) -> list[int]:
    """Image order when a policy first draws ``n_draws`` random values.

    Stands in for an augmentation policy: the two policies compared in
    an ablation construct different transforms and therefore consume
    different amounts of randomness before the first batch is drawn.
    """
    torch.manual_seed(0)
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    for _ in range(n_draws):
        torch.rand(1)
    return list(sampler)


def test_image_order_is_immune_to_augmentation_randomness():
    without_augmentation = _order_after_consuming(0)
    with_heavy_augmentation = _order_after_consuming(1000)

    assert without_augmentation == with_heavy_augmentation


def test_global_rng_ordering_would_not_be():
    """Documents the failure mode the local generator avoids.

    This is what ``DataLoader(shuffle=True)`` does: it permutes from
    the global generator, so the order shifts as soon as anything else
    consumes randomness first.
    """
    def global_order(n_draws):
        torch.manual_seed(0)
        for _ in range(n_draws):
            torch.rand(1)
        return torch.randperm(N_IMAGES).tolist()

    assert global_order(0) != global_order(1000)


def test_every_image_appears_exactly_once_per_epoch():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    order = list(sampler)

    assert sorted(order) == list(range(N_IMAGES))
    assert len(sampler) == N_IMAGES


def test_same_seed_and_epoch_give_the_same_order():
    first = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    second = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    assert list(first) == list(second)


def test_epoch_changes_the_order():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    first_epoch = list(sampler)
    sampler.set_epoch(1)

    assert list(sampler) != first_epoch
    assert sampler.epoch == 1


def test_repeated_iteration_without_set_epoch_repeats_the_order():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    assert list(sampler) == list(sampler)


def test_run_seed_changes_the_order():
    first = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    second = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED + 1)

    assert list(first) != list(second)


def test_non_positive_image_count_is_refused():
    with pytest.raises(ValueError, match="n_images must be positive"):
        ProportionalImageSampler(0, run_seed=RUN_SEED)


def test_negative_epoch_is_refused():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    with pytest.raises(ValueError, match="epoch must be >= 0"):
        sampler.set_epoch(-1)


def test_derive_seed_is_stable_and_in_range():
    value = derive_seed(RUN_SEED, 3)

    assert value == derive_seed(RUN_SEED, 3)
    assert derive_seed(RUN_SEED, 4) != value
    assert 0 <= value < 2 ** 64


def test_run_metadata_reports_exposure(split_csv):
    subset = load_split(split_csv, "train")
    sampler = ProportionalImageSampler(len(subset), run_seed=RUN_SEED)

    metadata = sampler_run_metadata(subset, sampler)

    assert metadata["strategy"] == "proportional_no_oversampling"
    assert metadata["oversampling"] is None
    assert metadata["steps_per_epoch"] == len(subset)
    assert metadata["split_id"] == "split_test"
    assert metadata["exposure"]["material"]["AS"] == pytest.approx(0.6)
    assert "scale_bin" in metadata["exposure"]


def test_run_metadata_refuses_a_mismatched_sampler(split_csv):
    subset = load_split(split_csv, "train")
    sampler = ProportionalImageSampler(len(subset) + 1, RUN_SEED)

    with pytest.raises(ValueError, match="Sampler covers"):
        sampler_run_metadata(subset, sampler)
