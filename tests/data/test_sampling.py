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

from materials_vision.data.sampling import (AUGMENT_DOMAIN, ORDER_DOMAIN,
                                            ProportionalImageSampler,
                                            decode_index, derive_seed,
                                            encode_index, sampler_run_metadata)
from materials_vision.data.split_io import load_split

N_IMAGES = 50
RUN_SEED = 1234


def _positions(sampler: ProportionalImageSampler) -> list[int]:
    """Dataset positions an epoch of this sampler visits.

    What the sampler yields carries the epoch as well as the position,
    so a test about ordering has to divide the two apart first.
    """
    return [
        decode_index(index, N_IMAGES)[1] for index in sampler
    ]


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
    return _positions(sampler)


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

    assert sorted(_positions(sampler)) == list(range(N_IMAGES))
    assert len(sampler) == N_IMAGES


def test_same_seed_and_epoch_give_the_same_order():
    first = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    second = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    assert list(first) == list(second)


def test_epoch_changes_the_order():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    first_epoch = _positions(sampler)
    sampler.set_epoch(1)

    assert _positions(sampler) != first_epoch
    assert sampler.epoch == 1


def test_the_epoch_travels_inside_the_index():
    """The load-bearing property for augmentation in worker processes.

    Positions repeat every epoch by construction, so the only thing
    that can tell two passes over the same image apart is the index
    itself. If these came out equal, augmentation would repeat one
    epoch's draws for the length of the run - which is the defect this
    encoding exists to make impossible.
    """
    sampler = ProportionalImageSampler(
        N_IMAGES, run_seed=RUN_SEED, shuffle=False
    )
    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert _positions(sampler) == list(range(N_IMAGES))
    assert set(first_epoch).isdisjoint(second_epoch)
    assert all(
        decode_index(index, N_IMAGES)[0] == 1 for index in second_epoch
    )


def test_index_encoding_round_trips():
    index = encode_index(epoch=7, position=13, n_images=N_IMAGES)

    assert decode_index(index, N_IMAGES) == (7, 13)


def test_decoding_against_a_meaningless_length_is_refused():
    with pytest.raises(ValueError, match="n_images must be positive"):
        decode_index(5, 0)


def test_unshuffled_sampling_keeps_the_given_order():
    sampler = ProportionalImageSampler(
        N_IMAGES, run_seed=RUN_SEED, positions=[4, 1, 9], shuffle=False
    )

    assert _positions(sampler) == [4, 1, 9]
    assert len(sampler) == 3


def test_restricting_positions_keeps_the_full_length_as_the_base():
    """A short rehearsal must still decode to the right positions.

    The epoch is packed against the dataset's length, not against how
    much of it is being drawn, because the dataset divides by its own
    length when it reads the index back.
    """
    sampler = ProportionalImageSampler(
        N_IMAGES, run_seed=RUN_SEED, positions=[2, 40], shuffle=False
    )
    sampler.set_epoch(3)

    assert list(sampler) == [3 * N_IMAGES + 2, 3 * N_IMAGES + 40]
    assert _positions(sampler) == [2, 40]


def test_repeated_iteration_without_set_epoch_repeats_the_order():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    assert list(sampler) == list(sampler)


def test_run_seed_changes_the_order():
    first = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)
    second = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED + 1)

    assert _positions(first) != _positions(second)


def test_non_positive_image_count_is_refused():
    with pytest.raises(ValueError, match="n_images must be positive"):
        ProportionalImageSampler(0, run_seed=RUN_SEED)


def test_an_empty_position_list_is_refused():
    with pytest.raises(ValueError, match="at least one image"):
        ProportionalImageSampler(N_IMAGES, RUN_SEED, positions=[])


def test_a_position_outside_the_dataset_is_refused():
    with pytest.raises(ValueError, match=r"must lie in \[0, 50\)"):
        ProportionalImageSampler(N_IMAGES, RUN_SEED, positions=[3, 50])


def test_negative_epoch_is_refused():
    sampler = ProportionalImageSampler(N_IMAGES, run_seed=RUN_SEED)

    with pytest.raises(ValueError, match="epoch must be >= 0"):
        sampler.set_epoch(-1)


def test_derive_seed_is_stable_and_in_range():
    value = derive_seed(RUN_SEED, 3, ORDER_DOMAIN)

    assert value == derive_seed(RUN_SEED, 3, ORDER_DOMAIN)
    assert derive_seed(RUN_SEED, 4, ORDER_DOMAIN) != value
    assert 0 <= value < 2 ** 64


def test_the_two_uses_of_one_run_seed_do_not_collide():
    """Image order and augmentation share a seed and a counter.

    Without a namespace between them the permutation of epoch zero and
    the augmentation of the first sample would be drawn from one value.
    Nothing visibly breaks today, because the two feed different
    generators - which is exactly the kind of coincidence that stops
    holding after an unrelated change.
    """
    assert (
        derive_seed(RUN_SEED, 0, ORDER_DOMAIN)
        != derive_seed(RUN_SEED, 0, AUGMENT_DOMAIN)
    )


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
