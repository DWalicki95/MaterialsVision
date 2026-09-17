"""Tests for the shared training configuration and loader builder.

Most of what this module holds is frozen configuration rather than
logic, and configuration is exactly what drifts unnoticed: this study
attributes a difference in segmentation quality to a difference in the
augmentation policy, and that inference collapses the moment two runs
also differ in the adaptation rank, in what is frozen, or in the
learning rate. The value tests below are therefore not tautologies
guarding constants against themselves - they pin the numbers a reader
of the study would cite, so that changing one requires changing a test
and noticing.
"""
import numpy as np
import pytest
import torch

from materials_vision.training import (BATCH_SIZE, DECODER_LEARNING_RATE,
                                       FREEZE_PARTS, LORA_LEARNING_RATE,
                                       LORA_RANK, N_OBJECTS_PER_BATCH,
                                       NUM_WORKERS, PEFT_KWARGS,
                                       CosineAnnealingIgnoringMetric,
                                       _trainable_groups, build_loader)


class _FakeRecord:
    def __init__(self, index):
        self.index = index
        self.image_id = f"img_{index}"


class _FakeSample:
    def __init__(self, image, labels, record):
        self.image = image
        self.labels = labels
        self.record = record


class _FakeSource:
    """Stands in for SampleSource, so the loader tests need no data."""

    def __init__(self, n_images=6, shape=(6, 8)):
        self._n = n_images
        self._shape = shape
        self.loaded = []

    def __len__(self):
        return self._n

    def load(self, index):
        self.loaded.append(index)
        image = np.full(self._shape, 100 + index, dtype=np.uint8)
        labels = np.zeros(self._shape, dtype=np.int32)
        labels[1:3, 1:4] = 1
        return _FakeSample(image, labels, _FakeRecord(index))


class _SeedIntoPixels:
    """Augmentation stand-in that paints its own seed into the image.

    Defined at module level and holding no state, so that it survives
    being pickled into a worker process - which the test that matters
    most here requires, because the defect it guards against only
    appears once loading leaves this process.
    """

    def __call__(self, image, labels, *, record, seed):
        return np.full_like(image, seed % 251), labels


@pytest.fixture
def single_process_loading(monkeypatch):
    """Load in this process, so a fake source stays observable.

    Worker processes would each get their own copy of the source and
    the record of what they loaded would never come back. Requested
    explicitly rather than applied to every test, because the tests
    that pin the frozen worker count must see the real value.
    """
    monkeypatch.setattr("materials_vision.training.NUM_WORKERS", 0)


def test_adaptation_touches_only_query_and_value_projections():
    assert PEFT_KWARGS["update_matrices"] == ["q", "v"]


def test_adaptation_covers_every_attention_block():
    assert PEFT_KWARGS["attention_layers_to_update"] == []


def test_adaptation_rank_is_eight():
    """Four times smaller than the rank the PEFT package recommends.

    Deliberate: their 32 buys capacity, and the baseline runs peak two
    epochs in and decline after, which is too much capacity for the
    data rather than too little.
    """
    assert PEFT_KWARGS["rank"] == LORA_RANK == 8


def test_adaptation_method_is_named_not_inherited():
    """The wrapper class is stated, not left to a library default.

    It happens to be that default today; stating it means a change
    upstream cannot silently change what this study trains.
    """
    from micro_sam.models.peft_sam import LoRASurgery

    assert PEFT_KWARGS["peft_module"] is LoRASurgery


def test_prompt_encoder_and_mask_decoder_are_frozen():
    assert set(FREEZE_PARTS) == {"prompt_encoder", "mask_decoder"}


def test_the_two_halves_do_not_share_a_learning_rate():
    """The correction and the decoder are in different regimes.

    One starts at zero and holds under a million parameters, the other
    starts pretrained and holds nine and a half million. A grid over
    both rates separated the correction's clearly, a point of instance
    F1 between its two values, and could not separate the decoder's at
    all. These are the values that grid settled on.
    """
    assert LORA_LEARNING_RATE == 3e-4
    assert DECODER_LEARNING_RATE == 1e-5
    assert DECODER_LEARNING_RATE < LORA_LEARNING_RATE


def test_one_image_per_step():
    assert BATCH_SIZE == 1


def test_instances_sampled_per_step():
    assert N_OBJECTS_PER_BATCH == 25


def test_worker_count_is_four():
    assert NUM_WORKERS == 4


def test_loader_yields_one_sample_per_batch(single_process_loading):
    loader = build_loader(_FakeSource())
    images, targets = next(iter(loader))
    assert images.shape[0] == 1
    assert targets.shape[0] == 1


def test_loader_covers_the_whole_source_by_default(single_process_loading):
    source = _FakeSource(n_images=6)
    loader = build_loader(source)
    assert len(loader) == 6


def test_indices_restrict_the_loader_to_those_positions(
    single_process_loading,
):
    source = _FakeSource(n_images=6)
    loader = build_loader(source, indices=[1, 4])
    list(loader)
    assert sorted(source.loaded) == [1, 4]


def test_order_is_kept_unless_shuffling_is_asked_for(single_process_loading):
    source = _FakeSource(n_images=6)
    list(build_loader(source, shuffle=False))
    assert source.loaded == [0, 1, 2, 3, 4, 5]


def test_no_policy_leaves_the_annotation_untouched(single_process_loading):
    """Without augmentation the labels reaching the targets are the
    ones the source produced, which is what the baseline condition and
    every validation pass require."""
    source = _FakeSource(n_images=1)
    loader = build_loader(source, policy=None)
    _, targets = next(iter(loader))
    # Channel zero of the decoder's target stack is the label image.
    instance_channel = targets[0, 0].numpy()
    assert set(np.unique(instance_channel)) == {0.0, 1.0}


def test_augmentation_differs_between_epochs_across_worker_processes():
    """The one test that would have caught a run's worst defect.

    Samples are prepared in worker processes holding their own copies
    of the dataset, so anything the training process sets on its copy
    reaches none of them. An augmentation keyed to such state repeats
    one epoch's draws for the whole run, and the model sees a single
    fixed augmented copy of the data - which looks exactly like a
    healthy run from the outside, loss curve included. Real workers are
    therefore the point of this test and it must not be made to load in
    process.
    """
    loader = build_loader(
        _FakeSource(n_images=8), policy=_SeedIntoPixels(),
        run_seed=11, shuffle=False,
    )

    first = [float(images.max()) for images, _ in loader]
    loader.sampler.set_epoch(1)
    second = [float(images.max()) for images, _ in loader]

    assert NUM_WORKERS > 0
    assert first != second


def test_image_order_survives_a_policy_that_draws_heavily():
    """Paired comparisons rest on this, and it is checked end to end.

    The sampler's own stream is tested in isolation elsewhere; what
    matters for the study is that the loader actually uses it, which
    only a test built through the loader can show.
    """
    def order_with(policy):
        torch.manual_seed(0)
        source = _FakeSource(n_images=8)
        loader = build_loader(
            source, policy=policy, run_seed=3, shuffle=True,
        )
        for _ in range(500):
            torch.rand(1)
        list(loader)
        return list(source.loaded)

    assert order_with(None) == order_with(_SeedIntoPixels())


def test_the_shared_encoder_reaches_only_one_parameter_group():
    """The decoder holds the encoder as a submodule, not a copy.

    Selecting the decoder's parameters by taking everything it owns
    would therefore hand the correction to the optimizer twice, in two
    groups, and step it at both rates every iteration.
    """
    encoder = torch.nn.Linear(4, 4)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    model = torch.nn.Module()
    model.encoder = encoder
    model.correction = torch.nn.Linear(4, 2)

    unetr = torch.nn.Module()
    unetr.encoder = encoder
    unetr.head = torch.nn.Linear(4, 3)

    correction, decoder = _trainable_groups(model, unetr)

    assert [tuple(p.shape) for p in correction] == [(2, 4), (2,)]
    assert [tuple(p.shape) for p in decoder] == [(3, 4), (3,)]
    correction_ids = {id(p) for p in correction}
    assert correction_ids.isdisjoint({id(p) for p in decoder})


def _rates(scheduler, optimizer, steps, metric=None):
    """Learning rates over a few epochs of a schedule."""
    observed = []
    for _ in range(steps):
        scheduler.step() if metric is None else scheduler.step(metric)
        observed.append(optimizer.param_groups[0]["lr"])
    return observed


def test_the_schedule_ignores_the_metric_the_trainer_hands_it():
    """The trainer steps its scheduler with a validation loss.

    Every scheduler but the plateau one reads that argument as an
    epoch number, so a cosine schedule handed 0.077 would look up the
    rate for epoch zero forever and hold its initial rate for the whole
    run while appearing to work.
    """
    def build(cls):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        return cls(optimizer, T_max=6), optimizer

    ours, our_optimizer = build(CosineAnnealingIgnoringMetric)
    plain, plain_optimizer = build(
        torch.optim.lr_scheduler.CosineAnnealingLR
    )

    with_metric = _rates(ours, our_optimizer, 6, metric=0.077)
    without = _rates(plain, plain_optimizer, 6)

    assert with_metric == pytest.approx(without)


def test_the_schedule_anneals_to_nothing_at_its_horizon():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = CosineAnnealingIgnoringMetric(optimizer, T_max=6)

    rates = _rates(scheduler, optimizer, 6, metric=0.077)

    assert rates[0] < 1.0
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] == pytest.approx(0.0, abs=1e-9)
