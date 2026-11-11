"""Tests for reading a checkpoint's adaptation back out of its weights.

The evaluation never learns how a checkpoint was trained except from
the checkpoint itself. These tests close the seam end to end: a model
is wrapped exactly as a run wraps it, its weights are saved the way the
trainer saves them, the configuration is read back, and a fresh model
built from that reading must accept the weights under a strict load.
"""
import pytest
import torch

from materials_vision.evaluation.inference import (APG_UNUSED_SETTINGS,
                                                   generate_kwargs,
                                                   peft_kwargs_from_state)
from materials_vision.evaluation.watershed import FROZEN_WATERSHED
from materials_vision.training import PEFT_KWARGS, build_peft_kwargs


def _saved_state(sam):
    """Weights as a run's checkpoint holds them, under the model's
    ``sam.`` prefix."""
    return {f"sam.{k}": v for k, v in sam.state_dict().items()}


def _strip(state):
    return {k[len("sam."):]: v for k, v in state.items()}


@pytest.mark.parametrize("method, kwargs", [
    ("lora", {}),
    ("lora", {"rank": 4}),
    ("lora", {"layers": [2, 3]}),
    ("lora", {"update_matrices": ["q", "k", "v", "mlp"]}),
    ("lora", {"layers": [1, 3], "update_matrices": ["v", "mlp"]}),
    ("lora", {"update_matrices": ["mlp"]}),
])
def test_a_low_rank_adaptation_is_read_back_as_it_was_built(
    tiny_sam, method, kwargs
):
    built = build_peft_kwargs(method, **kwargs)
    state = _saved_state(tiny_sam(built))

    read = peft_kwargs_from_state(state)

    assert read == built
    tiny_sam(read, seed=1).load_state_dict(_strip(state), strict=True)


def test_every_checkpoint_of_the_study_so_far_reads_as_the_frozen_wrapper(
    tiny_sam
):
    state = _saved_state(tiny_sam(PEFT_KWARGS))

    assert peft_kwargs_from_state(state) == PEFT_KWARGS


@pytest.mark.parametrize("method, kwargs", [
    ("late_ft", {"layers": [2, 3]}),
    ("full_ft", {}),
])
def test_full_weight_checkpoints_load_into_the_plain_model(
    tiny_sam, method, kwargs
):
    state = _saved_state(tiny_sam(build_peft_kwargs(method, **kwargs)))

    assert peft_kwargs_from_state(state) is None
    tiny_sam(None, seed=1).load_state_dict(_strip(state), strict=True)


def test_a_wrongly_rebuilt_model_refuses_the_weights(tiny_sam):
    """What makes a wrong reading loud: the load stays strict."""
    state = _saved_state(tiny_sam(build_peft_kwargs(layers=[2, 3])))

    with pytest.raises(RuntimeError):
        tiny_sam(PEFT_KWARGS).load_state_dict(_strip(state), strict=True)


def test_factors_of_different_ranks_are_refused(tiny_sam):
    state = _saved_state(tiny_sam(PEFT_KWARGS))
    key = next(k for k in state if "blocks.0.attn.qkv.w_a_linear_q" in k)
    state[key] = torch.zeros(3, state[key].shape[1])

    with pytest.raises(ValueError, match="rank"):
        peft_kwargs_from_state(state)


def test_blocks_disagreeing_on_projections_are_refused(tiny_sam):
    state = _saved_state(tiny_sam(PEFT_KWARGS))
    del state["sam.image_encoder.blocks.0.attn.qkv.w_a_linear_v.weight"]

    with pytest.raises(ValueError, match="projections"):
        peft_kwargs_from_state(state)


def test_the_prompt_based_path_is_not_handed_smoothing():
    from micro_sam.instance_segmentation import (
        AutomaticPromptGenerator, InstanceSegmentationWithDecoder)

    apg = object.__new__(AutomaticPromptGenerator)
    ais = object.__new__(InstanceSegmentationWithDecoder)

    apg_kwargs = generate_kwargs(apg, FROZEN_WATERSHED)
    ais_kwargs = generate_kwargs(ais, FROZEN_WATERSHED)

    assert ais_kwargs == FROZEN_WATERSHED.library_kwargs()
    assert set(ais_kwargs) - set(apg_kwargs) == set(APG_UNUSED_SETTINGS)


def test_every_argument_left_for_the_prompt_based_path_is_one_it_takes():
    """A renamed argument upstream would stop every APG evaluation."""
    import inspect

    from micro_sam.instance_segmentation import AutomaticPromptGenerator

    apg = object.__new__(AutomaticPromptGenerator)
    accepted = inspect.signature(AutomaticPromptGenerator.generate).parameters

    assert set(generate_kwargs(apg, FROZEN_WATERSHED)) <= set(accepted)
