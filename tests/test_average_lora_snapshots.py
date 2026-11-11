"""Tests for ``scripts/peft_sam/average_lora_snapshots.py``.

The claim the script makes is that its output is the exact average of
the snapshots' corrections, and that the evaluation loads it without
being told its rank. Both are checked through the real modules: the
library's low-rank wrapper computes the outputs, and the checkpoint
reader rebuilds the wrapper from the averaged weights.
"""
import argparse

import pytest
import torch

import scripts.peft_sam.average_lora_snapshots as cli
from materials_vision.evaluation.inference import peft_kwargs_from_state
from materials_vision.training import PEFT_KWARGS, build_peft_kwargs


def _snapshot_state(tiny_sam, seed, peft_kwargs=PEFT_KWARGS):
    """A snapshot's weights: shared base, factors drawn per snapshot."""
    sam = tiny_sam(peft_kwargs)
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for key, value in sam.state_dict().items():
        if cli.FIRST_FACTOR in key or cli.SECOND_FACTOR in key:
            value = torch.randn(value.shape, generator=generator)
        state[f"sam.{key}"] = value
    return state


def _strip(state):
    return {k[len("sam."):]: v for k, v in state.items()}


def _attention_output(tiny_sam, state, x):
    """Every block's qkv projection, as the loaded model computes it.

    The projection is linear in its correction, so its output averages
    exactly; a whole MLP does not, its activation sitting between the
    two corrected layers, and is checked through its factors instead.
    """
    sam = tiny_sam(peft_kwargs_from_state(state), seed=5)
    sam.load_state_dict(_strip(state), strict=True)
    return torch.cat(
        [block.attn.qkv(x).flatten() for block in sam.image_encoder.blocks]
    )


MIXED = build_peft_kwargs(
    layers=[1, 3], update_matrices=["q", "k", "v", "mlp"]
)


@pytest.mark.parametrize("peft_kwargs", [PEFT_KWARGS, MIXED])
def test_the_average_moves_every_projection_by_the_mean_correction(
    tiny_sam, peft_kwargs
):
    states = [_snapshot_state(tiny_sam, s, peft_kwargs) for s in (1, 2, 3)]
    x = torch.randn(1, 2, 2, 8)

    averaged = cli.average_low_rank(states)
    expected = torch.stack(
        [_attention_output(tiny_sam, s, x) for s in states]
    )

    assert torch.allclose(
        _attention_output(tiny_sam, averaged, x), expected.mean(dim=0),
        atol=1e-5,
    )


def test_every_pair_of_factors_averages_its_product_exactly(tiny_sam):
    """The MLP's factors included."""
    states = [_snapshot_state(tiny_sam, s, MIXED) for s in (1, 2, 3)]

    averaged = cli.average_low_rank(states)

    first_factors = [k for k in averaged if cli.FIRST_FACTOR in k]
    assert any(".mlp." in k for k in first_factors)
    for a_key in first_factors:
        b_key = a_key.replace(cli.FIRST_FACTOR, cli.SECOND_FACTOR)
        mean_product = torch.stack(
            [s[b_key] @ s[a_key] for s in states]
        ).mean(dim=0)
        assert torch.allclose(
            averaged[b_key] @ averaged[a_key], mean_product, atol=1e-5,
        )


def test_the_evaluation_reads_the_stacked_rank(tiny_sam):
    states = [_snapshot_state(tiny_sam, s) for s in range(4)]

    read = peft_kwargs_from_state(cli.average_low_rank(states))

    assert read["rank"] == 4 * PEFT_KWARGS["rank"]
    assert read["update_matrices"] == PEFT_KWARGS["update_matrices"]
    assert read["attention_layers_to_update"] == []


def test_the_snapshots_are_left_untouched(tiny_sam):
    states = [_snapshot_state(tiny_sam, s) for s in (1, 2)]
    before = [{k: v.clone() for k, v in s.items()} for s in states]

    cli.average_low_rank(states)

    for state, copy in zip(states, before):
        assert all(torch.equal(state[k], copy[k]) for k in state)


def test_snapshots_differing_outside_their_factors_are_refused(tiny_sam):
    states = [_snapshot_state(tiny_sam, s) for s in (1, 2)]
    key = "sam.image_encoder.neck.weight"
    states[1][key] = states[1][key] + 1

    with pytest.raises(ValueError, match="neck"):
        cli.average_low_rank(states)


def test_a_single_snapshot_is_refused(tiny_sam):
    with pytest.raises(ValueError, match="at least"):
        cli.average_low_rank([_snapshot_state(tiny_sam, 1)])


def test_snapshots_without_factors_are_refused(tiny_sam):
    states = [{f"sam.{k}": v for k, v in tiny_sam(None).state_dict().items()}
              for _ in range(2)]

    with pytest.raises(ValueError, match="no low-rank"):
        cli.average_low_rank(states)


def _write_run(tmp_path, tiny_sam, epochs, decoder_moves=False):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for epoch in epochs:
        decoder = {"head.weight": torch.ones(2, 2)}
        if decoder_moves:
            decoder["head.weight"] = decoder["head.weight"] * epoch
        torch.save({
            "model_state": _snapshot_state(tiny_sam, epoch),
            "decoder_state": decoder,
            "optimizer_state": {"unused": True},
        }, run_dir / f"epoch-{epoch}.pt")
    return run_dir


def test_the_saved_average_is_what_the_evaluation_loads(tmp_path, tiny_sam):
    run_dir = _write_run(tmp_path, tiny_sam, (1, 2, 3))
    out = run_dir / "avg-1-3.pt"

    status = cli.main([
        "--run-dir", str(run_dir), "--epochs", "1-3", "--out", str(out),
    ])

    assert status == cli.EXIT_OK
    saved = torch.load(out, weights_only=False)
    assert set(saved) >= {"model_state", "decoder_state"}
    assert "optimizer_state" not in saved
    assert peft_kwargs_from_state(saved["model_state"])["rank"] == (
        3 * PEFT_KWARGS["rank"]
    )


def test_a_run_whose_decoder_trained_is_refused(tmp_path, tiny_sam):
    run_dir = _write_run(tmp_path, tiny_sam, (1, 2), decoder_moves=True)

    status = cli.main([
        "--run-dir", str(run_dir), "--epochs", "1-2",
        "--out", str(run_dir / "avg.pt"),
    ])

    assert status == cli.EXIT_FAILED
    assert not (run_dir / "avg.pt").exists()


@pytest.mark.parametrize("out_name, epochs, existing", [
    ("epoch-40.pt", "1-2", False),
    ("avg.pt", "1-2", True),
    ("avg.pt", "1-5", False),
    ("avg.pt", "2", False),
])
def test_requests_that_would_mislead_are_refused(
    tmp_path, tiny_sam, out_name, epochs, existing
):
    run_dir = _write_run(tmp_path, tiny_sam, (1, 2))
    if existing:
        (run_dir / out_name).touch()

    status = cli.main([
        "--run-dir", str(run_dir), "--epochs", epochs,
        "--out", str(run_dir / out_name),
    ])

    assert status == cli.EXIT_FAILED


@pytest.mark.parametrize("text, epochs", [
    ("25-28", [25, 26, 27, 28]), ("3,1,2", [1, 2, 3]), ("1-2,5", [1, 2, 5]),
])
def test_epochs_are_read_from_ranges_and_lists(text, epochs):
    assert cli.parse_epochs(text) == epochs


@pytest.mark.parametrize("text", ["38-25", "0-3", "x", ""])
def test_unreadable_epochs_are_rejected(text):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_epochs(text)
