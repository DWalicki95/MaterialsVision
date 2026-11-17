"""Every call of ``build_segmenter`` must fit its current signature.

The scripts that load a checkpoint are not exercised by the suite: they
need weights and a GPU. When ``build_segmenter`` stopped taking the
adaptation and the rank as arguments, two callers kept passing them
positionally, where they landed in ``device`` and ``segmentation_mode``
and failed only once a run reached them. This reads every call out of
the source and binds it to the signature, so the next such change
fails here instead.
"""
import ast
import inspect
from pathlib import Path

import pytest

from materials_vision.evaluation.inference import build_segmenter

REPO_ROOT = Path(__file__).resolve().parents[1]

SEARCHED = ("scripts", "materials_vision")


def _called_name(call: ast.Call) -> str:
    """The bare name a call is made through, or an empty string."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(name: str) -> list[tuple[Path, ast.Call]]:
    """Every call of ``name`` in the searched source trees."""
    found = []
    for root in SEARCHED:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found.extend(
                (path, node) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and _called_name(node) == name
            )
    return found


CALLS = _calls("build_segmenter")


def test_callers_are_found():
    assert len(CALLS) >= 4


@pytest.mark.parametrize(
    "path, call", CALLS,
    ids=[f"{p.relative_to(REPO_ROOT)}:{c.lineno}" for p, c in CALLS],
)
def test_call_binds_to_signature(path, call):
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        pytest.skip("argument unpacking cannot be checked statically")
    signature = inspect.signature(build_segmenter)
    bound = signature.bind(
        *[object()] * len(call.args),
        **{keyword.arg: object() for keyword in call.keywords},
    )
    # The two trailing parameters take a device name and a mode name; a
    # positional argument reaching them is what the old calls did.
    assert len(call.args) <= 2, (
        f"{path.name}:{call.lineno} passes {len(call.args)} positional "
        f"arguments; device and segmentation_mode must be named."
    )
    assert "checkpoint" in bound.arguments
