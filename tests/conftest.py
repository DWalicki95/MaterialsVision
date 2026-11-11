"""Shared pytest fixtures: miniature data splits and a miniature SAM."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Formulations the miniature frozen split keeps in training in every
# cross-validation fold, as the real one does with VAB11.
CV_FORCED_TRAIN = ("VAB11",)

# (formulation, material, set of the frozen split, images, close-ups):
# the real split in miniature - one K and one VAB formulation in each
# evaluation set, three of each left in TRAIN for the remaining folds.
CV_SPEC = [
    ("AS3", "AS", "test", 2, 0),
    ("K3", "K", "test", 1, 0),
    ("VAB6", "VAB", "test", 1, 0),
    ("AS2", "AS", "val", 2, 0),
    ("K2", "K", "val", 1, 0),
    ("VAB5", "VAB", "val", 1, 0),
    ("AS10", "AS", "train", 3, 0),
    ("AS11", "AS", "train", 1, 0),
    ("AS12", "AS", "train", 2, 1),
    ("AS13", "AS", "train", 2, 0),
    ("AS14", "AS", "train", 1, 0),
    ("K1", "K", "train", 2, 1),
    ("K4", "K", "train", 1, 0),
    ("K6", "K", "train", 2, 0),
    ("VAB1", "VAB", "train", 2, 0),
    ("VAB2", "VAB", "train", 1, 0),
    ("VAB3", "VAB", "train", 2, 1),
    ("VAB11", "VAB", "train", 1, 1),
]


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the tests/fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def frozen_table() -> pd.DataFrame:
    """A frozen split table with the shape of the real one, in miniature."""
    rows = []
    for name, material, subset, n_images, n_closeups in CV_SPEC:
        for i in range(n_images + n_closeups):
            rows.append({
                "image_id": f"{name}_{i}",
                "formulation": name,
                "material": material,
                "microscope": "M1" if material == "AS" else "M2",
                "scale_bin": "outlier" if i >= n_images else "coarse",
                "pixel_size_um": 3.24,
                "n_instances": 10 + i,
                "source_path": f"/data/{name}_{i}.jpg",
                "split": subset,
                "used": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def frozen_split(tmp_path, frozen_table):
    """The frozen split on disk, with its metadata and a manifest.

    The manifest carries every column a sample source joins on, so a
    loader can be built from this split without any image existing;
    nothing is read until a sample is drawn.

    Returns
    -------
    tuple of Path
        The split table and the manifest.
    """
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_id": frozen_table["image_id"],
        "load_crop_bbox": "0,0,64,48",
        "mask_path": "/data/" + frozen_table["image_id"] + "_mask.tif",
        "width_px": 64,
        "height_px": 48,
        "n_channels": 1,
        "channels_identical": True,
        "n_instances": frozen_table["n_instances"],
        "q_max_i": 1.3,
    }).to_csv(manifest, index=False)
    split_csv = tmp_path / "split_mini.csv"
    frozen_table.to_csv(split_csv, index=False)
    (tmp_path / "split_mini_metadata.json").write_text(json.dumps({
        "split_id": "split_mini",
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "forced_train": list(CV_FORCED_TRAIN),
    }), encoding="utf-8")
    return split_csv, manifest


# Width and depth of the miniature encoder: small enough to build in
# milliseconds, with the module names of SAM's ViT, which is all the
# library's adaptation wrappers and the checkpoint reader look at.
TINY_SAM_DIM = 8
TINY_SAM_BLOCKS = 4


class _TinyMlp(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim, 2 * dim)
        self.lin2 = torch.nn.Linear(2 * dim, dim)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class _TinyAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = torch.nn.Linear(dim, 3 * dim)


class _TinyBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = _TinyAttention(dim)
        self.mlp = _TinyMlp(dim)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.patch_embed = torch.nn.Linear(dim, dim)
        self.blocks = torch.nn.ModuleList(
            _TinyBlock(dim) for _ in range(n_blocks)
        )
        self.neck = torch.nn.Linear(dim, dim)


class _TinySam(torch.nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.image_encoder = _TinyEncoder(dim, n_blocks)
        self.prompt_encoder = torch.nn.Linear(dim, dim)
        self.mask_decoder = torch.nn.Linear(dim, dim)


@pytest.fixture
def tiny_sam():
    """Build a miniature SAM, wrapped the way a run's adaptation wraps it.

    Returns a function taking the library's wrapper arguments (``None``
    for the plain model) and returning the model, with the parts every
    run freezes already frozen.
    """
    from micro_sam.models.peft_sam import PEFT_Sam

    def build(peft_kwargs=None, seed=0):
        torch.manual_seed(seed)
        sam = _TinySam(TINY_SAM_DIM, TINY_SAM_BLOCKS)
        if peft_kwargs is not None:
            sam = PEFT_Sam(sam, **dict(peft_kwargs)).sam
        for part in (sam.prompt_encoder, sam.mask_decoder):
            for parameter in part.parameters():
                parameter.requires_grad = False
        return sam

    return build
