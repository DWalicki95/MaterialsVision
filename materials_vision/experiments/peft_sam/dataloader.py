from pathlib import Path

from materials_vision.experiments.peft_sam.dataset import FoamSEMDataset
from torch.utils.data import DataLoader

import torch
from peft_sam.util import RawTrafo
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import MinInstanceSampler
import numpy as np


def get_data_loaders(
        train_dir,
        val_dir,
        desired_shape=(768, 768),
        batch_size=1,
        num_workers=4,
        train_dataset_kwargs=None,
        val_dataset_kwargs=None
):
    
    base_kwargs = dict(
        desired_shape=desired_shape,
        raw_transform=RawTrafo(
            desired_shape=desired_shape, triplicate_dims=True, do_padding=False
        ),
        label_transform=PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True
        ),
        sampler=MinInstanceSampler()
    )

    train_ds = FoamSEMDataset(
        img_dir=Path(train_dir),
        mask_dir=Path(train_dir),
        img_filename_suffix="_image.jpg",
        mask_filename_suffix="_masks.tif",
        **{**base_kwargs, **(train_dataset_kwargs or {})}
    )
    val_ds = FoamSEMDataset(
        img_dir=Path(val_dir),
        mask_dir=Path(val_dir),
        img_filename_suffix="_image.jpg",
        mask_filename_suffix="_masks.tif",
        **{**base_kwargs, **(val_dataset_kwargs or {})}
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        pin_memory=True,
    )

    return train_loader, val_loader


def _seed_worker(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2**32-1)
    np.random.seed(seed)
