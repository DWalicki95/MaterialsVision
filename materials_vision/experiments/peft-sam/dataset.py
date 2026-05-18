import logging
from pathlib import Path

import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import MinInstanceSampler
from peft_sam.util import RawTrafo

logger = logging.getLogger(__name__)


class FoamSEMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_dir,
            mask_dir, 
            img_filename_suffix='_image.tif',
            mask_filename_suffix='_masks.tif',
            desired_shape: tuple = (768, 768), 
            sampler=None,
        ):
        self.img_dir = img_dir
        self.img_suffix = img_filename_suffix
        self.mask_dir = mask_dir
        self.mask_filename_suffix = mask_filename_suffix
        self.desired_shape = desired_shape
        self.sample = sampler
        self.max_sampler_retries = 10  # how many times we try to crop

        self.pairs = self._load_pairs()

    def __len__(self):
        return len(self.img_masks)

    def _load_pairs(self):
        pairs = []
        img_files = sorted(self.img_dir.glob("*.tif"))
        masks_files = sorted(self.mask_dir.glob(".tif"))

        if len(img_files) != len(masks_files):
            logger.error("Number of images and masks doesn't fit. Correct it.")

        for img, mask in zip(img_files, masks_files):
            img_filename = img.stem
            mask_filename = mask.stem
            is_match = img_filename.removesuffix(self.img_suffix) == mask_filename.removesuffix(self.mask_filename_suffix)
            if not is_match:
                msg = 'Cannot match all images and masks.'
                logger.error(msg)
                raise DataLoaderError(msg)
            else:
                pairs.append((img, mask))

        return pairs

    def _verify_shapes(self):
        ph, pw = self.desired_shape
        for img_path, mask_path in self.pairs:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            H, W = img.shape[:2]
            if img.shape[:2] != mask.shape[:2]:
                raise DataLoaderError(
                    'Image and mask dimensions dont match: '
                    f'{img_path.name} {img.shape[:2]} vs '
                    f'{mask_path.name} {mask.shape[:2]}'
                )
            if H < ph or W < pw:
                raise DataLoaderError(
                    f"Image {img_path.name} is smaller tha patch: "
                    f"{img.shape[:2]} < {self.desired_shape}"
                )

    def _random_crop(self, img, mask):
        H, W = img.shape[:2]
        ph, pw = self.desired_shape
        x = np.random.randint(0, W - pw + 1)
        y = np.random.randint(0, H - ph + 1)
        img_patch = img[y:y+ph, x:x+pw]
        mask_patch = mask[y:y+ph, x:x+pw]
        return img_patch, mask_patch

    def _sample_valid_patch(self, image, mask):
        for _ in range(self.max_sampler_retries):
            img_patch, mask_patch = self._random_crop(image, mask)
            if self.sampler is None:
                return img_patch, mask_patch
            try:
                is_ok = self.sampler(img_patch, mask_patch)
            except Exception as e:
                logger.warning(f'Sampler error: {e}')
                continue
            if is_ok:
                return img_patch, mask_patch

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        image_patch, mask_patch = self._sample_valid_patch(image, mask)

        if self.raw_transform is not None:
            image_patch = self.raw_transform(image_path)
        if self.mask_transform is not None:
            mask_patch = self.mask_transform(mask_path)

        return image_patch, mask_patch


def _seed_worker(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2**32-1)
    np.random.seed(seed)


def get_data_loaders(
        train_dir,
        val_dir,
        desired_shape=(768, 768),
        batch_size=1,
        num_workers=4
):
    kwargs = dict(
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
        image_dir=Path(train_dir) / "images",
        mask_dir = Path(train_dir) / "masks",
        **kwargs
    )
    val_ds = FoamSEMDataset(
        image_dir=Path(train_dir) / "images",
        mask_dir = Path(train_dir) / "masks",
        **kwargs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        pin_memory=True
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


class DataLoaderError(Exception):
    pass
