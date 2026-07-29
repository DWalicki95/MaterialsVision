import logging
from pathlib import Path

import tifffile  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image

from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
from torch_em.transform.label import PerObjectDistanceTransform  # pyright: ignore[reportMissingImports]
from torch_em.data import MinInstanceSampler  # pyright: ignore[reportMissingImports]
from peft_sam.util import RawTrafo  # pyright: ignore[reportMissingImports]
from materials_vision.experiments.peft_sam.exceptions import DataLoaderError
from materials_vision.utils import find_image_mask_pairs

logger = logging.getLogger(__name__)


class FoamSEMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        desired_shape,
        img_filename_suffix="_image.tif",
        mask_filename_suffix="_masks.tif",
        transform=None,
        raw_transform=None,
        label_transform=None,
        sampler=None,
        max_sampler_retries=10,
        verify_shapes=True,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.desired_shape = tuple(desired_shape)
        self.img_suffix = img_filename_suffix
        self.mask_suffix = mask_filename_suffix
        self.raw_transform = raw_transform
        self.transform = transform
        self.label_transform = label_transform
        self.sampler = sampler
        self.max_sampler_retries = max_sampler_retries

        self.pairs = self._load_pairs()
        if not self.pairs:
            raise DataLoaderError(
                f"No image/mask pairs found in {img_dir} <-> {mask_dir}"
            )
        if verify_shapes:
            self._verify_shapes()

    def __len__(self):
        return len(self.pairs)

    def _load_pairs(self):
        try:
            pairs = find_image_mask_pairs(
                image_dir=self.img_dir,
                mask_dir=self.mask_dir,
                image_suffix=self.img_suffix,
                mask_suffix=self.mask_suffix,
                strict=True,
            )
        except ValueError as e:
            raise DataLoaderError(str(e)) from e
        return [(pair['image'], pair['mask']) for pair in pairs]

    def _verify_shapes(self):
        ph, pw = self.desired_shape
        for img_path, mask_path in self.pairs:
            try:
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img = np.array(Image.open(img_path).convert("RGB"))
                else:
                    img = tifffile.imread(img_path)
                mask = tifffile.imread(mask_path)
            except Exception as e:
                raise DataLoaderError(
                    f"Failed to read {img_path} or {mask_path}: {e}"
                ) from e
            if img.shape[:2] != mask.shape[:2]:
                raise DataLoaderError(
                    f"Shape mismatch: {img_path.name} {img.shape[:2]} "
                    f"vs {mask_path.name} {mask.shape[:2]}"
                )
            H, W = img.shape[:2]
            if H < ph or W < pw:
                raise DataLoaderError(
                    f"Image {img_path.name} smaller than patch: "
                    f"{img.shape[:2]} < {self.desired_shape}"
                )

    def _random_crop(self, img, mask):
        H, W = img.shape[:2]
        ph, pw = self.desired_shape
        y = np.random.randint(0, H - ph + 1)
        x = np.random.randint(0, W - pw + 1)
        return img[y:y + ph, x:x + pw], mask[y:y + ph, x:x + pw]

    def _sample_valid_patch(self, image, mask):
        img_patch, mask_patch = self._random_crop(image, mask)
        for _ in range(self.max_sampler_retries):
            img_patch, mask_patch = self._random_crop(image, mask)
            if self.sampler is None:
                return img_patch, mask_patch
            try:
                is_ok = self.sampler(img_patch, mask_patch)
            except Exception as e:
                logger.warning(f"Sampler error: {e}")
                continue
            if is_ok:
                return img_patch, mask_patch
        return img_patch, mask_patch  # fallback

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)
        if mask.dtype != np.uint16:
            mask = mask.astype(np.uint16)

        image_patch, mask_patch = self._sample_valid_patch(image, mask)

        # augmentation
        if self.transform is not None:
            image_patch, mask_patch = self.transform(image_patch, mask_patch)

        if self.raw_transform is not None:
            image_patch = torch.from_numpy(image_patch)
            image_patch = self.raw_transform(image_patch)
        if self.label_transform is not None:
            mask_patch = self.label_transform(mask_patch)

        return image_patch, mask_patch

  