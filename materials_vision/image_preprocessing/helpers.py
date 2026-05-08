import logging
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Dict
from torchvision.utils import draw_bounding_boxes, draw_keypoints, \
    draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


logger = logging.getLogger(__name__)


def find_image_mask_pairs(
    input_dir: Path,
    image_suffix: str = "_image.jpg",
    mask_suffix: str = "_masks.tif"
) -> List[Dict[str, Path]]:
    """
    Find matching image-mask pairs in directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing images and masks.
    image_suffix : str, optional
        Suffix for image files (default: "_image.jpg").
    mask_suffix : str, optional
        Suffix for mask files (default: "_masks.tif").

    Returns
    -------
    List[Dict[str, Path]]
        List of dicts with 'image', 'mask', and 'base_name' keys.
    """
    input_dir = Path(input_dir)
    pairs = []

    image_pattern = f"*{image_suffix}"
    for img_path in input_dir.glob(image_pattern):
        base_name = img_path.stem.replace(
            image_suffix.split('.')[0], ''
        )
        mask_path = input_dir / f"{base_name}{mask_suffix}"

        if mask_path.exists():
            pairs.append({
                'image': img_path,
                'mask': mask_path,
                'base_name': base_name
            })
            logger.info(
                f"Found pair: {img_path.name} <-> {mask_path.name}"
            )
        else:
            logger.warning(
                f"No mask found for image: {img_path.name}"
            )

    return pairs


def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            points = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target

                    # Conversion necessary because draw_bounding_boxes() only
                    # work with this specific format.
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat("xyxyxyxy")(boxes)
                elif isinstance(target, tv_tensors.KeyPoints):
                    points = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(
                    img, boxes, colors="yellow", width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(
                    img, masks.to(
                        torch.bool
                    ), colors=["green"] * masks.shape[0], alpha=.65)
            if points is not None:
                img = draw_keypoints(img, points, colors="red", radius=10)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
