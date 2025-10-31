from pathlib import Path
from typing import List
import logging

from PIL import Image

from materials_vision.image_preprocessing.image_transformation import (
    Augmentor
)
from materials_vision.utils import find_image_mask_pairs


logger = logging.getLogger(__name__)


def augment_dataset(
    input_dir: Path,
    image_suffix: str = "_image.jpg",
    mask_suffix: str = "_masks.tif",
    output_dir: Path = None,
    save_dataset: bool = True,
    apply_rotation: bool = True,
    apply_flip: bool = True,
    apply_contrast: bool = False,
    apply_gamma: bool = False,
    apply_poisson: bool = False,
    rotation_angles: List[int] = [90, 180, 270],
    contrast_range: tuple = (0.8, 1.2),
    gamma_range: tuple = (0.8, 1.2),
    poisson_noise_scale: float = 1.0
) -> Path:
    """
    Augment all image-mask pairs in directory and save results.

    Parameters
    ----------
    input_dir : Path
        Directory containing images and masks
    image_suffix : str, optional
        Suffix for image files (default: "_image.jpg")
    mask_suffix : str, optional
        Suffix for mask files (default: "_masks.tif")
    output_dir : Path, optional
        Output directory (default: parent/(dirname + '_augmented'))
    save_dataset : bool, optional
        Whether to save augmented images to disk (default: True)
    apply_rotation : bool, optional
        Apply rotation augmentation (default: True)
    apply_flip : bool, optional
        Apply flip augmentation (default: True)
    apply_contrast : bool, optional
        Apply contrast augmentation (default: False)
    apply_gamma : bool, optional
        Apply gamma correction (default: False)
    apply_poisson : bool, optional
        Apply Poisson noise (default: False)
    rotation_angles : List[int], optional
        Angles for rotation (default: [90, 180, 270])
    contrast_range : tuple, optional
        Range for contrast adjustment (default: (0.8, 1.2))
    gamma_range : tuple, optional
        Range for gamma correction (default: (0.8, 1.2))
    poisson_noise_scale : float, optional
        Scale factor for Poisson noise (default: 1.0)

    Returns
    -------
    Path
        Output directory path

    Notes
    -----
    Output directory is automatically created as:
    parent_dir / (current_dir_name + '_augmented')
    """
    input_dir = Path(input_dir)

    if save_dataset:
        if output_dir is None:
            output_dir = (
                input_dir.parent / f"{input_dir.name}_augmented"
            )
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    else:
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_augmented"
        else:
            output_dir = Path(output_dir)
        logger.info("Augmentation without saving to disk")

    # Find all image-mask pairs
    pairs = find_image_mask_pairs(
        input_dir, image_suffix, mask_suffix
    )

    if not pairs:
        logger.error(f"No image-mask pairs found in {input_dir}")
        return output_dir

    logger.info(f"Found {len(pairs)} image-mask pairs")

    # Initialize augmentor
    augmentor = Augmentor(
        rotation_angles=rotation_angles,
        contrast_range=contrast_range,
        gamma_range=gamma_range,
        poisson_noise_scale=poisson_noise_scale
    )

    # Process each pair
    total_augmented = 0
    for idx, pair in enumerate(pairs, 1):
        img_path = pair['image']
        mask_path = pair['mask']
        base_name = pair['base_name']

        logger.info(
            f"Processing [{idx}/{len(pairs)}]: {base_name}"
        )

        # Load image and mask
        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
        except Exception as e:
            logger.error(
                f"Error loading {base_name}: {str(e)}"
            )
            continue

        # Apply augmentations
        try:
            augmented_pairs = augmentor.augment(
                img, mask,
                apply_rotation=apply_rotation,
                apply_flip=apply_flip,
                apply_contrast=apply_contrast,
                apply_gamma=apply_gamma,
                apply_poisson=apply_poisson
            )
        except Exception as e:
            logger.error(
                f"Error augmenting {base_name}: {str(e)}"
            )
            continue

        # Save augmented dataset if enabled
        if save_dataset:
            try:
                # Detect image format from original file
                img_format = img_path.suffix.lstrip('.')

                augmentor.save_augmented_dataset(
                    output_dir=output_dir,
                    base_name=base_name,
                    img_suffix="_image",
                    mask_suffix="_masks",
                    img_format=img_format
                )
            except Exception as e:
                logger.error(
                    f"Error saving {base_name}: {str(e)}"
                )
                continue

        total_augmented += len(augmented_pairs)
        logger.info(
            f"  Generated {len(augmented_pairs)} augmented pairs"
        )

    if save_dataset:
        logger.info(
            f"\nAugmentation complete and saved!"
            f"\n  Input: {input_dir}"
            f"\n  Output: {output_dir}"
            f"\n  Original pairs: {len(pairs)}"
            f"\n  Total augmented: {total_augmented}"
        )
    else:
        logger.info(
            f"\nAugmentation complete (not saved)!"
            f"\n  Input: {input_dir}"
            f"\n  Original pairs: {len(pairs)}"
            f"\n  Total augmented: {total_augmented}"
        )

    return output_dir
