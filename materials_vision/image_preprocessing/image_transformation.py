from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import (
    hflip, vflip, adjust_contrast, adjust_gamma
)
from torchvision.transforms.v2 import functional as F

from materials_vision.image_preprocessing.helpers import plot


class Augmentor:
    """
    Image augmentation for Cellpose segmentation training.

    Applies geometric and intensity transformations to images and
    corresponding masks while maintaining their spatial correspondence.

    Parameters
    ----------
    rotation_angles : List[int], optional
        Angles for rotation augmentation (default: [90, 180, 270])
    contrast_range : Tuple[float, float], optional
        Range for contrast adjustment (default: (0.8, 1.2))
    gamma_range : Tuple[float, float], optional
        Range for gamma correction (default: (0.8, 1.2))
    poisson_noise_scale : float, optional
        Scale factor for Poisson noise (default: 1.0)

    Attributes
    ----------
    augmented_pairs : List[Tuple[Image.Image, Image.Image]]
        List of (image, mask) pairs after augmentation
    """

    def __init__(
        self,
        rotation_angles: List[int] = [90, 180, 270],
        contrast_range: Tuple[float, float] = (0.6, 1.2),
        gamma_range: Tuple[float, float] = (0.6, 1.4),
        poisson_noise_scale: float = 1.0
    ):
        self.rotation_angles = rotation_angles
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.poisson_noise_scale = poisson_noise_scale
        self.augmented_pairs: List[Tuple[Image.Image, Image.Image]] = []

    def rotate_image(
        self, img: Image.Image, mask: Image.Image, angle: int
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Rotate image and mask by specified angle.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask
        angle : int
            Rotation angle in degrees

        Returns
        -------
        Tuple[Image.Image, Image.Image]
            Rotated (image, mask) pair
        """
        rotated_img = img.rotate(angle, expand=True)
        rotated_mask = mask.rotate(angle, expand=True, fillcolor=0)
        return rotated_img, rotated_mask

    def flip_image(
        self, img: Image.Image, mask: Image.Image
    ) -> List[Tuple[Image.Image, Image.Image]]:
        """
        Generate horizontal, vertical, and combined flips.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask

        Returns
        -------
        List[Tuple[Image.Image, Image.Image]]
            List of flipped (image, mask) pairs
        """
        flipped = [
            (hflip(img), hflip(mask)),
            (vflip(img), vflip(mask)),
            (hflip(vflip(img)), hflip(vflip(mask)))
        ]
        return flipped

    def add_contrast(
        self, img: Image.Image, mask: Image.Image
    ) -> List[Tuple[Image.Image, Image.Image]]:
        """
        Apply contrast adjustments to image only.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask (unchanged)

        Returns
        -------
        List[Tuple[Image.Image, Image.Image]]
            List of contrast-adjusted (image, mask) pairs
        """
        lower_contrast = np.random.uniform(*self.contrast_range)
        higher_contrast = np.random.uniform(
            1.0, 2.0 + (1.0 - self.contrast_range[0])
        )

        contrasted = [
            (adjust_contrast(img, lower_contrast), mask),
            (adjust_contrast(img, higher_contrast), mask)
        ]
        return contrasted

    def add_gamma_correction(
        self, img: Image.Image, mask: Image.Image
    ) -> List[Tuple[Image.Image, Image.Image]]:
        """
        Apply gamma correction to image only.

        Generates two versions: one darker (lower gamma) and one brighter
        (higher gamma) to ensure visible differences for SEM images.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask (unchanged)

        Returns
        -------
        List[Tuple[Image.Image, Image.Image]]
            List of gamma-corrected (image, mask) pairs
        """
        # Lower gamma makes image darker (more aggressive range)
        # Sample from lower end of range to ensure visible darkening
        lower_gamma = np.random.uniform(
            self.gamma_range[0],
            (self.gamma_range[0] + 1.0) / 2  # Midpoint between min and 1.0
        )
        # Higher gamma makes image brighter
        # Sample from upper end of range to ensure visible brightening
        higher_gamma = np.random.uniform(
            (1.0 + self.gamma_range[1]) / 2,  # Midpoint between 1.0 and max
            self.gamma_range[1]
        )

        gamma_corrected = [
            (adjust_gamma(img, lower_gamma), mask),
            (adjust_gamma(img, higher_gamma), mask)
        ]
        return gamma_corrected

    def add_poisson_noise(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Add Poisson noise to image only.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask (unchanged)

        Returns
        -------
        Tuple[Image.Image, Image.Image]
            Noisy (image, mask) pair
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array *= self.poisson_noise_scale
        noisy = np.random.poisson(img_array * 255.0) / 255.0
        noisy = np.clip(noisy, 0, 1)
        noisy_img = Image.fromarray(
            (noisy * 255).astype(np.uint8)
        )
        return noisy_img, mask

    def augment(
        self,
        img: Image.Image,
        mask: Image.Image,
        apply_rotation: bool = True,
        apply_flip: bool = True,
        apply_contrast: bool = False,
        apply_gamma: bool = False,
        apply_poisson: bool = False
    ) -> List[Tuple[Image.Image, Image.Image]]:
        """
        Apply selected augmentations to image-mask pair.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask
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

        Returns
        -------
        List[Tuple[Image.Image, Image.Image]]
            List of augmented (image, mask) pairs
        """
        self.augmented_pairs = [(img, mask)]

        if apply_rotation:
            for angle in self.rotation_angles:
                pair = self.rotate_image(img, mask, angle)
                self.augmented_pairs.append(pair)

        if apply_flip:
            self.augmented_pairs.extend(self.flip_image(img, mask))

        if apply_contrast:
            self.augmented_pairs.extend(self.add_contrast(img, mask))

        if apply_gamma:
            self.augmented_pairs.extend(
                self.add_gamma_correction(img, mask)
            )

        if apply_poisson:
            self.augmented_pairs.append(
                self.add_poisson_noise(img, mask)
            )

        return self.augmented_pairs

    def visualize_augmentations(
        self,
        img: Image.Image,
        mask: Image.Image,
        apply_rotation: bool = False,
        apply_flip: bool = False,
        apply_contrast: bool = False,
        apply_gamma: bool = False,
        apply_poisson: bool = False,
        show_masks: bool = True,
        max_display: int = 10
    ) -> None:
        """
        Visualize selected augmentations for quick inspection.

        Parameters
        ----------
        img : Image.Image
            Input image
        mask : Image.Image
            Segmentation mask
        apply_rotation : bool, optional
            Apply rotation augmentation (default: False)
        apply_flip : bool, optional
            Apply flip augmentation (default: False)
        apply_contrast : bool, optional
            Apply contrast augmentation (default: False)
        apply_gamma : bool, optional
            Apply gamma correction (default: False)
        apply_poisson : bool, optional
            Apply Poisson noise (default: False)
        show_masks : bool, optional
            Overlay masks on images (default: True)
        max_display : int, optional
            Maximum number of augmentations to display (default: 10)

        Notes
        -----
        Displays original and augmented versions in a grid using matplotlib.
        When show_masks=True, masks are overlaid on images for visual
        inspection. When show_masks=False, only images are displayed
        (useful for inspecting intensity transformations like contrast,
        gamma, and noise).
        """
        # Generate augmented pairs
        augmented_pairs = self.augment(
            img, mask,
            apply_rotation=apply_rotation,
            apply_flip=apply_flip,
            apply_contrast=apply_contrast,
            apply_gamma=apply_gamma,
            apply_poisson=apply_poisson
        )

        # Limit display count
        display_pairs = augmented_pairs[:max_display]

        # Convert PIL images to tensors for visualization
        visualization_data = []
        labels = []

        for idx, (aug_img, aug_mask) in enumerate(display_pairs):
            # Convert image to tensor
            img_tensor = F.to_image(aug_img)
            img_tensor = F.to_dtype(img_tensor, torch.uint8, scale=True)

            if show_masks:
                # Convert mask to binary tensor
                mask_array = np.array(aug_mask)
                # Create binary mask (assuming mask has values > 0 for objects)
                mask_tensor = torch.from_numpy(mask_array > 0).unsqueeze(0)

                # Create tuple with image and mask dict for plot function
                visualization_data.append(
                    (img_tensor, {"masks": mask_tensor})
                )
            else:
                # Add only the image without mask overlay
                visualization_data.append(img_tensor)

            # Create label
            if idx == 0:
                labels.append("Original")
            else:
                labels.append(f"Aug {idx}")

        # Create grid layout (2 rows if more than 5 images)
        if len(visualization_data) <= 5:
            grid = [visualization_data]
            row_titles = None
        else:
            mid = (len(visualization_data) + 1) // 2
            grid = [
                visualization_data[:mid],
                visualization_data[mid:]
            ]
            row_titles = ["Original + Augmentations", "More Augmentations"]

        # Plot using the helper function
        plot(grid, row_title=row_titles)
        plt.show()
        plt.close()

    def save_augmented_dataset(
        self,
        output_dir: Path,
        base_name: str,
        img_suffix: str = "_image",
        mask_suffix: str = "_masks",
        img_format: str = "jpg"
    ) -> None:
        """
        Save augmented image-mask pairs to directory.

        Parameters
        ----------
        output_dir : Path
            Directory to save augmented images
        base_name : str
            Base name for saved files
        img_suffix : str, optional
            Suffix for image files (default: "_image")
        mask_suffix : str, optional
            Suffix for mask files (default: "_masks")
        img_format : str, optional
            Image file format (default: "jpg")

        Notes
        -----
        Images are saved in specified format, masks as TIF format.
        Naming follows Cellpose convention: basename_augN_image.jpg
        and basename_augN_masks.tif
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (aug_img, aug_mask) in enumerate(
            self.augmented_pairs
        ):
            img_path = (
                output_dir / f"{base_name}_aug{idx}{img_suffix}.{img_format}"
            )
            mask_path = (
                output_dir / f"{base_name}_aug{idx}{mask_suffix}.tif"
            )

            aug_img.save(img_path)
            aug_mask.save(mask_path, compression="tiff_deflate")
