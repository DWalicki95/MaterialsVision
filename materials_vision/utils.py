from materials_vision.config import DATA_TRAIN_TEST
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging


logger = logging.getLogger(__name__)


def get_train_and_test_dir(dataset_name: str):
    '''
    Returns train and test directories that are neccesary f.e. for cellpose
    model retraining.
    '''
    common_dir = DATA_TRAIN_TEST / dataset_name
    train_dir = common_dir / 'train'
    test_dir = common_dir / 'train'
    return str(train_dir), str(test_dir)


def create_current_time_output_directory(dir_base_path: Path):
    '''Creates output directory for f.e. for trained model'''
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(dir_base_path) / f"output_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
        Directory containing images and masks
    image_suffix : str, optional
        Suffix for image files (default: "_image.jpg")
    mask_suffix : str, optional
        Suffix for mask files (default: "_masks.tif")

    Returns
    -------
    List[Dict[str, Path]]
        List of dicts with 'image', 'mask', and 'base_name' keys
    """
    input_dir = Path(input_dir)
    pairs = []

    # Find all image files
    image_pattern = f"*{image_suffix}"
    for img_path in input_dir.glob(image_pattern):
        # Extract base name (remove suffix)
        base_name = img_path.stem.replace(
            image_suffix.split('.')[0], ''
        )

        # Look for corresponding mask
        mask_pattern = f"{base_name}{mask_suffix}"
        mask_path = input_dir / mask_pattern

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
