from materials_vision.config import DATA_TRAIN_TEST
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
import re


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


def extract_magnification_from_filename(filename: str) -> Optional[int]:
    """
    Extract magnification value from filename.

    The filename pattern is expected to be:
    [optional_prefix]SAMPLE_MAGNIFICATION_NUMBER_jpg.rf.HASH_masks.tif

    Examples
    --------
    >>> extract_magnification_from_filename("0ab7de9d-AS2_40_10_jpg.rf.209a8405481b2434b8436c3f3acd60fd_masks.tif")
    40
    >>> extract_magnification_from_filename("AS2_40_10_jpg.rf.209a8405481b2434b8436c3f3acd60fd_masks.tif")
    40
    >>> extract_magnification_from_filename("sample_100_5_jpg.rf.hash_masks.tif")
    100

    Parameters
    ----------
    filename : str
        The filename to parse

    Returns
    -------
    Optional[int]
        Magnification value if found, None otherwise
    """
    # Pattern to match: anything followed by underscore, then 2-4 digits
    # (magnification), then underscore, then digit(s), then _jpg.rf.
    # This captures the magnification value from patterns like "AS2_40_10_jpg.rf."
    pattern = r'_(\d{2,4})_\d+_jpg\.rf\.'

    match = re.search(pattern, filename)
    if match:
        magnification = int(match.group(1))
        logger.debug(f"Extracted magnification {magnification} from {filename}")
        return magnification

    logger.warning(f"Could not extract magnification from filename: {filename}")
    return None
