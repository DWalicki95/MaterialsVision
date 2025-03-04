from materials_vision.config import DATA_TRAIN_TEST
from pathlib import Path
import random
import shutil
import logging
from tqdm import tqdm


log = logging.getLogger(__name__)


def split_into_train_and_test_directory(
        dataset_store_path: Path,
        train_size: float = 0.8,
        dataset_name: str = 'synthetic_dataset_',
        image_suffix_convention: str = '_image',
        image_format: str = '.tiff',
        mask_suffix_convention: str = '_masks',
        mask_format: str = '.tiff'

):
    dataset_train_test_dir = DATA_TRAIN_TEST / dataset_name
    train_dir = dataset_train_test_dir / 'train'
    test_dir = dataset_train_test_dir / 'test'

    image_suffix = image_suffix_convention + image_format
    mask_suffix = mask_suffix_convention + mask_format

    all_images = list(dataset_store_path.glob(f'*{image_suffix}'))
    all_masks = list(dataset_store_path.glob(f'*{mask_suffix}'))

    if len(all_images) != len(all_masks):
        raise ValueError(
            f'Number of images and masks in {dataset_store_path} are not equal'
        )
    # randomly choose train files
    train_files_num = int(train_size * len(all_images))
    train_files = random.sample(all_images, train_files_num)
    test_files = [img for img in all_images if img not in train_files]
    # find matching masks
    train_masks = find_matching_masks(train_files, all_masks, mask_suffix)
    test_masks = find_matching_masks(test_files, all_masks, mask_suffix)
    # put files into desired directories
    copy_into_desired_directory(train_files, train_dir)
    copy_into_desired_directory(train_masks, train_dir)
    copy_into_desired_directory(test_files, test_dir)
    copy_into_desired_directory(test_masks, test_dir)
    log.info(
        'Split dataset into train and test and copied into desired catalogs'
    )


def find_matching_masks(
    files_list: list[Path],
    all_masks_list: list[Path],
    mask_suffix: str
) -> list[Path]:
    """Filters files to those that match images.

    Parameters
    ----------
    files_list : list[Path]
        List of all train or test files' paths (mask and images)
    all_masks_list : list[Path]
        List of all files' paths in dataset
    mask_suffix : str
        mask suffix in mask name convention, f.e. `_masks.tiff`

    Returns
    -------
    list[Path]
        List of chosen images masks.
    """
    matching_mask_paths = []
    for file in files_list:
        chosen_img_num = file.name.split('_')[-2]
        # name convention must be as below !
        chosen_mask_name = 'sample_' + str(chosen_img_num) + mask_suffix
        mask_path = get_full_mask_path(chosen_mask_name, all_masks_list)
        matching_mask_paths.append(mask_path)
    return matching_mask_paths


def get_full_mask_path(
    chosen_mask_name: str,
    all_masks_list: list[Path]
) -> Path:
    """Function tasks:
    1) Make sure if mask name with provided convention really exists.
    2) If file exists -> returns full path
    3) If not -> raise error

    Parameters
    ----------
    chosen_mask_name : str
        mask name created by `find_matching_mask` function with a specific
        convention
    all_masks_list : list[Path]
        List of all files' paths in dataset

    Returns
    -------
    Path
        Full path of mask

    Raises
    ------
    ValueError
        Error if file that should be in directory is not present. It might be
        changed name convention fault.
    """
    for mask in all_masks_list:
        if mask.name == chosen_mask_name:
            return mask
    raise ValueError(
            f'There is no mask {chosen_mask_name} in dataset.'
        )


def copy_into_desired_directory(files_list: list[Path], desired_folder: Path):
    '''Copy files from files_list into desired_folder directory'''
    desired_folder.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files_list, desc='Kopiowanie plików...'):
        shutil.copy2(file, desired_folder / file.name)
