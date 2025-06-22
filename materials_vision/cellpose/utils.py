import logging
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from materials_vision.config import DATA_TRAIN_TEST

log = logging.getLogger(__name__)


def split_into_train_and_test_directory(
        dataset_store_path: Path,
        train_size: float = 0.8,
        dataset_name: str = 'synthetic_dataset_',
        image_suffix_convention: str = '_image',
        image_format: str = '.tiff',
        mask_suffix_convention: str = '_masks',
        mask_format: str = '.tiff',
        subset_size: int = None
):
    """
    Splits the dataset into training and testing directories and copies
    corresponding images and masks.

    This function performs the following steps:
      1. Scans `dataset_store_path` for images and masks based on specified
         suffix conventions.
      2. Constructs (image, mask) pairs by swapping the image suffix for the
         mask suffix.
      3. Optionally selects a random subset of those pairs if `subset_size`
         is provided.
      4. Splits the resulting pairs into training and testing sets by the
         `train_size` ratio.
      5. Copies each image and its matched mask into the appropriate directory.

    Parameters
    ----------
    dataset_store_path : Path
        Directory where the raw images and masks live.
    train_size : float, optional
        Fraction of pairs to put into `train` (default: 0.8).
    dataset_name : str, optional
        Subfolder name under `DATA_TRAIN_TEST` to hold `train/` and `test/`
         (default: 'synthetic_dataset_').
    image_suffix_convention : str, optional
        Suffix before the image file extension (default: '_image').
    image_format : str, optional
        Image file extension, including the dot (default: '.tiff').
    mask_suffix_convention : str, optional
        Suffix before the mask file extension (default: '_masks').
    mask_format : str, optional
        Mask file extension, including the dot (default: '.tiff').
    subset_size : int, optional
        If set, randomly draw this many image–mask pairs before splitting;
        must not exceed the total number of available pairs.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the counts of raw image files and raw mask files differ.
    ValueError
        If an image filename does not end with the specified image suffix.
    ValueError
        If `subset_size` is larger than the available number of pairs.
        If the expected mask filename for an image cannot be found.
    """
    base_out = DATA_TRAIN_TEST / dataset_name
    train_dir = base_out / 'train'
    test_dir = base_out / 'test'

    img_suffix = image_suffix_convention + image_format
    msk_suffix = mask_suffix_convention + mask_format

    all_images = list(dataset_store_path.glob(f'*{img_suffix}'))
    all_masks = list(dataset_store_path.glob(f'*{msk_suffix}'))

    if len(all_images) != len(all_masks):
        raise ValueError(
            f'Found {len(all_images)} images but {len(all_masks)} masks in '
            f'{dataset_store_path}'
        )

    # build guaranteed-matched pairs by swapping suffixes
    pairs: list[tuple[Path, Path]] = []
    for img in all_images:
        if not img.name.endswith(img_suffix):
            raise ValueError(
                f'Image file "{img.name}" does not end with "{img_suffix}"'
            )
        prefix = img.name[: -len(img_suffix)]
        expected_mask_name = prefix + msk_suffix
        mask_path = get_full_mask_path(expected_mask_name, all_masks)
        pairs.append((img, mask_path))

    # optionally sub‑sample
    if subset_size is not None:
        if subset_size > len(pairs):
            raise ValueError(
                f'subset_size ({subset_size}) exceeds available pairs '
                f'({len(pairs)})'
            )
        pairs = random.sample(pairs, subset_size)

    # split into train / test
    n_train = int(train_size * len(pairs))
    train_pairs = random.sample(pairs, n_train)
    test_pairs = [p for p in pairs if p not in train_pairs]

    # Unzip and copy
    train_imgs, train_msks = zip(*train_pairs) if train_pairs else ([], [])
    test_imgs,  test_msks = zip(*test_pairs) if test_pairs else ([], [])

    copy_into_desired_directory(train_imgs, train_dir)
    copy_into_desired_directory(train_msks, train_dir)
    copy_into_desired_directory(test_imgs,  test_dir)
    copy_into_desired_directory(test_msks,  test_dir)

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
    for file in tqdm(files_list, desc='Copying files...'):
        shutil.copy2(file, desired_folder / file.name)
