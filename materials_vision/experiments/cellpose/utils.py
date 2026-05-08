import gc
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from cellpose import dynamics, io, train as _cp_train
from tqdm import tqdm

log = logging.getLogger(__name__)


def patch_cellpose_get_batch():
    """
    Patch ``cellpose.train._get_batch`` to accept
    and ignore the ``channel_axis`` keyword.

    Cellpose 4.0.8 passes ``channel_axis`` via
    ``**kwargs`` in file-based mode, but
    ``_get_batch`` does not accept it.
    """
    original = _cp_train._get_batch

    if getattr(original, "_mv_patched", False):
        return

    def _patched(*args, **kwargs):
        kwargs.pop("channel_axis", None)
        return original(*args, **kwargs)

    _patched._mv_patched = True
    _cp_train._get_batch = _patched
    log.debug(
        "Patched cellpose.train._get_batch "
        "for file-based mode compatibility"
    )


def get_train_test_file_paths(
    train_dir: str,
    test_dir: Optional[str] = None,
    image_filter: Optional[str] = None,
    mask_filter: str = "_masks",
    look_one_level_down: bool = False,
) -> Tuple[
    List[str], List[str],
    Optional[List[str]], Optional[List[str]],
]:
    """
    Gather image and mask file paths without loading data.

    Uses Cellpose's ``io.get_image_files`` and
    ``io.get_label_files`` to discover files on disk.

    Parameters
    ----------
    train_dir : str
        Path to training data directory.
    test_dir : str, optional
        Path to test data directory.
    image_filter : str, optional
        Suffix that identifies image files.
    mask_filter : str
        Suffix that identifies mask files.
    look_one_level_down : bool
        Search one level of subdirectories.

    Returns
    -------
    train_img_files : list of str
        Training image file paths.
    train_label_files : list of str
        Training mask file paths.
    test_img_files : list of str or None
        Test image file paths.
    test_label_files : list of str or None
        Test mask file paths.
    """
    train_img_files = io.get_image_files(
        train_dir,
        mask_filter,
        imf=image_filter,
        look_one_level_down=look_one_level_down,
    )
    train_label_files, _ = io.get_label_files(
        train_img_files,
        mask_filter,
        imf=image_filter,
    )
    if train_label_files is None:
        raise FileNotFoundError(
            f"No mask files found in {train_dir} "
            f"with filter '{mask_filter}'"
        )

    test_img_files = None
    test_label_files = None
    if test_dir is not None:
        test_img_files = io.get_image_files(
            test_dir,
            mask_filter,
            imf=image_filter,
            look_one_level_down=look_one_level_down,
        )
        test_label_files, _ = io.get_label_files(
            test_img_files,
            mask_filter,
            imf=image_filter,
        )
        if test_label_files is None:
            raise FileNotFoundError(
                f"No mask files found in {test_dir} "
                f"with filter '{mask_filter}'"
            )

    return (
        train_img_files,
        train_label_files,
        test_img_files,
        test_label_files,
    )


def precompute_flows_batched(
    label_files: List[str],
    image_files: List[str],
    batch_size: int = 10,
    device=None,
    redo_flows: bool = False,
) -> List[str]:
    """
    Compute flows in batches and save to disk.

    For each image, a ``_flows.tif`` file is saved next to
    the image file. Already-existing flow files are skipped
    unless ``redo_flows=True``.

    Parameters
    ----------
    label_files : list of str
        Paths to mask/label files.
    image_files : list of str
        Paths to corresponding image files.
    batch_size : int
        Number of images to process per batch.
    device : torch.device, optional
        Device for flow computation.
    redo_flows : bool
        Recompute even if flow files exist.

    Returns
    -------
    list of str
        Paths to the ``_flows.tif`` files.
    """
    import numpy as np

    nimg = len(label_files)
    flow_files = [
        os.path.splitext(f)[0] + "_flows.tif"
        for f in image_files
    ]

    if not redo_flows:
        todo_idx = [
            i for i in range(nimg)
            if not os.path.exists(flow_files[i])
        ]
    else:
        todo_idx = list(range(nimg))

    if not todo_idx:
        log.info(
            "All %d flow files already cached, "
            "skipping precomputation",
            nimg,
        )
        return flow_files

    log.info(
        "Pre-computing flows for %d / %d images "
        "(batch_size=%d)",
        len(todo_idx),
        nimg,
        batch_size,
    )

    for start in tqdm(
        range(0, len(todo_idx), batch_size),
        desc="Flow batches",
    ):
        batch_idx = todo_idx[start:start + batch_size]
        labels_batch = [
            io.imread(label_files[i]) for i in batch_idx
        ]
        img_files_batch = [
            image_files[i] for i in batch_idx
        ]

        # Ensure labels are 3D [1, H, W]
        labels_batch = [
            lbl if lbl.ndim >= 3
            else lbl[np.newaxis, :, :]
            for lbl in labels_batch
        ]

        dynamics.labels_to_flows(
            labels_batch,
            files=img_files_batch,
            device=device,
            return_flows=False,
        )

        del labels_batch
        gc.collect()

    log.info("Flow precomputation complete")
    return flow_files


def filter_by_min_masks(
    image_files: List[str],
    label_files: List[str],
    flow_files: List[str],
    min_train_masks: int = 5,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Remove images whose masks have fewer instances
    than ``min_train_masks``.

    Works around a Cellpose 4.0.8 bug where
    ``_process_train_test`` crashes in file-based
    mode after this filtering step.

    Parameters
    ----------
    image_files : list of str
        Image file paths.
    label_files : list of str
        Mask file paths.
    flow_files : list of str
        Flow file paths.
    min_train_masks : int
        Minimum number of mask instances required.

    Returns
    -------
    tuple of (list, list, list)
        Filtered image, label, and flow file paths.
    """
    import numpy as np
    from cellpose import utils as cp_utils

    keep_img = []
    keep_lbl = []
    keep_flw = []
    n_removed = 0

    for i in range(len(image_files)):
        lbl = io.imread(label_files[i])
        if lbl.ndim >= 3:
            lbl = lbl[0]
        _, dall = cp_utils.diameters(lbl)
        n_masks = len(dall)

        if n_masks >= min_train_masks:
            keep_img.append(image_files[i])
            keep_lbl.append(label_files[i])
            keep_flw.append(flow_files[i])
        else:
            n_removed += 1

    if n_removed > 0:
        log.warning(
            "%d images with fewer than %d masks "
            "removed from training set",
            n_removed,
            min_train_masks,
        )

    return keep_img, keep_lbl, keep_flw


def split_into_train_and_test_directory(
        dataset_store_path: Path,
        output_root: Path,
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
    output_root : Path
        Root directory under which the ``<dataset_name>/train`` and
        ``<dataset_name>/test`` subfolders are created.
    train_size : float, optional
        Fraction of pairs to put into `train` (default: 0.8).
    dataset_name : str, optional
        Subfolder name under ``output_root`` to hold ``train/`` and
        ``test/`` (default: 'synthetic_dataset_').
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
    base_out = Path(output_root) / dataset_name
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
