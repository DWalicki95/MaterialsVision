import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_pixel_sizes() -> dict:
    """
    Load SEM pixel size calibration from shared YAML.

    Returns
    -------
    dict
        Mapping of magnification (int) to pixel size in µm/px (float).
    """
    path = Path(__file__).parent / "config" / "sem_calibration.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["pixel_sizes"]


def create_current_time_output_directory(dir_base_path: Path):
    """
    Create timestamped output directory.

    Parameters
    ----------
    dir_base_path : Path
        Parent directory where the output directory will be created.

    Returns
    -------
    Path
        Path to the created directory.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(dir_base_path) / f"output_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_image_mask_pairs(
    image_dir: Path,
    mask_dir: Optional[Path] = None,
    image_suffix: str = "_image.jpg",
    mask_suffix: str = "_masks.tif",
    strict: bool = False,
) -> List[Dict[str, Path]]:
    """
    Find matching image-mask pairs by filename suffix.

    Parameters
    ----------
    image_dir : Path
        Directory containing image files.
    mask_dir : Path, optional
        Directory containing mask files (default: same as `image_dir`,
        for the common case where images and masks are co-located).
    image_suffix : str, optional
        Suffix (including extension) identifying image files
        (default: "_image.jpg").
    mask_suffix : str, optional
        Suffix (including extension) identifying mask files
        (default: "_masks.tif").
    strict : bool, optional
        If True, raise `ValueError` when any image has no matching mask
        or any mask has no matching image. If False, log a warning and
        skip unmatched images; unmatched masks are silently ignored
        (default: False).

    Returns
    -------
    List[Dict[str, Path]]
        List of dicts with 'image', 'mask', and 'base_name' keys, one
        per matched pair.

    Raises
    ------
    ValueError
        If `strict` is True and an image has no matching mask, or a
        mask has no matching image.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else image_dir

    pairs = []
    matched_mask_names = set()
    for img_path in sorted(image_dir.glob(f"*{image_suffix}")):
        base_name = img_path.name[:-len(image_suffix)]
        mask_path = mask_dir / f"{base_name}{mask_suffix}"

        if mask_path.exists():
            pairs.append({
                'image': img_path,
                'mask': mask_path,
                'base_name': base_name,
            })
            matched_mask_names.add(mask_path.name)
        elif strict:
            raise ValueError(
                f"No mask found for image: {img_path.name}"
            )
        else:
            logger.warning(
                f"No mask found for image: {img_path.name}"
            )

    if strict:
        orphan_masks = [
            mask_path.name
            for mask_path in sorted(mask_dir.glob(f"*{mask_suffix}"))
            if mask_path.name not in matched_mask_names
        ]
        if orphan_masks:
            raise ValueError(
                f"Masks with no matching image: {orphan_masks}"
            )

    return pairs
