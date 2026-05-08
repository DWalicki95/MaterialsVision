import logging
import re
from typing import Optional


logger = logging.getLogger(__name__)


def extract_magnification_from_filename(filename: str) -> Optional[int]:
    """
    Extract magnification value from filename.

    The filename pattern is expected to be:
    [optional_prefix]SAMPLE_MAGNIFICATION_NUMBER_jpg.rf.HASH_masks.tif

    Examples
    --------
    >>> extract_magnification_from_filename(
    ...     "0ab7de9d-AS2_40_10_jpg.rf.209a_masks.tif"
    ... )
    40
    >>> extract_magnification_from_filename(
    ...     "sample_100_5_jpg.rf.hash_masks.tif"
    ... )
    100

    Parameters
    ----------
    filename : str
        The filename to parse.

    Returns
    -------
    Optional[int]
        Magnification value if found, None otherwise.
    """
    pattern = r'_(\d{2,4})_\d+_jpg\.rf\.'
    match = re.search(pattern, filename)
    if match:
        magnification = int(match.group(1))
        logger.debug(
            f"Extracted magnification {magnification} from {filename}"
        )
        return magnification

    logger.warning(
        f"Could not extract magnification from filename: {filename}"
    )
    return None
