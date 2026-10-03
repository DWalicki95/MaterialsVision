"""
Serving the frozen split to Cellpose's file-based trainer.

Cellpose trains from files on disk: an image, its instance mask, and a
flow field it derives from the mask once and stores beside the image.
The samples this study trains and scores on are not files, though -
they are what ``SampleSource.load`` produces, with the microscope's
information panel cropped away and the instances the cut passed through
rebuilt. Pointing Cellpose at the source images would train it on
different pixels and different instances than the model it is compared
against. So each sample is written out exactly as the source prepares
it, and Cellpose reads that.

**One cache for every split.** What a sample looks like depends only on
the manifest - the crop box and the mask - and not on which subset or
fold it belongs to. A single directory named by image therefore serves
the selection runs, the final model and every cross-validation fold,
and the flow field of an image is computed once however many runs use
it.

**Staleness is decided by content, not by bookkeeping.** A file is
rewritten only when what the source now prepares differs from what is
on disk, and a rewrite deletes the flow field derived from the old
version. Recording hashes in a sidecar would do the same job with one
more thing that can fall out of step with the files it describes.

**Only what a run trains on is written.** TEST is never exported by
the selection runs or the final model; it reaches the disk only as the
training material of the folds that hold other formulations out, and
through the same locked split reader as everything else.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from materials_vision.data.samples import SampleSource

logger = logging.getLogger(__name__)

IMAGE_SUFFIX = "_image.tif"

MASK_SUFFIX = "_masks.tif"

# Cellpose names the flow field after the image it belongs to, by
# replacing the image's extension with this suffix.
FLOWS_SUFFIX = "_flows.tif"


class CacheExportError(ValueError):
    """Raised when a source cannot be exported to the cache."""


@dataclass(frozen=True)
class CachedSubset:
    """The files one run hands to the Cellpose trainer.

    Parameters
    ----------
    image_ids : tuple of str
        In the source's sampling order.
    image_paths, mask_paths : tuple of Path
        Aligned with ``image_ids``.
    n_written : int
        Samples whose image or mask was written by this export, because
        it was new or had changed.
    n_flows_invalidated : int
        Flow fields deleted because the sample they were derived from
        changed. Cellpose recomputes them before training.
    """

    image_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    n_written: int
    n_flows_invalidated: int

    def __len__(self) -> int:
        return len(self.image_ids)


def image_path(cache_dir: Path, image_id: str) -> Path:
    """Where one sample's cropped image is cached.

    Parameters
    ----------
    cache_dir : Path
    image_id : str

    Returns
    -------
    Path
    """
    return cache_dir / f"{image_id}{IMAGE_SUFFIX}"


def mask_path(cache_dir: Path, image_id: str) -> Path:
    """Where one sample's instance mask is cached.

    Parameters
    ----------
    cache_dir : Path
    image_id : str

    Returns
    -------
    Path
    """
    return cache_dir / f"{image_id}{MASK_SUFFIX}"


def flows_path(image_file: Path) -> Path:
    """Where Cellpose stores the flow field derived for an image.

    Parameters
    ----------
    image_file : Path

    Returns
    -------
    Path
    """
    return image_file.with_name(image_file.stem + FLOWS_SUFFIX)


def export_source(source: SampleSource, cache_dir: Path) -> CachedSubset:
    """Write every sample of a source to the cache, as it prepares them.

    Parameters
    ----------
    source : SampleSource
        The subset a run trains on, already opened through the split
        reader, so the lock on TEST applies here as everywhere else.
    cache_dir : Path
        Created if missing.

    Returns
    -------
    CachedSubset

    Raises
    ------
    CacheExportError
        If the source is empty, or names one image twice - two samples
        sharing a file name would overwrite each other.
    """
    records = source.records
    if not records:
        raise CacheExportError("Cannot export an empty sample source.")
    image_ids = tuple(record.image_id for record in records)
    duplicated = sorted({i for i in image_ids if image_ids.count(i) > 1})
    if duplicated:
        raise CacheExportError(
            f"The source names {len(duplicated)} image(s) more than "
            f"once, e.g. {duplicated[:5]}."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_invalidated = 0
    for position, image_id in enumerate(image_ids):
        prepared = source.load(position)
        image_file = image_path(cache_dir, image_id)
        image_changed = _write_if_changed(image_file, prepared.image)
        mask_changed = _write_if_changed(
            mask_path(cache_dir, image_id), prepared.labels
        )
        if image_changed or mask_changed:
            n_written += 1
            n_invalidated += _remove_if_present(flows_path(image_file))

    logger.info(
        "Cache %s: %d sample(s), %d written, %d flow field(s) "
        "invalidated.", cache_dir, len(image_ids), n_written,
        n_invalidated,
    )
    return CachedSubset(
        image_ids=image_ids,
        image_paths=tuple(image_path(cache_dir, i) for i in image_ids),
        mask_paths=tuple(mask_path(cache_dir, i) for i in image_ids),
        n_written=n_written,
        n_flows_invalidated=n_invalidated,
    )


def _write_if_changed(path: Path, array: np.ndarray) -> bool:
    """Write an array unless the file already holds exactly it.

    The comparison includes the dtype: a mask read back at a different
    integer width is the same labelling, but the file would no longer
    be the one the source prepares, and "exactly what the source
    prepares" is the whole contract of this cache.

    Parameters
    ----------
    path : Path
    array : np.ndarray

    Returns
    -------
    bool
        Whether the file was written.
    """
    existing = _read_if_present(path)
    if (
        existing is not None
        and existing.dtype == array.dtype
        and np.array_equal(existing, array)
    ):
        return False
    # Written beside the target and moved into place, so an interrupted
    # export leaves either the old file or the new one, never a partial
    # file that a later comparison would read as a changed sample.
    partial = path.with_name(path.name + ".partial")
    tifffile.imwrite(partial, array)
    partial.replace(path)
    return True


def _read_if_present(path: Path) -> Optional[np.ndarray]:
    """Read a cached array, or nothing if there is none."""
    if not path.exists():
        return None
    return tifffile.imread(path)


def _remove_if_present(path: Path) -> int:
    """Delete a file if it exists; return how many were deleted."""
    if not path.exists():
        return 0
    path.unlink()
    return 1
