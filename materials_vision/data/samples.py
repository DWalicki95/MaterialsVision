"""
Turning one row of the frozen split into a training-ready sample.

This is the deterministic half of the input pipeline, and deliberately
knows nothing about torch. It reads an image and its instance mask,
removes the microscope's information panel, rebuilds the instances the
cut passed through, and hands back a single working channel with a
label image numbered ``1..n``.

Everything after that point - augmentation, and the resize, padding
and normalization the model performs on its own input - happens
elsewhere. Keeping this layer pure means the part with the domain
rules in it can be tested without a GPU, and the same loading path
serves evaluation, which needs the ground truth at content resolution
rather than the decoder targets training wants.

**One working channel.** The images are monochrome. Those from one
microscope are stored as RGB with three identical channels, those from
the other as single-channel grayscale; both collapse to one channel
here, and the triplication the model expects is applied at the very
end of the pipeline rather than carried through it.

**Why the manifest is needed alongside the split.** The split table
says which images belong to which set, but the crop box, the mask path
and the source dimensions live in the manifest. Joining them here
keeps a single source of truth for each fact instead of copying the
crop rule into a second file.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from materials_vision.data.instances import apply_content_crop, parse_crop_bbox
from materials_vision.data.masks import load_instance_mask
from materials_vision.data.split_io import SplitSubset

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "image_id", "load_crop_bbox", "mask_path", "width_px", "height_px",
    "n_channels", "channels_identical", "n_instances", "q_max_i",
)

GRAYSCALE_MODE = "L"

RGB_MODE = "RGB"


class SampleSourceError(ValueError):
    """Raised when the split and the manifest cannot be reconciled, or
    a sample on disk contradicts what the manifest records."""


@dataclass(frozen=True)
class SampleRecord:
    """Everything known about one sample before it is read.

    Parameters
    ----------
    index : int
        Position within the source, i.e. the index a sampler yields.
    image_id : str
    formulation : str
        Grouping unit of the split; needed for per-formulation
        reporting.
    material : str
    microscope : str
    scale_bin : str
    pixel_size_um : float
        Physical scale, in micrometres per pixel.
    q_max_i : float
        Largest scale factor this image may be magnified by before it
        would exceed the finest resolution present in the dataset.
        Unused until scale augmentation exists; carried here so that
        adding it later does not mean re-plumbing this layer.
    source_path : Path
    mask_path : Path
    crop_bbox : tuple of int
        ``(x0, y0, x1, y1)`` content region to crop to.
    n_instances_expected : int
        Instance count the manifest recorded for this image, used to
        detect a mask that no longer matches the manifest it was built
        from.
    """

    index: int
    image_id: str
    formulation: str
    material: str
    microscope: str
    scale_bin: str
    pixel_size_um: float
    q_max_i: float
    source_path: Path
    mask_path: Path
    crop_bbox: tuple[int, int, int, int]
    n_instances_expected: int


@dataclass(frozen=True)
class PreparedSample:
    """One image/label pair, cropped and ready for augmentation.

    Parameters
    ----------
    record : SampleRecord
    image : np.ndarray
        ``(H, W)`` uint8, one working channel, cropped to the content
        region.
    labels : np.ndarray
        ``(H, W)`` instance labels numbered ``1..n_instances`` with 0
        as background.
    border_instance : np.ndarray
        Boolean per instance, indexed by ``id - 1``: whether it touches
        an edge of the cropped frame. Such instances have a truncated
        shape, so morphological measurements on them are meaningless.
    n_cut_by_crop : int
    n_dropped_below_min_area : int
    n_dropped_disconnected : int
        Bookkeeping from the crop, so a run can report what removing
        the panel cost instead of leaving it invisible.
    """

    record: SampleRecord
    image: np.ndarray
    labels: np.ndarray
    border_instance: np.ndarray
    n_cut_by_crop: int
    n_dropped_below_min_area: int
    n_dropped_disconnected: int

    @property
    def n_instances(self) -> int:
        """Instances surviving the crop.

        Returns
        -------
        int
        """
        return int(self.border_instance.size)


class SampleSource:
    """Reads samples of one split subset, in a fixed order.

    Parameters
    ----------
    subset : SplitSubset
        Output of ``split_io.load_split``.
    manifest : pandas.DataFrame
        The frozen manifest the split was derived from.
    min_fragment_area_px2 : float
        Smallest area a crop-truncated instance may keep and still
        count as an instance.
    check_dense_ids : bool, optional
        Passed to the mask reader.

    Raises
    ------
    SampleSourceError
        If the manifest lacks a required column, does not cover every
        image of the subset, or records a three-channel image whose
        channels are not identical - collapsing such an image to one
        channel would silently discard information.
    """

    def __init__(
        self,
        subset: SplitSubset,
        manifest: pd.DataFrame,
        *,
        min_fragment_area_px2: float,
        check_dense_ids: bool = True,
    ) -> None:
        self._min_fragment_area_px2 = float(min_fragment_area_px2)
        self._check_dense_ids = check_dense_ids
        self._records = _build_records(subset, manifest)
        logger.info(
            "Sample source over %s of split %s: %d image(s).",
            subset.subset.upper(), subset.split_id, len(self._records),
        )

    def __len__(self) -> int:
        return len(self._records)

    def record(self, index: int) -> SampleRecord:
        """Return the record of one sample without reading any file.

        Parameters
        ----------
        index : int

        Returns
        -------
        SampleRecord
        """
        return self._records[index]

    @property
    def records(self) -> tuple[SampleRecord, ...]:
        """All records, in sampling order.

        Returns
        -------
        tuple of SampleRecord
        """
        return self._records

    def load(self, index: int) -> PreparedSample:
        """Read one sample and prepare it for augmentation.

        Parameters
        ----------
        index : int

        Returns
        -------
        PreparedSample

        Raises
        ------
        SampleSourceError
            If the image is missing, is not one of the expected
            monochrome layouts, disagrees with the dimensions the
            manifest records, or carries a different number of
            instances than the manifest expects.
        """
        record = self._records[index]
        image = _read_working_channel(record)
        labels = load_instance_mask(
            record.mask_path,
            expected_shape=image.shape,
            check_dense_ids=self._check_dense_ids,
        )

        cropped = apply_content_crop(
            image, labels, record.crop_bbox,
            min_fragment_area_px2=self._min_fragment_area_px2,
        )
        if cropped.n_input_instances != record.n_instances_expected:
            raise SampleSourceError(
                f"{record.image_id}: mask holds "
                f"{cropped.n_input_instances} instance(s) but the "
                f"manifest records {record.n_instances_expected}. The "
                f"mask and the manifest are out of sync; rebuild the "
                f"masks with scripts/build_instance_masks.py."
            )

        return PreparedSample(
            record=record,
            image=cropped.image,
            labels=cropped.labels,
            border_instance=cropped.border_instance,
            n_cut_by_crop=cropped.n_cut_by_crop,
            n_dropped_below_min_area=cropped.n_dropped_below_min_area,
            n_dropped_disconnected=cropped.n_dropped_disconnected,
        )


def _build_records(
    subset: SplitSubset, manifest: pd.DataFrame
) -> tuple[SampleRecord, ...]:
    """Join the subset with the manifest into per-sample records."""
    missing_columns = [
        c for c in MANIFEST_COLUMNS if c not in manifest.columns
    ]
    if missing_columns:
        raise SampleSourceError(
            f"Manifest is missing column(s) the sample source needs: "
            f"{missing_columns}"
        )

    # Both tables carry n_instances. They agree in practice, but a
    # silent merge would let the split's copy win; the manifest is the
    # authoritative record, so drop the duplicate before joining
    # rather than depend on which side pandas prefers.
    duplicated = [
        c for c in MANIFEST_COLUMNS
        if c != "image_id" and c in subset.table.columns
    ]
    joined = subset.table.drop(columns=duplicated).merge(
        manifest[list(MANIFEST_COLUMNS)], on="image_id", how="left",
    )
    unmatched = joined.loc[
        joined["load_crop_bbox"].isna(), "image_id"
    ].tolist()
    if unmatched:
        raise SampleSourceError(
            f"{len(unmatched)} image(s) of subset {subset.subset!r} "
            f"are absent from the manifest, e.g. {unmatched[:5]}. The "
            f"split and the manifest must describe the same dataset."
        )

    _check_channels(joined)

    records = []
    for position, row in enumerate(joined.itertuples(index=False)):
        crop_bbox = parse_crop_bbox(row.load_crop_bbox)
        if crop_bbox is None:
            raise SampleSourceError(
                f"{row.image_id}: the manifest records no "
                f"load_crop_bbox, so there is no content region to "
                f"crop to"
            )
        records.append(
            SampleRecord(
                index=position,
                image_id=str(row.image_id),
                formulation=str(row.formulation),
                material=str(row.material),
                microscope=str(row.microscope),
                scale_bin=str(row.scale_bin),
                pixel_size_um=float(row.pixel_size_um),
                q_max_i=float(row.q_max_i),
                source_path=Path(row.source_path),
                mask_path=Path(row.mask_path),
                crop_bbox=crop_bbox,
                n_instances_expected=int(row.n_instances),
            )
        )
    return tuple(records)


def _check_channels(joined: pd.DataFrame) -> None:
    """Verify every multi-channel image is genuinely monochrome.

    Checked once over the whole subset rather than per sample: the
    answer cannot change between epochs, and paying for it on every
    read would be pointless.
    """
    multichannel = joined[joined["n_channels"] > 1]
    if multichannel.empty:
        return
    suspect = multichannel[~multichannel["channels_identical"].eq(True)]
    if not suspect.empty:
        raise SampleSourceError(
            f"{len(suspect)} image(s) have more than one channel "
            f"without identical channel content, e.g. "
            f"{suspect['image_id'].tolist()[:5]}. Collapsing them to "
            f"one working channel would silently discard information."
        )


def _read_working_channel(record: SampleRecord) -> np.ndarray:
    """Read an image and reduce it to its single working channel.

    Parameters
    ----------
    record : SampleRecord

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8.

    Raises
    ------
    SampleSourceError
        If the file is missing or stored in an unexpected mode.
    """
    if not record.source_path.exists():
        raise SampleSourceError(
            f"{record.image_id}: image not found at "
            f"{record.source_path}"
        )
    with Image.open(record.source_path) as handle:
        mode = handle.mode
        if mode not in (GRAYSCALE_MODE, RGB_MODE):
            raise SampleSourceError(
                f"{record.image_id}: image mode {mode!r} is neither "
                f"{GRAYSCALE_MODE!r} nor {RGB_MODE!r}; these are "
                f"monochrome SEM images and no other layout is "
                f"expected"
            )
        array = np.asarray(handle)

    if mode == RGB_MODE:
        array = array[:, :, 0]
    return np.ascontiguousarray(array)


def read_manifest(path: Path, columns: Optional[list[str]] = None):
    """Read the frozen manifest for use with ``SampleSource``.

    A thin helper so callers do not have to remember that the sample
    source needs the manifest as well as the split.

    Parameters
    ----------
    path : Path
    columns : list of str, optional
        Restrict to these columns; defaults to all.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    SampleSourceError
        If the file does not exist.
    """
    if not path.exists():
        raise SampleSourceError(f"Manifest not found: {path}")
    return pd.read_csv(path, usecols=columns)
