"""Reading the ground truth of the classically segmented images.

The annotation is read exactly as it was for the segmentation models:
the frozen split and manifest, the instance masks they point to, and
the content crop with its fragment rule, all through ``SampleSource``.
A classical figure and a model figure therefore rest on the same
annotated pores, not on a second rasterization of the same polygons.

The classical predictions exist for a handful of images, chosen before
the split was drawn, so they fall into whichever subsets the split put
their formulations in. Each subset is opened separately and narrowed
to those images. Some of them may sit in TEST. No model is scored here
and nothing is chosen, but TEST is still read, so it has to be
unlocked explicitly and the reader logs that it was.
"""
import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from materials_vision.data.samples import SampleSource, read_manifest
from materials_vision.data.split_io import (SUBSETS, LockedTestSetError,
                                            SplitLoadError, load_split)
from materials_vision.training import A_MIN_FRAGMENT_PX2

logger = logging.getLogger(__name__)

TEST_SUBSET = "test"


def sources_for_images(
    split_csv: Path,
    manifest_csv: Path,
    image_ids: Iterable[str],
    *,
    allow_test: bool = False,
) -> tuple[SampleSource, ...]:
    """Sample sources covering exactly the given images.

    Parameters
    ----------
    split_csv, manifest_csv : Path
    image_ids : iterable of str
    allow_test : bool, optional
        Required when any of the images belongs to TEST.

    Returns
    -------
    tuple of SampleSource
        One per subset that holds any of the images, in split order.

    Raises
    ------
    LockedTestSetError
        If some of the images are in TEST and ``allow_test`` is False.
    SplitLoadError
        If some image is absent from the split or excluded from use
        there; scoring the rest would quietly change the population.
    """
    wanted = set(image_ids)
    membership = _subset_membership(split_csv, wanted)
    in_test = sorted(
        image_id for image_id, subset in membership.items()
        if subset == TEST_SUBSET
    )
    if in_test and not allow_test:
        raise LockedTestSetError(
            f"{len(in_test)} image(s) with classical predictions belong "
            f"to TEST: {in_test}. Pass --unlock-test to read their "
            f"annotation; no model is scored on it."
        )

    manifest = read_manifest(manifest_csv)
    sources = []
    for subset_name in SUBSETS:
        if subset_name not in membership.values():
            continue
        subset = load_split(
            split_csv, subset_name,
            allow_test=subset_name == TEST_SUBSET,
        )
        narrowed = subset.table[subset.table["image_id"].isin(wanted)]
        sources.append(SampleSource(
            replace(subset, table=narrowed.reset_index(drop=True)),
            manifest,
            min_fragment_area_px2=A_MIN_FRAGMENT_PX2,
        ))
    return tuple(sources)


def _subset_membership(
    split_csv: Path, wanted: set[str]
) -> dict[str, str]:
    """Which subset each wanted image is in, read from the split table.

    Only the ``split`` and ``used`` columns are read, so learning that
    an image is in TEST does not open TEST.
    """
    if not split_csv.exists():
        raise SplitLoadError(f"Split table not found: {split_csv}")
    table = pd.read_csv(split_csv, usecols=["image_id", "split", "used"])
    rows = table[table["image_id"].isin(wanted)]
    unused = sorted(rows.loc[~rows["used"].astype(bool), "image_id"])
    if unused:
        raise SplitLoadError(
            f"Image(s) {unused} are excluded from use in {split_csv}, "
            f"so they have no evaluable annotation."
        )
    absent = sorted(wanted - set(rows["image_id"]))
    if absent:
        raise SplitLoadError(
            f"Image(s) {absent} have predictions but no row in "
            f"{split_csv}."
        )
    return dict(zip(rows["image_id"], rows["split"]))
