"""
Reading the frozen dataset split at training and evaluation time.

Everything that needs to know which images belong to which set goes
through ``load_split``. Centralizing it has one purpose beyond
convenience: the test set can only be guarded by whoever opens the
file, so the lock lives here rather than in any single dataloader.

The split itself is produced once by
``scripts/create_dataset_split.py`` and then reused unchanged by every
training run, so that results from different runs stay comparable. It
is never regenerated per run.
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUBSETS: tuple[str, str, str] = ("train", "val", "test")

REQUIRED_COLUMNS = (
    "image_id", "formulation", "material", "microscope", "scale_bin",
    "source_path", "split", "used",
)


class SplitLoadError(ValueError):
    """Raised when the split table is missing, malformed, or does not
    match the manifest it was derived from."""


class LockedTestSetError(RuntimeError):
    """Raised on any attempt to read TEST without unlocking it.

    TEST is held back untouched until the very end of the experiment
    and opened exactly once, to produce the headline numbers. It must
    take part in no decision along the way - every choice of
    hyperparameter, checkpoint or augmentation policy is made on
    VALIDATION - because a test set consulted during development stops
    measuring generalization and starts measuring how well the
    development loop fitted it. Requiring an explicit flag makes an
    accidental read impossible and a deliberate one visible in the
    log.
    """


@dataclass(frozen=True)
class SplitSubset:
    """One set of the frozen split, ready to build a dataset from.

    Parameters
    ----------
    split_id : str
        Identifier of the split, e.g. ``"split_v1"``.
    subset : str
        ``train``, ``val`` or ``test``.
    table : pandas.DataFrame
        The subset's rows, sorted by ``image_id`` so the order never
        depends on how the CSV happened to be written.
    n_excluded_unused : int
        Rows of this subset dropped because ``used`` was False, i.e.
        ``scale_outlier`` images whose formulation landed outside
        TRAIN. They follow their formulation and are discarded rather
        than relocated: all images of one formulation come from a
        single synthesis and are strongly correlated, so putting some
        in TRAIN and others in an evaluation set would let the model
        be scored on material it effectively trained on.
    """

    split_id: str
    subset: str
    table: pd.DataFrame
    n_excluded_unused: int

    def __len__(self) -> int:
        return len(self.table)

    @property
    def image_ids(self) -> tuple[str, ...]:
        """Image identifiers, in sampling order.

        Returns
        -------
        tuple of str
        """
        return tuple(self.table["image_id"])

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Source image paths, in sampling order.

        Returns
        -------
        tuple of Path
        """
        return tuple(Path(p) for p in self.table["source_path"])

    def exposure(self, column: str) -> dict[str, float]:
        """Share of training steps each value of ``column`` receives.

        Under proportional sampling with ``batch_size = 1`` every
        image gets one step per epoch, so a group's share of images is
        exactly its share of optimizer steps. Recording this next to
        the per-material metrics answers the obvious question about
        any weak cross-section: how much training did it actually get.

        Parameters
        ----------
        column : str
            Manifest column to group by, e.g. ``"material"`` or
            ``"scale_bin"``.

        Returns
        -------
        dict of str to float
            Value to share in ``[0, 1]``, ordered by descending share.

        Raises
        ------
        SplitLoadError
            If ``column`` is not present in the table.
        """
        if column not in self.table.columns:
            raise SplitLoadError(
                f"Column {column!r} is not in the split table "
                f"(available: {sorted(self.table.columns)})"
            )
        if not len(self.table):
            return {}
        counts = self.table[column].value_counts()
        return {
            str(value): count / len(self.table)
            for value, count in counts.items()
        }


def load_split(
    split_csv: Path,
    subset: str,
    *,
    allow_test: bool = False,
    verify_manifest: Optional[Path] = None,
) -> SplitSubset:
    """Load one subset of the frozen split.

    Parameters
    ----------
    split_csv : Path
        Split table written by ``scripts/create_dataset_split.py``.
    subset : str
        ``train``, ``val`` or ``test``.
    allow_test : bool, optional
        Required to read TEST. Passing it is logged at WARNING level,
        so opening the test set always leaves a trace.
    verify_manifest : Path, optional
        Manifest to check the split against. When given, its SHA-256
        must equal the ``manifest_sha256`` recorded in the split's
        metadata; this catches a split silently outliving the manifest
        it was derived from.

    Returns
    -------
    SplitSubset

    Raises
    ------
    LockedTestSetError
        If ``subset`` is ``"test"`` and ``allow_test`` is False.
    SplitLoadError
        If the file is missing or malformed, if ``subset`` is not a
        known set name, if the subset is empty, or if
        ``verify_manifest`` does not match.
    """
    if subset not in SUBSETS:
        raise SplitLoadError(
            f"Unknown subset {subset!r}, expected one of {list(SUBSETS)}"
        )
    if subset == "test" and not allow_test:
        raise LockedTestSetError(
            "TEST is locked. It is opened exactly once, at the end of "
            "the experiment, and takes part in no decision before "
            "that - development choices are made on VALIDATION. Pass "
            "allow_test=True only when that final moment has come."
        )

    if not split_csv.exists():
        raise SplitLoadError(f"Split table not found: {split_csv}")

    table = pd.read_csv(split_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SplitLoadError(
            f"Split table {split_csv} is missing column(s): {missing}"
        )

    metadata = _load_metadata(split_csv)
    split_id = str(metadata.get("split_id", split_csv.stem))
    if verify_manifest is not None:
        _verify_manifest(metadata, verify_manifest, split_csv)

    rows = table[table["split"] == subset]
    if rows.empty:
        raise SplitLoadError(
            f"Split {split_id} assigns no image to {subset!r}"
        )
    used = rows[rows["used"].astype(bool)]
    n_excluded = len(rows) - len(used)
    if used.empty:
        raise SplitLoadError(
            f"Every image of {subset!r} in split {split_id} is marked "
            f"used=False"
        )

    used = used.sort_values("image_id").reset_index(drop=True)

    if subset == "test":
        logger.warning(
            "TEST UNLOCKED: reading %d image(s) of split %s. This must "
            "be the single, final evaluation.", len(used), split_id,
        )
    logger.info(
        "Loaded %s of split %s: %d image(s)%s.",
        subset.upper(), split_id, len(used),
        f", {n_excluded} excluded as used=False" if n_excluded else "",
    )
    return SplitSubset(
        split_id=split_id,
        subset=subset,
        table=used,
        n_excluded_unused=n_excluded,
    )


def _load_metadata(split_csv: Path) -> Mapping[str, object]:
    """Read the split's metadata sidecar, or an empty mapping.

    Parameters
    ----------
    split_csv : Path

    Returns
    -------
    Mapping
    """
    path = split_csv.with_name(f"{split_csv.stem}_metadata.json")
    if not path.exists():
        logger.warning(
            "No split metadata next to %s: cannot check which manifest "
            "this split came from.", split_csv,
        )
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _verify_manifest(
    metadata: Mapping[str, object], manifest: Path, split_csv: Path
) -> None:
    """Check that a manifest is the one the split was derived from.

    Parameters
    ----------
    metadata : Mapping
    manifest : Path
    split_csv : Path

    Raises
    ------
    SplitLoadError
        If the manifest is missing, the metadata records no hash, or
        the hashes differ.
    """
    expected = metadata.get("manifest_sha256")
    if expected is None:
        raise SplitLoadError(
            f"Cannot verify {manifest}: the metadata of {split_csv} "
            f"records no manifest_sha256"
        )
    if not manifest.exists():
        raise SplitLoadError(f"Manifest not found: {manifest}")

    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != expected:
        raise SplitLoadError(
            f"Manifest mismatch: {manifest} hashes to {actual[:12]}... "
            f"but split {split_csv.stem} was built from "
            f"{str(expected)[:12]}.... The split is only valid for the "
            f"manifest it was derived from."
        )
    logger.debug("Manifest verified against split metadata: %s", manifest)
