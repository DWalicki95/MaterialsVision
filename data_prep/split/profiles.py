"""
Aggregation of the frozen inventory manifest into per-formulation
profiles, plus the integrity checks that must hold before a grouped
split is meaningful at all.

The split never sees individual images: it operates exclusively on the
profiles built here.
"""
import logging

import pandas as pd

from data_prep.split.models import FormulationProfile

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "image_id",
    "formulation",
    "material",
    "microscope",
    "scale_bin",
    "n_instances",
    "file_hash",
)


class SplitDataError(ValueError):
    """Raised when the manifest cannot support a grouped split."""


def load_manifest(path) -> pd.DataFrame:
    """Read the frozen manifest and check the columns the split needs.

    Parameters
    ----------
    path : Path
        Manifest CSV.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    SplitDataError
        If the file is missing or lacks a required column.
    """
    if not path.exists():
        raise SplitDataError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SplitDataError(
            f"Manifest {path} is missing required column(s): {missing}"
        )
    logger.info(
        "Loaded manifest %s: %d row(s), %d formulation(s).",
        path, len(df), df["formulation"].nunique(),
    )
    return df


def check_grouping_integrity(df: pd.DataFrame) -> list[str]:
    """Verify the assumptions that make formulation-level grouping safe.

    Grouping by formulation only prevents leakage if a formulation is
    a well-defined, self-contained unit: it must not span materials or
    microscopes, and no image content may be shared across
    formulations. The last check is the one that would silently defeat
    the whole split - two formulations holding byte-identical files
    would put the same image on both sides of the TRAIN/TEST boundary
    no matter how the formulations are assigned.

    Parameters
    ----------
    df : pandas.DataFrame
        Manifest.

    Returns
    -------
    list of str
        Human-readable violations, empty when the manifest is sound.
    """
    violations = []

    for column in ("material", "microscope"):
        spanning = (
            df.groupby("formulation")[column].nunique() > 1
        )
        for formulation in spanning[spanning].index:
            values = sorted(
                df.loc[df["formulation"] == formulation, column]
                .dropna().unique()
            )
            violations.append(
                f"Formulation {formulation} spans multiple "
                f"{column} values: {values}"
            )

    shared = df.groupby("file_hash")["formulation"].nunique()
    for file_hash in shared[shared > 1].index:
        names = sorted(
            df.loc[df["file_hash"] == file_hash, "formulation"].unique()
        )
        violations.append(
            f"file_hash {file_hash[:12]}... is shared by formulations "
            f"{names}: identical image content would cross the split "
            f"boundary"
        )

    return violations


def build_formulation_profiles(
    df: pd.DataFrame,
) -> tuple[FormulationProfile, ...]:
    """Aggregate the manifest into one profile per formulation.

    Parameters
    ----------
    df : pandas.DataFrame
        Manifest, already checked by ``check_grouping_integrity``.

    Returns
    -------
    tuple of FormulationProfile
        Sorted by formulation name, so the profile order does not
        depend on manifest row order.

    Raises
    ------
    SplitDataError
        If ``check_grouping_integrity`` would fail, or a formulation
        has no microscope assigned (which would make the M2 coverage
        condition of III.4 unverifiable).
    """
    violations = check_grouping_integrity(df)
    if violations:
        raise SplitDataError(
            "Manifest cannot support a grouped split:\n  - "
            + "\n  - ".join(violations)
        )

    profiles = []
    for formulation, group in df.groupby("formulation", sort=True):
        microscopes = group["microscope"].dropna().unique()
        if len(microscopes) != 1:
            raise SplitDataError(
                f"Formulation {formulation} has no unambiguous "
                f"microscope: {sorted(microscopes)}"
            )
        bins = group["scale_bin"].value_counts()
        profiles.append(
            FormulationProfile(
                formulation=str(formulation),
                material=str(group["material"].iloc[0]),
                microscope=str(microscopes[0]),
                n_images=len(group),
                n_coarse=int(bins.get("coarse", 0)),
                n_fine=int(bins.get("fine", 0)),
                n_outlier=int(bins.get("outlier", 0)),
                n_instances=int(group["n_instances"].sum()),
            )
        )

    n_binned = sum(
        p.n_coarse + p.n_fine + p.n_outlier for p in profiles
    )
    if n_binned != len(df):
        raise SplitDataError(
            f"scale_bin is unset for {len(df) - n_binned} manifest "
            f"row(s); every image must fall into exactly one bin"
        )

    logger.info(
        "Built %d formulation profile(s) across material(s) %s.",
        len(profiles), sorted({p.material for p in profiles}),
    )
    return tuple(profiles)
