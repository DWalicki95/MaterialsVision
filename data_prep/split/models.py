"""
Data models for the dataset split pipeline.

Everything that flows between the profiling, search and reporting
stages is a frozen dataclass, so a candidate split can never be
mutated after it has been scored.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

SETS: tuple[str, str, str] = ("train", "val", "test")

SCALE_BINS: tuple[str, str, str] = ("coarse", "fine", "outlier")


@dataclass(frozen=True)
class CostWeights:
    """Weights of the split balance cost terms.

    All terms are L1 deviations from the target share of a set, so the
    weights are directly comparable to one another.

    Parameters
    ----------
    images : float
        Weight on the share of evaluable images per set. The dominant
        term: formulations hold 4 to 40 images each, so a quota
        expressed in formulations alone controls nothing that matters
        for evaluation power.
    instances : float
        Weight on the share of ground-truth instances per set. Tracks
        statistical power, which scales with instances rather than
        with images.
    cell_default : float
        Weight on the share of each non-empty ``material x scale_bin``
        cell per set. This is the stratification term proper.
    cell_overrides : Mapping[str, float]
        Per-cell weight overrides, keyed ``"<material>:<scale_bin>"``.
        Used for cells that a three-way split cannot reach, where a
        full-weight term would contribute a near-constant offset and
        distort the argmin without expressing a real preference.
    lost_outlier_image : float
        Penalty per ``scale_outlier`` image that ends up outside
        TRAIN. Such an image is unusable for evaluation - its
        resolution is several times finer than the rest, so it is a
        visually different task - and it is dropped entirely rather
        than moved back to TRAIN on its own, which would split its
        formulation across two sets and let the model be scored on
        material it trained on. The penalty is small because the loss
        is small; it only nudges the search away from splits that
        waste those few images.
    """

    images: float = 3.0
    instances: float = 1.0
    cell_default: float = 2.0
    cell_overrides: Mapping[str, float] = field(default_factory=dict)
    lost_outlier_image: float = 0.02

    def cell_weight(self, material: str, scale_bin: str) -> float:
        """Return the cost weight of one ``material x scale_bin`` cell.

        Parameters
        ----------
        material : str
        scale_bin : str

        Returns
        -------
        float
        """
        return float(
            self.cell_overrides.get(
                f"{material}:{scale_bin}", self.cell_default
            )
        )


@dataclass(frozen=True)
class SplitConstraints:
    """Hard admissibility conditions a candidate split must satisfy.

    A candidate failing any of these is rejected outright and never
    scored. They encode what the evaluation must be able to report -
    both microscopes and both scale bins present in every set - plus
    the minimum cross-section sizes below which a reported
    per-material or per-scale number would not support any conclusion.

    Parameters
    ----------
    min_m2_formulations_per_set : int
        Minimum formulations acquired on microscope M2 (materials K,
        VAB) in every set. Without this the evaluation does not
        measure cross-microscope transfer at all.
    min_scale_bin_images_per_set : int
        Minimum images of each of ``coarse`` and ``fine`` in every
        set.
    min_eval_fine_images : int
        Minimum ``fine`` images in VALIDATION and in TEST. The ``fine``
        bin is about a tenth of the dataset, so a global metric is
        dominated by ``coarse`` and says almost nothing about how the
        model handles the other resolution. Deciding whether
        scale-varying augmentation helps therefore rests on the
        per-``scale_bin`` cross-section, and a set too thin to report
        it is useless for that decision however good its global
        metric looks.
    min_eval_images_by_material : Mapping[str, int]
        Minimum evaluable images per material in VALIDATION and in
        TEST, keyed by material. Materials absent from the mapping are
        unconstrained.
    """

    min_m2_formulations_per_set: int = 1
    min_scale_bin_images_per_set: int = 1
    min_eval_fine_images: int = 8
    min_eval_images_by_material: Mapping[str, int] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class MinFragmentAreaConfig:
    """Configuration of the ``A_min_fragment`` calibration.

    Parameters
    ----------
    inventory_config : Path
        Inventory configuration YAML, reused as the single source of
        truth for the Label Studio export paths. The manifest stores
        no per-instance areas, so they are recomputed from polygons.
    percentile : float
        Percentile of the TRAIN instance-area distribution to freeze,
        in percent. Set low (P1) on purpose: the value becomes the
        floor below which a clipped instance fragment is discarded,
        and the intent is to never create a label smaller than
        anything an annotator actually drew.
    exclude_scale_outlier : bool
        Whether to exclude ``scale_outlier`` images - the handful of
        close-ups several times finer than the rest - from the
        calibration. Instance area in pixels grows with the square of
        the resolution, so leaving them in would drag the threshold
        upward on evidence that does not represent the dataset. Both
        values are always reported.
    """

    inventory_config: Path
    percentile: float = 1.0
    exclude_scale_outlier: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """Fully resolved split run configuration.

    Parameters
    ----------
    split_id : str
        Identifier of this split, e.g. ``"split_v1"``. Used in output
        filenames and recorded in every run's metadata; a new split
        requires a new id, never an overwrite.
    manifest_path : Path
        Frozen inventory manifest the split is derived from.
    output_dir : Path
        Directory the split artifacts are written to.
    seed : int
        Seed of the candidate generator. Together with
        ``n_candidates`` and the cost function it makes the chosen
        split fully reproducible.
    n_candidates : int
        Number of candidate assignments to generate and score.
    quotas : Mapping[str, tuple[int, int, int]]
        Formulation counts ``(train, val, test)`` per material. Must
        sum to the number of formulations of that material in the
        manifest.
    forced_train : tuple of str
        Formulations pinned to TRAIN before any randomization, with
        the reason recorded in the split report.
    target_shares : Mapping[str, float]
        Target share of each set, keyed by set name.
    constraints : SplitConstraints
    cost_weights : CostWeights
    min_fragment_area : MinFragmentAreaConfig, optional
        ``None`` disables the ``A_min_fragment`` calibration step.
    """

    split_id: str
    manifest_path: Path
    output_dir: Path
    seed: int
    n_candidates: int
    quotas: Mapping[str, tuple[int, int, int]]
    forced_train: tuple[str, ...]
    target_shares: Mapping[str, float]
    constraints: SplitConstraints
    cost_weights: CostWeights
    min_fragment_area: Optional[MinFragmentAreaConfig] = None


@dataclass(frozen=True)
class FormulationProfile:
    """Everything the split needs to know about one formulation.

    The formulation is the atomic unit of the split: it is never
    divided, so all quantities here are already aggregated over its
    images.

    Parameters
    ----------
    formulation : str
        Formulation token, e.g. ``"AS26"``.
    material : str
        ``AS``, ``K`` or ``VAB``.
    microscope : str
        ``M1`` or ``M2``.
    n_images : int
        All images of this formulation.
    n_coarse, n_fine, n_outlier : int
        Images per ``scale_bin``.
    n_instances : int
        Ground-truth instances summed over the formulation's images.
    """

    formulation: str
    material: str
    microscope: str
    n_images: int
    n_coarse: int
    n_fine: int
    n_outlier: int
    n_instances: int

    @property
    def n_eval_images(self) -> int:
        """Images usable for evaluation (all bins except ``outlier``).

        Returns
        -------
        int
        """
        return self.n_coarse + self.n_fine


@dataclass(frozen=True)
class SetStats:
    """Aggregate of one set of a candidate split.

    Parameters
    ----------
    name : str
        ``train``, ``val`` or ``test``.
    formulations : tuple of str
        Member formulations, sorted.
    n_images, n_eval_images : int
    n_coarse, n_fine, n_outlier : int
    n_instances : int
    n_m2_formulations : int
        Formulations acquired on microscope M2.
    images_by_material : Mapping[str, int]
        Evaluable images per material.
    images_by_cell : Mapping[tuple[str, str], int]
        Images per ``(material, scale_bin)`` cell.
    """

    name: str
    formulations: tuple[str, ...]
    n_images: int
    n_eval_images: int
    n_coarse: int
    n_fine: int
    n_outlier: int
    n_instances: int
    n_m2_formulations: int
    images_by_material: Mapping[str, int]
    images_by_cell: Mapping[tuple[str, str], int]


@dataclass(frozen=True)
class SplitResult:
    """The chosen split plus the evidence for why it was chosen.

    Parameters
    ----------
    assignment : Mapping[str, str]
        Formulation to set name.
    cost : float
        Balance cost of the chosen candidate.
    stats : Mapping[str, SetStats]
        Per-set aggregates, keyed by set name.
    profiles : tuple of FormulationProfile
        Profiles the search ran on, in manifest order.
    n_generated : int
        Candidates generated.
    n_feasible : int
        Candidates that passed the hard constraints. A low ratio is a
        signal that the constraints are close to infeasible and the
        chosen split sits in a narrow corner of the space.
    """

    assignment: Mapping[str, str]
    cost: float
    stats: Mapping[str, SetStats]
    profiles: tuple[FormulationProfile, ...]
    n_generated: int
    n_feasible: int


@dataclass(frozen=True)
class FragmentAreaResult:
    """Outcome of the ``A_min_fragment`` calibration on TRAIN.

    Parameters
    ----------
    a_min_fragment_px2 : float
        The frozen value: the configured percentile of the TRAIN
        ground-truth instance area distribution, in source pixels
        squared, measured after the deterministic crop to
        ``load_crop_bbox``.
    percentile : float
    n_images : int
        TRAIN images the distribution was measured on.
    n_instances : int
        Instances contributing to the frozen value.
    excluded_scale_outlier : bool
        Whether ``scale_outlier`` images were excluded.
    value_including_outliers_px2 : float
        The same percentile computed with ``scale_outlier`` images
        included, reported as a diagnostic either way.
    by_scale_bin_px2 : Mapping[str, float]
        The same percentile computed within each ``scale_bin``.
        Diagnostic only: instance area in pixels scales with the
        square of the resolution ratio, so a single global value is
        necessarily dominated by the 88% ``coarse`` majority.
    n_instances_lost_to_crop : int
        Instances that rasterized to zero pixels inside
        ``load_crop_bbox``, i.e. annotated entirely within the
        cropped-away information panel.
    """

    a_min_fragment_px2: float
    percentile: float
    n_images: int
    n_instances: int
    excluded_scale_outlier: bool
    value_including_outliers_px2: float
    by_scale_bin_px2: Mapping[str, float]
    n_instances_lost_to_crop: int
