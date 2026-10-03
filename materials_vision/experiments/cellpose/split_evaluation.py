"""
Scoring a Cellpose-SAM snapshot the way PEFT-SAM snapshots are scored.

The comparison is only as fair as the measuring stick is shared, so
nothing here measures anything: a snapshot is run over the images of a
subset as ``SampleSource`` prepares them, its output is turned into an
instance label image, and that label image goes through
``evaluate_image`` - the same matching, the same size bins, the same
boundary tolerance and the same exclusion of close-ups as every
PEFT-SAM figure.

**How instances are produced.** The network predicts, per pixel, a
probability of lying inside a pore and a flow vector pointing towards
that pore's centre. Pixels above a probability threshold follow the
flows until they converge, pixels converging together form an
instance, and instances whose predicted flows disagree with the flows
their own shape implies are discarded as implausible. The two
thresholds - on the probability and on that disagreement - are what
the post-processing calibration tunes, as the watershed thresholds are
for PEFT-SAM.

**How far the flows are followed is a setting too, and it matters
here.** Cellpose follows them for 200 steps at native resolution, a
number that suits objects tens of pixels across. The pores of this
dataset are a median 116 pixels across and reach almost 900, and a
pixel further from its pore's centre than the steps allow never
arrives: the pore is split or discarded. Measured on VALIDATION with
the snapshot and thresholds fixed, raising the steps from 200 to 1000
brought the recall of the largest quarter of pores from 0.59 to 0.89
without touching the network's output, while shrinking the image
instead - the library's other remedy for large objects - lost the
smallest pores. The steps are therefore part of every setting, and the
recipe chooses them on VALIDATION with the thresholds.

**The network runs once per image, whatever the number of settings.**
Almost all the cost is the network; following the flows is cheap. So
the flows are predicted once and grown into instances once per
setting, which is what makes a grid of twenty settings affordable. The
two steps are the same library calls Cellpose makes in one go, with
the values it uses for an image scored at native resolution named
below rather than inherited.

**A physical area filter is applied after Cellpose's own.** Cellpose
removes instances below a minimum size in pixels. The PEFT-SAM
deployment additionally drops instances below 200 square micrometres,
converted per image from its pixel size, and the variant reported
alongside uses exactly that function.
"""
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

from materials_vision.evaluation.aggregate import (ImageEvaluation,
                                                   evaluate_image)
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.inference import drop_small_instances

logger = logging.getLogger(__name__)

# Flow-following steps Cellpose uses for an image that is not rescaled.
# Its default of 200 is scaled by the rescaling factor, one here.
LIBRARY_NITER = 200

# Steps the snapshot is chosen under in the second round of selection,
# the smallest value at which the largest pores were recovered in the
# diagnosis, and the steps the thresholds are then calibrated over.
SELECTION_NITER = 1000

CALIBRATION_NITERS = (1000, 2000)

# Cellpose's default: instances covering more than this share of the
# image are discarded as failures of the flow following.
MAX_SIZE_FRACTION = 0.4

# Batch of tiles the network is run on, and their size and overlap;
# Cellpose's defaults for inference.
EVAL_BATCH_SIZE = 8

TILE_PX = 256

TILE_OVERLAP = 0.1

FLOW_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)

CELLPROB_THRESHOLDS = (-2.0, -1.0, 0.0, 1.0, 2.0)

# The physical area below which the PEFT-SAM deployment drops predicted
# instances, reported here as the variant alongside.
MIN_INSTANCE_AREA_UM2 = 200.0


@dataclass(frozen=True)
class CellposeSettings:
    """How a snapshot's flows are turned into instances.

    Parameters
    ----------
    flow_threshold : float
        Largest disagreement between predicted flows and the flows of
        an instance's own shape at which the instance is kept.
    cellprob_threshold : float
        Probability logit above which a pixel may belong to an
        instance.
    min_size_px : int
        Instances smaller than this, in pixels, are removed by Cellpose.
    min_instance_area_um2 : float
        Instances smaller than this physical area are removed
        afterwards. Zero keeps all.
    niter : int
        Steps the flows are followed for.
    """

    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size_px: int = 15
    min_instance_area_um2: float = 0.0
    niter: int = LIBRARY_NITER

    def label(self) -> str:
        """Spell out every value, so no two settings share a name.

        Returns
        -------
        str
            For example ``ft0.4_cp0_m15``, with ``_n<steps>`` appended
            when the steps are not the library's and ``_a<um2>`` when
            the physical area filter is on. The suffixes are left out
            at their defaults so that every figure recorded before they
            existed keeps the name it was reported under.
        """
        name = (
            f"ft{self.flow_threshold:g}_cp{self.cellprob_threshold:g}"
            f"_m{self.min_size_px:d}"
        )
        if self.niter != LIBRARY_NITER:
            name += f"_n{self.niter:d}"
        if self.min_instance_area_um2 > 0:
            name += f"_a{self.min_instance_area_um2:g}"
        return name

    def with_area_filter(
        self, min_area_um2: float = MIN_INSTANCE_AREA_UM2
    ) -> "CellposeSettings":
        """The same setting with the physical area filter on.

        Parameters
        ----------
        min_area_um2 : float, optional

        Returns
        -------
        CellposeSettings
        """
        return replace(self, min_instance_area_um2=min_area_um2)


# The library's own post-processing, which the first round of selection
# chose the snapshot under.
LIBRARY_DEFAULT = CellposeSettings()


def calibration_grid(
    niters: Sequence[int] = (LIBRARY_NITER,)
) -> tuple[CellposeSettings, ...]:
    """Every setting the calibration on VALIDATION chooses among.

    Parameters
    ----------
    niters : sequence of int, optional
        Flow-following steps to cross with the thresholds. The first
        round of selection calibrated at the library's steps alone.

    Returns
    -------
    tuple of CellposeSettings
        Steps outermost, then flow threshold, so a printed grid reads
        block by block and row by row.

    Raises
    ------
    ValueError
        If no steps are given or one is not positive.
    """
    if not niters or any(niter < 1 for niter in niters):
        raise ValueError(f"Steps must be positive, got {tuple(niters)}.")
    return tuple(
        CellposeSettings(
            flow_threshold=flow, cellprob_threshold=cellprob, niter=niter,
        )
        for niter in niters
        for flow in FLOW_THRESHOLDS
        for cellprob in CELLPROB_THRESHOLDS
    )


def load_snapshot(snapshot: Any, gpu: bool = True) -> Any:
    """Load a trained snapshot for inference.

    Parameters
    ----------
    snapshot : Path
    gpu : bool, optional

    Returns
    -------
    cellpose.models.CellposeModel
    """
    from cellpose import models

    return models.CellposeModel(gpu=gpu, pretrained_model=str(snapshot))


def predict_flows(model: Any, image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Run the network once over one image at native resolution.

    Parameters
    ----------
    model : cellpose.models.CellposeModel
    image : np.ndarray
        ``(H, W)``, one working channel as the sample source prepares it.
        Not modified.

    Returns
    -------
    tuple of np.ndarray
        The flow field ``(2, H, W)`` and the probability logits
        ``(H, W)``.
    """
    _, flows, _ = model.eval(
        image,
        batch_size=EVAL_BATCH_SIZE,
        diameter=None,
        normalize=True,
        augment=False,
        tile_overlap=TILE_OVERLAP,
        bsize=TILE_PX,
        compute_masks=False,
    )
    return flows[1], flows[2]


def masks_from_flows(
    flow_field: np.ndarray,
    cellprob: np.ndarray,
    settings: CellposeSettings,
    *,
    device: Any,
    pixel_size_um: Optional[float] = None,
) -> np.ndarray:
    """Grow instances from predicted flows under one setting.

    Parameters
    ----------
    flow_field : np.ndarray
        ``(2, H, W)``.
    cellprob : np.ndarray
        ``(H, W)``.
    settings : CellposeSettings
    device : torch.device
        Where the flows are followed.
    pixel_size_um : float, optional
        Required when the setting filters by physical area.

    Returns
    -------
    np.ndarray
        ``(H, W)`` int32 labels, background zero.

    Raises
    ------
    ValueError
        If the setting filters by physical area and no pixel size is
        given.
    """
    from cellpose import dynamics

    if settings.min_instance_area_um2 > 0 and pixel_size_um is None:
        raise ValueError(
            "This setting drops instances below "
            f"{settings.min_instance_area_um2:g} um2, which needs the "
            "image's pixel size."
        )
    masks = dynamics.resize_and_compute_masks(
        flow_field, cellprob,
        niter=settings.niter,
        cellprob_threshold=settings.cellprob_threshold,
        flow_threshold=settings.flow_threshold,
        min_size=settings.min_size_px,
        max_size_fraction=MAX_SIZE_FRACTION,
        device=device,
    )
    labels = np.asarray(masks).astype(np.int32)
    if settings.min_instance_area_um2 <= 0:
        return labels
    assert pixel_size_um is not None
    return drop_small_instances(
        labels, settings.min_instance_area_um2, pixel_size_um
    )


def score_settings(
    model: Any,
    source: Any,
    positions: Sequence[int],
    size_bins: Any,
    settings: Sequence[CellposeSettings],
    boundary_scales: tuple[float, ...] = (DECISION_SCALE,),
) -> dict[CellposeSettings, list[ImageEvaluation]]:
    """Measure one snapshot under several settings.

    Parameters
    ----------
    model : cellpose.models.CellposeModel
    source : SampleSource
    positions : sequence of int
        Which samples of the source to measure, in order.
    size_bins : SizeBins
    settings : sequence of CellposeSettings
    boundary_scales : tuple of float, optional
        One tolerance by default, as for PEFT-SAM snapshots.

    Returns
    -------
    dict
        Per setting, the per-image evaluations in the order given.

    Raises
    ------
    ValueError
        If no setting is given, which would run the network for
        nothing.
    """
    if not settings:
        raise ValueError("At least one setting is needed to score under.")
    measured: dict[CellposeSettings, list[ImageEvaluation]] = {
        setting: [] for setting in settings
    }
    for position in positions:
        prepared = source.load(position)
        flow_field, cellprob = predict_flows(model, prepared.image)
        for setting in settings:
            measured[setting].append(evaluate_image(
                prepared.record, prepared.labels,
                masks_from_flows(
                    flow_field, cellprob, setting, device=model.device,
                    pixel_size_um=prepared.record.pixel_size_um,
                ),
                size_bins=size_bins,
                boundary_scales=boundary_scales,
            ))
    return measured
