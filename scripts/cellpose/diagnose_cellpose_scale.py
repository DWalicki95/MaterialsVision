#!/usr/bin/env python3
"""
Why the chosen Cellpose snapshot loses the largest pores: a diagnosis.

On VALIDATION the chosen snapshot recovers the middle two size classes
as well as PEFT-SAM does but loses both ends, the largest pores most:
0.59 of them against 0.94. Two mechanisms of Cellpose's own could do
that, and they call for different remedies, so they are measured apart
on the same snapshot and the same images.

* **Flow following that stops too early.** At native resolution
  Cellpose follows the predicted flows for 200 steps, and a pixel
  further than that from its pore's centre may never arrive, leaving
  the pore split or discarded. The pores here reach 889 pixels across.
  If this is the cause, more steps recover the pores from the very
  same network output, and the remedy is a post-processing setting.
* **Objects larger than the network's working scale.** The model was
  trained on 256-pixel crops, narrower than a tenth of the pores.
  Cellpose's answer is its ``diameter`` setting, which shrinks the
  image before the network and scales the number of steps up to match.
  If this is the cause, only the shrunken input recovers them, and the
  remedy belongs to the recipe - training included.

Nothing is chosen here. The figures go to a JSON report and into the
experiment plan, and the owner of the study decides what follows. The
thresholds are held at the library's defaults, under which the
snapshot was chosen, so that the two mechanisms are the only things
varied.

Examples
--------
    $ python scripts/cellpose/diagnose_cellpose_scale.py \\
        --checkpoint checkpoints/cellpose/selection/\\
cpsam_lr1e-5_seed20260907/models/cpsam_lr1e-5_seed20260907_epoch_0140
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from materials_vision.evaluation import (DECISION_SCALE, aggregate,
                                         evaluate_image, load_size_bins)
from materials_vision.experiments.cellpose.split_evaluation import (
    EVAL_BATCH_SIZE, LIBRARY_DEFAULT, MAX_SIZE_FRACTION, TILE_OVERLAP, TILE_PX,
    load_snapshot, predict_flows)
from materials_vision.logging_config import setup_logging
from materials_vision.training import build_source

logger = logging.getLogger(__name__)

EXIT_OK = 0

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

# Flow-following steps at native resolution; 200 is the library's.
NITERS = (200, 500, 1000, 2000)

# Nominal object diameters handed to Cellpose, which shrinks the image
# by 30 / diameter and scales the steps by the same factor inverted.
DIAMETERS = (60.0, 90.0)


def niter_masks(model: Any, flows: tuple, niter: int) -> np.ndarray:
    """Grow instances from native-resolution flows with more steps.

    Parameters
    ----------
    model : cellpose.models.CellposeModel
    flows : tuple of np.ndarray
        From ``predict_flows``.
    niter : int

    Returns
    -------
    np.ndarray
    """
    from cellpose import dynamics

    flow_field, cellprob = flows
    masks = dynamics.resize_and_compute_masks(
        flow_field, cellprob, niter=niter,
        cellprob_threshold=LIBRARY_DEFAULT.cellprob_threshold,
        flow_threshold=LIBRARY_DEFAULT.flow_threshold,
        min_size=LIBRARY_DEFAULT.min_size_px,
        max_size_fraction=MAX_SIZE_FRACTION, device=model.device,
    )
    return np.asarray(masks).astype(np.int32)


def diameter_masks(
    model: Any, image: np.ndarray, diameter: float
) -> np.ndarray:
    """Segment through Cellpose's own rescaling to a nominal diameter.

    Parameters
    ----------
    model : cellpose.models.CellposeModel
    image : np.ndarray
    diameter : float

    Returns
    -------
    np.ndarray
        On the image's own frame; Cellpose resizes back before growing
        instances.
    """
    masks, _, _ = model.eval(
        image, batch_size=EVAL_BATCH_SIZE, diameter=diameter,
        normalize=True, augment=False, tile_overlap=TILE_OVERLAP,
        bsize=TILE_PX, flow_threshold=LIBRARY_DEFAULT.flow_threshold,
        cellprob_threshold=LIBRARY_DEFAULT.cellprob_threshold,
        min_size=LIBRARY_DEFAULT.min_size_px,
    )
    return np.asarray(masks).astype(np.int32)


def summary(evaluations: list) -> dict[str, Any]:
    """The figures the diagnosis is read on.

    Parameters
    ----------
    evaluations : list of ImageEvaluation

    Returns
    -------
    dict
    """
    result = asdict(aggregate(evaluations))
    return {
        key: result[key]
        for key in ("f1", "precision", "recall", "pore_count_error",
                    "merges_per_100_gt", "splits_per_100_gt",
                    "size_bin_recall", "n_images", "n_gt")
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument("--n-images", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("checkpoints/cellpose/selection/eval/"
                     "scale_diagnosis.json"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Measure both mechanisms on VALIDATION.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    source = build_source(args.split, args.manifest, "val")
    size_bins = load_size_bins(args.size_bins)
    model = load_snapshot(args.checkpoint)
    positions = list(range(len(source)))
    if 0 < args.n_images < len(source):
        stride = len(source) / args.n_images
        positions = [int(step * stride) for step in range(args.n_images)]

    variants = [f"native_niter{n}" for n in NITERS] + [
        f"diameter{d:g}" for d in DIAMETERS
    ]
    measured: dict[str, list] = {variant: [] for variant in variants}
    started_s = time.perf_counter()
    for count, position in enumerate(positions, 1):
        prepared = source.load(position)
        flows = predict_flows(model, prepared.image)
        predictions = {
            f"native_niter{n}": niter_masks(model, flows, n) for n in NITERS
        }
        predictions.update({
            f"diameter{d:g}": diameter_masks(model, prepared.image, d)
            for d in DIAMETERS
        })
        for variant, labels in predictions.items():
            measured[variant].append(evaluate_image(
                prepared.record, prepared.labels, labels,
                size_bins=size_bins, boundary_scales=(DECISION_SCALE,),
            ))
        if count % 10 == 0:
            logger.info("%d / %d image(s), %.0f s.", count, len(positions),
                        time.perf_counter() - started_s)

    report = {
        "checkpoint": str(args.checkpoint),
        "thresholds": LIBRARY_DEFAULT.label(),
        "variants": {v: summary(measured[v]) for v in variants},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    for variant, figures in report["variants"].items():
        logger.info(
            "%-18s F1 %.4f  P %.4f  R %.4f  count %+.3f  recall by size %s",
            variant, figures["f1"], figures["precision"], figures["recall"],
            figures["pore_count_error"],
            " ".join(f"{b['label']} {b['recall']:.3f}"
                     for b in figures["size_bin_recall"]),
        )
    logger.info("Wrote %s.", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
