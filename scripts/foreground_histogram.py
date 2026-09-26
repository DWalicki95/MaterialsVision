#!/usr/bin/env python3
"""
Anchor 1 of the foreground threshold: where the confidences actually lie.

The foreground map is a confidence per pixel. A threshold turns it into
a decision, and moving the threshold changes something only where
pixels are: if the confidences pile up near zero and near one with an
empty valley between them, a threshold anywhere in the valley decides
the same thing, and sweeping it there measures nothing. The histogram
of confidences therefore fixes the range the foreground grid has to
span - read on TRAIN, per material, from the stored decoder output of
the calibration snapshot, before any threshold is scored.

It is read both raw and after each blur the grid will try, because the
blur runs before the threshold and reshapes the very distribution the
threshold is read against.

Examples
--------
    $ python scripts/foreground_histogram.py
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from materials_vision.evaluation.decoder_cache import load_maps
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (POSTPROCESSING_EXPERIMENT,
                                       configure_tracking, ensure_experiment)

logger = logging.getLogger(__name__)

EXIT_OK = 0

DEFAULT_CACHE = Path(
    "checkpoints/postprocessing/decoder_cache/b0_seed20260907_epoch-12"
)

DEFAULT_OUT = Path("checkpoints/postprocessing/foreground_histogram")

# The smoothing values the foreground grid will try; zero is no blur.
SMOOTHING_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

N_BINS = 100

# Shares of the pixels whose confidence lies between the two modes,
# read as the valley's edges: below the lower one the map is background
# for any threshold anyone would try, above the upper one foreground.
VALLEY_QUANTILES = (0.01, 0.99)

SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")


def smoothed(foreground: np.ndarray, sigma: float) -> np.ndarray:
    """The foreground map as the segmenter sees it before thresholding.

    Uses the library's own smoothing, so that the distribution read
    here is the one the threshold is actually applied to.
    """
    if sigma <= 0:
        return foreground
    from micro_sam.instance_segmentation import _apply_smoothing

    return _apply_smoothing(foreground, sigma, None, None)


def valley(counts: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    """Edges and weight of the region between the two modes.

    The histogram is split at 0.5; the two modes are the fullest bins on
    either side, and the valley is the stretch between them. Reported
    with the share of pixels it holds, which is how much any threshold
    placed inside it can change.

    Parameters
    ----------
    counts : np.ndarray
    edges : np.ndarray

    Returns
    -------
    dict
    """
    centres = (edges[:-1] + edges[1:]) / 2
    low = centres < 0.5
    low_mode = centres[low][np.argmax(counts[low])]
    high_mode = centres[~low][np.argmax(counts[~low])]
    inside = (centres > low_mode) & (centres < high_mode)
    total = counts.sum()
    cumulative = np.cumsum(counts[inside]) / max(counts[inside].sum(), 1)
    inner = centres[inside]
    return {
        "low_mode": float(low_mode),
        "high_mode": float(high_mode),
        "share_between_modes": float(counts[inside].sum() / total),
        "minimum_at": float(inner[np.argmin(counts[inside])])
        if inner.size else float("nan"),
        **{
            f"q{int(q * 100):02d}_between_modes": float(
                inner[np.searchsorted(cumulative, q)]
            ) if inner.size else float("nan")
            for q in VALLEY_QUANTILES
        },
    }


def collect(cache: Path) -> dict:
    """Histograms per material and per blur, pooled over images."""
    index = json.loads((cache / "index.json").read_text())
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    counts: dict = defaultdict(lambda: np.zeros(N_BINS, dtype=np.int64))
    for entry in index["images"]:
        foreground = load_maps(cache / entry["path"])["foreground"]
        for sigma in SMOOTHING_SIGMAS:
            values = np.clip(smoothed(foreground, sigma), 0.0, 1.0)
            histogram, _ = np.histogram(values, bins=edges)
            counts[(entry["material"], sigma)] += histogram
            counts[("all", sigma)] += histogram
    return {"edges": edges, "counts": dict(counts)}


def figure(collected: dict, path: Path) -> Path:
    """One panel per blur, the three materials overlaid, log counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = collected["edges"]
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for axis, sigma in zip(axes.ravel(), SMOOTHING_SIGMAS):
        for material, colour in zip(("AS", "K", "VAB"), SERIES_COLOURS):
            counts = collected["counts"][(material, sigma)]
            share = counts / counts.sum()
            axis.step(centres, share, where="mid", color=colour,
                      linewidth=2, label=material)
        axis.set_yscale("log")
        axis.set_title(
            "no blur" if sigma == 0 else f"blur sigma {sigma:g} px",
            loc="left",
        )
        axis.axvline(0.5, color="#52514e", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("foreground confidence")
    for axis in axes[:, 0]:
        axis.set_ylabel("share of pixels")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Foreground confidence, calibration sample (TRAIN), "
        "b0_seed20260907 epoch 12", x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Histogram the confidences, locate the valley, record both.

    Returns
    -------
    int
        Process exit code.
    """
    import mlflow

    args = parse_args(argv)
    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)
    collected = collect(args.cache)
    edges = collected["edges"]
    summary = {
        f"{material}/sigma{sigma:g}": valley(counts, edges)
        for (material, sigma), counts in sorted(
            collected["counts"].items(), key=lambda item: str(item[0])
        )
    }
    report = args.out / "foreground_histogram.json"
    report.write_text(json.dumps({
        "summary": summary,
        "edges": edges.tolist(),
        "counts": {
            f"{material}/sigma{sigma:g}": counts.tolist()
            for (material, sigma), counts in collected["counts"].items()
        },
    }, indent=2))
    plot = figure(collected, args.out / "foreground_histogram.png")

    configure_tracking()
    with mlflow.start_run(
        experiment_id=ensure_experiment(POSTPROCESSING_EXPERIMENT),
        run_name="step5_foreground_histogram",
        tags={"part": "I", "plan_step": "5", "subset": "train"},
    ):
        mlflow.log_param("snapshot", "b0_seed20260907/epoch-12")
        for key, values in summary.items():
            mlflow.log_metrics({
                f"{key}/{name}".replace(".", "_"): value
                for name, value in values.items() if value == value
            })
        mlflow.log_artifact(str(report))
        mlflow.log_artifact(str(plot))
    for key, values in summary.items():
        logger.info("%s: %s", key, {k: round(v, 3) for k, v in values.items()})
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
