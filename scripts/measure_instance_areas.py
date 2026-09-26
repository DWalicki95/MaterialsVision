#!/usr/bin/env python3
"""
Step 4 of the optimization plan, descriptive part: how small are the
instances the watershed produces, set against how small real pores are.

The minimum instance size is meant to remove fragments the watershed
leaves behind - a stray seed, a splinter at the edge of a mask, a
remnant of the decoder's block pattern - and nothing the annotation
still counts as a pore. Where it belongs is read off two distributions
side by side: the areas of annotated pores, and the areas of predicted
instances, the latter split into those matched to a pore and those
matched to nothing. If fragments exist as a population of their own,
they show up as unmatched instances below the point where annotated
pores end, and the threshold goes into the gap between the two.

This script only measures. It reads the stored decoder output of the
calibration snapshot, grows instances at today's frozen setting with no
minimum size, and writes the distributions per scale bin, in square
pixels and square micrometres. Choosing the threshold and applying the
safety gate - the threshold applied to the annotation must remove no
pore - come after the distributions have been looked at.

The annotated side is read twice: on the calibration sample, to compare
like with like, and on the whole of TRAIN, because the gate has to hold
on every annotated pore, not only on those of the sample.

Examples
--------
    $ python scripts/measure_instance_areas.py
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from materials_vision.data.instances import border_instance_labels
from materials_vision.evaluation.aggregate import SCALE_OUTLIER_BIN
from materials_vision.evaluation.decoder_cache import (load_maps,
                                                       replay_segmenter)
from materials_vision.evaluation.inference import segment
from materials_vision.evaluation.matching import match_instances
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   WatershedParams)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (POSTPROCESSING_EXPERIMENT,
                                       configure_tracking, ensure_experiment,
                                       postprocessing_id)
from materials_vision.training import build_source

logger = logging.getLogger(__name__)

EXIT_OK = 0

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_CACHE = Path(
    "checkpoints/postprocessing/decoder_cache/b0_seed20260907_epoch-12"
)

DEFAULT_OUT = Path("checkpoints/postprocessing/instance_areas")

LOW_PERCENTILES = (0.1, 0.5, 1, 5, 50)

# Categories of instance, in the order they are drawn.
CATEGORIES = (
    "gt_interior", "gt_border", "pred_matched", "pred_unmatched",
)

SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#e34948")

TEXT_SECONDARY = "#52514e"


def instance_areas(labels: np.ndarray) -> dict[int, float]:
    """Area of every instance, by label."""
    counts = np.bincount(labels.ravel())
    return {
        label: float(counts[label])
        for label in np.flatnonzero(counts)
        if label != 0
    }


def split_gt(labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Annotated areas, interior and cut by the frame."""
    areas = instance_areas(labels)
    border = set(border_instance_labels(labels).tolist())
    interior = [a for label, a in areas.items() if label not in border]
    cut = [a for label, a in areas.items() if label in border]
    return interior, cut


def split_pred(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> tuple[list[float], list[float]]:
    """Predicted areas, matched to a pore and matched to nothing."""
    areas = instance_areas(pred_labels)
    match = match_instances(gt_labels, pred_labels)
    matched_ids = {pair.pred_id for pair in match.pairs}
    matched = [a for label, a in areas.items() if label in matched_ids]
    unmatched = [a for label, a in areas.items() if label not in matched_ids]
    return matched, unmatched


def measure_sample(args) -> dict:
    """Both sides on the calibration sample, per scale bin.

    Returns
    -------
    dict
        ``{scale_bin: {category: {"px2": [...], "um2": [...]}}}``.
    """
    source = build_source(args.split, args.manifest, "train")
    index = json.loads((args.cache / "index.json").read_text())
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in index["images"]:
        sample = source.load(entry["index"])
        pred = segment(
            replay_segmenter(load_maps(args.cache / entry["path"])),
            args.setting,
        )
        interior, cut = split_gt(sample.labels)
        matched, unmatched = split_pred(sample.labels, pred)
        pixel_um2 = sample.record.pixel_size_um ** 2
        bin_ = sample.record.scale_bin
        for category, values in zip(
            CATEGORIES, (interior, cut, matched, unmatched)
        ):
            table[bin_][category]["px2"].extend(values)
            table[bin_][category]["um2"].extend(
                value * pixel_um2 for value in values
            )
    return table


def measure_train_gt(args) -> dict:
    """Annotated areas on the whole of TRAIN, close-ups excluded."""
    source = build_source(args.split, args.manifest, "train")
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for position, record in enumerate(source.records):
        if record.scale_bin == SCALE_OUTLIER_BIN:
            continue
        labels = source.load(position).labels
        interior, cut = split_gt(labels)
        pixel_um2 = record.pixel_size_um ** 2
        for category, values in (
            ("gt_interior", interior), ("gt_border", cut)
        ):
            table[record.scale_bin][category]["px2"].extend(values)
            table[record.scale_bin][category]["um2"].extend(
                value * pixel_um2 for value in values
            )
    return table


def describe(values: list[float]) -> dict:
    """Count, minimum and low percentiles of one set of areas."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        **{
            f"p{level:g}": float(np.percentile(array, level))
            for level in LOW_PERCENTILES
        },
    }


def below(values: list[float], threshold: float) -> int:
    """How many areas lie strictly below a threshold."""
    return int(np.count_nonzero(np.asarray(values) < threshold))


def summarize(sample: dict, train: dict) -> dict:
    """The figures the threshold will be read from.

    For every scale bin and unit: the distributions, and how many
    predicted instances - matched and not - lie below the smallest
    interior pore annotated anywhere in TRAIN.
    """
    summary = {}
    for bin_ in sorted(sample):
        summary[bin_] = {}
        for unit in ("px2", "um2"):
            train_min = min(train[bin_]["gt_interior"][unit])
            row = {
                category: describe(sample[bin_][category][unit])
                for category in CATEGORIES
            }
            row["train_gt_interior"] = describe(
                train[bin_]["gt_interior"][unit]
            )
            row["train_gt_border"] = describe(train[bin_]["gt_border"][unit])
            row["pred_below_train_gt_interior_min"] = {
                "matched": below(
                    sample[bin_]["pred_matched"][unit], train_min
                ),
                "unmatched": below(
                    sample[bin_]["pred_unmatched"][unit], train_min
                ),
            }
            summary[bin_][unit] = row
    return summary


def area_figure(sample: dict, train: dict, unit: str, path: Path) -> Path:
    """Area histograms per scale bin, the four categories overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins_ = sorted(sample)
    figure, axes = plt.subplots(
        1, len(bins_), figsize=(7 * len(bins_), 4.2), squeeze=False
    )
    label = "px²" if unit == "px2" else "µm²"
    for axis, bin_ in zip(axes[0], bins_):
        values = [
            np.asarray(sample[bin_][c][unit]) for c in CATEGORIES
        ]
        pooled = np.concatenate([v for v in values if v.size])
        edges = np.logspace(
            np.log10(max(pooled.min(), 1)), np.log10(pooled.max()), 60
        )
        for category, colour, data in zip(
            CATEGORIES, SERIES_COLOURS, values
        ):
            if data.size:
                axis.hist(
                    data, bins=edges, histtype="step", linewidth=2,
                    color=colour, label=f"{category} (n={data.size})",
                )
        train_min = min(train[bin_]["gt_interior"][unit])
        axis.axvline(train_min, color=TEXT_SECONDARY, linestyle="--",
                     linewidth=1)
        axis.annotate(
            f"smallest interior pore in TRAIN: {train_min:.0f} {label}",
            (train_min, 0.97), xycoords=("data", "axes fraction"),
            rotation=90, va="top", ha="right", fontsize=8,
            color=TEXT_SECONDARY,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"instance area [{label}]")
        axis.set_ylabel("instances")
        axis.set_title(f"{bin_}: calibration sample", loc="left")
        axis.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def record_in_mlflow(
    summary: dict, files: list[Path], setting: WatershedParams
) -> str:
    """One run holding the summary and the figures."""
    import mlflow

    configure_tracking()
    experiment_id = ensure_experiment(POSTPROCESSING_EXPERIMENT)
    repeated = setting != FROZEN_WATERSHED
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="step4_instance_areas"
        + (f"_at_{postprocessing_id(setting)}" if repeated else ""),
        tags={"part": "I", "plan_step": "6" if repeated else "4",
              "subset": "train"},
    ) as run:
        mlflow.log_params({
            "snapshot": "b0_seed20260907/epoch-12",
            "watershed": postprocessing_id(setting),
        })
        for bin_, units in summary.items():
            for unit, row in units.items():
                for category, stats in row.items():
                    for key, value in stats.items():
                        name = f"{bin_}/{unit}/{category}/{key}"
                        mlflow.log_metric(name.replace(".", "_"), value)
        for path in files:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--seeding", type=float, nargs=3, default=None,
        metavar=("CENTRE", "BOUNDARY", "DISTANCE_SMOOTHING"),
        help="Grow instances at this seeding instead of the frozen one, "
             "to repeat the derivation at a calibrated setting.",
    )
    parser.add_argument(
        "--foreground", type=float, nargs=2, default=None,
        metavar=("THRESHOLD", "SMOOTHING"),
    )
    args = parser.parse_args(argv)
    # Always without any size filter: the derivation has to see the
    # fragments a filter would remove.
    fields = FROZEN_WATERSHED.to_kwargs()
    if args.seeding is not None:
        fields.update(zip(
            ("center_distance_threshold", "boundary_distance_threshold",
             "distance_smoothing"), args.seeding,
        ))
    if args.foreground is not None:
        fields.update(zip(
            ("foreground_threshold", "foreground_smoothing"),
            args.foreground,
        ))
    fields.update(min_size=0, min_instance_area_um2=0.0)
    args.setting = WatershedParams(**fields)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Measure both distributions, write them and record them.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)
    sample = measure_sample(args)
    train = measure_train_gt(args)
    summary = summarize(sample, train)
    report = args.out / "instance_areas.json"
    report.write_text(json.dumps(summary, indent=2))
    raw = args.out / "instance_areas_raw.json"
    raw.write_text(json.dumps({"sample": sample, "train_gt": train}))
    figures = [
        area_figure(sample, train, unit, args.out / f"areas_{unit}.png")
        for unit in ("px2", "um2")
    ]
    run_id = record_in_mlflow(summary, [report, *figures], args.setting)
    logger.info("Wrote %s; MLflow run %s.", report, run_id)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
