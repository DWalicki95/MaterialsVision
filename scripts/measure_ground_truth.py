#!/usr/bin/env python3
"""
Step 3 of the optimization plan: what the training annotation says,
before any model is involved.

Three quantities anchor the post-processing calibration to the material
rather than to the metric being optimized:

* **Porosity per material**, the target the foreground threshold is
  matched against. Averaged over images, as the evaluation's own
  porosity error is, so the threshold is read with the same measure the
  study reports.
* **Wall thickness**, the upper limit on how far the foreground map may
  be blurred. Measured only where two pores face each other, and read at
  its low percentiles, because the thinnest wall is the one a blur
  breaches first. Set against the period of the block pattern the
  decoder's upsampling leaves - the thing the blur is there to remove -
  the two lengths say whether any blur can remove the pattern without
  merging pores.
* **Areas of the pores the frame does not cut**, per scale bin, the
  lower edge the minimum instance size is later placed below.

All three are measured on TRAIN, never on VALIDATION, and without the
close-ups, which the evaluation does not score. Lengths are in pixels of
the frame, the resolution at which the decoder's maps are thresholded
and smoothed, and are converted to micrometres per image through the
pixel size the manifest records.

Examples
--------
    $ python scripts/measure_ground_truth.py
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from materials_vision.evaluation.aggregate import SCALE_OUTLIER_BIN
from materials_vision.evaluation.ground_truth import (
    areal_porosity, encoder_grid_period_px, interior_instance_areas_px2,
    wall_crossings)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (POSTPROCESSING_EXPERIMENT,
                                       configure_tracking, ensure_experiment)
from materials_vision.training import build_source

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_OUT = Path("checkpoints/postprocessing/gt_measurements")

SUBSET = "train"

PERCENTILES = (1, 5, 10, 25, 50, 75, 90)

# Lower percentiles of the area distribution, where the smallest real
# pores are; the minimum itself is reported beside them, not used,
# because one annotator's slip sets it.
AREA_PERCENTILES = (0.1, 0.5, 1, 5, 50)

# Categorical slots of the project's chart palette, in fixed order.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")

TEXT_SECONDARY = "#52514e"


@dataclass
class Group:
    """Measurements pooled over the images of one material or bin."""

    porosity: list[float] = field(default_factory=list)
    crossing_px: list[np.ndarray] = field(default_factory=list)
    wall_min_px: list[np.ndarray] = field(default_factory=list)
    wall_min_um: list[np.ndarray] = field(default_factory=list)
    n_walls: int = 0
    n_touching_walls: int = 0
    area_px2: list[np.ndarray] = field(default_factory=list)
    area_um2: list[np.ndarray] = field(default_factory=list)
    grid_period_px: list[float] = field(default_factory=list)
    n_images: int = 0


def measure_image(sample) -> dict:
    """Everything this step reads off one annotated image.

    Parameters
    ----------
    sample : PreparedSample

    Returns
    -------
    dict
    """
    walls = wall_crossings(sample.labels)
    areas_px2 = interior_instance_areas_px2(
        sample.labels, sample.border_instance
    )
    pixel_um = sample.record.pixel_size_um
    return {
        "porosity": areal_porosity(sample.labels),
        "crossing_px": walls.thickness_px.astype(np.float32),
        "wall_min_px": walls.per_wall_min_px,
        "wall_min_um": walls.per_wall_min_px * pixel_um,
        "n_walls": int(walls.per_wall_min_px.size),
        "n_touching_walls": walls.n_touching_walls,
        "area_px2": areas_px2,
        "area_um2": areas_px2 * pixel_um ** 2,
        "grid_period_px": encoder_grid_period_px(sample.labels.shape),
    }


def add_to(group: Group, measured: dict) -> None:
    """Pool one image's measurements into a group."""
    group.porosity.append(measured["porosity"])
    group.crossing_px.append(measured["crossing_px"])
    group.wall_min_px.append(measured["wall_min_px"])
    group.wall_min_um.append(measured["wall_min_um"])
    group.n_walls += measured["n_walls"]
    group.n_touching_walls += measured["n_touching_walls"]
    group.area_px2.append(measured["area_px2"])
    group.area_um2.append(measured["area_um2"])
    group.grid_period_px.append(measured["grid_period_px"])
    group.n_images += 1


def percentiles(values: np.ndarray, levels) -> dict[str, float]:
    """Named percentiles, or nothing for an empty sample."""
    if values.size == 0:
        return {}
    return {
        f"p{level:g}": float(np.percentile(values, level))
        for level in levels
    }


def summarize(group: Group) -> dict:
    """Reduce a group to the figures the plan reads.

    Parameters
    ----------
    group : Group

    Returns
    -------
    dict
    """
    porosity = np.asarray(group.porosity)
    wall_min_px = _joined(group.wall_min_px)
    area_px2 = _joined(group.area_px2)
    area_um2 = _joined(group.area_um2)
    return {
        "n_images": group.n_images,
        "porosity": {
            "mean": float(porosity.mean()),
            "std": float(porosity.std(ddof=1)) if porosity.size > 1
            else float("nan"),
            "min": float(porosity.min()),
            "max": float(porosity.max()),
        },
        "wall_crossing_px": percentiles(
            _joined(group.crossing_px), PERCENTILES
        ),
        "wall_min_px": percentiles(wall_min_px, PERCENTILES),
        "wall_min_um": percentiles(_joined(group.wall_min_um), PERCENTILES),
        # Walls the annotation drew with background between the pores.
        # Where it drew them touching, the thickness is zero by
        # convention, and the low percentiles of the full distribution
        # would report that convention instead of the material.
        "wall_min_px_drawn": percentiles(
            wall_min_px[wall_min_px > 0], PERCENTILES
        ),
        "n_walls": group.n_walls,
        "touching_wall_share": (
            group.n_touching_walls / group.n_walls if group.n_walls
            else float("nan")
        ),
        "grid_period_px": {
            "min": float(min(group.grid_period_px)),
            "max": float(max(group.grid_period_px)),
            "median": float(np.median(group.grid_period_px)),
        },
        "walls_thinner_than_grid_period_share": _share_below(
            wall_min_px, float(np.median(group.grid_period_px))
        ),
        "drawn_walls_thinner_than_grid_period_share": _share_below(
            wall_min_px[wall_min_px > 0],
            float(np.median(group.grid_period_px)),
        ),
        "n_interior_instances": int(area_px2.size),
        "interior_area_px2": {
            "min": float(area_px2.min()) if area_px2.size else float("nan"),
            **percentiles(area_px2, AREA_PERCENTILES),
        },
        "interior_area_um2": {
            "min": float(area_um2.min()) if area_um2.size else float("nan"),
            **percentiles(area_um2, AREA_PERCENTILES),
        },
    }


def _joined(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0)


def _share_below(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.count_nonzero(values < threshold) / values.size)


def collect(args: argparse.Namespace) -> tuple[dict, dict, list]:
    """Measure every scored TRAIN image, grouped two ways.

    Returns
    -------
    by_material, by_scale_bin : dict of str to Group
    per_image : list of dict
        Porosity and wall figures per image, for the record.
    """
    source = build_source(args.split, args.manifest, SUBSET)
    by_material: dict[str, Group] = defaultdict(Group)
    by_scale_bin: dict[str, Group] = defaultdict(Group)
    per_image = []
    n_close_ups = 0
    for index, record in enumerate(source.records):
        if record.scale_bin == SCALE_OUTLIER_BIN:
            n_close_ups += 1
            continue
        measured = measure_image(source.load(index))
        add_to(by_material[record.material], measured)
        add_to(by_scale_bin[record.scale_bin], measured)
        per_image.append({
            "image_id": record.image_id,
            "material": record.material,
            "scale_bin": record.scale_bin,
            "microscope": record.microscope,
            "pixel_size_um": record.pixel_size_um,
            "porosity": measured["porosity"],
            "n_walls": measured["n_walls"],
            "wall_min_p10_px": percentiles(
                measured["wall_min_px"], (10,)
            ).get("p10"),
            "grid_period_px": measured["grid_period_px"],
        })
        if len(per_image) % 50 == 0:
            logger.info("Measured %d image(s).", len(per_image))
    logger.info(
        "Measured %d TRAIN image(s); %d close-up(s) left out.",
        len(per_image), n_close_ups,
    )
    return by_material, by_scale_bin, per_image


def porosity_figure(by_material: dict, destination: Path) -> Path:
    """Per-image porosity of each material, with its mean marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    for position, (name, colour) in enumerate(
        zip(sorted(by_material), SERIES_COLOURS)
    ):
        values = np.asarray(by_material[name].porosity) * 100
        jitter = rng.uniform(-0.15, 0.15, values.size)
        axis.scatter(
            position + jitter, values, s=12, color=colour, alpha=0.6,
            linewidths=0,
        )
        axis.hlines(
            values.mean(), position - 0.3, position + 0.3, color=colour,
            linewidth=2,
        )
        axis.annotate(
            f"{values.mean():.1f}%  (n={values.size})",
            (position + 0.32, values.mean()), va="center", fontsize=9,
            color=TEXT_SECONDARY,
        )
    axis.set_xticks(range(len(by_material)))
    axis.set_xticklabels(sorted(by_material))
    axis.set_ylabel("areal porosity per image [%]")
    axis.set_title("Annotated porosity, TRAIN", loc="left")
    _recede(axis)
    return _save(figure, destination / "porosity_by_material.png")


def wall_figure(by_scale_bin: dict, destination: Path) -> Path:
    """Cumulative distribution of the thinnest point of every wall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for name, colour in zip(sorted(by_scale_bin), SERIES_COLOURS):
        group = by_scale_bin[name]
        values = np.sort(_joined(group.wall_min_px))
        if values.size == 0:
            continue
        axis.step(
            values, np.arange(1, values.size + 1) / values.size,
            where="post", color=colour, linewidth=2,
            label=f"{name} (n={values.size} walls)",
        )
        period = float(np.median(group.grid_period_px))
        axis.axvline(period, color=colour, linewidth=1, linestyle="--")
        axis.annotate(
            f"grid period {period:.1f} px", (period, 0.04),
            rotation=90, fontsize=8, color=TEXT_SECONDARY,
            xytext=(3, 0), textcoords="offset points",
        )
    # Linear through one pixel so the walls drawn as touching, at zero,
    # stay on the axis instead of being dropped by a log scale.
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlim(left=0)
    axis.set_xlabel("thinnest point of a wall [px of the frame]")
    axis.set_ylabel("share of walls at or below")
    axis.set_title("Walls between facing pores, TRAIN", loc="left")
    axis.legend(frameon=False)
    _recede(axis)
    return _save(figure, destination / "wall_thickness_by_scale_bin.png")


def area_figure(by_scale_bin: dict, destination: Path) -> Path:
    """Distribution of interior pore areas per scale bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    all_areas = _joined([
        _joined(group.area_um2) for group in by_scale_bin.values()
    ])
    bins = np.logspace(
        np.log10(max(all_areas.min(), 1e-3)),
        np.log10(all_areas.max()), 60,
    )
    for name, colour in zip(sorted(by_scale_bin), SERIES_COLOURS):
        values = _joined(by_scale_bin[name].area_um2)
        axis.hist(
            values, bins=bins, histtype="step", color=colour, linewidth=2,
            label=f"{name} (n={values.size})",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("area of a pore the frame does not cut [µm²]")
    axis.set_ylabel("pores")
    axis.set_title("Interior pore areas, TRAIN", loc="left")
    axis.legend(frameon=False)
    _recede(axis)
    return _save(figure, destination / "interior_areas_by_scale_bin.png")


def _recede(axis) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)


def _save(figure, path: Path) -> Path:
    import matplotlib.pyplot as plt

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def flatten_metrics(prefix: str, summary: dict) -> dict[str, float]:
    """Turn a nested summary into MLflow metric names."""
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            metrics.update(flatten_metrics(name, value))
        elif isinstance(value, (int, float)) and value == value:
            metrics[name.replace(".", "_")] = float(value)
    return metrics


def record_in_mlflow(report: dict, files: list[Path], args) -> str:
    """Log the summary, the JSON and the figures as one MLflow run."""
    import mlflow

    configure_tracking()
    experiment_id = ensure_experiment(POSTPROCESSING_EXPERIMENT)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="step3_gt_measurements",
        tags={"part": "I", "plan_step": "3", "subset": SUBSET},
    ) as run:
        mlflow.log_params({
            "subset": SUBSET, "close_ups": "excluded",
            "split": str(args.split), "manifest": str(args.manifest),
        })
        for group_kind in ("by_material", "by_scale_bin"):
            for name, summary in report[group_kind].items():
                mlflow.log_metrics(
                    flatten_metrics(f"{group_kind}/{name}", summary)
                )
        for path in files:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Measure, write the report and the figures, and record them.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    by_material, by_scale_bin, per_image = collect(args)
    report = {
        "subset": SUBSET,
        "close_ups": "excluded",
        "by_material": {k: summarize(v) for k, v in by_material.items()},
        "by_scale_bin": {k: summarize(v) for k, v in by_scale_bin.items()},
        "per_image": per_image,
    }
    report_path = args.out / "gt_measurements.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s.", report_path)
    figures = [
        porosity_figure(by_material, args.out),
        wall_figure(by_scale_bin, args.out),
        area_figure(by_scale_bin, args.out),
    ]
    run_id = record_in_mlflow(report, [report_path, *figures], args)
    logger.info("Recorded as MLflow run %s.", run_id)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
