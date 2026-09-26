#!/usr/bin/env python3
"""
Step 5 of the optimization plan: the two post-processing grids.

Both grids are read off the stored decoder output of the calibration
snapshot, on the calibration sample of TRAIN, so no model runs here and
every setting sees exactly the same predictions. The minimum physical
instance area derived in step 4 applies throughout.

* **Knob A** - verification of the seeding thresholds: the grid of the
  2026-09-14 calibration crossed with three values of distance
  smoothing. The setting in use holds unless some setting beats its F1
  by more than one noise band.
* **Knob B** - the foreground threshold crossed with the foreground
  smoothing, at the seeding setting knob A leaves. The threshold is
  anchored per material where matched pores are drawn neither larger
  nor smaller than annotated; porosity is read beside it.

Every setting becomes a child run in MLflow, with its parameters and its
figures pooled and per material, so the grid can be explored in the
interface; the decision computed by the pre-registered rule is written
to the parent run and to the JSON report.

Examples
--------
    $ python scripts/postprocessing_grid.py --knob A
    $ python scripts/postprocessing_grid.py --knob B \\
        --seeding 0.30 0.40 1.6
"""
import argparse
import itertools
import json
import logging
import multiprocessing
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from materials_vision.evaluation.aggregate import (aggregate, cross_sections,
                                                   evaluate_image)
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.decoder_cache import (load_maps,
                                                       replay_segmenter)
from materials_vision.evaluation.inference import segment
from materials_vision.evaluation.postprocessing_calibration import (
    seeding_decision, zero_crossing)
from materials_vision.evaluation.size_bins import load_size_bins
from materials_vision.evaluation.watershed import (MIN_INSTANCE_AREA_UM2,
                                                   WatershedParams)
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (POSTPROCESSING_EXPERIMENT,
                                       configure_tracking, ensure_experiment,
                                       postprocessing_id)
from materials_vision.training import build_source

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

DEFAULT_CACHE = Path(
    "checkpoints/postprocessing/decoder_cache/b0_seed20260907_epoch-12"
)

DEFAULT_OUT = Path("checkpoints/postprocessing/grids")

# Knob A: the 2026-09-14 grid, crossed with distance smoothing.
CENTRE_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
BOUNDARY_THRESHOLDS = (0.35, 0.40, 0.45, 0.50)
DISTANCE_SMOOTHINGS = (1.0, 1.6, 2.4)
SEEDING_IN_USE = (0.30, 0.40, 1.6)

# Knob A widened past every edge the first grid's winner sat on, as the
# edge rule requires; decided on the union with the first grid.
#
# Stage 1 (2026-09-24): past all three edges of the first grid's winner.
# Stage 2 (2026-09-24): one step further in distance smoothing only, the
# one edge the union's winner still sat on.
EXTENSIONS = {
    1: ((0.10, 0.15, 0.20, 0.25), (0.45, 0.50, 0.55, 0.60),
        (0.5, 1.0, 1.6, 2.4, 3.2)),
    2: ((0.20, 0.25, 0.30), (0.45, 0.50, 0.55, 0.60), (4.0, 4.8)),
}


def report_stem(stage: int) -> str:
    """File stem of a knob A grid's report; stage 0 is the first grid."""
    return {0: "knobA", 1: "knobA_extended"}.get(
        stage, f"knobA_extended{stage}"
    )


# Knob B: the populated floor of the confidence histogram, and the
# blurs from no blur up to one wider than most walls.
FOREGROUND_THRESHOLDS = tuple(np.round(np.arange(0.20, 0.801, 0.05), 2))
FOREGROUND_SMOOTHINGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
FOREGROUND_IN_USE = (0.5, 1.0)

MATERIALS = ("AS", "K", "VAB")

# Pooled figures shown per setting; the same names per material.
FIGURES = (
    "f1", "precision", "recall", "pore_count_error",
    "merges_per_100_gt", "splits_per_100_gt", "mean_porosity_error_pp",
    "median_diameter_log_ratio", "median_diameter_error", "n_pred",
    "n_gt",
)

_WORKER: dict = {}


def knob_a_grid(
    centres=CENTRE_THRESHOLDS,
    boundaries=BOUNDARY_THRESHOLDS,
    smoothings=DISTANCE_SMOOTHINGS,
) -> list[WatershedParams]:
    """Seeding thresholds crossed with distance smoothing."""
    return [
        WatershedParams(
            center_distance_threshold=c, boundary_distance_threshold=b,
            distance_smoothing=d,
            min_instance_area_um2=MIN_INSTANCE_AREA_UM2,
        )
        for d, c, b in itertools.product(smoothings, centres, boundaries)
    ]


def setting_of(row: dict) -> WatershedParams:
    """The setting a report row describes."""
    names = WatershedParams.__dataclass_fields__
    return WatershedParams(**{k: v for k, v in row.items() if k in names})


def result_of(row: dict, n_gt: int) -> SimpleNamespace:
    """The figures the choice rule reads, from a report row.

    ``n_gt`` is passed in because the first grid's report predates the
    column; every setting of a grid is scored on the same images, so
    it is one number for the whole grid.
    """
    return SimpleNamespace(
        f1=row["f1"], n_gt=n_gt,
        pore_count_error=row["pore_count_error"],
        merges_per_100_gt=row["merges_per_100_gt"],
        splits_per_100_gt=row["splits_per_100_gt"],
        boundary_f1={DECISION_SCALE: row.get("boundary_f1", 0.0)},
    )


def union_with_first_grid(
    first_rows: list[dict], new_rows: list[dict]
) -> tuple[dict, int]:
    """Earlier grids and the new one as one mapping, checked on overlap.

    ``first_rows`` may hold the rows of several earlier grids; a setting
    they share is checked the same way as one shared with the new grid.

    Returns
    -------
    union : dict of WatershedParams to SimpleNamespace
    n_overlap : int
        Settings scored in both grids.

    Raises
    ------
    RuntimeError
        If a setting scored twice gave different figures: the replay is
        deterministic, so a difference means the two grids did not see
        the same predictions and cannot be read together.
    """
    n_gts = {
        row["n_gt"] for row in [*first_rows, *new_rows] if "n_gt" in row
    }
    if len(n_gts) != 1:
        raise RuntimeError(f"Settings were scored on different sets: {n_gts}")
    n_gt = n_gts.pop()
    union: dict = {}
    n_overlap = 0
    for row in [*first_rows, *new_rows]:
        setting = setting_of(row)
        if setting in union:
            n_overlap += 1
            if abs(union[setting].f1 - row["f1"]) > 1e-12:
                raise RuntimeError(
                    f"{setting}: F1 {union[setting].f1} in the first grid "
                    f"but {row['f1']} now."
                )
        union[setting] = result_of(row, n_gt)
    return union, n_overlap


def knob_b_grid(seeding: tuple[float, float, float]) -> list[WatershedParams]:
    """Foreground threshold crossed with foreground smoothing."""
    centre, boundary, distance = seeding
    return [
        WatershedParams(
            center_distance_threshold=centre,
            boundary_distance_threshold=boundary,
            distance_smoothing=distance,
            foreground_threshold=float(t), foreground_smoothing=s,
            min_instance_area_um2=MIN_INSTANCE_AREA_UM2,
        )
        for s, t in itertools.product(
            FOREGROUND_SMOOTHINGS, FOREGROUND_THRESHOLDS
        )
    ]


def _init_worker(split, manifest, size_bins, cache, settings) -> None:
    """Open the data once per worker process.

    Each worker is held to one thread. The work is already split across
    processes, and numerical libraries that each start a thread per core
    in every process multiply into far more threads than cores: the
    first full grid took five times its estimate that way.
    """
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    threadpool_limits(1)
    setup_logging()
    _WORKER["source"] = build_source(split, manifest, "train")
    _WORKER["size_bins"] = load_size_bins(size_bins)
    _WORKER["cache"] = cache
    _WORKER["settings"] = settings


def _score_image(entry: dict) -> list:
    """Every setting of the grid on one image."""
    sample = _WORKER["source"].load(entry["index"])
    maps = load_maps(_WORKER["cache"] / entry["path"])
    segmenter = replay_segmenter(maps)
    return [
        evaluate_image(
            sample.record, sample.labels,
            segment(segmenter, setting,
                    pixel_size_um=sample.record.pixel_size_um),
            size_bins=_WORKER["size_bins"],
            boundary_scales=(DECISION_SCALE,),
        )
        for setting in _WORKER["settings"]
    ]


def score_grid(args, settings: list[WatershedParams]) -> dict:
    """Aggregate every setting, pooled and per material."""
    index = json.loads((args.cache / "index.json").read_text())["images"]
    if args.limit > 0:
        index = index[:args.limit]
    per_setting: list[list] = [[] for _ in settings]
    with multiprocessing.get_context("fork").Pool(
        args.workers, initializer=_init_worker,
        initargs=(args.split, args.manifest, args.size_bins, args.cache,
                  settings),
    ) as pool:
        for done, evaluations in enumerate(
            pool.imap_unordered(_score_image, index), start=1
        ):
            for position, evaluation in enumerate(evaluations):
                per_setting[position].append(evaluation)
            if done % 10 == 0:
                logger.info("%d / %d image(s) scored.", done, len(index))
    return {
        setting: {
            "overall": aggregate(evaluations, label="overall"),
            "materials": {
                section.label.split("=")[1]: section
                for section in cross_sections(evaluations, "material")
            },
        }
        for setting, evaluations in zip(settings, per_setting)
    }


def figures_of(scored: dict) -> dict[str, float]:
    """Flat metric names for one setting."""
    metrics = {}
    for name in FIGURES:
        value = getattr(scored["overall"], name)
        if np.isfinite(value):
            metrics[name] = float(value)
    metrics["boundary_f1"] = float(
        scored["overall"].boundary_f1.get(DECISION_SCALE, float("nan"))
    )
    for material, section in scored["materials"].items():
        for name in FIGURES:
            value = getattr(section, name)
            if np.isfinite(value):
                metrics[f"{material}/{name}"] = float(value)
    return metrics


def decide_knob_a(overall: dict) -> dict:
    """The pre-registered seeding rule applied to a grid.

    Parameters
    ----------
    overall : dict of WatershedParams to result
        Pooled figures per setting; must include the setting in use.
    """
    reference = WatershedParams(
        center_distance_threshold=SEEDING_IN_USE[0],
        boundary_distance_threshold=SEEDING_IN_USE[1],
        distance_smoothing=SEEDING_IN_USE[2],
        min_instance_area_um2=MIN_INSTANCE_AREA_UM2,
    )
    decision = seeding_decision(overall, reference)
    decision["chosen"] = asdict(decision["chosen"])
    decision["tied"] = [
        {**asdict(s), "f1": overall[s].f1,
         "pore_count_error": overall[s].pore_count_error}
        for s in decision["tied"]
    ]
    return decision


def _along_threshold(results: dict, smoothing: float) -> list[tuple]:
    """Settings at one smoothing, in increasing threshold order."""
    return sorted(
        ((s, r) for s, r in results.items()
         if s.foreground_smoothing == smoothing),
        key=lambda item: item[0].foreground_threshold,
    )


def decide_knob_b(results: dict) -> dict:
    """Per-material anchors and the per-smoothing F1 of the grid.

    For every smoothing: where the median signed diameter drift of each
    material crosses zero, where its porosity error does, and the F1 at
    every threshold. The rule itself - a narrow band of the three
    anchors excluding 0.5, F1 not falling - is read from this by the
    owner of the study, as pre-registered.
    """
    readings = {}
    for smoothing in FOREGROUND_SMOOTHINGS:
        rows = _along_threshold(results, smoothing)
        thresholds = [s.foreground_threshold for s, _ in rows]
        reading = {"f1": [r["overall"].f1 for _, r in rows],
                   "thresholds": thresholds, "materials": {}}
        for material in MATERIALS:
            sections = [r["materials"].get(material) for _, r in rows]
            reading["materials"][material] = {
                "ecd_zero_at": zero_crossing(thresholds, [
                    s.median_diameter_log_ratio if s else float("nan")
                    for s in sections
                ]),
                "porosity_zero_at": zero_crossing(thresholds, [
                    s.mean_porosity_error_pp if s else float("nan")
                    for s in sections
                ]),
            }
        readings[f"{smoothing:g}"] = reading
    return readings


def record(args, results: dict, decision: dict, files: list[Path]) -> str:
    """Parent run with the decision, one child run per setting."""
    import mlflow

    configure_tracking()
    experiment_id = ensure_experiment(POSTPROCESSING_EXPERIMENT)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"step5_knob{args.knob}"
        + report_stem(args.extend)[len("knobA"):],
        tags={"part": "I", "plan_step": "5", "knob": args.knob,
              "subset": "train", "extension_stage": str(args.extend)},
    ) as parent:
        mlflow.log_params({
            "snapshot": "b0_seed20260907/epoch-12",
            "n_images": results[next(iter(results))]["overall"].n_images,
            "n_settings": len(results),
            "min_instance_area_um2": MIN_INSTANCE_AREA_UM2,
        })
        for path in files:
            mlflow.log_artifact(str(path))
        for setting, scored in results.items():
            with mlflow.start_run(
                experiment_id=experiment_id, nested=True,
                run_name=postprocessing_id(setting),
            ):
                mlflow.log_params(asdict(setting))
                mlflow.log_metrics(figures_of(scored))
        return parent.info.run_id


def knob_a_figure(overall: dict, path: Path) -> Path:
    """F1 and pore count error over the seeding grid, per smoothing.

    Draws whatever settings are given on the union of their values, so
    two grids read together appear as one landscape; combinations
    neither grid scored stay blank.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = sorted({s.center_distance_threshold for s in overall})
    boundaries = sorted({s.boundary_distance_threshold for s in overall})
    smoothings = sorted({s.distance_smoothing for s in overall})
    by_key = {
        (s.center_distance_threshold, s.boundary_distance_threshold,
         s.distance_smoothing): r
        for s, r in overall.items()
    }
    fig, axes = plt.subplots(2, len(smoothings),
                             figsize=(4.5 * len(smoothings), 8),
                             squeeze=False)
    for column, smoothing in enumerate(smoothings):
        for row, (name, cmap) in enumerate(
            (("f1", "viridis"), ("pore_count_error", "RdBu_r"))
        ):
            grid = np.array([[
                getattr(by_key[(c, b, smoothing)], name)
                if (c, b, smoothing) in by_key else np.nan
                for b in boundaries] for c in centres])
            axis = axes[row, column]
            limit = (np.nanmax(np.abs(grid))
                     if name == "pore_count_error" else None)
            shown = axis.imshow(
                grid, cmap=cmap, aspect="auto",
                vmin=-limit if limit else None, vmax=limit,
            )
            for i, j in itertools.product(range(grid.shape[0]),
                                          range(grid.shape[1])):
                if np.isfinite(grid[i, j]):
                    axis.text(j, i, f"{grid[i, j]:.3f}", ha="center",
                              va="center", fontsize=7, color="white")
            axis.set_xticks(range(len(boundaries)),
                            [f"{b:g}" for b in boundaries])
            axis.set_yticks(range(len(centres)),
                            [f"{c:g}" for c in centres])
            axis.set_xlabel("boundary distance threshold")
            axis.set_ylabel("centre distance threshold")
            axis.set_title(f"{name}, distance smoothing {smoothing:g}",
                           loc="left", fontsize=10)
            fig.colorbar(shown, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _figure(scored: dict, name: str, material: Optional[str]) -> float:
    """One figure of a setting, pooled or for a material it holds."""
    if material is None:
        return getattr(scored["overall"], name)
    section = scored["materials"].get(material)
    return getattr(section, name) if section else float("nan")


def knob_b_figure(results: dict, path: Path) -> Path:
    """F1, signed diameter drift and porosity error along the threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#4a3aa7")
    panels = (
        ("f1", "instance F1 (pooled)", None),
        ("median_diameter_log_ratio", "median log(d_pred/d_GT)", "AS"),
        ("median_diameter_log_ratio", "median log(d_pred/d_GT)", "K"),
        ("median_diameter_log_ratio", "median log(d_pred/d_GT)", "VAB"),
        ("mean_porosity_error_pp", "porosity error [pp]", None),
        ("pore_count_error", "pore count error (pooled)", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for axis, (name, label, material) in zip(axes.ravel(), panels):
        for smoothing, colour in zip(FOREGROUND_SMOOTHINGS, colours):
            rows = _along_threshold(results, smoothing)
            values = [
                _figure(r, name, material) for _, r in rows
            ]
            axis.plot([s.foreground_threshold for s, _ in rows], values,
                      color=colour, linewidth=2, marker="o", markersize=3,
                      label=f"blur {smoothing:g}")
        if name != "f1":
            axis.axhline(0, color="#52514e", linewidth=1)
        axis.axvline(0.5, color="#52514e", linestyle="--", linewidth=1)
        axis.set_title(f"{label}" + (f", {material}" if material else ""),
                       loc="left", fontsize=10)
        axis.set_xlabel("foreground threshold")
        axis.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_table(results: dict, path: Path) -> Path:
    """One CSV row per setting, every figure a column."""
    import csv

    rows = [{**asdict(s), **figures_of(r)} for s, r in results.items()]
    columns = sorted({key for row in rows for key in row},
                     key=lambda k: (k not in rows[0], k))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knob", required=True, choices=("A", "B"))
    parser.add_argument(
        "--seeding", type=float, nargs=3, default=SEEDING_IN_USE,
        metavar=("CENTRE", "BOUNDARY", "DISTANCE_SMOOTHING"),
        help="Seeding setting knob B runs at: the one knob A leaves.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--extend", type=int, default=0, choices=[0, *EXTENSIONS],
        help="Knob A only: score this stage of widening and decide on "
             "its union with every earlier stage's report.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only the first this many images, for a dry run; results "
             "are then not recorded in MLflow.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Score one grid, apply its rule, write and record everything.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.extend and args.knob != "A":
        logger.error("--extend applies to knob A only.")
        return EXIT_FAILED
    if args.knob == "B":
        settings = knob_b_grid(tuple(args.seeding))
    elif args.extend:
        settings = knob_a_grid(*EXTENSIONS[args.extend])
    else:
        settings = knob_a_grid()
    logger.info("Knob %s: %d setting(s).", args.knob, len(settings))
    results = score_grid(args, settings)
    rows = [{**asdict(s), **figures_of(r)} for s, r in results.items()]

    extra: dict = {}
    if args.knob == "B":
        decision = decide_knob_b(results)
        drawn = results
    elif args.extend and not args.limit:
        earlier = [
            row
            for stage in range(args.extend)
            for row in json.loads(
                (args.out / f"{report_stem(stage)}.json").read_text()
            )["settings"]
        ]
        overall, n_overlap = union_with_first_grid(earlier, rows)
        logger.info(
            "Union of %d grid(s): %d setting(s); %d scored more than "
            "once, all identical.", args.extend + 1, len(overall),
            n_overlap,
        )
        decision = decide_knob_a(overall)
        extra = {"n_union": len(overall), "n_overlap": n_overlap}
        drawn = overall
    else:
        overall = {s: r["overall"] for s, r in results.items()}
        # A dry run of the widened grid lacks the setting in use, which
        # only the first grid holds, so there is nothing to decide.
        decision = (
            {"dry_run": True} if args.extend
            else decide_knob_a(overall)
        )
        drawn = overall

    stem = (
        (report_stem(args.extend) if args.knob == "A" else "knobB")
        + ("_dryrun" if args.limit else "")
    )
    report = args.out / f"{stem}.json"
    report.write_text(json.dumps({
        "knob": args.knob,
        "seeding": list(args.seeding),
        "decision": decision,
        **extra,
        "settings": rows,
    }, indent=2, default=str))
    table = write_table(results, args.out / f"{stem}.csv")
    plot = (knob_a_figure if args.knob == "A" else knob_b_figure)(
        drawn, args.out / f"{stem}.png"
    )
    logger.info("Decision: %s", json.dumps(decision, default=str)[:2000])
    if args.limit:
        logger.info("Dry run on %d image(s); not recorded.", args.limit)
        return EXIT_OK
    run_id = record(args, results, decision, [report, table, plot])
    logger.info("Wrote %s; MLflow run %s.", report, run_id)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
