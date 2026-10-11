"""
Build the data behind the MaterialsVision demo viewer from a real run.

Segments every TEST image of the chosen formulations with the deployed
checkpoint, timing each image end to end (encoder, decoder and
post-processing, then the quantitative analysis), and writes what the
viewer reads: ``data.json``, the images as JPEG and the predicted
instance labels as PNG. The expert annotation of the same images is
analysed the same way and stored next to the prediction, so the viewer
can show how far the model's numbers are from the expert's.

Run it on the machine the demo claims speed for: the recorded times
are replayed as they were measured, and the GPU name goes into the
data with them.

Usage
-----
PYTHONPATH=. python scripts/build_demo_data.py \\
    --checkpoint /path/to/epoch-12.pt \\
    --split /home/dwalicki/dane/splits/split_v1.csv \\
    --manifest /home/dwalicki/dane/manifests/v3/manifest_v3.csv \\
    --formulations AS19 AS3 \\
    --output demo_data
"""
import argparse
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

from data_prep.inventory.manifest import INSTRUMENT_TO_MICROSCOPE
from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.evaluation.matching import match_instances
from materials_vision.inspection.model import load_segmenter, predict
from materials_vision.logging_config import setup_logging
from materials_vision.quantitative_analysis.macroscopic_metrics import \
    calculate_all_metrics
from materials_vision.quantitative_analysis.quantitative_analysis import (
    EDGE_WEIGHT_KEY, PorousMaterialAnalyzer, ellipse_angle_to_image_direction)
from materials_vision.training import build_source
from materials_vision.utils import load_pixel_sizes_by_instrument

logger = logging.getLogger(__name__)

# The viewer shows images at this fraction of their resolution: enough
# to read single pores, small enough to keep the page light. Metrics
# are always computed at full resolution.
DISPLAY_SCALE = 0.8

JPEG_QUALITY = 85

# Relative tolerance when matching a recorded pixel size to the SEM
# calibration; calibrated sizes of neighbouring magnifications differ
# by at least 20%, rounding in the manifest by far less than 1%.
PIXEL_SIZE_REL_TOLERANCE = 0.01

# Labels are packed into the red (low byte) and green (high byte)
# channels of an RGB PNG, the encoding the viewer decodes.
MAX_PACKED_LABEL = 2 ** 16 - 1

# Records name a microscope by its manifest label (M1, M2), while the
# SEM calibration is keyed by instrument model (TM3000, SU8000).
MICROSCOPE_TO_INSTRUMENT = {
    microscope: instrument
    for instrument, microscope in INSTRUMENT_TO_MICROSCOPE.items()
}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--formulations", nargs="+", required=True,
        help="TEST formulations to include, in display order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default=None,
        help="Torch device; default lets the library pick the GPU",
    )
    return parser.parse_args()


def select_records(
    records: Tuple[SampleRecord, ...], formulations: List[str]
) -> Dict[str, List[SampleRecord]]:
    """
    TEST records of each requested formulation, in source order.

    Raises
    ------
    ValueError
        If a formulation has no TEST image, which usually means a typo
        or a formulation that belongs to TRAIN or VALIDATION.
    """
    selected = {
        name: [r for r in records if r.formulation == name]
        for name in formulations
    }
    missing = [name for name, recs in selected.items() if not recs]
    if missing:
        available = sorted({r.formulation for r in records})
        raise ValueError(
            f"No TEST images for {missing}; TEST holds {available}."
        )
    return selected


def magnification_from_calibration(
    microscope: str, pixel_size_um: float
) -> Optional[int]:
    """
    Nominal magnification whose calibrated pixel size matches the image.

    Read from the calibration rather than the file name, whose format
    differs between data sources.

    Parameters
    ----------
    microscope : str
        Manifest label of the microscope (``M1``, ``M2``).
    pixel_size_um : float
        Pixel size recorded for the image.

    Returns
    -------
    int or None
        None when the microscope is unknown, its instrument is not
        calibrated or no magnification matches; the viewer then leaves
        the magnification out.
    """
    instrument = MICROSCOPE_TO_INSTRUMENT.get(microscope)
    calibration = load_pixel_sizes_by_instrument().get(instrument, {})
    for magnification, calibrated_um in calibration.items():
        if abs(calibrated_um - pixel_size_um) <= (
            PIXEL_SIZE_REL_TOLERANCE * calibrated_um
        ):
            return int(magnification)
    logger.warning(
        f"No calibrated magnification for {microscope} ({instrument}) at "
        f"{pixel_size_um} um/px"
    )
    return None


def synchronize(device: str) -> None:
    """Wait for queued GPU work, so a timer measures finished work."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed_predict(
    segmenter: Any, sample: PreparedSample, device: str
) -> Tuple[np.ndarray, float]:
    """Segment one sample and return the labels and wall time in s."""
    synchronize(device)
    start = time.perf_counter()
    labels = predict(segmenter, sample)
    synchronize(device)
    return labels, time.perf_counter() - start


def analyze_labels(
    labels: np.ndarray,
    pixel_size_um: float,
    work_dir: Path,
    name: str,
) -> Tuple[PorousMaterialAnalyzer, Dict[str, Any], float]:
    """
    Run the full quantitative analysis on one label image.

    The analyzer reads masks from disk only, so the labels pass through
    a TIFF in ``work_dir``.

    Returns
    -------
    analyzer : PorousMaterialAnalyzer
    results : dict
        As returned by ``analyze_all``.
    seconds : float
        Wall time of loading and analysing.
    """
    mask_path = work_dir / f"{name}_masks.tif"
    tifffile.imwrite(mask_path, labels.astype(np.uint32))
    start = time.perf_counter()
    analyzer = PorousMaterialAnalyzer(
        str(mask_path),
        pixel_size_um,
        work_dir / "analysis",
        generate_plots=False,
        plot_boundary_rejection=False,
    )
    results = analyzer.analyze_all(generate_excel_report=False)
    return analyzer, results, time.perf_counter() - start


def pore_entries(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-pore values the viewer shows, rounded for a small file."""
    pores = results["morphology_individual"]
    angles = np.array([p["ellipse_angle"] for p in pores], dtype=float)
    directions = ellipse_angle_to_image_direction(angles)
    return [
        {
            "l": int(p["pore_id"]),
            "deq": round(float(p["equivalent_diameter"]), 1),
            "area": round(float(p["area"]), 0),
            "circ": round(float(p["circularity"]), 3),
            "ar": round(float(p["aspect_ratio"]), 3),
            "ang": round(float(p["ellipse_angle"]), 1),
            "dir": round(float(direction), 1),
            "cn": p.get("coordination_number"),
            "w": round(float(p[EDGE_WEIGHT_KEY]), 3),
        }
        for p, direction in zip(pores, directions)
    ]


def image_summary(
    analyzer: PorousMaterialAnalyzer, results: Dict[str, Any]
) -> Dict[str, Any]:
    """Per-image descriptors shown in the detail panel."""
    measured = {p.label for p in analyzer.props}
    excluded = sorted(
        int(p.label) for p in analyzer.props_all if p.label not in measured
    )
    diameters = [
        p["equivalent_diameter"] for p in results["morphology_individual"]
    ]
    cn_stats = results["topology_metrics"]["coordination_number_stats"]
    return {
        "porosity": float(results["global_descriptors"]["porosity"]),
        "anisotropy": float(results["global_descriptors"]["anisotropy"]),
        "nnd_um": float(results["spatial_metrics"]["mean"]),
        "cn_mean": float(cn_stats["mean"]) if cn_stats else None,
        "fractal": float(results["topology_metrics"]["fractal_dimension"]),
        "n_pores": len(measured),
        "n_excluded": len(excluded),
        "mean_deq_um": float(np.mean(diameters)) if diameters else None,
        "d50_um": float(np.median(diameters)) if diameters else None,
        "excluded": excluded,
    }


def agreement(gt: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    """True positives, false positives and false negatives at IoU 0.5."""
    match = match_instances(gt, pred)
    return {
        "tp": len(match.pairs),
        "fp": len(match.unmatched_pred_ids),
        "fn": len(match.unmatched_gt_ids),
    }


def save_assets(
    image: np.ndarray, labels: np.ndarray, out_dir: Path, image_id: str
) -> Tuple[int, int]:
    """
    Write the display JPEG and packed-label PNG of one image.

    Returns
    -------
    tuple of int
        Display width and height in pixels.

    Raises
    ------
    ValueError
        If a label does not fit the 16-bit packing.
    """
    if labels.max() > MAX_PACKED_LABEL:
        raise ValueError(f"{image_id}: label {labels.max()} exceeds 16 bit")
    height, width = labels.shape
    size = (round(width * DISPLAY_SCALE), round(height * DISPLAY_SCALE))

    Image.fromarray(image).resize(size, Image.BILINEAR).save(
        out_dir / "img" / f"{image_id}.jpg", quality=JPEG_QUALITY
    )
    packed = np.zeros((height, width, 3), dtype=np.uint8)
    packed[..., 0] = labels & 0xFF
    packed[..., 1] = labels >> 8
    Image.fromarray(packed).resize(size, Image.NEAREST).save(
        out_dir / "lbl" / f"{image_id}.png"
    )
    return size


def json_ready(value: Any) -> Any:
    """
    Plain Python values with NaN as null.

    ``json.dump`` would write NaN as a bare token that browsers refuse
    to parse, and a metric can be undefined - an image with too few
    pores has no nearest-neighbour distance.
    """
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def macro_metrics(pores: List[Dict]) -> Dict[str, Any]:
    """
    Material-level metrics of pooled pores.

    A frame with fewer than four pores has no coordination numbers, so
    the column can be missing altogether; it is added empty rather than
    letting the metric calculation fail.
    """
    frame = pd.DataFrame(pores)
    for column in ("equivalent_diameter", "aspect_ratio", "circularity",
                   "ellipse_angle", "coordination_number"):
        if column not in frame.columns:
            frame[column] = np.nan
    return calculate_all_metrics(frame)


def process_image(
    segmenter: Any,
    sample: PreparedSample,
    device: str,
    work_dir: Path,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict], List[Dict]]:
    """Segment, time, analyse and save one image."""
    record = sample.record
    pred_labels, infer_s = timed_predict(segmenter, sample, device)
    analyzer, results, analysis_s = analyze_labels(
        pred_labels, record.pixel_size_um, work_dir, record.image_id
    )
    expert_analyzer, expert_results, _ = analyze_labels(
        sample.labels, record.pixel_size_um, work_dir,
        f"{record.image_id}_expert",
    )
    expert_summary = image_summary(expert_analyzer, expert_results)
    del expert_summary["excluded"]
    width, height = save_assets(
        sample.image, pred_labels, out_dir, record.image_id
    )
    entry = {
        "id": record.image_id,
        "mag": magnification_from_calibration(
            record.microscope, record.pixel_size_um
        ),
        "px_um": record.pixel_size_um,
        "w": width,
        "h": height,
        "full_w": int(pred_labels.shape[1]),
        "full_h": int(pred_labels.shape[0]),
        "t_infer_s": round(infer_s, 3),
        "t_analysis_s": round(analysis_s, 3),
        "agreement": agreement(sample.labels, pred_labels),
        "pores": pore_entries(results),
        **image_summary(analyzer, results),
        "expert": expert_summary,
    }
    logger.info(
        f"{record.image_id}: {entry['n_pores']} pores, "
        f"inference {infer_s:.2f} s, analysis {analysis_s:.2f} s"
    )
    return (
        entry,
        results["morphology_individual"],
        expert_results["morphology_individual"],
    )


def formulation_entry(
    name: str,
    records: List[SampleRecord],
    images: List[Dict[str, Any]],
    pores: List[Dict],
    expert_pores: List[Dict],
) -> Dict[str, Any]:
    """Per-formulation block of ``data.json``."""
    totals = {
        key: sum(im["agreement"][key] for im in images)
        for key in ("tp", "fp", "fn")
    }
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    return {
        "name": name,
        "material": records[0].material,
        "instrument": MICROSCOPE_TO_INSTRUMENT.get(
            records[0].microscope, records[0].microscope
        ),
        "images": images,
        "porosity_mean": float(np.mean([im["porosity"] for im in images])),
        "macro": macro_metrics(pores),
        "expert_macro": macro_metrics(expert_pores),
        "agreement": {
            **totals,
            "f1": 2 * totals["tp"] / denominator if denominator else None,
        },
    }


def gpu_name(device: str) -> str:
    """Human-readable name of the device the times were measured on."""
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device))
    return "CPU"


def main() -> None:
    setup_logging()
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for sub in ("img", "lbl"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    source = build_source(args.split, args.manifest, "test", allow_test=True)
    selected = select_records(source.records, args.formulations)

    load_start = time.perf_counter()
    segmenter = load_segmenter(args.checkpoint, device=device)
    load_s = time.perf_counter() - load_start

    # The first call pays for CUDA kernel selection and memory pools;
    # it is timed and reported separately rather than charged to an
    # image the demo replays.
    warmup_sample = source.load(selected[args.formulations[0]][0].index)
    _, warmup_s = timed_predict(segmenter, warmup_sample, device)

    formulations = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for name, records in selected.items():
            images, pores, expert_pores = [], [], []
            for record in records:
                entry, image_pores, image_expert_pores = process_image(
                    segmenter, source.load(record.index), device,
                    work_dir, args.output,
                )
                images.append(entry)
                pores.extend(image_pores)
                expert_pores.extend(image_expert_pores)
            formulations.append(
                formulation_entry(name, records, images, pores, expert_pores)
            )

    data = {
        "run": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": args.checkpoint.name,
            "device": gpu_name(device),
            "torch": torch.__version__,
            "model_load_s": round(load_s, 2),
            "warmup_s": round(warmup_s, 2),
            "display_scale": DISPLAY_SCALE,
            "subset": "test",
        },
        "samples": formulations,
    }
    with open(args.output / "data.json", "w") as f:
        json.dump(json_ready(data), f, separators=(",", ":"),
                  allow_nan=False)
    logger.info(f"Demo data written to {args.output}")


if __name__ == "__main__":
    main()
