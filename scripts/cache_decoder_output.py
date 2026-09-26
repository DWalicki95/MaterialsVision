#!/usr/bin/env python3
"""
Run one checkpoint over the calibration sample once and keep its output.

Every grid of part I - the seeding thresholds, the foreground threshold
and smoothing, the minimum instance size - grows instances from the same
predictions under many settings. This script produces those predictions
once, for the images the calibration sample fixed, and writes the
decoder's three maps per image to disk. Everything after it runs on the
CPU from those files.

**The replay is checked against the model before it is trusted.** For
the first few images the instances grown from the stored maps are
compared, pixel for pixel, with the instances the live segmenter grows,
under the frozen setting and under the extremes of the grids. A single
differing pixel stops the script: a calibration read from a replay that
does not reproduce the model would calibrate something else.

Examples
--------
The snapshot the plan fixed for part I:
    $ python scripts/cache_decoder_output.py

Check the whole path on one image without touching the GPU:
    $ python scripts/cache_decoder_output.py --device cpu --limit 1 \\
        --out /tmp/decoder_cache_check
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from materials_vision.evaluation.decoder_cache import (decoder_maps, load_maps,
                                                       replay_segmenter,
                                                       save_maps)
from materials_vision.evaluation.inference import (build_segmenter, segment,
                                                   to_rgb)
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   WatershedParams)
from materials_vision.logging_config import setup_logging
from materials_vision.training import (LORA_RANK, PEFT_KWARGS, build_source,
                                       prepare_geometry)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SAMPLE = Path("checkpoints/postprocessing/calibration_sample.json")

# Fixed by the plan: the last snapshot of the first baseline seed.
DEFAULT_CHECKPOINT = Path(
    "checkpoints/e0_v2/checkpoints/b0_seed20260907/epoch-12.pt"
)

DEFAULT_OUT = Path(
    "checkpoints/postprocessing/decoder_cache/b0_seed20260907_epoch-12"
)

MODEL_TYPE = "vit_l"

N_VERIFIED = 3

# The frozen setting and the corners of the part I grids, where a
# replay that differed from the model would differ most.
VERIFICATION_SETTINGS = (
    FROZEN_WATERSHED,
    WatershedParams(0.30, 0.40, foreground_threshold=0.3,
                    foreground_smoothing=0.0),
    WatershedParams(0.30, 0.40, foreground_threshold=0.7,
                    foreground_smoothing=3.0, distance_smoothing=2.4),
)


def load_sample(path: Path) -> list[dict]:
    """The images the calibration sample fixed.

    Parameters
    ----------
    path : Path

    Returns
    -------
    list of dict
    """
    return json.loads(path.read_text())["images"]


def verify_replay(segmenter, path: Path, image_id: str) -> None:
    """Stop unless the stored maps grow exactly the live instances.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
        Initialized on the image the maps were stored from.
    path : Path
    image_id : str

    Raises
    ------
    RuntimeError
        On the first setting under which the two differ.
    """
    replayed = replay_segmenter(load_maps(path))
    for setting in VERIFICATION_SETTINGS:
        live = segment(segmenter, setting)
        stored = segment(replayed, setting)
        if not np.array_equal(live, stored):
            raise RuntimeError(
                f"{image_id}: the replay differs from the model under "
                f"{setting} in {int(np.count_nonzero(live != stored))} "
                f"pixel(s)."
            )
    logger.info(
        "%s: replay identical to the model under %d setting(s).",
        image_id, len(VERIFICATION_SETTINGS),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device", default=None,
        help="Force a device, e.g. 'cpu'. Default: the library's choice.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only the first this many images; 0 means all.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute images whose maps are already stored.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Compute, store and verify the maps for the sample.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    if not args.checkpoint.exists():
        logger.error("No such checkpoint: %s.", args.checkpoint)
        return EXIT_FAILED

    images = load_sample(args.sample)
    if args.limit > 0:
        images = images[:args.limit]
    source = build_source(args.split, args.manifest, "train")
    prepare_geometry()
    segmenter = build_segmenter(
        args.checkpoint, MODEL_TYPE, PEFT_KWARGS, LORA_RANK,
        device=args.device,
    )

    index = []
    started_s = time.perf_counter()
    for position, entry in enumerate(images):
        record = source.record(entry["index"])
        if record.image_id != entry["image_id"]:
            logger.error(
                "Sample entry %d names %s but the split holds %s there; "
                "the sample was drawn from a different split.",
                entry["index"], entry["image_id"], record.image_id,
            )
            return EXIT_FAILED
        path = args.out / f"{record.image_id}.npz"
        verify = position < N_VERIFIED
        if path.exists() and not args.force and not verify:
            index.append({**entry, "path": path.name})
            continue
        sample = source.load(entry["index"])
        segmenter.initialize(to_rgb(sample.image))
        save_maps(path, decoder_maps(segmenter))
        if verify:
            try:
                verify_replay(segmenter, path, record.image_id)
            except RuntimeError:
                logger.exception("Replay check failed; stopping.")
                return EXIT_FAILED
        index.append({**entry, "path": path.name,
                      "shape": list(sample.image.shape)})
        if (position + 1) % 10 == 0:
            logger.info("%d / %d image(s).", position + 1, len(images))

    (args.out / "index.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "sample": str(args.sample),
        "n_images": len(index),
        "n_verified": min(N_VERIFIED, len(index)),
        "images": index,
    }, indent=2))
    logger.info(
        "Stored the decoder output for %d image(s) under %s in %.1f min.",
        len(index), args.out, (time.perf_counter() - started_s) / 60,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
