"""
The decoder's output, computed once and replayed under many settings.

Calibrating the post-processing means growing instances from the same
predictions under dozens of watershed settings. The encoder and the
decoder are nearly the whole cost of that and do not depend on the
setting, so their output is computed once per image, written to disk,
and every setting afterwards reads it back - on the CPU, and without the
GPU being needed again.

**Replay goes through the library's own code.** The segmenter keeps the
three maps the decoder produced and grows instances from them in
``generate``; nothing else of its state is read there. A replayed
segmenter is the library's class with those three maps put back, so the
watershed that runs is the one inference runs, not a copy of it that
could drift. Maps are stored at full precision for the same reason: a
lossy copy could move a pixel across a threshold and make the replay
disagree with the model it stands for.
"""
from pathlib import Path

import numpy as np

MAP_NAMES = ("foreground", "center_distances", "boundary_distances")


def decoder_maps(segmenter) -> dict[str, np.ndarray]:
    """Take the three maps out of an initialized segmenter.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder

    Returns
    -------
    dict of str to np.ndarray

    Raises
    ------
    RuntimeError
        If the segmenter has not been run on an image.
    """
    if not segmenter.is_initialized:
        raise RuntimeError("The segmenter has not been run on an image.")
    return {
        "foreground": np.asarray(segmenter._foreground, dtype=np.float32),
        "center_distances": np.asarray(
            segmenter._center_distances, dtype=np.float32
        ),
        "boundary_distances": np.asarray(
            segmenter._boundary_distances, dtype=np.float32
        ),
    }


def save_maps(path: Path, maps: dict[str, np.ndarray]) -> None:
    """Write one image's maps.

    Parameters
    ----------
    path : Path
    maps : dict of str to np.ndarray
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: maps[name] for name in MAP_NAMES})


def load_maps(path: Path) -> dict[str, np.ndarray]:
    """Read one image's maps back.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict of str to np.ndarray

    Raises
    ------
    ValueError
        If a map is missing or the three disagree in shape.
    """
    with np.load(path) as stored:
        missing = [name for name in MAP_NAMES if name not in stored]
        if missing:
            raise ValueError(f"{path} lacks {', '.join(missing)}.")
        maps = {name: stored[name] for name in MAP_NAMES}
    if len({array.shape for array in maps.values()}) != 1:
        raise ValueError(f"{path}: the maps differ in shape.")
    return maps


def replay_segmenter(maps: dict[str, np.ndarray]):
    """A segmenter that grows instances from stored maps.

    Parameters
    ----------
    maps : dict of str to np.ndarray

    Returns
    -------
    InstanceSegmentationWithDecoder
        Holding no model; only ``generate`` may be called on it.
    """
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

    segmenter = InstanceSegmentationWithDecoder.__new__(
        InstanceSegmentationWithDecoder
    )
    segmenter._predictor = None
    segmenter._decoder = None
    segmenter._foreground = maps["foreground"]
    segmenter._center_distances = maps["center_distances"]
    segmenter._boundary_distances = maps["boundary_distances"]
    segmenter._i = None
    segmenter._is_initialized = True
    return segmenter
