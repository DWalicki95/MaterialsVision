"""
Loading a checkpoint and segmenting one image with it.

Thin on purpose: both functions hand straight to the ones the scoring
script uses, adding only what a notebook would otherwise have to
remember - installing the encoder input geometry before the model is
built, and passing each image's pixel size to the post-processing that
filters by physical area.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from materials_vision.data.samples import PreparedSample
from materials_vision.evaluation.inference import (build_segmenter, segment,
                                                   to_rgb)
from materials_vision.evaluation.watershed import (DEPLOYED_POSTPROCESSING,
                                                   WatershedParams)
from materials_vision.training import LORA_RANK, PEFT_KWARGS, prepare_geometry

logger = logging.getLogger(__name__)

# Architecture of the image encoder every checkpoint of this study was
# trained on; the weights themselves come from the checkpoint.
MODEL_TYPE = "vit_l"

# The pretrained generalist every checkpoint of this study started
# adaptation from - see PEFT_KWARGS's neighbourhood in training.py and
# the P0 pilot that chose it over the electron-microscopy generalist.
# Its own name carries the encoder architecture, unlike MODEL_TYPE above,
# which names it separately from a checkpoint file because a fine-tuned
# checkpoint holds no such name itself.
PRETRAINED_MODEL_TYPE = "vit_l_lm"


def load_segmenter(
    checkpoint: Path,
    model_type: str = MODEL_TYPE,
    device: Optional[str] = None,
) -> Any:
    """Load a checkpoint the way the scoring script does.

    Installs the encoder input geometry first: without it the model
    still runs, but sees its images at a size it was never trained on.

    Parameters
    ----------
    checkpoint : Path
    model_type : str, optional
    device : str, optional
        ``None`` lets the library pick the GPU when one is visible.

    Returns
    -------
    InstanceSegmentationWithDecoder
    """
    prepare_geometry()
    return build_segmenter(
        Path(checkpoint), model_type, PEFT_KWARGS, LORA_RANK, device=device
    )


def load_pretrained_segmenter(
    model_type: str = PRETRAINED_MODEL_TYPE,
    device: Optional[str] = None,
) -> Any:
    """Load the generalist checkpoint fine-tuning started from, as is.

    No low-rank correction and no fine-tuned decoder: the encoder and
    the instance decoder are exactly what ``micro_sam`` ships for this
    model type. This is the zero-shot baseline every trained checkpoint
    in this study is measured as an improvement over.

    Parameters
    ----------
    model_type : str, optional
    device : str, optional
        ``None`` lets the library pick the GPU when one is visible.

    Returns
    -------
    InstanceSegmentationWithDecoder

    Raises
    ------
    ValueError
        If the library's checkpoint for ``model_type`` carries no
        instance decoder to segment with.
    """
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder, get_predictor_and_decoder)

    prepare_geometry()
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type, checkpoint_path=None, device=device,
    )
    return InstanceSegmentationWithDecoder(predictor, decoder)


def predict(
    segmenter: Any,
    sample: PreparedSample,
    settings: WatershedParams = DEPLOYED_POSTPROCESSING,
) -> np.ndarray:
    """Segment one prepared sample.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
    sample : PreparedSample
        Already cropped to its content region, as a source loads it.
    settings : WatershedParams, optional
        Defaults to the post-processing the deployed model runs with.

    Returns
    -------
    np.ndarray
        ``(H, W)`` instance labels on the sample's frame.
    """
    segmenter.initialize(to_rgb(sample.image))
    return segment(
        segmenter, settings, pixel_size_um=sample.record.pixel_size_um
    )
