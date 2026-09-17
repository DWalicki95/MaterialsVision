"""
Running a trained checkpoint the way it will actually be used.

Three separate questions need a checkpoint turned into instance masks -
how well it segments, how fast its batch statistics should move, and
where the watershed thresholds belong - and all three want the same
loading and the same forward pass. Keeping that in one place means an
answer to one of them cannot rest on a slightly different inference
path than an answer to another.

**The adaptation has to be rebuilt before the weights will fit.** A
checkpoint stores the corrected weights but not the shape of the
wrapper they belong in, and the weights only load into a wrapper built
at the same rank. Rather than ask a caller to remember which rank a
given file was trained at, the rank is read back out of the weights:
the first factor of the correction has it as its leading dimension.

**The images are monochrome and the encoder takes three channels.** The
one working channel is repeated rather than converted, and values are
put on the 0-255 scale the predictor checks for.
"""
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from materials_vision.evaluation.aggregate import (ImageEvaluation,
                                                   evaluate_image)
from materials_vision.evaluation.boundary import DECISION_SCALE
from materials_vision.evaluation.watershed import (FROZEN_WATERSHED,
                                                   WatershedParams)

logger = logging.getLogger(__name__)

RGB_CHANNELS = 3

GREY_LEVELS = 255.0


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Present one working channel the way the predictor expects it.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)``.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8.
    """
    scaled = image.astype(np.float32)
    if scaled.max() <= 1.0:
        scaled = scaled * GREY_LEVELS
    stacked = np.repeat(scaled[..., None], RGB_CHANNELS, axis=-1)
    return np.clip(stacked, 0, GREY_LEVELS).astype(np.uint8)


def detect_lora_rank(checkpoint: Path, default_rank: int) -> int:
    """Read the size of the low-rank correction out of a checkpoint.

    Asking the caller to remember invites exactly the mistake this
    replaces: a rank-32 checkpoint refused by a rank-8 wrapper,
    hundreds of lines of shape mismatches, and an hour of training with
    nothing to show.

    Parameters
    ----------
    checkpoint : Path
    default_rank : int
        Returned, with a warning, if the checkpoint holds no low-rank
        weights at all.

    Returns
    -------
    int
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = state.get("model_state", state)
    for key, value in model_state.items():
        if key.endswith("w_a_linear_q.weight"):
            return int(value.shape[0])
    logger.warning(
        "%s holds no low-rank weights; assuming rank %d.",
        checkpoint, default_rank,
    )
    return default_rank


def build_segmenter(
    checkpoint: Path,
    model_type: str,
    peft_kwargs: dict[str, Any],
    default_rank: int,
) -> Any:
    """Load a trained checkpoint as an automatic instance segmenter.

    Parameters
    ----------
    checkpoint : Path
    model_type : str
        Architecture of the image encoder the checkpoint was trained
        on, not the name of a pretrained checkpoint.
    peft_kwargs : dict
        The adaptation configuration the checkpoint was trained under.
        Its rank is replaced by the one read from the weights.
    default_rank : int
        Used only when the checkpoint carries no correction.

    Returns
    -------
    InstanceSegmentationWithDecoder
    """
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder, get_predictor_and_decoder)

    rebuilt = dict(peft_kwargs)
    rebuilt["rank"] = detect_lora_rank(checkpoint, default_rank)
    logger.info(
        "%s: rebuilding the wrapper at rank %d.", checkpoint,
        rebuilt["rank"],
    )
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=str(checkpoint),
        peft_kwargs=rebuilt,
    )
    return InstanceSegmentationWithDecoder(predictor, decoder)


def segment(
    segmenter: Any, settings: WatershedParams = FROZEN_WATERSHED
) -> np.ndarray:
    """Grow instances from decoder output that is already computed.

    Separated from the encoding step so that several settings can be
    scored on one image without recomputing the expensive part: the
    encoder and decoder run once per image, the watershed on top of
    them is cheap.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
        Already initialized on the image.
    settings : WatershedParams, optional

    Returns
    -------
    np.ndarray
        Instance labels on the same frame, background zero.
    """
    segmentation = segmenter.generate(
        output_mode="instance_segmentation", **settings.to_kwargs()
    )
    # The watershed hands back floating-point labels. They are whole
    # numbers, but an identifier that can be compared for equality
    # should not be a float, and the metrics count on integers.
    return np.asarray(segmentation).astype(np.int32)


def score_settings(
    segmenter: Any,
    source: Any,
    positions: Sequence[int],
    size_bins: Any,
    settings: Sequence[WatershedParams] = (FROZEN_WATERSHED,),
    boundary_scales: tuple[float, ...] = (DECISION_SCALE,),
) -> dict[WatershedParams, list[ImageEvaluation]]:
    """Measure one checkpoint under several watershed settings.

    Each image is encoded once and then grown into instances once per
    setting, which is what makes a second setting nearly free: the
    encoder and the decoder are almost the whole cost, and the
    watershed on top of them is not.

    One boundary tolerance is scored by default rather than the three
    the metric can report. Boundary agreement dominates the cost of an
    evaluation, and the extra tolerances answer how sensitive the score
    is to the tolerance - a question worth asking once of a finished
    model rather than of every snapshot.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
    source : SampleSource
    positions : sequence of int
        Which samples of the source to measure, in order.
    size_bins : SizeBins
    settings : sequence of WatershedParams, optional
    boundary_scales : tuple of float, optional

    Returns
    -------
    dict
        Per setting, the per-image evaluations, in the order given.
    """
    measured: dict[WatershedParams, list[ImageEvaluation]] = {
        setting: [] for setting in settings
    }
    for position in positions:
        prepared = source.load(position)
        segmenter.initialize(to_rgb(prepared.image))
        for setting in settings:
            measured[setting].append(evaluate_image(
                prepared.record, prepared.labels,
                segment(segmenter, setting),
                size_bins=size_bins,
                boundary_scales=boundary_scales,
            ))
    return measured


def decoder_modules(segmenter: Any) -> Any:
    """The part of the segmenter that is not the frozen encoder.

    Exposed because recalibrating the decoder's batch statistics has to
    reach its normalization layers, and the segmenter holds the decoder
    behind an adapter rather than as a plain attribute.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder

    Returns
    -------
    torch.nn.Module
    """
    return segmenter._decoder


def recalibrate_batch_statistics(
    segmenter: Any,
    images: list[np.ndarray],
    reset: bool = True,
) -> int:
    """Recompute the decoder's batch statistics over fixed images.

    The decoder normalizes over the batch, and training ran with one
    image per batch, so its running statistics track the handful of
    images seen most recently rather than the training distribution. A
    snapshot therefore carries whichever images happened to end its
    epoch, and two snapshots an epoch apart can differ for that reason
    alone rather than because the model changed.

    Averaging over one fixed set of images removes that difference from
    the comparison. The averaging is cumulative rather than
    exponential, so the result is the plain mean over the images given
    and does not depend on their order.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
        Already loaded; its embeddings are recomputed here per image.
    images : list of np.ndarray
        One working channel each. The same list must be used for every
        checkpoint being compared, or the comparison reintroduces the
        difference it is removing.
    reset : bool, optional
        Discard the statistics the checkpoint carries first. On is the
        honest setting: keeping them would average the new images into
        whatever the end of an epoch happened to leave behind.

    Returns
    -------
    int
        How many normalization layers were recalibrated, so a caller
        can notice a zero rather than assume it worked.
    """
    kinds = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    )
    decoder = decoder_modules(segmenter)
    layers = [
        module for module in decoder.modules()
        if isinstance(module, kinds)
    ]
    if not layers:
        return 0

    previous_momentum = []
    for layer in layers:
        previous_momentum.append(layer.momentum)
        if reset:
            layer.reset_running_stats()
        # None asks for a cumulative average over the forward passes
        # below, which is the mean over exactly these images.
        layer.momentum = None
        layer.train()

    with torch.no_grad():
        for image in images:
            segmenter.initialize(to_rgb(image))

    for layer, momentum in zip(layers, previous_momentum):
        layer.momentum = momentum
        layer.eval()
    return len(layers)


def prepared_images(source: Any, positions: Optional[list[int]]) -> list:
    """Read the working channels of a fixed set of samples.

    Parameters
    ----------
    source : SampleSource
    positions : list of int, optional
        ``None`` reads all of them.

    Returns
    -------
    list of np.ndarray
    """
    chosen = range(len(source)) if positions is None else positions
    return [source.load(position).image for position in chosen]
