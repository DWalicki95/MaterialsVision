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
the same way: the same rank, the same projections, the same blocks.
Rather than ask a caller to remember how a given file was trained, all
three are read back out of the weights' names and shapes. A checkpoint
without low-rank factors - a full-weight adaptation - is loaded into
the plain model. Loading stays strict, so anything this reading gets
wrong fails loudly instead of leaving weights behind.

**Two ways to turn decoder output into instances.** ``ais`` grows them
by a watershed from the decoder's distance maps, which is what every
figure of this study so far used. ``apg`` derives a point prompt per
seed and lets SAM's own mask decoder draw each instance, which is the
only inference path that uses the interactive branch the joint
objective trains.

**The images are monochrome and the encoder takes three channels.** The
one working channel is repeated rather than converted, and values are
put on the 0-255 scale the predictor checks for.
"""
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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

SEGMENTATION_MODES = ("ais", "apg")

# Watershed settings the prompt-based path has no step for: it seeds
# from unsmoothed maps and draws instances with SAM's mask decoder.
APG_UNUSED_SETTINGS = ("foreground_smoothing", "distance_smoothing")

# Order in which the library names the projections a correction can sit
# on; the rebuilt configuration lists them in it.
_LORA_MATRICES = ("q", "k", "v", "mlp")

# First factor of a correction, in an attention projection or in the
# MLP, with the block it sits in.
_LORA_FACTOR_KEY = re.compile(
    r"image_encoder\.blocks\.(\d+)\."
    r"(?:attn\.qkv\.w_a_linear_([qkv])|mlp\.w_a_linear_1)\.weight$"
)

_BLOCK_KEY = re.compile(r"image_encoder\.blocks\.(\d+)\.")


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


def read_model_state(checkpoint: Path) -> Mapping[str, Any]:
    """The model weights of a checkpoint, mapped rather than read.

    Parameters
    ----------
    checkpoint : Path

    Returns
    -------
    mapping of str to torch.Tensor
        Tensors are memory-mapped, so reading names and shapes does not
        pull a gigabyte of weights off the disk.
    """
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True,
    )
    return state.get("model_state", state)


def peft_kwargs_from_state(
    model_state: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """Read how a checkpoint's encoder was wrapped out of its weights.

    Parameters
    ----------
    model_state : mapping of str to torch.Tensor

    Returns
    -------
    dict or None
        The library's wrapper arguments: the rank from the first
        factor's leading dimension, the projections that carry factors,
        and the blocks that do - an empty list when it is every block,
        which is how the frozen configuration states it. ``None`` when
        the checkpoint holds no factors at all.

    Raises
    ------
    ValueError
        If the factors disagree on their rank or on which projections
        they sit on, a layout no single wrapper can rebuild.
    """
    from micro_sam.models.peft_sam import LoRASurgery

    matrices_per_block: dict[int, set[str]] = {}
    ranks = set()
    for key, value in model_state.items():
        match = _LORA_FACTOR_KEY.search(key)
        if match is None:
            continue
        block = int(match.group(1))
        matrices_per_block.setdefault(block, set()).add(
            match.group(2) or "mlp"
        )
        ranks.add(int(value.shape[0]))
    if not matrices_per_block:
        return None
    if len(ranks) != 1:
        raise ValueError(f"The low-rank factors disagree on rank: {ranks}.")
    layouts = {frozenset(m) for m in matrices_per_block.values()}
    if len(layouts) != 1:
        raise ValueError(
            "Blocks carry factors on different projections: "
            f"{sorted(sorted(layout) for layout in layouts)}."
        )
    every_block = {
        int(match.group(1)) for key in model_state
        if (match := _BLOCK_KEY.search(key)) is not None
    }
    adapted = set(matrices_per_block)
    (layout,) = layouts
    return {
        "rank": ranks.pop(),
        "peft_module": LoRASurgery,
        "update_matrices": [m for m in _LORA_MATRICES if m in layout],
        "attention_layers_to_update": (
            [] if adapted == every_block else sorted(adapted)
        ),
        "quantize": False,
    }


def _segmenter_class(segmentation_mode: str) -> type:
    """The library class that turns decoder output into instances.

    Parameters
    ----------
    segmentation_mode : str
        One of :data:`SEGMENTATION_MODES`.

    Returns
    -------
    type

    Raises
    ------
    ValueError
        For an unknown mode.
    """
    from micro_sam.instance_segmentation import (
        AutomaticPromptGenerator, InstanceSegmentationWithDecoder)

    classes = {
        "ais": InstanceSegmentationWithDecoder,
        "apg": AutomaticPromptGenerator,
    }
    if segmentation_mode not in classes:
        raise ValueError(
            f"segmentation_mode must be one of {SEGMENTATION_MODES}, got "
            f"{segmentation_mode!r}."
        )
    return classes[segmentation_mode]


def describe_adaptation(peft_kwargs: Optional[dict[str, Any]]) -> str:
    """Say in a few words how an encoder was adapted, for the logs.

    Parameters
    ----------
    peft_kwargs : dict or None
        From :func:`peft_kwargs_from_state`.

    Returns
    -------
    str
    """
    if peft_kwargs is None:
        return "no low-rank factors (plain encoder weights)"
    layers = peft_kwargs["attention_layers_to_update"]
    return (
        f"rank {peft_kwargs['rank']} on "
        f"{','.join(peft_kwargs['update_matrices'])} in "
        f"{'every block' if not layers else f'blocks {layers}'}"
    )


def build_segmenter(
    checkpoint: Path,
    model_type: str,
    device: Optional[str] = None,
    segmentation_mode: str = "ais",
) -> Any:
    """Load a trained checkpoint as an automatic instance segmenter.

    Parameters
    ----------
    checkpoint : Path
    model_type : str
        Architecture of the image encoder the checkpoint was trained
        on, not the name of a pretrained checkpoint.
    device : str, optional
        ``None`` lets the library pick, which means the GPU when one is
        visible. Forcing the CPU keeps a check off a GPU that a training
        run is using to the last gigabyte.
    segmentation_mode : str, optional
        One of :data:`SEGMENTATION_MODES`.

    Returns
    -------
    InstanceSegmentationWithDecoder
        Or its prompt-based subclass for ``apg``.
    """
    from micro_sam.instance_segmentation import get_predictor_and_decoder

    segmenter_class = _segmenter_class(segmentation_mode)
    peft_kwargs = peft_kwargs_from_state(read_model_state(checkpoint))
    logger.info(
        "%s: rebuilding the encoder with %s.", checkpoint,
        describe_adaptation(peft_kwargs),
    )
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=str(checkpoint),
        device=device,
        peft_kwargs=peft_kwargs,
    )
    return segmenter_class(predictor, decoder)


def build_pretrained_segmenter(
    model_type: str,
    device: Optional[str] = None,
    segmentation_mode: str = "ais",
) -> Any:
    """Load a generalist checkpoint as shipped, the zero-shot baseline.

    No low-rank correction and no fine-tuned decoder: the encoder and
    the instance decoder are exactly what ``micro_sam`` ships for this
    model type.

    Parameters
    ----------
    model_type : str
        A registry name that carries an instance decoder, such as
        ``vit_l_lm``.
    device : str, optional
    segmentation_mode : str, optional
        One of :data:`SEGMENTATION_MODES`.

    Returns
    -------
    InstanceSegmentationWithDecoder

    Raises
    ------
    ValueError
        If the library's checkpoint for ``model_type`` carries no
        instance decoder to segment with.
    """
    from micro_sam.instance_segmentation import get_predictor_and_decoder

    segmenter_class = _segmenter_class(segmentation_mode)
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type, checkpoint_path=None, device=device,
    )
    return segmenter_class(predictor, decoder)


def generate_kwargs(segmenter: Any, settings: WatershedParams) -> dict:
    """The keyword arguments a segmenter's ``generate`` accepts.

    Parameters
    ----------
    segmenter : InstanceSegmentationWithDecoder
    settings : WatershedParams

    Returns
    -------
    dict
        The library's watershed arguments, without the smoothing ones
        when the segmenter is prompt-based: it has no smoothing step,
        and passing them would stop it with a type error.
    """
    from micro_sam.instance_segmentation import AutomaticPromptGenerator

    kwargs = settings.library_kwargs()
    if isinstance(segmenter, AutomaticPromptGenerator):
        for name in APG_UNUSED_SETTINGS:
            del kwargs[name]
    return kwargs


def segment(
    segmenter: Any,
    settings: WatershedParams = FROZEN_WATERSHED,
    pixel_size_um: Optional[float] = None,
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
    pixel_size_um : float, optional
        The image's pixel size. Required when the setting filters by
        physical area, and unused otherwise.

    Returns
    -------
    np.ndarray
        Instance labels on the same frame, background zero, numbered
        consecutively from one.

    Raises
    ------
    ValueError
        If the setting filters by physical area and no pixel size is
        given, which would otherwise have to be guessed.
    """
    segmentation = segmenter.generate(
        output_mode="instance_segmentation",
        **generate_kwargs(segmenter, settings),
    )
    # The watershed hands back floating-point labels. They are whole
    # numbers, but an identifier that can be compared for equality
    # should not be a float, and the metrics count on integers.
    labels = np.asarray(segmentation).astype(np.int32)
    if settings.min_instance_area_um2 <= 0:
        return labels
    if pixel_size_um is None:
        raise ValueError(
            "This setting drops instances below "
            f"{settings.min_instance_area_um2:g} um2, which needs the "
            "image's pixel size."
        )
    return drop_small_instances(
        labels, settings.min_instance_area_um2, pixel_size_um
    )


def drop_small_instances(
    labels: np.ndarray, min_area_um2: float, pixel_size_um: float
) -> np.ndarray:
    """Remove instances smaller than a physical area.

    Parameters
    ----------
    labels : np.ndarray
        ``(H, W)`` instance labels, background zero. Not modified.
    min_area_um2 : float
    pixel_size_um : float
        Micrometres per pixel of this image.

    Returns
    -------
    np.ndarray
        A new label image, the surviving instances renumbered
        consecutively from one in their original order.

    Raises
    ------
    ValueError
        For a pixel size that is not positive.
    """
    if pixel_size_um <= 0:
        raise ValueError(f"Pixel size must be positive, got {pixel_size_um}.")
    min_area_px = min_area_um2 / pixel_size_um ** 2
    areas_px = np.bincount(labels.ravel())
    keep = areas_px >= min_area_px
    keep[0] = False
    renumbered = np.zeros(areas_px.size, dtype=np.int32)
    renumbered[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return renumbered[labels]


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
                segment(
                    segmenter, setting,
                    pixel_size_um=prepared.record.pixel_size_um,
                ),
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
