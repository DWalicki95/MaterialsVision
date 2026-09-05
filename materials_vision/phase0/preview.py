"""
The image as the model receives it, which is not the image we hand it.

Two of the acceptance criteria of Phase 0 - that thin walls and small
pores do not disappear, and that a synthetic wall stays visible - are
about structures a few pixels across, and a few pixels is a quantity
that only means something at a stated resolution. The resolution that
counts is the model's, after its own resizing, so this module rebuilds
that step for the panel.

**Two resolutions, because the training path does not do what the plan
assumes.** ``TrainableSAM.preprocess`` resizes through
``ResizeLongestSide.apply_image_torch``, and that function reads the
target size off ``image.shape[0]`` and ``image.shape[1]`` of a tensor
documented as ``BxCxHxW`` - the batch and the channel count, not the
height and the width. The consequence is that the content size depends
on how many images are in the batch and how many channels each has,
and not at all on the geometry of the image:

===============  ==================
input            content afterwards
===============  ==================
(1, 3,  960, 1280)   (341, 1024)
(1, 3,  890, 1280)   (341, 1024)
(1, 3, 1280,  960)   (341, 1024)
(1, 3,  512,  512)   (341, 1024)
(2, 3,  960, 1280)   (683, 1024)
(1, 1,  512,  512)  (1024, 1024)
===============  ==================

For three-channel input with a batch of one - our configuration - every
image is squeezed to 341 by 1024, a vertical scale of 0.355 against a
horizontal 0.8. Single-channel square patches, which is what the
upstream loaders produce, come out square and undamaged, which is why
the defect has gone unnoticed there.

The defect is corrected on the training path by
``materials_vision.sam_geometry``, so ``MODE_ISOTROPIC`` is what the
model now receives and is what a panel is rendered at. ``MODE_AS_IS``
reproduces the uncorrected behaviour and stays available for one
purpose: showing what the correction changed. Both are computed here
from named functions rather than by calling whichever version of the
library happens to be installed - delegating would make the two modes
identical the moment the correction is in place, which is exactly when
the comparison is worth having. A panel records which mode it was
rendered at, because a wall judged at one is not a wall judged at the
other.

**Normalization is deliberately not applied.** The model subtracts a
mean and divides by a standard deviation, which is a linear map: it
moves no structure and changes nothing a reviewer can see once the
result is displayed on its own scale. Applying it would only make the
padded region print as a mid grey rather than black. The geometry -
the resize and the padding - is the part that decides whether a wall
survives, and that is reproduced exactly.
"""
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from materials_vision.sam_geometry import (SAM_CANVAS_PX, resize_isotropic,
                                           resize_upstream_defect)

logger = logging.getLogger(__name__)

# The uncorrected library behaviour, kept for comparison only.
MODE_AS_IS = "as_is"

# The longest side scaled to the canvas, proportions preserved: what
# the corrected training path does and what a panel is judged at.
MODE_ISOTROPIC = "isotropic"

MODES = (MODE_AS_IS, MODE_ISOTROPIC)


@dataclass(frozen=True)
class ModelInput:
    """One image at the resolution the encoder works in.

    Parameters
    ----------
    image : np.ndarray
        ``(1024, 1024)`` uint8, content in the top-left corner and the
        remainder padded, exactly as the encoder receives it.
    content_shape : tuple of int
        Height and width the content occupies inside the canvas.
    scale_y, scale_x : float
        What the vertical and horizontal dimensions were multiplied by.
        Equal under ``MODE_ISOTROPIC``; under ``MODE_AS_IS`` they are
        not, and their ratio is the aspect distortion.
    padding_share : float
        Fraction of the canvas holding no content.
    mode : str
    """

    image: np.ndarray
    content_shape: tuple[int, int]
    scale_y: float
    scale_x: float
    padding_share: float
    mode: str

    @property
    def aspect_distortion(self) -> float:
        """How far from square the pixels came out.

        Returns
        -------
        float
            One when the resize preserved shape; two means a circle
            arrives as an ellipse twice as wide as it is tall.
        """
        return max(self.scale_x, self.scale_y) / min(
            self.scale_x, self.scale_y
        )


def to_model_input(image: np.ndarray, *, mode: str) -> ModelInput:
    """Put one working channel through the encoder's own resizing.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` working channel, in the resolution the dataloader
        hands over.
    mode : str
        ``MODE_AS_IS`` or ``MODE_ISOTROPIC``. No default: which of the
        two is the truth is a decision about the experiment, not about
        this function.

    Returns
    -------
    ModelInput

    Raises
    ------
    ValueError
        If the mode is not one of the two, or the image is not 2-D.
    """
    if mode not in MODES:
        raise ValueError(
            f"unknown preprocessing mode {mode!r}; expected one of "
            f"{list(MODES)}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"expected one working channel of shape (H, W), got "
            f"{image.shape}"
        )

    height_px, width_px = image.shape
    batched = torch.from_numpy(
        np.repeat(image[None, None], 3, axis=1).astype(np.float32)
    )
    resized = _resize(batched, mode)
    content_h, content_w = resized.shape[-2:]

    canvas = F.pad(
        resized,
        (0, SAM_CANVAS_PX - content_w, 0, SAM_CANVAS_PX - content_h),
    )
    rendered = np.clip(
        np.rint(canvas[0, 0].numpy()), 0, 255
    ).astype(np.uint8)
    content_px = content_h * content_w
    return ModelInput(
        image=rendered,
        content_shape=(int(content_h), int(content_w)),
        scale_y=content_h / height_px,
        scale_x=content_w / width_px,
        padding_share=1.0 - content_px / SAM_CANVAS_PX ** 2,
        mode=mode,
    )


def _resize(batched: torch.Tensor, mode: str) -> torch.Tensor:
    """Resize a ``(1, 3, H, W)`` batch the way the chosen mode says.

    Both formulas come from ``sam_geometry``, which is also where the
    training path's correction lives, so a panel and a training sample
    are resized by the same code.
    """
    if mode == MODE_AS_IS:
        return resize_upstream_defect(batched)
    return resize_isotropic(batched)


def place_mask_on_canvas(
    mask: np.ndarray, model_input: ModelInput, threshold: float = 0.25
) -> np.ndarray:
    """Follow a source-resolution mask onto the encoder's canvas.

    Resized the same way the image was, then thresholded, so a pixel
    counts as covered when the source region behind it really does
    contain the mask. Sampling the mask at the new grid instead would
    be wrong for exactly the structures worth measuring: a wall two
    pixels across becomes 1.6 after the resize, and a grid sample can
    step over it entirely and report the pore beside it.

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` boolean, at source resolution.
    model_input : ModelInput
        The result the mask is being followed into.
    threshold : float, optional
        Share of a canvas pixel that must come from the mask. The
        default keeps a thin structure that lands between two pixels
        while excluding the faint spill of the interpolation.

    Returns
    -------
    np.ndarray
        ``(1024, 1024)`` boolean, in the canvas's frame.
    """
    batched = torch.from_numpy(
        mask.astype(np.float32)[None, None]
    ).repeat(1, 3, 1, 1)
    resized = _resize(batched, model_input.mode)
    height, width = model_input.content_shape
    placed = np.zeros(model_input.image.shape, dtype=bool)
    placed[:height, :width] = (
        resized[0, 0].numpy() > threshold
    )
    return placed


def to_model_coordinates(
    box: tuple[int, int, int, int], model_input: ModelInput
) -> tuple[int, int, int, int]:
    """Map a region of the source image into the encoder's canvas.

    Used to show the same pore or wall twice, once as annotated and
    once at the resolution the model sees it, without hunting for it a
    second time.

    Parameters
    ----------
    box : tuple of int
        ``(y0, x0, y1, x1)`` in the source image.
    model_input : ModelInput

    Returns
    -------
    tuple of int
        The same region inside the canvas, clipped to it.
    """
    y0, x0, y1, x1 = box
    scaled = (
        int(np.floor(y0 * model_input.scale_y)),
        int(np.floor(x0 * model_input.scale_x)),
        int(np.ceil(y1 * model_input.scale_y)),
        int(np.ceil(x1 * model_input.scale_x)),
    )
    return (
        max(0, scaled[0]),
        max(0, scaled[1]),
        min(SAM_CANVAS_PX, scaled[2]),
        min(SAM_CANVAS_PX, scaled[3]),
    )
