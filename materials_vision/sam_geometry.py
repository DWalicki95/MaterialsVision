"""
Making the model receive the image at the geometry it was trained for.

SAM accepts a square canvas of 1024 pixels and expects an image to
reach it scaled so that its longer side fills that canvas, proportions
intact, the remainder padded. That is what its pretraining saw and what
its own inference path still does.

The training path does something else. ``TrainableSAM.preprocess``
resizes through ``ResizeLongestSide.apply_image_torch``, and that
function computes the target size from ``image.shape[0]`` and
``image.shape[1]`` of a tensor its own docstring describes as
``BxCxHxW``. Those two axes are the batch and the channels. For a
three-channel image in a batch of one the arithmetic runs on 1 and 3
instead of on the image's height and width:

.. code-block:: text

    scale  = 1024 / max(1, 3) = 341.33
    height = 1 x 341.33 -> 341
    width  = 3 x 341.33 -> 1024

Every image therefore arrives as 341 by 1024 whatever its real shape,
and the result changes with the batch size - two images per batch give
683 by 1024 - which is how a reader can tell this is a defect rather
than a convention: the geometry of a picture cannot depend on how many
pictures travel beside it.

**Why it has gone unnoticed.** With one channel the same formula gives
1024 by 1024, and a square patch resized to a square canvas is exactly
right. The upstream loaders pad to square single-channel patches, so
the defect never shows there. It shows here because this pipeline makes
two deliberate choices that upstream does not: whole images rather than
square patches, and the working channel triplicated to RGB before the
model.

**Why it matters more than a scale factor.** Inference is unaffected:
``SamPredictor.set_image`` resizes through ``apply_image``, the numpy
path, where the two axes read really are height and width. Left alone,
the model would be fine-tuned on images squeezed 2.25 times vertically
and then run on undistorted ones. The squeeze is also directional - a
wall two pixels across survives horizontally at 1.6 pixels and vanishes
vertically at 0.71 - so it would teach an anisotropy the material does
not have, in an experiment that measures anisotropy.

The correction is to compute the target size from the spatial axes,
which is what ``apply_image`` does for numpy arrays. Everything else -
the interpolation, its settings, the padding, the normalization - is
left exactly as it was.
"""
import logging
from typing import Sequence

import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger(__name__)

# Side of the square canvas the image encoder consumes.
SAM_CANVAS_PX = 1024

# Attribute stamped on the replacement so that applying the correction
# twice does nothing and a later library fix can be told apart from it.
PATCH_MARKER = "_materials_vision_isotropic"

# The geometries this pipeline feeds the model, and the content sizes
# a corrected preprocessing has to produce for them. Content, not file:
# the panel of the second microscope is cropped before the model sees
# anything.
CONTENT_GEOMETRIES: tuple[tuple[int, int], ...] = (
    (960, 1280),
    (890, 1280),
    (1280, 960),
    (1280, 890),
)


class SamGeometryError(RuntimeError):
    """Raised when the model would receive a geometry nobody chose."""


def expected_content_shape(
    height_px: int, width_px: int, canvas_px: int = SAM_CANVAS_PX
) -> tuple[int, int]:
    """Content size a correctly preprocessed image should occupy.

    Parameters
    ----------
    height_px, width_px : int
        Geometry handed to the model, i.e. after the content crop.
    canvas_px : int, optional

    Returns
    -------
    tuple of int
        ``(height, width)`` inside the canvas.
    """
    return ResizeLongestSide.get_preprocess_shape(
        height_px, width_px, canvas_px
    )


def resize_isotropic(batched: torch.Tensor) -> torch.Tensor:
    """Resize a ``(B, C, H, W)`` batch, longest side to the canvas.

    Parameters
    ----------
    batched : torch.Tensor

    Returns
    -------
    torch.Tensor
        The batch at the target size, proportions preserved.
    """
    target = expected_content_shape(
        int(batched.shape[-2]), int(batched.shape[-1])
    )
    return F.interpolate(
        batched, target, mode="bilinear", align_corners=False,
        antialias=True,
    )


def resize_upstream_defect(batched: torch.Tensor) -> torch.Tensor:
    """Resize the way the unpatched library does, for comparison.

    Kept so that the two geometries can be shown side by side and so
    that the defect is described by running code rather than by a
    comment. Never used on the training path.

    Parameters
    ----------
    batched : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    target = ResizeLongestSide.get_preprocess_shape(
        int(batched.shape[0]), int(batched.shape[1]), SAM_CANVAS_PX
    )
    return F.interpolate(
        batched, target, mode="bilinear", align_corners=False,
        antialias=True,
    )


def isotropic_apply_image_torch(
    self: ResizeLongestSide, image: torch.Tensor
) -> torch.Tensor:
    """Replacement for the library method, reading the spatial axes.

    Parameters
    ----------
    self : ResizeLongestSide
        Bound instance; its ``target_length`` is honoured, so a model
        with a different canvas keeps working.
    image : torch.Tensor
        ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
    """
    target = ResizeLongestSide.get_preprocess_shape(
        int(image.shape[-2]), int(image.shape[-1]), self.target_length
    )
    return F.interpolate(
        image, target, mode="bilinear", align_corners=False,
        antialias=True,
    )


# Stamped here rather than at install time: the marker says what this
# function is, not what has been done with it, and setting it during
# the install would leave the attribute behind on a function that is no
# longer in place.
setattr(isotropic_apply_image_torch, PATCH_MARKER, True)


def patch_resize_longest_side() -> bool:
    """Install the correction, once, for this process.

    Every entry point that trains or evaluates a model has to call this
    before the model is built. It is a replacement of a library method,
    which is a heavy thing to do quietly, so it says so in the log and
    it is verified rather than trusted: ``verify_preprocess_geometry``
    reads the geometry back out of the library afterwards.

    Returns
    -------
    bool
        True if the correction was installed, False if it was already
        in place or the library no longer needs it. The second case is
        the one to watch for: it means the upstream defect was fixed
        and this module can be retired.
    """
    current = ResizeLongestSide.apply_image_torch
    if getattr(current, PATCH_MARKER, False):
        return False

    if _library_is_correct():
        logger.info(
            "ResizeLongestSide.apply_image_torch already reads the "
            "spatial axes; the correction in "
            "materials_vision.sam_geometry is no longer needed and can "
            "be removed."
        )
        return False

    ResizeLongestSide.apply_image_torch = isotropic_apply_image_torch
    logger.warning(
        "Replaced ResizeLongestSide.apply_image_torch: the library "
        "computes the target size from the batch and channel axes, "
        "which squeezes every three-channel image to 341x1024. The "
        "replacement reads height and width, so a %s image reaches "
        "the encoder as %s.",
        CONTENT_GEOMETRIES[0],
        expected_content_shape(*CONTENT_GEOMETRIES[0]),
    )
    return True


def verify_preprocess_geometry(
    geometries: Sequence[tuple[int, int]] = CONTENT_GEOMETRIES,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Check what the library actually does to each geometry.

    Run after ``patch_resize_longest_side`` and before training. It
    calls the library function rather than the replacement, so it
    catches both a correction that failed to install and a library
    update that changed the behaviour underneath it.

    Parameters
    ----------
    geometries : Sequence of tuple, optional
        ``(height, width)`` pairs to check.

    Returns
    -------
    dict
        Each geometry mapped to the content size it produces.

    Raises
    ------
    SamGeometryError
        If any geometry does not come out at its expected content size.
    """
    transform = ResizeLongestSide(SAM_CANVAS_PX)
    measured: dict[tuple[int, int], tuple[int, int]] = {}
    wrong = []
    for height_px, width_px in geometries:
        probe = torch.zeros((1, 3, height_px, width_px))
        resized = transform.apply_image_torch(probe)
        content = (int(resized.shape[-2]), int(resized.shape[-1]))
        expected = expected_content_shape(height_px, width_px)
        measured[(height_px, width_px)] = content
        if content != expected:
            wrong.append(
                f"{height_px}x{width_px} -> {content[0]}x{content[1]}, "
                f"expected {expected[0]}x{expected[1]}"
            )

    if wrong:
        raise SamGeometryError(
            "the model would receive a geometry nobody chose: "
            + "; ".join(wrong)
            + ". Call patch_resize_longest_side() before building the "
              "model."
        )
    logger.info(
        "Preprocessing geometry verified: %s.",
        ", ".join(
            f"{h}x{w} -> {content[0]}x{content[1]}"
            for (h, w), content in measured.items()
        ),
    )
    return measured


def _library_is_correct() -> bool:
    """Whether the library already reads the spatial axes.

    Decided by measurement, not by inspecting the source: a rectangular
    three-channel probe comes out at its proper content size only if
    the target was computed from height and width.
    """
    probe = torch.zeros((1, 3, 960, 1280))
    resized = ResizeLongestSide(SAM_CANVAS_PX).apply_image_torch(probe)
    content = (int(resized.shape[-2]), int(resized.shape[-1]))
    return content == expected_content_shape(960, 1280)
