"""
Checks that an augmented sample is still a valid training sample.

Augmentation is the one stage that rewrites both the image and its
annotation, and its failures are quiet. An interpolated label id, a
mask that drifted a pixel from its image, an instance broken into two
pieces - none of these raises anything on its own. They surface much
later as a model that trains but does not learn, and by then nothing
points back here. So the sample is checked where it is produced.

Two kinds of failure have to be told apart, because they deserve
opposite responses.

A **failed draw** is expected. A random window may not contain enough
instances; a pore may be too small to divide. The transformation
retries a bounded number of times, then leaves the sample untouched and
records why. This is normal and must not stop anything - but it is
never silent, because a sample that skipped a transformation still
counts as that family's sample when results are compared.

An **integrity error** is a defect in the code. A label id that no
longer exists, a mask and an image of different sizes, a target that
disagrees with the mask it came from. Nothing downstream can repair it
and training on it produces a meaningless number, so it stops the run.

The checks are split by cost. Everything here except
:func:`check_connectivity` is a handful of array passes and runs on
every sample of every policy. Connectivity needs its own labelling pass
over the frame, so it runs only where an instance can actually be
broken apart: where a window is cut out of the frame, and where a wall
is drawn across a pore. A policy that only moves or reshades pixels
cannot break connectivity, and paying for the check there would be a
cost on every training step in exchange for nothing.
"""
import logging

import numpy as np
from skimage.measure import label as connected_components

from materials_vision.data.instances import is_densely_numbered

logger = logging.getLogger(__name__)

MAX_INTENSITY = 255.0


class IntegrityError(RuntimeError):
    """Raised when an augmented sample is not a valid training sample.

    Deliberately not a ``ValueError``: this is not a caller passing bad
    input, it is the pipeline having produced something it must never
    produce, and the only correct response is to stop.
    """


def check_sample(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    context: str,
    expect_instances: bool = True,
) -> None:
    """Verify the properties every augmented sample must have.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` working channel.
    labels : np.ndarray
        ``(H, W)`` instance labels, 0 as background.
    context : str
        What produced the sample, used in the error message. Without it
        a failure names a property but not a culprit.
    expect_instances : bool, optional
        Whether the labels must still hold at least one instance. An
        empty mask is a legitimate result only if the input was empty
        too; produced from a populated input it means the labels were
        lost somewhere.

    Raises
    ------
    IntegrityError
        If any property is violated.
    """
    if image.ndim != 2:
        raise IntegrityError(
            f"{context}: image has shape {image.shape}; the pipeline "
            f"works on one channel of shape (H, W)"
        )
    if labels.ndim != 2:
        raise IntegrityError(
            f"{context}: labels have shape {labels.shape}; a label "
            f"image must be 2-D"
        )
    if image.shape != labels.shape:
        raise IntegrityError(
            f"{context}: image {image.shape} and labels "
            f"{labels.shape} describe different frames"
        )

    _check_intensities(image, context)
    _check_label_values(labels, context, expect_instances)


def check_labels_preserved(
    before: np.ndarray, after: np.ndarray, *, context: str
) -> None:
    """Verify a transformation only rearranged the instances.

    Photometry must not touch the mask at all, and an orientation
    change may only move its pixels. Both are covered by one property:
    the multiset of instance areas is unchanged. It catches an id that
    vanished, an id that appeared, and - the failure worth the check -
    a mask resampled with interpolation, which blends neighbouring ids
    into values that were never annotated and shifts every area.

    It is not a check for transformations that legitimately change the
    mask; those have their own.

    Parameters
    ----------
    before, after : np.ndarray
        Label images from either side of the transformation.
    context : str

    Raises
    ------
    IntegrityError
        If any instance changed size, appeared or disappeared.
    """
    areas_before = np.bincount(before.ravel())[1:]
    areas_after = np.bincount(after.ravel())[1:]
    if areas_before.size != areas_after.size or not np.array_equal(
        np.sort(areas_before), np.sort(areas_after)
    ):
        raise IntegrityError(
            f"{context}: instance areas changed - "
            f"{int(np.count_nonzero(areas_before))} instance(s) in, "
            f"{int(np.count_nonzero(areas_after))} out. This "
            f"transformation may only rearrange the mask, so a change "
            f"means ids were interpolated, dropped or invented"
        )


def check_mask_untouched(
    before: np.ndarray, after: np.ndarray, *, context: str
) -> None:
    """Verify the mask is bitwise identical.

    The strictest form, for transformations that change brightness
    only. Equal areas would not be enough: a mask could in principle
    be shifted and still report the same areas.

    Parameters
    ----------
    before, after : np.ndarray
    context : str

    Raises
    ------
    IntegrityError
        If a single pixel or the dtype differs.
    """
    if before.dtype != after.dtype or not np.array_equal(before, after):
        raise IntegrityError(
            f"{context}: the mask changed. A transformation that only "
            f"alters brightness must leave it bitwise identical"
        )


def check_connectivity(labels: np.ndarray, *, context: str) -> None:
    """Verify every instance occupies a single connected region.

    An instance in two pieces is not a cosmetic problem. The decoder's
    targets are built per instance from a distance transform, so two
    pieces produce two basins and teach the model to divide a pore the
    annotation regards as whole.

    Costs one labelling pass over the frame, so it belongs only where a
    transformation can actually disconnect something.

    Parameters
    ----------
    labels : np.ndarray
    context : str

    Raises
    ------
    IntegrityError
        If any instance is split across more than one region.
    """
    n_instances = int(labels.max())
    if n_instances == 0:
        return
    components = connected_components(
        labels, background=0, connectivity=1
    )
    if int(components.max()) != n_instances:
        broken = _disconnected_ids(labels, components)
        raise IntegrityError(
            f"{context}: instance(s) {broken[:10]} occupy more than "
            f"one region; per-instance training targets would teach "
            f"the model to split a pore the annotation keeps whole"
        )


def _disconnected_ids(
    labels: np.ndarray, components: np.ndarray
) -> list[int]:
    """Instance ids occupying more than one connected region."""
    n_components = int(components.max())
    origin = np.zeros(n_components + 1, dtype=np.int64)
    origin[components.ravel()] = labels.ravel()
    ids, counts = np.unique(origin[1:], return_counts=True)
    return [
        int(instance_id)
        for instance_id, count in zip(ids, counts)
        if instance_id > 0 and count > 1
    ]


def _check_intensities(image: np.ndarray, context: str) -> None:
    """Reject images that left the representable intensity range."""
    if image.dtype == np.uint8:
        return
    if not np.issubdtype(image.dtype, np.floating):
        raise IntegrityError(
            f"{context}: image dtype is {image.dtype}; the working "
            f"channel is either uint8 or floating point"
        )
    if not np.isfinite(image).all():
        raise IntegrityError(
            f"{context}: image holds NaN or infinity"
        )
    if image.min() < 0.0 or image.max() > MAX_INTENSITY:
        raise IntegrityError(
            f"{context}: image values span "
            f"[{float(image.min()):.3f}, {float(image.max()):.3f}], "
            f"outside the [0, {MAX_INTENSITY:.0f}] range the model "
            f"normalizes from"
        )


def _check_label_values(
    labels: np.ndarray, context: str, expect_instances: bool
) -> None:
    """Reject label images that cannot index per-instance arrays."""
    if not np.issubdtype(labels.dtype, np.integer):
        raise IntegrityError(
            f"{context}: labels have dtype {labels.dtype}; instance "
            f"ids must be integers, and a floating-point label image "
            f"is the signature of an interpolated mask"
        )
    if labels.min() < 0:
        raise IntegrityError(
            f"{context}: labels hold negative values; 0 is background "
            f"and instances are positive"
        )
    if expect_instances and int(labels.max()) == 0:
        raise IntegrityError(
            f"{context}: the mask came out empty although the sample "
            f"had instances; a sample with no target teaches nothing "
            f"and must not reach the trainer unnoticed"
        )
    if not is_densely_numbered(labels):
        raise IntegrityError(
            f"{context}: instance ids leave gaps. Later stages count "
            f"instances as the maximum id and index per-instance "
            f"arrays by id - 1, so a gap silently misaligns them"
        )
