"""
Transformations that move pixels.

They act on the image and the mask together, with the mask resampled by
nearest neighbour so that no id is ever blended with its neighbour into
a value nobody annotated.

Right now this holds one family: the eight symmetries of the square.
They are the cheapest augmentation available, because a quarter turn
and a mirror only reorder pixels - no arithmetic touches an intensity
and no instance changes area.

A rectangular frame comes out transposed after an odd number of quarter
turns. It is deliberately not stretched back. Stretching would change
the aspect ratio, and with it the elongation and orientation of every
pore, which are exactly the quantities the model is meant to reproduce
faithfully. The frame is left as it is; the model scales the longer
side to its own input size either way, and since the longer side has
the same length before and after the turn, the scale factor does not
move.
"""
import albumentations as A

from materials_vision.augmentation.config import OrientationConfig


def build_orientation(config: OrientationConfig) -> A.D4:
    """Build the orientation transformation.

    Parameters
    ----------
    config : OrientationConfig

    Returns
    -------
    A.D4
        Draws uniformly from the eight symmetries of the square,
        applying the same one to image and mask.
    """
    return A.D4(p=config.p)
