"""
The one place a computed image is turned back into a real one.

Every transformation that changes brightness works in floating point,
because the arithmetic that produces the change - a ramp, a smooth
field, a multiplication - has no meaning in whole grey levels. The
result then has to become an image again, and the two steps that turn
it back are easy to get subtly wrong: dropping the rounding shifts
every value down by up to one level, and dropping the clip lets a
bright pore wrap around to black.

Neither failure announces itself. A shift of half a grey level is
invisible by eye and changes every intensity statistic a run reports;
a wrapped pixel looks like a speck of dirt in exactly the images where
the augmentation was strongest. Both are avoided by having one
conversion rather than one per family.
"""
import numpy as np


def to_source_dtype(
    values: np.ndarray, like: np.ndarray
) -> np.ndarray:
    """Return computed values as an image of the original type.

    Parameters
    ----------
    values : np.ndarray
        The computed result, normally floating point.
    like : np.ndarray
        The array whose type and range the result must match.

    Returns
    -------
    np.ndarray
        ``values`` rounded and clipped to what ``like``'s type can
        hold, in that type. A floating-point original is returned
        unchanged apart from its type, since there is nothing to
        round it to.
    """
    if np.issubdtype(like.dtype, np.integer):
        limits = np.iinfo(like.dtype)
        values = np.clip(np.rint(values), limits.min, limits.max)
    return values.astype(like.dtype, copy=False)
