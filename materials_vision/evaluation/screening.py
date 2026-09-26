"""
The screening rule of the fine-tuning experiments, as registered.

An arm - one change to how the model is fine-tuned - is compared with a
reference trained three times at three seeds. Every quantity is read as
the mean over the last three epochs of a twelve-epoch run, because the
deployed model is the last snapshot and no validation set is left to
pick a peak with; a peak would also reward a jagged curve, which has
more chances to touch a high value.

The noise band is the spread of the three reference seeds. An arm
passes when it clears the reference by more than that band on pooled
instance F1, is no worse than the worst reference seed on any guard
metric, and does not pull the second microscope's main material below
the reference's range. The last condition exists because a gain on the
dominant material can hide a loss on a minor one in a pooled figure.

When both arms pass, the one that changes only the step length is
preferred unless the other beats it by more than the band: it changes
one number and leaves the model the same size.
"""
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

LATE_EPOCHS = (10, 11, 12)

# Below this a spread of three seeds is too small to trust as a band,
# and twice the standard deviation of the three is used instead.
MIN_RANGE_BAND = 0.002

# Guard metrics and whether a larger value is better.
GUARDS = {
    "merges_per_100_gt": False,
    "splits_per_100_gt": False,
    "abs_pore_count_error": False,
    "boundary_f1": True,
}

GUARD_MATERIAL = "K"


@dataclass(frozen=True)
class ArmVerdict:
    """How one arm fares against the reference.

    Parameters
    ----------
    gain : float
        Late-epoch pooled F1 of the arm minus the mean of the reference
        seeds' late-epoch F1.
    clears_band : bool
    guards_failed : tuple of str
        Guard metrics on which the arm is worse than the worst seed.
    material_ok : bool
    passes : bool
    """

    gain: float
    clears_band: bool
    guards_failed: tuple[str, ...]
    material_ok: bool
    passes: bool


def late_mean(
    by_epoch: Mapping[int, float], epochs: Sequence[int] = LATE_EPOCHS
) -> float:
    """Mean of a curve over the late epochs.

    Parameters
    ----------
    by_epoch : mapping of int to float
    epochs : sequence of int, optional

    Returns
    -------
    float

    Raises
    ------
    KeyError
        If a late epoch is missing: a mean over whichever of them exist
        would compare unequal things.
    """
    return sum(by_epoch[epoch] for epoch in epochs) / len(epochs)


def noise_band(reference_f1: Sequence[float]) -> tuple[float, str]:
    """The band an arm has to clear.

    Parameters
    ----------
    reference_f1 : sequence of float
        Late-epoch pooled F1 of each reference seed.

    Returns
    -------
    band : float
    how : str
        ``"range"`` or ``"2sigma"``.

    Raises
    ------
    ValueError
        With fewer than two seeds, which have no spread.
    """
    if len(reference_f1) < 2:
        raise ValueError("A noise band needs at least two seeds.")
    spread = max(reference_f1) - min(reference_f1)
    if spread >= MIN_RANGE_BAND:
        return spread, "range"
    mean = sum(reference_f1) / len(reference_f1)
    variance = sum((f - mean) ** 2 for f in reference_f1) / (
        len(reference_f1) - 1
    )
    return 2 * math.sqrt(variance), "2sigma"


def worst(values: Sequence[float], higher_is_better: bool) -> float:
    """The worst of several seeds' values."""
    return min(values) if higher_is_better else max(values)


def screen_arm(
    arm: Mapping[str, float],
    references: Sequence[Mapping[str, float]],
    band: float,
) -> ArmVerdict:
    """Apply the three conditions to one arm.

    Parameters
    ----------
    arm : mapping of str to float
        Late-epoch figures: ``f1``, every name in ``GUARDS`` and
        ``K/f1``.
    references : sequence of mapping
        The same for each reference seed.
    band : float

    Returns
    -------
    ArmVerdict
    """
    reference_mean = sum(r["f1"] for r in references) / len(references)
    gain = arm["f1"] - reference_mean
    failed = []
    for name, higher_is_better in GUARDS.items():
        limit = worst([r[name] for r in references], higher_is_better)
        worse = (
            arm[name] < limit if higher_is_better else arm[name] > limit
        )
        if worse:
            failed.append(name)
    material_key = f"{GUARD_MATERIAL}/f1"
    material_ok = arm[material_key] >= min(
        r[material_key] for r in references
    )
    clears = gain > band
    return ArmVerdict(
        gain=gain,
        clears_band=clears,
        guards_failed=tuple(failed),
        material_ok=material_ok,
        passes=clears and not failed and material_ok,
    )


def branch(
    capacity: ArmVerdict, step: ArmVerdict, band: float
) -> tuple[str, Optional[str]]:
    """Which branch the screening sends the study down.

    Parameters
    ----------
    capacity : ArmVerdict
        Arm A1: higher rank at the same rate.
    step : ArmVerdict
        Arm A2: same rank at a higher rate.
    band : float

    Returns
    -------
    reading : str
    branch : str or None
        ``"capacity"``, ``"rate"`` or ``None`` when neither passes and
        the configuration stays as it is.
    """
    if capacity.passes and step.passes:
        if capacity.gain - step.gain > band:
            return "both pass, capacity ahead by more than the band", \
                "capacity"
        return "both pass; step length decides", "rate"
    if capacity.passes:
        return "capacity decides", "capacity"
    if step.passes:
        return "step length decides; a higher rank does not help", "rate"
    return "adaptation of the encoder is not the bottleneck", None
