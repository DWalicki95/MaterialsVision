"""
The settings that turn the decoder's output into separate pores.

The decoder does not predict instances. It predicts, for every pixel,
whether it lies inside a pore and how far it is from that pore's centre
and from its boundary. Instances appear afterwards: seeds are taken
where both distances are small, and a watershed grows them until they
meet. How many pores that produces depends on the thresholds below as
much as on the model, which is why they have to be stated somewhere
rather than inherited silently from a library default.

**Why they are frozen.** Two runs compared to attribute a difference to
augmentation must differ in the augmentation alone. Whatever these
values are, they apply to every run equally, so freezing them is what
the comparison needs; choosing them well is a separate question that
affects how good the models are, not what is learned about
augmentation.

**Why the frozen value is nonetheless worth choosing.** The thresholds
are read against the scale of the distance maps, and fine-tuning moves
that scale. Left at the value that suited the untrained decoder, the
main metric measures the drift of that scale as well as the quality of
the model: on one baseline run, a snapshot scoring 0.807 at a centre
threshold of 0.5 scores 0.833 at 0.3, with the same predictions
underneath and only the seeding changed. A calibrated value keeps the
metric closer to what it claims to measure.

**Why the robustness thresholds exist.** A single calibrated value
invites the objection that it was chosen on runs without augmentation
and therefore flatters them. Reporting the main metric at several
thresholds answers it with evidence rather than argument: a policy that
wins at one threshold and loses at the others has not won.
"""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WatershedParams:
    """One setting of the step that separates touching pores.

    Defaults are the library's own, which is where this study started
    and what every run before the first calibration used.

    Parameters
    ----------
    center_distance_threshold : float
        Predicted distance from a pore's centre below which a pixel may
        seed an instance. Lower values seed more selectively, so fewer
        pores are split in two.
    boundary_distance_threshold : float
        The same for the predicted distance from a pore's boundary;
        both conditions must hold for a pixel to seed.
    foreground_threshold : float
        Predicted foreground probability above which a pixel can belong
        to any instance at all.
    foreground_smoothing : float
        Width of the blur applied to the foreground map before
        thresholding, which removes the block pattern the decoder's
        upsampling leaves behind.
    distance_smoothing : float
        Width of the blur applied to both distance maps, which merges
        seeds that a noisy map would otherwise leave separate.
    min_size : int
        Instances smaller than this are dropped from the result. Zero
        keeps all of them, which is what the annotation does: filtering
        predictions by a size the ground truth still counts would
        penalize the model for pores it did find.
    """

    center_distance_threshold: float = 0.5
    boundary_distance_threshold: float = 0.5
    foreground_threshold: float = 0.5
    foreground_smoothing: float = 1.0
    distance_smoothing: float = 1.6
    min_size: int = 0

    def to_kwargs(self) -> dict[str, Any]:
        """Render as the keyword arguments the segmenter accepts.

        Returns
        -------
        dict
        """
        return asdict(self)

    def label(self) -> str:
        """Short name for the setting, for a results table.

        Named against the frozen setting rather than against the
        dataclass's own defaults, because what a reader of a results
        table needs to know is how a row departs from what the study
        actually runs, not how it departs from what the library ships.
        Only the centre threshold appears where only it differs, since
        that is the one the robustness cross-section varies; anything
        else gets a fuller name so two rows can never read alike.

        Returns
        -------
        str
        """
        varied = {
            field: value
            for field, value in self.to_kwargs().items()
            if value != getattr(FROZEN_WATERSHED, field)
        }
        if not varied:
            return "frozen"
        if set(varied) == {"center_distance_threshold"}:
            return f"cdt={self.center_distance_threshold:g}"
        return ",".join(
            f"{field}={value:g}" for field, value in sorted(varied.items())
        )


# The library's own values, which every run before the first
# calibration was scored under. Kept nameable so that a result from
# before that date can be reproduced, and so that the size of the
# correction stays visible.
LIBRARY_DEFAULT_WATERSHED = WatershedParams()

# Calibrated 2026-09-14 on 48 training images, against a snapshot two
# epochs into a run, by scoring a six by four grid of the two seeding
# thresholds. Chosen among the settings tied on instance F1 - nineteen
# of twenty-four were, the grid being flat near its top - as the one
# whose pore count was least wrong, at six parts in a thousand.
#
# The correction is not cosmetic. Read against the library's values, a
# snapshot five epochs into a run scores 0.823; read against these, it
# scores 0.856, and the epoch a run peaks at moves from its first to
# its fifth. The library's values were tuned for a decoder that had not
# been fine-tuned, and fine-tuning moves the scale of the distance maps
# these thresholds are read against, so they drift further out of step
# the longer a run goes on.
FROZEN_WATERSHED = WatershedParams(
    center_distance_threshold=0.30,
    boundary_distance_threshold=0.40,
)

# Centre thresholds the main metric is reported at alongside the frozen
# one. They bracket it closely rather than spanning the whole plausible
# range, because the question they answer is whether a result depends
# on the calibration having landed exactly where it did. Not a search:
# three points show whether a result survives the choice, and more
# would turn a robustness check into an opportunity to pick a winner.
ROBUSTNESS_CENTER_THRESHOLDS = (0.25, 0.30, 0.35)


def robustness_series(
    base: WatershedParams = FROZEN_WATERSHED,
    thresholds: tuple[float, ...] = ROBUSTNESS_CENTER_THRESHOLDS,
) -> tuple[WatershedParams, ...]:
    """Build the settings a robustness cross-section is reported over.

    Parameters
    ----------
    base : WatershedParams, optional
        Everything other than the centre threshold is taken from here,
        so the cross-section varies one thing.
    thresholds : tuple of float, optional

    Returns
    -------
    tuple of WatershedParams
        One per threshold, in the order given, with duplicates of the
        base removed so the base is never scored twice.
    """
    series = []
    for threshold in thresholds:
        candidate = WatershedParams(
            **{**base.to_kwargs(), "center_distance_threshold": threshold}
        )
        if candidate not in series:
            series.append(candidate)
    return tuple(series)
