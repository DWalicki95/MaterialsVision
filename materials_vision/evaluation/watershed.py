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
        Instances smaller than this many pixels are dropped by the
        library. Zero keeps all of them, which is what the annotation
        does: filtering predictions by a size the ground truth still
        counts would penalize the model for pores it did find.
    min_instance_area_um2 : float
        Instances smaller than this physical area are dropped after the
        watershed, in this project's own layer. A pixel threshold would
        be wrong for part of the data, because the same area covers
        more than twice as many pixels in one scale bin as in the
        other; this one is converted per image from its pixel size.
        Zero disables it.
    """

    center_distance_threshold: float = 0.5
    boundary_distance_threshold: float = 0.5
    foreground_threshold: float = 0.5
    foreground_smoothing: float = 1.0
    distance_smoothing: float = 1.6
    min_size: int = 0
    min_instance_area_um2: float = 0.0

    def to_kwargs(self) -> dict[str, Any]:
        """Render as the keyword arguments of this setting.

        Returns
        -------
        dict
        """
        return asdict(self)

    def library_kwargs(self) -> dict[str, Any]:
        """Render as the keyword arguments the library's segmenter takes.

        Returns
        -------
        dict
            Everything except the physical area filter, which the
            library does not know and which is applied afterwards.
        """
        kwargs = self.to_kwargs()
        del kwargs["min_instance_area_um2"]
        return kwargs

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

# Smallest predicted instance kept, in square micrometres, placed low in
# the gap between the watershed's fragments and everything larger. The
# gap depends on the seeding, so there is one value per seeding, each
# derived on the calibration snapshot. Applied to the annotation neither
# removes a single pore, border pores included.
#
# At the seeding of FROZEN_WATERSHED the fragments reach 98 um2 and
# nothing else comes before the smallest annotated interior pore in
# TRAIN, at 535 um2.
MIN_INSTANCE_AREA_FROZEN_SEEDING_UM2 = 200.0

# At the seeding calibrated on 2026-09-24 the fragments reach only 63
# um2, but a second group of unmatched instances begins at 189, so 200
# would cut through that group.
MIN_INSTANCE_AREA_UM2 = 100.0

# The post-processing calibrated on 2026-09-24, on TRAIN, against the
# last snapshot of the first baseline seed. It is what the fine-tuning
# experiments were scored and screened under. It is not what the
# deployed model runs with; see DEPLOYED_POSTPROCESSING for why. Two
# changes and three confirmations:
#
# * Seeding. The thresholds of 2026-09-14 were calibrated on a snapshot
#   two epochs into a run; by the twelfth epoch the distance maps had
#   drifted and those thresholds cut pores in two, overcounting by five
#   percent. Across 164 seeding settings the rule chose centre 0.25,
#   boundary 0.50 and distance smoothing 3.2, where the pore count is
#   right and F1 is higher by 0.006.
# * A minimum physical instance area, see MIN_INSTANCE_AREA_UM2.
# * The foreground threshold stays at one half. Matched pores are drawn
#   about one percent too wide in diameter there, and the threshold
#   that would correct it, 0.62 to 0.66 depending on the material,
#   costs F1 and loses three to five percent of the pores.
# * The foreground smoothing stays at 1.0: six values from none to 3.0
#   were within 0.0011 of each other in F1.
CALIBRATED_2026_09_24 = WatershedParams(
    center_distance_threshold=0.25,
    boundary_distance_threshold=0.50,
    foreground_threshold=0.5,
    foreground_smoothing=1.0,
    distance_smoothing=3.2,
    min_size=0,
    min_instance_area_um2=MIN_INSTANCE_AREA_UM2,
)

# What the deployed model runs with: the seeding of FROZEN_WATERSHED and
# the fragment filter derived for that seeding.
#
# The calibrated seeding won on the images it was tuned on and did not
# carry over. On VALIDATION, five runs read at their last three epochs,
# it tied the frozen seeding on instance F1 while merging more pores and
# undercounting them by about seven percent instead of three. On the
# tuning images themselves it absorbed small pores into their
# neighbours: of the annotated pores between 300 and 800 square pixels,
# 9 percent were found under it against 37 percent under the frozen
# seeding, because the stronger blur of the distance maps runs a small
# pore's seed into its neighbour's. The pooled figures the calibration
# was chosen on could not show that cost - small pores are a few percent
# of the population - but the lower tail of the pore size distribution
# is a reported material quantity.
#
# With F1 tied on held-out images, the deployed model keeps the seeding
# that preserves small pores and whose test figure was measured: the
# score reported on TEST was taken under FROZEN_WATERSHED, and the
# fragment filter changes about one prediction in a thousand, so that
# figure still describes the deployed model. What it gives up is the
# pore count on the tuning images, where it overcounts by five percent;
# the sign of that error differs between image sets, and no single
# setting removes it on all of them.
DEPLOYED_POSTPROCESSING = WatershedParams(
    **{
        **FROZEN_WATERSHED.to_kwargs(),
        "min_instance_area_um2": MIN_INSTANCE_AREA_FROZEN_SEEDING_UM2,
    }
)

# Named post-processing configurations a checkpoint can be scored under.
POSTPROCESSING_CONFIGS = {
    "frozen": FROZEN_WATERSHED,
    "calibrated_2026_09_24": CALIBRATED_2026_09_24,
    "deployed": DEPLOYED_POSTPROCESSING,
}

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
