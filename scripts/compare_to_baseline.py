#!/usr/bin/env python3
"""
What one augmentation family changed, measured against the baseline.

A run under a candidate policy and the unaugmented baseline runs are
scored the same way, on the same images, and this puts the two curves
side by side. The comparison it produces is the unit the whole study
reports in: "adding this family moved instance F1 by so much, against a
noise floor of so much".

**The noise floor comes from the baseline, not from a table.** The
baseline was trained three times, changing only the random seed. The
spread between those three peaks is what this pipeline produces when
nothing real has changed, and it is recomputed here from the same file
the baseline curves come from, so it cannot drift out of step with
them. A candidate that moves the metric by less than that spread has
not been shown to do anything, however suggestive the direction.

**Two readings, one run.** A candidate is judged first on a screening
share of the budget and then, if it survives, on the whole of it.
Because a snapshot is kept every epoch, both readings come out of the
same series: the screening one restricted to the epochs inside that
share, the full one over all of them. The restriction applies to the
baseline too, or the two would be read at different amounts of
training and the difference would include that.

**The peak, not the last epoch.** Each run is represented by its best
snapshot, chosen by the same rule that picks a checkpoint to keep:
instance F1 first, then boundary agreement, then the error in the
number of pores, then merges and splits together. Using the last epoch
instead would compare where two curves happened to end rather than
what each policy can do.

**Every threshold is read separately.** Instances are cut out of the
predicted distance maps by a watershed whose seeding threshold is
frozen, and the decision is read at that frozen value. The neighbouring
values are reported beside it because a result that only holds at one
threshold is a property of that threshold rather than of the policy -
and each one is allowed to find its own peak epoch, since a snapshot
that peaks under one seeding rule need not peak under another.

**A slice of the validation set can be read the same way.** Some
families are adopted for what they do to part of the data rather than
to all of it, and the pooled metric can be dominated by the part they
were never meant to change. Any breakdown the scoring layer writes out
- by scale bin, by material - can therefore be compared exactly as the
whole set is, against the same baseline runs and with its own noise
floor measured on that same slice.

A slice is reported twice, and the two numbers can differ a great deal.
Letting the slice choose its own peak snapshot asks what a policy can
do there. Reading the slice at the peak of the whole set asks what the
model somebody would actually keep achieves there - a stricter
question, and usually a much wider floor, because the baseline seeds
peak at different epochs and a slice whose quality varies with training
is then read at three different points of its own curve.

**Several rows of one axis can be read as one slice.** The subset a
family was adopted for is not always a row the scoring layer writes
out: it can be the union of a few of them. Written ``axis=one+other``,
such a view is pooled from the counts the rows carry, exactly as the
scoring layer would have pooled the images had it been asked for that
grouping in the first place, so the result is the figure itself and not
an average of figures.

Pooling matters more than it looks, because a noise floor narrows as a
slice grows. Two rows of six and twelve images can each be too small to
separate anything from seed noise while their union separates it
comfortably, and reading them apart would then report "nothing shown"
about a subset where something was.

Only the figures that follow from the stored counts appear in a pooled
view. Boundary agreement does not: it is pooled over outline pixels,
and the artefact keeps the result rather than the pixel counts behind
it, so there is nothing to add up. A weighted average of the rows would
look like the same quantity and not be comparable with it, so the field
is left out rather than approximated, and the rule that picks a peak
runs without its second tie-break. Where that rule then ends in a tie,
the comparison stops instead of choosing arbitrarily.

**What this does not do.** It does not pronounce a verdict. The band
around zero is reported, and so are the metrics a family was adopted to
improve, because a candidate that costs a little overall F1 while
clearly fixing what it was meant to fix is a judgement call that
belongs to the person reading the table.

Examples
--------
Compare the orientation run against the baseline:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e1/e1_curves.json

Against a baseline scored at a different threshold:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e1/e1_curves.json \\
        --baseline checkpoints/e0_v2/e0_v2_curves_cdt025.json

Decide a scale run on the bins it was meant to change:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e2/e2_curves.json \\
        --slice scale_bin=fine --slice scale_bin=coarse

Decide a tonal run on one instrument against the other, where the
second instrument is two of the materials pooled:
    $ python scripts/compare_to_baseline.py \\
        --candidate checkpoints/e3/e3_curves.json \\
        --slice material=AS --slice material=K+VAB
"""
import argparse
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_BASELINE = Path("checkpoints/e0_v2/e0_v2_curves.json")

# Below this the spread of three runs is too small to be a usable
# floor, and twice the sample deviation is used instead: three draws
# can land close together by chance, and a floor of almost zero would
# call every later difference significant.
MIN_USABLE_SPREAD = 0.002

# The tolerance boundary agreement is decided at, as a share of the
# mean annotated pore diameter of the image.
DECISION_SCALE = "0.1"

# Share of the budget the screening reading is taken at.
SCREENING_FRACTION = 0.6

# Which runs in the baseline artefact are the baseline. The file also
# holds a calibration run trained under every family at once, which
# exists to show how much later a rich policy converges and takes no
# part in attributing anything. Letting it into the floor would widen
# the floor with a difference that is not a seed difference, and would
# pull the mean it is measured against towards a policy rather than
# away from one.
BASELINE_RUN_PREFIX = "b0_"

# The whole validation set, as opposed to one slice of it.
OVERALL = "overall"

# Separates the values of an axis that a view pools into one slice.
POOLED_JOIN = "+"

# Counts a pooled view adds up. Everything it reports is derived from
# these, so a pooled figure is the figure itself rather than an average
# of the rows' figures - the two differ whenever the rows hold
# different numbers of pores, which is the usual case.
POOLED_COUNTS = (
    "n_images", "n_scale_outliers_excluded", "n_gt", "n_pred",
    "n_true_positives", "n_false_positives", "n_false_negatives",
    "n_merges", "n_splits",
)


class AmbiguousPeak(Exception):
    """Two snapshots the ranking rule has no way to choose between.

    Deliberately not a ``ValueError``: a reading that cannot be taken
    because the window holds no snapshot is an ordinary, expected gap
    and is reported as one, whereas this means the rule ran and came
    out undecided. Sharing an exception type would let the second be
    logged as the first and the comparison continue without it.
    """


@dataclass(frozen=True)
class Snapshot:
    """One scored snapshot of one run under one setting.

    Two sets of figures, because choosing which snapshot represents a
    run and reporting what that snapshot achieved are separate
    questions. They coincide for the whole validation set and come
    apart on a slice of it: the snapshot a run peaks at overall is not
    the one it peaks at on any given subset of the images.

    Attributes
    ----------
    run, setting : str
    epoch : int
    figures : dict
        What is reported.
    ranking_figures : dict
        What decides which snapshot of a curve is its peak. The same
        object as ``figures`` unless a slice is being read at a peak
        chosen elsewhere.
    """

    run: str
    setting: str
    epoch: int
    figures: dict
    ranking_figures: dict


def parse_key(key: str) -> tuple[str, str, int]:
    """Split a result key into the run, the setting and the epoch.

    Keys are written as ``run/epoch-N`` when one setting was scored and
    ``run/epoch-N@setting`` when several were. Both forms are accepted
    so that a candidate scored at the frozen setting alone compares
    against a baseline scored across a sweep, and the other way round.

    Parameters
    ----------
    key : str

    Returns
    -------
    tuple
        ``(run, setting, epoch)``; the setting is ``"frozen"`` when the
        key carries none.

    Raises
    ------
    ValueError
        If the key carries no epoch number, which means it names
        something other than a snapshot of a run.
    """
    body, _, setting = key.partition("@")
    run, _, snapshot = body.rpartition("/")
    match = re.search(r"epoch-(\d+)", snapshot)
    if match is None:
        raise ValueError(
            f"Result key {key!r} names no epoch, so it cannot be placed "
            f"on a curve"
        )
    return run or body, setting or "frozen", int(match.group(1))


def pool_rows(rows: list[dict], label: str) -> dict:
    """Combine several rows of one axis into the figures of their union.

    The counts add up, and every figure reported here is recomputed
    from the totals rather than averaged from the rows, because the
    rows hold different numbers of pores and an average of ratios is
    not the ratio of the sums.

    What cannot be recomputed is left out. Boundary agreement is pooled
    over outline pixels and the artefact stores only its result, so the
    quantity has no addends here; a weighted average would wear the
    same name without being the same measurement.

    Parameters
    ----------
    rows : list of dict
        Rows of one section, each covering a disjoint set of images.
    label : str
        What to call the union.

    Returns
    -------
    dict
        The same field names the rows use, minus the ones that cannot
        be pooled from counts.

    Raises
    ------
    ValueError
        If the union holds no annotated pore, since every figure below
        is a ratio to that count.
    """
    totals = {
        field: sum(row[field] for row in rows) for field in POOLED_COUNTS
    }
    n_gt = totals["n_gt"]
    if n_gt <= 0:
        raise ValueError(
            f"View {label!r} pools {len(rows)} row(s) holding no "
            f"annotated pore, so there is nothing to score it against"
        )
    true_positives = totals["n_true_positives"]
    denominator = (
        2 * true_positives
        + totals["n_false_positives"]
        + totals["n_false_negatives"]
    )
    predicted = true_positives + totals["n_false_positives"]
    return {
        **totals,
        "label": label,
        "precision": true_positives / predicted if predicted else 0.0,
        "recall": true_positives / n_gt,
        "f1": 2 * true_positives / denominator if denominator else 0.0,
        "merges_per_100_gt": 100 * totals["n_merges"] / n_gt,
        "splits_per_100_gt": 100 * totals["n_splits"] / n_gt,
        "pore_count_error": (totals["n_pred"] - n_gt) / n_gt,
    }


def figures_for(entry: dict, view: str) -> dict:
    """The figures of one view of a scored snapshot.

    A snapshot is scored once and reported several ways: pooled over
    the whole validation set, and broken down along the axes the
    scoring layer writes out beside it. A view names one of those. The
    breakdown rows carry the same field names as the pooled figures, so
    everything downstream - the rule that picks a peak, the noise
    floor, the readings - works on a slice without knowing it is one.

    Parameters
    ----------
    entry : dict
        One scored snapshot as the artefact holds it.
    view : str
        ``"overall"``, or a row label as the artefact writes it, e.g.
        ``"scale_bin=fine"`` or ``"material=AS"``. The part before the
        equals sign names the axis and locates the section it lives in.
        Several values of one axis joined by ``+``, as in
        ``"material=K+VAB"``, name their union and are pooled from the
        rows' counts.

    Returns
    -------
    dict

    Raises
    ------
    KeyError
        If the view is absent. Raised rather than skipped: a view
        missing from some snapshots would shorten one curve and leave
        the two sides of a comparison read over different amounts of
        training, which is the one thing the pairing exists to prevent.
    """
    if view == OVERALL:
        return entry[OVERALL]
    axis, _, value = view.partition("=")
    if not value:
        raise KeyError(
            f"View {view!r} names no value; write it as the artefact "
            f"labels a row, e.g. 'scale_bin=fine'"
        )
    section = f"per_{axis}"
    rows = entry.get(section)
    if rows is None:
        raise KeyError(
            f"This artefact carries no {section!r} section, so "
            f"{view!r} cannot be read from it; it holds "
            f"{sorted(entry)}"
        )
    by_label = {row.get("label"): row for row in rows}
    wanted = [f"{axis}={part}" for part in value.split(POOLED_JOIN)]
    missing = [label for label in wanted if label not in by_label]
    if missing:
        raise KeyError(
            f"No row labelled {', '.join(repr(m) for m in missing)} in "
            f"{section!r}; it holds {sorted(by_label)}"
        )
    if len(wanted) == 1:
        return by_label[wanted[0]]
    return pool_rows([by_label[label] for label in wanted], view)


def load_snapshots(
    path: Path,
    setting_label: Optional[str] = None,
    view: str = OVERALL,
    rank_on: Optional[str] = None,
) -> list[Snapshot]:
    """Read a scoring artefact into a flat list of snapshots.

    Parameters
    ----------
    path : Path
    setting_label : str, optional
        Call every snapshot in this file this setting, overriding what
        the keys say. An artefact scored under one setting carries no
        setting in its keys at all, so a baseline scored at one
        threshold cannot otherwise be matched against a candidate
        scored across a sweep, and the pair would be skipped as
        one-sided rather than compared.
    view : str, optional
        Which view of each snapshot to report; the whole validation set
        by default.
    rank_on : str, optional
        Which view decides where a curve peaks, when that should not be
        the view being reported. Reading a slice at the peak of the
        whole set answers a different question - what one model, chosen
        the usual way, achieved on that slice - and gives a harsher
        noise floor, because the three baseline seeds then peak at
        epochs far enough apart that the slice is read at three
        different points of its own curve.

    Returns
    -------
    list of Snapshot

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If a view is absent from a snapshot.
    """
    payload = json.loads(path.read_text())
    snapshots = []
    for key, entry in payload.items():
        run, setting, epoch = parse_key(key)
        figures = figures_for(entry, view)
        snapshots.append(Snapshot(
            run, setting_label or setting, epoch, figures,
            figures if rank_on is None else figures_for(entry, rank_on),
        ))
    return snapshots


def selection_key(figures: dict) -> tuple:
    """Rank one snapshot by the rule that picks a checkpoint to keep.

    Instance F1 decides; where two snapshots tie on it, boundary
    agreement decides, then the error in the number of pores, then
    merges and splits together. The tie-breaks matter because two
    snapshots can reach the same F1 while one traces pore walls
    faithfully and the other cuts corners, or while one merges
    neighbouring pores as often as the other splits single ones.

    A view pooled from several rows carries no boundary agreement, so
    there the chain runs without its second link. Every snapshot of one
    curve is read through the same view, so the keys within a curve stay
    comparable; what a shorter chain costs is a tie it can no longer
    break, and :func:`peak` refuses rather than picking one.

    Parameters
    ----------
    figures : dict
        The pooled figures of one snapshot.

    Returns
    -------
    tuple
        Sorts descending: larger is better.
    """
    boundary = figures.get("boundary_f1")
    tail = (
        -abs(figures["pore_count_error"]),
        -(figures["merges_per_100_gt"] + figures["splits_per_100_gt"]),
    )
    if boundary is None:
        return (figures["f1"], *tail)
    return (figures["f1"], boundary[DECISION_SCALE], *tail)


def peak(
    snapshots: list[Snapshot], within_epoch: Optional[int] = None
) -> Snapshot:
    """The best snapshot of one curve.

    Best by the figures a snapshot is ranked on, which are its own
    unless it is carrying a slice to be read at a peak chosen on
    something else.

    Parameters
    ----------
    snapshots : list of Snapshot
        One run under one setting.
    within_epoch : int, optional
        Consider only epochs up to and including this one, for a
        reading taken at part of the budget.

    Returns
    -------
    Snapshot

    Raises
    ------
    ValueError
        If no snapshot falls inside the window, which would otherwise
        surface as an unexplained missing row.
    AmbiguousPeak
        If the ranking rule ends in a tie it cannot break, which
        happens only on a pooled view, where it runs without boundary
        agreement. Choosing between the tied snapshots would be
        arbitrary, and an arbitrary choice on one side of a paired
        comparison becomes a difference attributed to the policy.
    """
    eligible = [
        snapshot for snapshot in snapshots
        if within_epoch is None or snapshot.epoch <= within_epoch
    ]
    if not eligible:
        raise ValueError(
            f"No snapshot at or below epoch {within_epoch}; the curve "
            f"holds epochs {sorted(s.epoch for s in snapshots)}"
        )
    best = max(eligible, key=lambda s: selection_key(s.ranking_figures))
    key = selection_key(best.ranking_figures)
    tied = [s for s in eligible if selection_key(s.ranking_figures) == key]
    if len(tied) > 1:
        raise AmbiguousPeak(
            f"Epochs {sorted(s.epoch for s in tied)} of {best.run!r} "
            f"rank identically on every figure this view carries, so "
            f"which one represents the run cannot be decided"
        )
    return best


def group_curves(
    snapshots: list[Snapshot],
) -> dict[tuple[str, str], list[Snapshot]]:
    """Gather snapshots into one curve per run and setting.

    Parameters
    ----------
    snapshots : list of Snapshot

    Returns
    -------
    dict
        Keyed by ``(run, setting)``, each value sorted by epoch.
    """
    curves: dict[tuple[str, str], list[Snapshot]] = {}
    for snapshot in snapshots:
        curves.setdefault((snapshot.run, snapshot.setting), []).append(
            snapshot
        )
    return {
        key: sorted(value, key=lambda s: s.epoch)
        for key, value in curves.items()
    }


def significance_threshold(peaks: list[float]) -> float:
    """The floor below which a difference is not evidence of anything.

    The spread of the baseline peaks, except where three runs happen to
    land almost on top of each other: a floor of nearly zero would call
    every later difference significant, so twice the sample deviation
    takes over there.

    Parameters
    ----------
    peaks : list of float
        One peak per baseline seed.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If fewer than two peaks are given, since one run shows no
        spread at all.
    """
    if len(peaks) < 2:
        raise ValueError(
            f"A noise floor needs at least two baseline runs, got "
            f"{len(peaks)}"
        )
    spread = max(peaks) - min(peaks)
    if spread >= MIN_USABLE_SPREAD:
        return spread
    return max(spread, 2 * statistics.stdev(peaks))


def screening_epoch(n_epochs: int, fraction: float) -> int:
    """Last epoch lying within the screening share of the budget.

    Rounded down to the epoch below the share, because a reading can
    only be taken where a snapshot exists. The rounding shortens both
    sides of the comparison equally, which is what keeps it paired.

    Parameters
    ----------
    n_epochs : int
    fraction : float

    Returns
    -------
    int
    """
    return max(1, int(n_epochs * fraction))


def verdict(delta: float, floor: float) -> str:
    """Where a difference falls relative to the noise floor.

    Deliberately not a decision. A candidate inside the band may still
    be adopted on the strength of the error it was meant to fix, and
    one outside it may still be rejected as too expensive; both are
    judgements for the reader.

    Parameters
    ----------
    delta : float
    floor : float

    Returns
    -------
    str
    """
    if delta >= floor:
        return "above the floor"
    if delta <= -floor:
        return "below the floor"
    return "inside the band"


def format_row(label: str, snapshot: Snapshot) -> str:
    """One line of the per-snapshot table.

    Parameters
    ----------
    label : str
    snapshot : Snapshot

    Returns
    -------
    str
    """
    figures = snapshot.figures
    boundary = figures.get("boundary_f1")
    shown = (
        "     -" if boundary is None
        else f"{boundary[DECISION_SCALE]:.4f}"
    )
    return (
        f"{label:<26} ep{snapshot.epoch:>3}  "
        f"F1 {figures['f1']:.4f}  "
        f"bF1 {shown}  "
        f"count {100 * figures['pore_count_error']:+6.1f}%  "
        f"merge {figures['merges_per_100_gt']:5.2f}  "
        f"split {figures['splits_per_100_gt']:5.2f}"
    )


def report_setting(
    setting: str,
    candidate: list[Snapshot],
    baselines: list[list[Snapshot]],
    within_epoch: Optional[int],
    heading: str,
    view: str = OVERALL,
) -> dict:
    """Compare one candidate curve against the baseline curves.

    Parameters
    ----------
    setting : str
        Watershed setting these curves were scored under.
    candidate : list of Snapshot
    baselines : list of list of Snapshot
        One curve per baseline seed.
    within_epoch : int, optional
        Restrict both sides to epochs up to this one.
    heading : str
        What this reading is, for the log.
    view : str, optional
        Which view of the validation set these curves carry. Recorded
        so that a summary file holding several views says which is
        which, rather than leaving them to be told apart by their
        position in the list.

    Returns
    -------
    dict
        The figures behind the lines logged, for the summary file.
    """
    candidate_peak = peak(candidate, within_epoch)
    baseline_peaks = [peak(curve, within_epoch) for curve in baselines]
    values = [snapshot.figures["f1"] for snapshot in baseline_peaks]
    mean = statistics.fmean(values)
    floor = significance_threshold(values)
    delta = candidate_peak.figures["f1"] - mean

    logger.info("")
    logger.info("%s, seeding threshold %s, view %s", heading, setting, view)
    for snapshot in baseline_peaks:
        logger.info("  %s", format_row(snapshot.run, snapshot))
    logger.info(
        "  %-26s        F1 %.4f  (mean of %d seed(s))",
        "baseline mean", mean, len(values),
    )
    logger.info("  %s", format_row(candidate_peak.run, candidate_peak))
    logger.info(
        "  delta %+.4f against a floor of %.4f: %s",
        delta, floor, verdict(delta, floor),
    )
    return {
        "setting": setting,
        "view": view,
        "reading": heading,
        "within_epoch": within_epoch,
        "candidate": {
            "run": candidate_peak.run,
            "epoch": candidate_peak.epoch,
            **{
                field: candidate_peak.figures[field]
                for field in ("f1", "pore_count_error",
                              "merges_per_100_gt", "splits_per_100_gt")
            },
            "boundary_f1": candidate_peak.figures.get("boundary_f1"),
        },
        "baseline_peaks": {
            snapshot.run: {"epoch": snapshot.epoch,
                           "f1": snapshot.figures["f1"]}
            for snapshot in baseline_peaks
        },
        "baseline_mean_f1": mean,
        "significance_floor": floor,
        "delta_f1": delta,
        "verdict": verdict(delta, floor),
    }


def select_baseline_runs(
    curves: dict[tuple[str, str], list[Snapshot]], prefix: str
) -> dict[tuple[str, str], list[Snapshot]]:
    """Keep only the runs that are the baseline condition.

    The scoring artefact of the calibration set holds a run trained
    under every augmentation family at once alongside the unaugmented
    seeds. That run calibrated the budget and takes no part in
    attribution: inside the floor it would widen the floor with a
    difference that is not a difference of seed, and inside the mean it
    would move the reference towards a policy.

    Parameters
    ----------
    curves : dict
    prefix : str
        Run names starting with this are the baseline.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If nothing matches, which would otherwise produce a comparison
        against an empty reference.
    """
    kept = {
        key: curve for key, curve in curves.items()
        if key[0].startswith(prefix)
    }
    dropped = sorted({run for run, _ in curves if not run.startswith(prefix)})
    for run in dropped:
        logger.info(
            "Excluded %r from the baseline: it does not take part in "
            "attribution.", run,
        )
    if not kept:
        raise ValueError(
            f"No run in the baseline artefact starts with {prefix!r}; "
            f"it holds {sorted({run for run, _ in curves})}"
        )
    return kept


def resolve_settings(
    candidate: dict[tuple[str, str], list[Snapshot]],
    baseline: dict[tuple[str, str], list[Snapshot]],
) -> list[str]:
    """Settings both sides were scored under, frozen one first.

    A setting present on only one side cannot be compared, and silently
    dropping it would leave a reader wondering why a threshold they
    asked for is missing, so it is named in the log instead.

    Parameters
    ----------
    candidate, baseline : dict

    Returns
    -------
    list of str
    """
    on_candidate = {setting for _, setting in candidate}
    on_baseline = {setting for _, setting in baseline}
    shared = on_candidate & on_baseline
    for setting in sorted((on_candidate | on_baseline) - shared):
        logger.warning(
            "Setting %r was scored on only one side; skipping it.",
            setting,
        )
    return sorted(shared, key=lambda name: (name != "frozen", name))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path, required=True,
        help="Scoring artefact of the run under the candidate policy.",
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE,
        help="Scoring artefact of the unaugmented baseline runs.",
    )
    parser.add_argument(
        "--screening-fraction", type=float, default=SCREENING_FRACTION,
        help="Share of the budget the screening reading is taken at.",
    )
    parser.add_argument(
        "--baseline-setting", default=None,
        help="Treat every snapshot of the baseline artefact as this "
             "watershed setting, e.g. 'cdt=0.25' for a baseline "
             "rescored at that threshold, whose keys carry no setting "
             "of their own.",
    )
    parser.add_argument(
        "--baseline-run-prefix", default=BASELINE_RUN_PREFIX,
        help="Runs in the baseline artefact whose names start with "
             "this are the baseline; the rest are excluded from the "
             "floor and the mean.",
    )
    parser.add_argument(
        "--slice", action="append", default=[], metavar="AXIS=VALUE",
        help="Also compare on this slice of the validation set, "
             "labelled as the scoring artefact labels it, e.g. "
             "'scale_bin=fine'; join values with '+', as in "
             "'material=K+VAB', to read their union, pooled from the "
             "rows' counts. Repeat for several. Each slice is "
             "reported twice: once letting it choose its own peak "
             "snapshot, and once read at the peak of the whole set.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the comparison as JSON.",
    )
    return parser.parse_args(argv)


def views_to_compare(slices: list[str]) -> list[tuple[str, Optional[str]]]:
    """Every view to report, as ``(view, view that ranks it)``.

    The whole validation set always comes first. A slice is reported
    twice, because the two readings answer different questions and
    neither alone is honest. Letting the slice pick its own peak asks
    what the policy can do there; reading it at the peak of the whole
    set asks what the model someone would actually keep achieves there,
    and carries a wider noise floor for it. Reporting both leaves the
    stricter number visible instead of only the one that favours a
    result.

    Parameters
    ----------
    slices : list of str
        Slice labels, in the order they were asked for.

    Returns
    -------
    list of tuple
        ``None`` as the second element means the view ranks itself.
    """
    views: list[tuple[str, Optional[str]]] = [(OVERALL, None)]
    for label in slices:
        views.append((label, None))
        views.append((label, OVERALL))
    return views


def main(argv: Optional[list[str]] = None) -> int:
    """Compare a candidate run against the baseline and report.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    for path in (args.candidate, args.baseline):
        if not path.exists():
            logger.error("No such scoring artefact: %s.", path)
            return EXIT_FAILED

    summary = []
    cut = None
    for view, rank_on in views_to_compare(args.slice):
        label = view if rank_on is None else f"{view} at {rank_on} peak"
        try:
            candidate = group_curves(load_snapshots(
                args.candidate, view=view, rank_on=rank_on,
            ))
            baseline = select_baseline_runs(
                group_curves(load_snapshots(
                    args.baseline, args.baseline_setting,
                    view=view, rank_on=rank_on,
                )),
                args.baseline_run_prefix,
            )
        except (KeyError, ValueError) as failure:
            logger.error("View %s: %s", label, failure)
            return EXIT_FAILED
        if not candidate:
            logger.error("%s holds no snapshots.", args.candidate)
            return EXIT_FAILED

        if cut is None:
            n_epochs = max(
                snapshot.epoch
                for curve in candidate.values() for snapshot in curve
            )
            cut = screening_epoch(n_epochs, args.screening_fraction)
            logger.info(
                "Candidate budget %d epoch(s); screening reading at "
                "epoch %d (%.0f%% of it).",
                n_epochs, cut, 100 * args.screening_fraction,
            )

        for setting in resolve_settings(candidate, baseline):
            candidate_curves = [
                curve for (_, name), curve in candidate.items()
                if name == setting
            ]
            baseline_curves = [
                curve for (_, name), curve in baseline.items()
                if name == setting
            ]
            for curve in candidate_curves:
                readings = (
                    (cut, f"Screening reading (through epoch {cut})"),
                    (None, "Full-budget reading"),
                )
                for within, heading in readings:
                    try:
                        summary.append(report_setting(
                            setting, curve, baseline_curves, within,
                            heading, label,
                        ))
                    except AmbiguousPeak as failure:
                        logger.error(
                            "%s, seeding threshold %s, view %s: %s.",
                            heading, setting, label, failure,
                        )
                        return EXIT_FAILED
                    except ValueError as failure:
                        # An artefact holding only each run's peak
                        # snapshot supports the full-budget reading but
                        # not a partial one, because a peak can lie
                        # outside the window. Saying so beats both
                        # crashing and quietly comparing fewer seeds
                        # than the floor was built from.
                        logger.warning(
                            "%s, seeding threshold %s, view %s: not "
                            "available (%s).",
                            heading, setting, label, failure,
                        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("")
        logger.info("Wrote %s.", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
