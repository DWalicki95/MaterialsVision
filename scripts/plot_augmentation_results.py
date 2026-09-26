#!/usr/bin/env python3
"""
Figures for the augmentation study: learning curves and attribution.

Everything drawn here is read from scoring artefacts already on disk;
nothing is recomputed from a model. Every figure is written together
with a table holding the same numbers, so that a value can be read
exactly rather than estimated off a plotted mark.

**Learning curves, one policy per panel.** Instance F1 on VALIDATION
over the twelve epochs of each run, at the frozen watershed setting.
Several of the policies compared differ by less than a thousandth, so
drawn on one plot their curves would be a tangle in which no single
run can be followed. Each policy gets a panel of its own instead, over
the same axes, and every panel carries the unaugmented baseline in the
background - the spread of its three seeds and their mean - so that
each policy is read against the reference it was judged by. The epoch
each run is represented by, its peak on VALIDATION, is marked, since
that is the snapshot every comparison used.

**Attribution, in the two views the study measures.** The upper panel
is the gain each policy shows over the baseline when added; the lower
one is what removing a family from the adopted composite costs. Both
share one axis of instance F1 difference, and both carry the noise band
of plus or minus the significance floor, which is what separates a
difference from the spread between seeds. A mark inside the band is a
difference this study cannot tell from chance.

**TEST, once it has been scored.** One row per snapshot scored on TEST:
the three unaugmented seeds, whose spread is shaded as the reference
for noise, then the policies trained on TRAIN, then the final model.
The final model sits below a rule and is labelled with the data it
learned from, because it saw VALIDATION as well and so cannot be read
as one more policy in the comparison above it. Until TEST has been
scored only the VALIDATION figures are drawn.

Examples
--------
Write the figures and tables next to the scoring artefacts:
    $ python scripts/plot_augmentation_results.py

Write them elsewhere, as vector graphics only:
    $ python scripts/plot_augmentation_results.py \\
        --out-dir reports/figures --formats pdf
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from materials_vision.logging_config import setup_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

ARTEFACTS = Path("checkpoints")

DEFAULT_OUT_DIR = ARTEFACTS / "figures" / "augmentation"

N_EPOCHS = 12

B0_CURVES = ARTEFACTS / "e0_v2" / "e0_v2_curves.json"

B0_RUNS = ("b0_seed20260907", "b0_seed20260908", "b0_seed20260909")

# Surface, ink and one series colour. A single hue is enough because
# every panel carries one policy, and identity comes from the panel
# title; the baseline and the noise band are reference, not series,
# and are drawn in neutral tones so that they recede behind the data.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NOISE_BAND = "#f0efec"
SERIES = "#2a78d6"


class Policy(NamedTuple):
    """One augmentation policy and where its scores are stored.

    Parameters
    ----------
    key : str
        Short identifier, used in the tables.
    label : str
        How the figures name it.
    curves : Path
        Scoring artefact holding the run's twelve snapshots.
    run : str
        The run's name inside that artefact.
    versus_b0 : Path
        The run's comparison against the unaugmented baseline.
    """

    key: str
    label: str
    curves: Path
    run: str
    versus_b0: Path


POLICIES = (
    Policy(
        "d4", "D4 (FULL_A)",
        ARTEFACTS / "e1" / "e1_curves.json", "d4_seed20260907",
        ARTEFACTS / "e1" / "e1_vs_b0.json",
    ),
    Policy(
        "d4_tonal", "D4 + tonalna",
        ARTEFACTS / "e3" / "e3_curves.json", "tonal_seed20260907",
        ARTEFACTS / "e3" / "e3_vs_b0.json",
    ),
    Policy(
        "d4_blur", "D4 + blur",
        ARTEFACTS / "e4" / "e4_curves.json", "blur_seed20260907",
        ARTEFACTS / "e4" / "e4_vs_b0.json",
    ),
    Policy(
        "full_b", "FULL_B = D4 + tonalna + blur (przyjęta)",
        ARTEFACTS / "e8" / "e8_full_b_curves.json", "full_b_seed20260907",
        ARTEFACTS / "e8" / "e8_full_b_vs_b0.json",
    ),
    Policy(
        "no_d4", "tonalna + blur (bez D4)",
        ARTEFACTS / "e8" / "e8_full_b_no_d4_curves.json",
        "full_b_no_d4_seed20260907",
        ARTEFACTS / "e8" / "e8_full_b_no_d4_vs_b0.json",
    ),
)

COMPOSITE = "full_b"

# Removing a family from the composite leaves one of the other runs:
# the family named first, the run it leaves behind second.
REMOVALS = (
    ("D4", "no_d4"),
    ("tonalna", "d4_blur"),
    ("blur", "d4_tonal"),
)


TEST_SCORES = ARTEFACTS / "final" / "test_scores.json"


class Snapshot(NamedTuple):
    """One snapshot scored on TEST.

    Parameters
    ----------
    key : str
        Its name in the TEST artefact: run directory and snapshot.
    label : str
        How the figure names it.
    trained_on : str
        The subsets it learned from.
    """

    key: str
    label: str
    trained_on: str


# Each at the epoch chosen on VALIDATION before TEST was opened.
B0_TEST = (
    Snapshot("b0_seed20260907/epoch-3", "B0, seed 0907", "TRAIN"),
    Snapshot("b0_seed20260908/epoch-8", "B0, seed 0908", "TRAIN"),
    Snapshot("b0_seed20260909/epoch-6", "B0, seed 0909", "TRAIN"),
)

POLICIES_TEST = (
    Snapshot("d4_seed20260907/epoch-10", "D4 = FULL_A", "TRAIN"),
    Snapshot(
        "full_b_seed20260907/epoch-8", "FULL_B, polityka przyjęta", "TRAIN"
    ),
)

# Its epoch was fixed before it was trained, since no held-out set was
# left to choose one on.
FINAL_TEST = Snapshot(
    "full_b_trainval_seed20260907/epoch-12", "FULL_B, model finalny",
    "TRAIN+VAL",
)


class Reading(NamedTuple):
    """A run's full-budget reading against the baseline.

    Parameters
    ----------
    peak_f1 : float
        Instance F1 at the snapshot chosen on VALIDATION.
    epoch : int
        That snapshot's epoch.
    delta : float
        Its difference from the mean of the baseline seeds' peaks.
    floor : float
        The significance floor the difference is judged against.
    """

    peak_f1: float
    epoch: int
    delta: float
    floor: float


def load_curve(path: Path, run: str, n_epochs: int = N_EPOCHS) -> np.ndarray:
    """Instance F1 of every epoch of one run, at the frozen setting.

    Artefacts scored at a single setting name a snapshot by run and
    epoch alone; those scored at several add the setting after an
    ``@``. Both forms are accepted, and only the frozen setting is read.

    Parameters
    ----------
    path : Path
    run : str
    n_epochs : int, optional

    Returns
    -------
    numpy.ndarray
        One value per epoch, epoch one first.

    Raises
    ------
    KeyError
        If any epoch of the run is missing at the frozen setting.
    """
    table = json.loads(path.read_text())
    values = []
    for epoch in range(1, n_epochs + 1):
        names = (f"{run}/epoch-{epoch}", f"{run}/epoch-{epoch}@frozen")
        found = [name for name in names if name in table]
        if not found:
            raise KeyError(
                f"{path} holds no epoch {epoch} of {run} at the frozen "
                f"setting"
            )
        values.append(table[found[0]]["overall"]["f1"])
    return np.asarray(values)


def _full_budget_entry(path: Path) -> dict:
    """The comparison's full-budget reading at the frozen setting.

    Only the pooled view is taken. Comparisons made for a family with a
    pre-registered slice also hold readings on that slice, and those
    answer a narrower question than the one these figures show.
    Artefacts written before slices existed carry no view at all and
    are pooled by construction.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If the artefact does not hold exactly one such reading.
    """
    entries = [
        entry for entry in json.loads(path.read_text())
        if entry["setting"] == "frozen"
        and entry.get("view", "overall") == "overall"
        and entry["reading"].startswith("Full-budget")
    ]
    if len(entries) != 1:
        raise ValueError(
            f"{path} holds {len(entries)} full-budget reading(s) at the "
            f"frozen setting, expected one"
        )
    return entries[0]


def load_reading(path: Path) -> Reading:
    """Read a run's comparison against the baseline.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Reading
    """
    entry = _full_budget_entry(path)
    return Reading(
        peak_f1=entry["candidate"]["f1"],
        epoch=int(entry["candidate"]["epoch"]),
        delta=entry["delta_f1"],
        floor=entry["significance_floor"],
    )


def load_baseline_peaks(path: Path) -> dict[str, tuple[int, float]]:
    """Where each baseline seed peaks, as the comparison chose it.

    Parameters
    ----------
    path : Path
        Any comparison against the baseline; all of them record the
        same peaks.

    Returns
    -------
    dict of str to tuple of (int, float)
        Epoch and instance F1 per baseline run.
    """
    peaks = _full_budget_entry(path)["baseline_peaks"]
    return {
        run: (int(peak["epoch"]), float(peak["f1"]))
        for run, peak in peaks.items()
    }


def verdict(delta: float, floor: float) -> str:
    """Name the side of the noise band a difference falls on.

    Parameters
    ----------
    delta, floor : float

    Returns
    -------
    str
    """
    if delta > floor:
        return "powyżej pasma szumu"
    if delta < -floor:
        return "poniżej pasma szumu"
    return "w paśmie szumu"


def style_axes(axes: plt.Axes) -> None:
    """Recessive chrome: hairline axes, horizontal hairline grid.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
    """
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(
        colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8,
        length=3, width=0.8,
    )
    axes.set_axisbelow(True)


def plot_curves(
    curves: dict[str, np.ndarray],
    readings: dict[str, Reading],
    baseline: np.ndarray,
    baseline_peaks: dict[str, tuple[int, float]],
) -> plt.Figure:
    """Draw the learning curves, one policy per panel.

    Parameters
    ----------
    curves : dict of str to numpy.ndarray
        Per policy key, instance F1 per epoch.
    readings : dict of str to Reading
    baseline : numpy.ndarray
        Baseline curves, one row per seed in the order of ``B0_RUNS``.
    baseline_peaks : dict

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = np.arange(1, baseline.shape[1] + 1)
    everything = np.concatenate([baseline.ravel(), *curves.values()])
    low = everything.min() - 0.004
    high = everything.max() + 0.006

    figure, panels = plt.subplots(
        2, 3, figsize=(11.0, 6.6), sharex=True, sharey=True,
        layout="constrained", facecolor=SURFACE,
    )
    flat = panels.ravel()

    axes = flat[0]
    style_axes(axes)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    for run, row in zip(B0_RUNS, baseline):
        axes.plot(epochs, row, color=INK_MUTED, linewidth=1.2)
        epoch, f1 = baseline_peaks[run]
        axes.plot(
            epoch, f1, "o", markersize=6, color=INK_MUTED,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
        )
    mean_peak = np.mean([f1 for _, f1 in baseline_peaks.values()])
    axes.set_title(
        f"B0: bez augmentacji, 3 seedy\nśrednia szczytów {mean_peak:.4f}",
        loc="left", fontsize=9, color=INK_PRIMARY,
    )

    spread_low, spread_high = baseline.min(axis=0), baseline.max(axis=0)
    spread_mean = baseline.mean(axis=0)
    for axes, policy in zip(flat[1:], POLICIES):
        style_axes(axes)
        axes.grid(axis="y", color=GRID, linewidth=0.6)
        axes.fill_between(
            epochs, spread_low, spread_high, color=GRID, linewidth=0,
        )
        axes.plot(epochs, spread_mean, color=INK_MUTED, linewidth=1.2)
        axes.plot(epochs, curves[policy.key], color=SERIES, linewidth=2)
        reading = readings[policy.key]
        axes.plot(
            reading.epoch, reading.peak_f1, "o", markersize=8,
            color=SERIES, markeredgecolor=SURFACE, markeredgewidth=2,
        )
        axes.annotate(
            f"{reading.peak_f1:.4f} (ep. {reading.epoch})",
            xy=(reading.epoch, reading.peak_f1), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8,
            color=INK_PRIMARY,
        )
        axes.set_title(
            f"{policy.label}\nΔ wobec B0 {reading.delta:+.4f}, "
            f"{verdict(reading.delta, reading.floor)}",
            loc="left", fontsize=9, color=INK_PRIMARY,
        )

    for axes in flat:
        axes.set_ylim(low, high)
        axes.set_xticks(epochs)
    for axes in panels[-1]:
        axes.set_xlabel("epoka", fontsize=8.5, color=INK_SECONDARY)
    for axes in panels[:, 0]:
        axes.set_ylabel(
            "instance F1, VALIDATION", fontsize=8.5, color=INK_SECONDARY
        )

    figure.legend(
        handles=[
            Patch(color=GRID, label="B0: zakres 3 seedów"),
            Line2D([], [], color=INK_MUTED, linewidth=1.2,
                   label="B0: średnia (w panelu B0: pojedyncze seedy)"),
            Line2D([], [], color=SERIES, linewidth=2,
                   label="polityka augmentacji"),
            Line2D([], [], color=SERIES, marker="o", linestyle="none",
                   markersize=8, markeredgecolor=SURFACE,
                   markeredgewidth=2,
                   label="epoka wybrana na VALIDATION"),
        ],
        loc="outside lower center", ncol=4, frameon=False, fontsize=8.5,
        labelcolor=INK_SECONDARY,
    )
    figure.suptitle(
        "Krzywe uczenia na VALIDATION "
        "(watershed zamrożony: środek 0.30, granica 0.40)",
        fontsize=11, color=INK_PRIMARY,
    )
    return figure


def removal_costs(readings: dict[str, Reading]) -> list[tuple[str, float]]:
    """What removing each family from the composite costs.

    The cost is the composite's score minus the score of the run
    without that family, so a positive cost means the family was
    contributing. Both scores are differences from the same baseline,
    so the baseline cancels and the cost is the difference of their
    peaks.

    Parameters
    ----------
    readings : dict of str to Reading

    Returns
    -------
    list of tuple of (str, float)
        Family name and cost, in the order of ``REMOVALS``.
    """
    composite = readings[COMPOSITE].delta
    return [
        (family, composite - readings[remaining].delta)
        for family, remaining in REMOVALS
    ]


def _lollipops(
    axes: plt.Axes, labels: list[str], values: list[float], floor: float
) -> None:
    """Horizontal lollipops from zero, with the noise band behind them.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
    labels : list of str
        Top row first.
    values : list of float
    floor : float
    """
    style_axes(axes)
    rows = np.arange(len(labels))[::-1]
    axes.axvspan(-floor, floor, color=NOISE_BAND, linewidth=0)
    axes.axvline(0.0, color=AXIS, linewidth=0.8)
    for row, value in zip(rows, values):
        axes.plot([0.0, value], [row, row], color=SERIES, linewidth=2)
        axes.plot(
            value, row, "o", markersize=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2,
        )
        axes.annotate(
            f"{value:+.4f}", xy=(value, row),
            xytext=(8 if value >= 0 else -8, 0),
            textcoords="offset points", va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8, color=INK_PRIMARY,
        )
    axes.set_yticks(rows)
    axes.set_yticklabels(labels, fontsize=8.5, color=INK_PRIMARY)
    axes.set_ylim(-0.7, len(labels) - 0.3)
    axes.tick_params(axis="y", length=0)


def plot_attribution(
    readings: dict[str, Reading], floor: float
) -> plt.Figure:
    """Draw both views of attribution over one axis.

    Parameters
    ----------
    readings : dict of str to Reading
    floor : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    costs = removal_costs(readings)
    added = [readings[policy.key].delta for policy in POLICIES]
    removed = [cost for _, cost in costs]
    everything = added + removed + [floor, -floor]
    low, high = min(everything) - 0.003, max(everything) + 0.004

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.0, 5.4), sharex=True, layout="constrained",
        facecolor=SURFACE, gridspec_kw={"height_ratios": [5, 3]},
    )
    _lollipops(upper, [policy.label for policy in POLICIES], added, floor)
    upper.set_title(
        "Zysk przy dodaniu: Δ instance F1 wobec B0",
        loc="left", fontsize=9.5, color=INK_PRIMARY,
    )
    upper.annotate(
        f"pasmo szumu ±{floor:.4f}", xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"), xytext=(0, -2),
        textcoords="offset points", ha="center", va="top", fontsize=7.5,
        color=INK_MUTED,
    )
    _lollipops(
        lower, [f"bez: {family}" for family, _ in costs], removed, floor
    )
    lower.set_title(
        "Koszt usunięcia rodziny z FULL_B: "
        "F1(FULL_B) − F1(FULL_B bez rodziny)",
        loc="left", fontsize=9.5, color=INK_PRIMARY,
    )
    lower.set_xlim(low, high)
    lower.set_xlabel(
        "różnica instance F1 na VALIDATION", fontsize=8.5,
        color=INK_SECONDARY,
    )
    figure.suptitle(
        "Atrybucja rodzin augmentacji, dwa widoki",
        fontsize=11, color=INK_PRIMARY,
    )
    return figure


def plot_test(scores: dict[str, dict]) -> plt.Figure:
    """Draw every snapshot scored on TEST on one axis of instance F1.

    Parameters
    ----------
    scores : dict
        The TEST artefact, keyed by snapshot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    baseline = np.array([scores[s.key]["overall"]["f1"] for s in B0_TEST])
    mean = baseline.mean()
    policies = [scores[s.key]["overall"]["f1"] for s in POLICIES_TEST]
    final = scores[FINAL_TEST.key]["overall"]["f1"]
    everything = [*baseline, *policies, final]

    labels = (
        ["B0, 3 seedy (TRAIN)"]
        + [f"{s.label} ({s.trained_on})" for s in POLICIES_TEST]
        + [f"{FINAL_TEST.label} ({FINAL_TEST.trained_on}, inne dane)"]
    )
    rows = np.arange(len(labels))[::-1]

    figure, axes = plt.subplots(
        figsize=(8.0, 3.4), layout="constrained", facecolor=SURFACE
    )
    style_axes(axes)
    axes.axvspan(baseline.min(), baseline.max(), color=NOISE_BAND,
                 linewidth=0)
    axes.axvline(mean, color=AXIS, linewidth=0.8)
    axes.axhline(rows[-1] + 0.5, color=GRID, linewidth=0.8)

    axes.plot(
        baseline, [rows[0]] * len(baseline), "o", markersize=6,
        color=INK_MUTED, markeredgecolor=SURFACE, markeredgewidth=1.5,
    )
    axes.annotate(
        f"średnia {mean:.4f}, rozstęp {np.ptp(baseline):.4f}",
        xy=(baseline.max(), rows[0]), xytext=(8, 0),
        textcoords="offset points", va="center", fontsize=8,
        color=INK_PRIMARY,
    )
    for row, value in zip(rows[1:], [*policies, final]):
        axes.plot(
            value, row, "o", markersize=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2,
        )
        axes.annotate(
            f"{value:.4f} (Δ wobec B0 {value - mean:+.4f})",
            xy=(value, row), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=8, color=INK_PRIMARY,
        )

    axes.set_yticks(rows)
    axes.set_yticklabels(labels, fontsize=8.5, color=INK_PRIMARY)
    axes.tick_params(axis="y", length=0)
    axes.set_ylim(-0.7, len(labels) - 0.3)
    axes.set_xlim(min(everything) - 0.004, max(everything) + 0.014)
    axes.set_xlabel("instance F1, TEST", fontsize=8.5, color=INK_SECONDARY)
    n_images = scores[FINAL_TEST.key]["overall"]["n_images"]
    axes.set_title(
        f"Ocena na TEST ({n_images} obrazów, jednorazowo)\n"
        f"pasmo: rozstęp trzech seedów B0",
        loc="left", fontsize=9.5, color=INK_PRIMARY,
    )
    return figure


def write_test_table(destination: Path, scores: dict[str, dict]) -> None:
    """Write every TEST score, the table twin of the TEST figure.

    Parameters
    ----------
    destination : Path
    scores : dict
    """
    mean = np.mean([scores[s.key]["overall"]["f1"] for s in B0_TEST])
    header = [
        "snapshot", "label", "trained_on", "f1", "precision", "recall",
        "boundary_f1", "merges_per_100_gt", "splits_per_100_gt",
        "pore_count_error", "delta_f1_vs_b0_mean",
    ]
    rows = []
    for snapshot in (*B0_TEST, *POLICIES_TEST, FINAL_TEST):
        overall = scores[snapshot.key]["overall"]
        boundary = next(iter(overall["boundary_f1"].values()))
        rows.append([
            snapshot.key, snapshot.label, snapshot.trained_on,
            f"{overall['f1']:.4f}", f"{overall['precision']:.4f}",
            f"{overall['recall']:.4f}", f"{boundary:.4f}",
            f"{overall['merges_per_100_gt']:.2f}",
            f"{overall['splits_per_100_gt']:.2f}",
            f"{overall['pore_count_error']:+.4f}",
            f"{overall['f1'] - mean:+.4f}",
        ])
    _write_csv(destination, header, rows)


def write_curve_table(
    destination: Path,
    curves: dict[str, np.ndarray],
    baseline: np.ndarray,
) -> None:
    """Write every curve as a row, the table twin of the curve figure.

    Parameters
    ----------
    destination : Path
    curves : dict of str to numpy.ndarray
    baseline : numpy.ndarray
    """
    header = ["run", "policy"] + [
        f"epoch_{epoch}" for epoch in range(1, baseline.shape[1] + 1)
    ]
    rows = [
        [run, "B0"] + [f"{value:.4f}" for value in row]
        for run, row in zip(B0_RUNS, baseline)
    ]
    rows += [
        [policy.run, policy.label]
        + [f"{value:.4f}" for value in curves[policy.key]]
        for policy in POLICIES
    ]
    _write_csv(destination, header, rows)


def write_attribution_table(
    destination: Path, readings: dict[str, Reading], floor: float
) -> None:
    """Write both views of attribution, the twin of that figure.

    Parameters
    ----------
    destination : Path
    readings : dict of str to Reading
    floor : float
    """
    header = [
        "view", "item", "peak_f1", "epoch", "delta_f1",
        "significance_floor", "verdict",
    ]
    rows = [
        [
            "added_vs_b0", policy.label,
            f"{readings[policy.key].peak_f1:.4f}",
            readings[policy.key].epoch,
            f"{readings[policy.key].delta:+.4f}", f"{floor:.4f}",
            verdict(readings[policy.key].delta, floor),
        ]
        for policy in POLICIES
    ]
    rows += [
        [
            "removed_from_full_b", family, "", "", f"{cost:+.4f}",
            f"{floor:.4f}", verdict(cost, floor),
        ]
        for family, cost in removal_costs(readings)
    ]
    _write_csv(destination, header, rows)


def _write_csv(destination: Path, header: list, rows: list) -> None:
    """Write one table.

    Parameters
    ----------
    destination : Path
    header : list
    rows : list of list
    """
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Wrote %s.", destination)


def save(figure: plt.Figure, stem: Path, formats: list[str]) -> None:
    """Write a figure in every format asked for, then release it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
    stem : Path
        Destination without an extension.
    formats : list of str
    """
    for extension in formats:
        destination = stem.with_suffix(f".{extension}")
        figure.savefig(destination, dpi=200, facecolor=SURFACE)
        logger.info("Wrote %s.", destination)
    plt.close(figure)


def common_floor(readings: dict[str, Reading]) -> float:
    """The one significance floor every reading shares.

    All runs are compared against the same baseline, so they must be
    judged against the same floor; a mismatch would mean the artefacts
    came from different baselines, and drawing them on one axis would
    be misleading.

    Parameters
    ----------
    readings : dict of str to Reading

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the readings carry different floors.
    """
    floors = {round(reading.floor, 6) for reading in readings.values()}
    if len(floors) != 1:
        raise ValueError(
            f"Readings are judged against different floors: "
            f"{sorted(floors)}"
        )
    return floors.pop()


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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        help="File formats to write each figure in.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Draw the figures and write their tables.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    missing = [
        path for policy in POLICIES
        for path in (policy.curves, policy.versus_b0)
        if not path.exists()
    ]
    if not B0_CURVES.exists():
        missing.append(B0_CURVES)
    if missing:
        logger.error(
            "Scoring artefact(s) not found: %s.",
            ", ".join(str(path) for path in missing),
        )
        return EXIT_FAILED

    curves = {
        policy.key: load_curve(policy.curves, policy.run)
        for policy in POLICIES
    }
    readings = {
        policy.key: load_reading(policy.versus_b0) for policy in POLICIES
    }
    baseline = np.stack([load_curve(B0_CURVES, run) for run in B0_RUNS])
    baseline_peaks = load_baseline_peaks(POLICIES[0].versus_b0)
    floor = common_floor(readings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save(
        plot_curves(curves, readings, baseline, baseline_peaks),
        args.out_dir / "curves_validation", args.formats,
    )
    save(
        plot_attribution(readings, floor),
        args.out_dir / "attribution_validation", args.formats,
    )
    write_curve_table(
        args.out_dir / "curves_validation.csv", curves, baseline
    )
    write_attribution_table(
        args.out_dir / "attribution_validation.csv", readings, floor
    )

    if not TEST_SCORES.exists():
        logger.info(
            "TEST has not been scored yet (%s is absent); only the "
            "VALIDATION figures were drawn.", TEST_SCORES,
        )
        return EXIT_OK
    scores = json.loads(TEST_SCORES.read_text())
    expected = [s.key for s in (*B0_TEST, *POLICIES_TEST, FINAL_TEST)]
    absent = [key for key in expected if key not in scores]
    if absent:
        logger.error(
            "%s lacks snapshot(s) %s.", TEST_SCORES, ", ".join(absent)
        )
        return EXIT_FAILED
    save(plot_test(scores), args.out_dir / "test", args.formats)
    write_test_table(args.out_dir / "test.csv", scores)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
