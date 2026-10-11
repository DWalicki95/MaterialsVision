#!/usr/bin/env python3
"""
Draw the classical-baseline comparison as static figures for slides.

The figures show what ``compare_classical_baseline.py`` computes - the
models pooled over the five folds of ``cv5_v1``, the classical variants
over the ten images they were run on - read afresh on every run, so a
figure cannot drift from the reports. Each is written as PNG for
pasting and as SVG for scaling without loss.

**One layout, one colour per method.** Every figure lists the methods
in the same rows, top to bottom, and every method keeps its colour in
every figure, so a reader who learnt it on the first slide does not
relearn it on the next. The method's name stands beside every bar or
line, so identity never rests on colour alone; the populations are
stated once, in the note under each figure, to keep the slide quiet.

**The classical figures are an estimate, and look it.** Their labels
say so; in the size-class figure their lines are also dashed. In
the F1 figure the individual folds of the models and the individual
images of the classical methods are drawn over the pooled bar, which
shows how much a pooled figure over ten images can be trusted.

**Figures that are a mean of folds are marked** with ``*`` in the
panel title and explained in the footnote: for the models, medians and
distribution distances cannot be pooled from the fold reports.

Examples
--------
    $ PYTHONPATH=. python scripts/plot_classical_comparison.py
"""
import argparse
import logging
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import scripts.compare_classical_baseline as comparison
from materials_vision.evaluation import (SIZE_BIN_LABELS, SizeBins,
                                         SizeBinsLoadError, load_size_bins)
from materials_vision.logging_config import setup_logging
from scripts.peft_sam.summarize_cv import SummaryError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_OUT_DIR = Path("outputs/morphological_evaluation/figures")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

# Chart ink and chrome. The categorical slots below were checked for
# colour-vision deficiency as an ordered set (blue, orange, aqua,
# yellow); aqua and yellow fall below 3:1 against the surface, which is
# why every bar carries its method's name beside it.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METHOD_STYLES = {
    "peft_sam": ("PEFT-SAM", "#2a78d6"),
    "cellpose": ("Cellpose-SAM", "#eb6834"),
    "interactive": ("Watershed interaktywny", "#1baf7a"),
    "marker": ("Watershed", "#eda100"),
}

ESTIMATE = "szacunkowy"

BAR_HEIGHT = 0.52
GROUP_GAP = 0.45
DPI = 200
MINUS = "−"

FOLD_MEAN_NOTE = "* Modele: średnia z 5 foldów."
POPULATION_NOTE = (
    "Modele: walidacja krzyżowa cv5_v1, 699 obrazów, każdy oceniony "
    "przez model, który nie widział jego formulacji. Metody klasyczne: "
    "szacunek na 10 obrazach AS."
)


def method_key(row: dict[str, Any]) -> str:
    """Which method a comparison row belongs to.

    Parameters
    ----------
    row : dict

    Returns
    -------
    str
        A key of ``METHOD_STYLES``.

    Raises
    ------
    ValueError
        For a method without a style: a colour invented on the fly
        would break the one-colour-per-method rule across figures.
    """
    method = row["method"]
    if method.startswith("PEFT-SAM"):
        return "peft_sam"
    if method.startswith("Cellpose-SAM"):
        return "cellpose"
    if method == "classical interactive_watershed":
        return "interactive"
    if method == "classical marker_watershed":
        return "marker"
    raise ValueError(f"No figure style for method {method!r}")


def is_classical(row: dict[str, Any]) -> bool:
    """Whether a row is a classical estimate rather than a model.

    Parameters
    ----------
    row : dict

    Returns
    -------
    bool
    """
    return "classical_only" in row


def format_number(value: float, digits: int, signed: bool = False) -> str:
    """A number as written in Polish: decimal comma, true minus sign.

    Parameters
    ----------
    value : float
    digits : int
    signed : bool, optional
        Show ``+`` on positive values.

    Returns
    -------
    str
        ``b.d.`` (no data) where the value is undefined.
    """
    if value is None or not math.isfinite(value):
        return "b.d."
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", MINUS).replace(".", ",")


def row_label(row: dict[str, Any]) -> str:
    """The label beside a method's bar.

    The population each row rests on is stated once, in the note under
    the figure; beside the bar only the estimate is flagged.

    Parameters
    ----------
    row : dict

    Returns
    -------
    str
    """
    name = METHOD_STYLES[method_key(row)][0]
    if is_classical(row):
        return f"{name}\n({ESTIMATE})"
    return name


def row_positions(rows: Sequence[dict[str, Any]]) -> list[float]:
    """Vertical positions, top to bottom, with a gap before the
    classical rows so the estimate reads as a separate group."""
    positions, y = [], 0.0
    for index, row in enumerate(rows):
        if index and is_classical(row) and not is_classical(rows[index - 1]):
            y += GROUP_GAP
        positions.append(y)
        y += 1.0
    return positions


def apply_style() -> None:
    """Quiet chrome shared by every figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def _quiet_axes(ax, *, value_axis: str = "x") -> None:
    """Hairline grid on the value axis, no frame, no tick marks."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=value_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _decimal_comma(digits: int) -> FuncFormatter:
    """Axis ticks with a decimal comma."""
    return FuncFormatter(lambda value, _: format_number(value, digits))


def _titles(
    fig,
    title: str,
    subtitle: str,
    notes: Sequence[str],
    axis_band_in: float,
) -> None:
    """Title and subtitle above, wrapped notes below, in text ink.

    The notes are wrapped to the figure's width and the plot's bottom
    margin is raised to clear them and the axis labels beneath the
    plot, ``axis_band_in`` inches tall, so nothing is clipped at the
    edge or drawn over the axis.
    """
    width_in, height_in = fig.get_size_inches()
    lines = [
        line
        for note in notes
        for line in textwrap.wrap(note, width=int(width_in * 15.5))
    ]
    line_in = 0.16
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold",
             color=INK, va="top")
    fig.text(0.01, 1 - 0.42 / height_in,
             "\n".join(textwrap.wrap(subtitle, width=int(width_in * 13))),
             fontsize=10, color=INK_SECONDARY, va="top", linespacing=1.4)
    for index, line in enumerate(lines):
        y_in = 0.08 + line_in * (len(lines) - 1 - index)
        fig.text(0.01, y_in / height_in, line, fontsize=8, color=MUTED,
                 va="bottom")
    notes_in = 0.08 + line_in * len(lines) + 0.06
    fig.subplots_adjust(bottom=(notes_in + axis_band_in) / height_in)


def _method_axis(ax, rows, positions) -> None:
    """Method labels on the left, first method on top."""
    ax.set_yticks(positions)
    ax.set_yticklabels([row_label(row) for row in rows], fontsize=9.5)
    ax.set_ylim(positions[-1] + 0.7, positions[0] - 0.7)


def figure_f1(rows: Sequence[dict[str, Any]]):
    """Pooled F1 per method, with its folds or images over the bar.

    Parameters
    ----------
    rows : sequence of dict

    Returns
    -------
    matplotlib.figure.Figure
    """
    positions = row_positions(rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.80, bottom=0.2)
    for y, row in zip(positions, rows):
        color = METHOD_STYLES[method_key(row)][1]
        ax.barh(y, row["f1"], height=BAR_HEIGHT, color=color)
        spread = (list(row["per_image"].values()) if is_classical(row)
                  else row["f1_per_section"])
        ax.scatter(spread, [y] * len(spread), s=30, facecolor=SURFACE,
                   edgecolor=INK, linewidth=1, zorder=3)
        tip = max([row["f1"], *spread])
        ax.text(tip + 0.015, y, format_number(row["f1"], 3), va="center",
                fontsize=10, fontweight="bold", color=INK)
    _method_axis(ax, rows, positions)
    _quiet_axes(ax)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(_decimal_comma(1))
    _titles(
        fig, "Jakość segmentacji porów: F1",
        "Kółka: pojedyncze foldy (modele) lub obrazy (metody klasyczne)",
        [POPULATION_NOTE],
        axis_band_in=0.35,
    )
    return fig


def equivalent_diameter_um(area_um2: float) -> float:
    """Diameter of the circle with the given area.

    Parameters
    ----------
    area_um2 : float

    Returns
    -------
    float
        In micrometres.
    """
    return 2.0 * math.sqrt(area_um2 / math.pi)


def size_class_labels(size_bins: SizeBins) -> list[str]:
    """Class labels with their ranges as equivalent diameters, in µm.

    The classes are defined on area, but an area in tens of thousands
    of square micrometres is hard to picture; a diameter is not. The
    conversion is monotonic, so the diameter boundaries delimit exactly
    the same pores as the area boundaries.

    Parameters
    ----------
    size_bins : SizeBins

    Returns
    -------
    list of str
    """
    edges = [
        format_number(equivalent_diameter_um(edge), 0)
        for edge in size_bins.edges_um2
    ]
    return [
        f"Q1\n< {edges[0]} µm",
        f"Q2\n{edges[0]}–{edges[1]} µm",
        f"Q3\n{edges[1]}–{edges[2]} µm",
        f"Q4\n> {edges[2]} µm",
    ]


def figure_size_recall(rows: Sequence[dict[str, Any]], size_bins: SizeBins):
    """Recall per pore size class, one line per method.

    Parameters
    ----------
    rows : sequence of dict
    size_bins : SizeBins

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80)
    xs = list(range(len(SIZE_BIN_LABELS)))
    for row in rows:
        color = METHOD_STYLES[method_key(row)][1]
        values = [row["size_bin_recall"][label] for label in SIZE_BIN_LABELS]
        ax.plot(xs, values, color=color, linewidth=2,
                linestyle="--" if is_classical(row) else "-",
                marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=2,
                solid_capstyle="round", zorder=3)
        # Each line is named where the methods lie furthest apart, at
        # the smallest pores, which takes the place of a legend.
        ax.text(-0.12, values[0],
                f"{row_label(row).replace(chr(10), ' ')}  "
                f"{format_number(values[0], 2)}",
                ha="right", va="center", fontsize=9, color=INK)
    _quiet_axes(ax, value_axis="y")
    ax.set_xticks(xs)
    ax.set_xticklabels(size_class_labels(size_bins), fontsize=9,
                       color=INK)
    # The direct labels sit left of the first class, inside the plot,
    # so the axis is widened to the left to make room for them.
    ax.set_xlim(-2.05, len(xs) - 0.85)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(_decimal_comma(1))
    _titles(
        fig, "Recall według wielkości porów",
        "Klasy wielkości porów Q1–Q4 (średnica równoważna)",
        [POPULATION_NOTE],
        axis_band_in=0.55,
    )
    return fig


@dataclass(frozen=True)
class Panel:
    """One panel of a bar figure.

    Parameters
    ----------
    title : str
    value : callable
        Row to the figure drawn.
    digits : int
        Decimals in the value labels.
    signed : bool
        Whether the figure has a direction (prediction minus
        annotation); such panels are centred on zero.
    scale : float
        Factor applied before display, e.g. 100 for a percentage.
    unit : str
        Appended to the value labels.
    """

    title: str
    value: Callable[[dict[str, Any]], float]
    digits: int
    signed: bool
    scale: float = 1.0
    unit: str = ""


def _descriptor(name: str) -> Callable[[dict[str, Any]], float]:
    """Read one figure from a row's descriptor errors."""
    return lambda row: row["descriptor_errors"][name]


def figure_panels(
    rows: Sequence[dict[str, Any]],
    panels: Sequence[Panel],
    title: str,
    subtitle: str,
    notes: Sequence[str],
):
    """Bars per method, one panel per figure, side by side.

    Each panel has its own scale - the figures are in different units
    - and a signed panel is centred on zero, so over- and
    under-estimates sit on opposite sides.

    Parameters
    ----------
    rows : sequence of dict
    panels : sequence of Panel
    title, subtitle : str
    notes : sequence of str

    Returns
    -------
    matplotlib.figure.Figure
    """
    positions = row_positions(rows)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(2.9 * len(panels) + 2.6, 4.4),
        sharey=True,
    )
    axes = list(axes) if len(panels) > 1 else [axes]
    fig.subplots_adjust(left=2.3 / (2.9 * len(panels) + 2.6), right=0.98,
                        top=0.74, wspace=0.25)
    for ax, panel in zip(axes, panels):
        values = [panel.value(row) * panel.scale for row in rows]
        finite = [abs(v) for v in values if math.isfinite(v)] or [1.0]
        reach = max(finite) * 1.55 or 1.0
        for y, row, value in zip(positions, rows, values):
            if not math.isfinite(value):
                ax.text(0, y, "b.d.", va="center", ha="center",
                        fontsize=9, color=MUTED)
                continue
            ax.barh(y, value, height=BAR_HEIGHT,
                    color=METHOD_STYLES[method_key(row)][1])
            pad = reach * 0.03
            ax.text(value + (pad if value >= 0 else -pad), y,
                    format_number(value, panel.digits, panel.signed)
                    + panel.unit,
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=9.5, color=INK)
        if panel.signed:
            ax.set_xlim(-reach, reach)
        else:
            ax.set_xlim(0, reach)
        ax.axvline(0, color=BASELINE, linewidth=1, zorder=2)
        ax.set_title(panel.title, loc="left", fontsize=10.5, pad=8)
        _quiet_axes(ax)
        ax.xaxis.set_major_formatter(_decimal_comma(panel.digits - 1
                                                    if panel.digits else 0))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, symmetric=panel.signed))
    _method_axis(axes[0], rows, positions)
    _titles(fig, title, subtitle, notes, axis_band_in=0.35)
    return fig


def figure_count_porosity(rows):
    """Pore count error and porosity error."""
    return figure_panels(
        rows,
        [
            Panel("Błąd liczby porów", lambda row: row["pore_count_error"],
                  1, True, 100, "%"),
            Panel("Błąd porowatości [pp]",
                  _descriptor("mean_porosity_error_pp"), 1, True),
        ],
        "Liczba porów i porowatość",
        "Predykcja − adnotacja",
        [POPULATION_NOTE],
    )


def figure_diameter_distribution(rows):
    """Distance and shifts of the equivalent-diameter distribution."""
    return figure_panels(
        rows,
        [
            Panel("Odległość Wassersteina [µm] *",
                  _descriptor("wasserstein_um"), 1, False),
            Panel("Przesunięcie mediany [µm] *",
                  _descriptor("median_diameter_drift_um"), 1, True),
            Panel("Zmiana rozstępu IQR [µm] *",
                  _descriptor("iqr_drift_um"), 1, True),
        ],
        "Rozkład średnicy równoważnej porów",
        "Predykcja − adnotacja",
        [POPULATION_NOTE, FOLD_MEAN_NOTE],
    )


def figure_pair_shape(rows):
    """Per-pair diameter and shape errors."""
    return figure_panels(
        rows,
        [
            Panel("Błąd średnicy równoważnej *",
                  _descriptor("median_diameter_error"), 1, False, 100, "%"),
            Panel("Błąd wydłużenia *",
                  _descriptor("median_elongation_error"), 1, False, 100,
                  "%"),
            Panel("Błąd kąta orientacji [°] *",
                  _descriptor("median_angle_error_deg"), 1, False),
        ],
        "Kształt odnalezionych porów",
        "Mediana błędu w dopasowanych parach",
        [POPULATION_NOTE, FOLD_MEAN_NOTE],
    )


def figure_orientation(rows):
    """Orientation concentration differences, both statistics."""
    return figure_panels(
        rows,
        [
            Panel("Δ R: jedna oś *",
                  _descriptor("orientation_resultant_difference"), 3, True),
            Panel("Δ R4: osie prostopadłe *",
                  _descriptor("orientation_grid_resultant_difference"), 3,
                  True),
        ],
        "Orientacja porów",
        "Różnica koncentracji orientacji, predykcja − adnotacja",
        [POPULATION_NOTE, FOLD_MEAN_NOTE],
    )


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    """Write a figure as PNG and SVG, and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : Path
    stem : str

    Returns
    -------
    list of Path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=DPI)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def draw_all(
    rows: Sequence[dict[str, Any]], size_bins: SizeBins, out_dir: Path
) -> list[Path]:
    """Draw every figure of the comparison.

    Parameters
    ----------
    rows : sequence of dict
        Rows of ``compare_classical_baseline.compare``.
    size_bins : SizeBins
    out_dir : Path

    Returns
    -------
    list of Path
        Every file written.
    """
    for row in rows:
        method_key(row)
    plt.switch_backend("Agg")
    apply_style()
    figures = {
        "01_f1": figure_f1(rows),
        "02_recall_klasy_wielkosci": figure_size_recall(rows, size_bins),
        "03_liczba_porow_porowatosc": figure_count_porosity(rows),
        "04_rozklad_srednic": figure_diameter_distribution(rows),
        "05_ksztalt_par": figure_pair_shape(rows),
        "06_orientacja": figure_orientation(rows),
    }
    written = []
    for stem, fig in figures.items():
        written += save(fig, out_dir, stem)
    logger.info("Wrote %d figure(s) to %s.", len(figures), out_dir)
    return written


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
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument(
        "--classical-scores", type=Path,
        default=comparison.DEFAULT_CLASSICAL_SCORES,
    )
    parser.add_argument(
        "--recipe", type=Path, default=comparison.DEFAULT_RECIPE
    )
    parser.add_argument(
        "--cellpose-eval-dir", type=Path,
        default=comparison.DEFAULT_SAVE_ROOT / "eval",
    )
    parser.add_argument(
        "--peft-sam-eval-dir", type=Path,
        default=comparison.DEFAULT_EVAL_DIR,
    )
    parser.add_argument(
        "--folds-dir", type=Path, default=comparison.DEFAULT_FOLDS_DIR
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Draw the figures.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    try:
        rows = comparison.compare(args)["rows"]
        size_bins = load_size_bins(args.size_bins)
        draw_all(rows, size_bins, args.out_dir)
    except (SummaryError, SizeBinsLoadError, FileNotFoundError, KeyError,
            ValueError) as error:
        logger.error("%s", error)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
