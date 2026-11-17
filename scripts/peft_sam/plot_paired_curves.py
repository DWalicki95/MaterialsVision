#!/usr/bin/env python3
"""
Validation-loss curves of the Stage 1 variants against their reference.

Every run of Stage 1 shares the seed, and so the order of the data,
with the reference, and that order sets most of the epoch-to-epoch
wobble of the validation loss. Drawn on their own, the curves of six
runs mostly show that shared wobble, and the differences the study is
about - a thousandth or two - are hard to read off them. The figure
therefore has two panels over the same epochs: the curves themselves,
and each variant's difference from the reference at every epoch, in
which the shared wobble cancels. The window the paired comparison is
judged on is shaded in both.

Everything is read from MLflow; nothing is recomputed from a model. A
table with the same numbers is written next to the figure, so that a
value can be read exactly rather than off a plotted mark.

Examples
--------
    $ PYTHONPATH=. python scripts/peft_sam/plot_paired_curves.py
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

from materials_vision.logging_config import setup_logging
from scripts.peft_sam.compare_paired_runs import _metric_by_epoch, _runs_in

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

METRIC = "epoch/val_instance_loss"

REFERENCE_RUN = "full_b_seed20260907"
REFERENCE_EXPERIMENT = "peft_sam_training"
VARIANT_EXPERIMENT = "peft_sam_methods"

# Plan order, which is also the colour order: identity stays with the
# variant whatever subset is drawn.
VARIANTS = (
    ("full_b_both_e12_L12-23_lora0.0003_dec1e-05_train_seed20260907",
     "V1 late LoRA q,v (bloki 12-23)"),
    ("full_b_both_e12_qkvmlp_lora0.0003_dec1e-05_train_seed20260907",
     "V2 LoRA q,k,v,mlp"),
    ("full_b_both_e12_L12-23_qkvmlp_lora0.0003_dec1e-05_train_seed20260907",
     "V3 late LoRA q,k,v,mlp (bloki 12-23)"),
    ("full_b_both_e12_lateft_L12-23_dec1e-05_enc1e-05_train_seed20260907",
     "V4 late FT (bloki 12-23)"),
    ("full_b_both_e12_fullft_dec1e-05_enc1e-05_train_seed20260907",
     "V5 full FT"),
)

WINDOW = (7, 12)

DEFAULT_OUT = Path("checkpoints/figures/peft_methods/"
                   "stage1_val_instance_loss.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WINDOW_FILL = "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

LINE_WIDTH = 1.8
MARKER_SIZE = 5.0


def differences(
    reference: Mapping[int, float], candidate: Mapping[int, float]
) -> dict[int, float]:
    """Candidate minus reference at every epoch both have.

    Parameters
    ----------
    reference, candidate : mapping of int to float
        Figure per epoch.

    Returns
    -------
    dict of int to float
        Keyed by epoch, ascending.
    """
    shared = sorted(set(reference) & set(candidate))
    return {epoch: candidate[epoch] - reference[epoch] for epoch in shared}


def window_mean(
    values: Mapping[int, float], window: tuple[int, int]
) -> Optional[float]:
    """Mean over an epoch window, or ``None`` if an epoch is missing.

    A mean over fewer epochs than the window would not be the figure
    the paired comparison reports, so it is not returned at all.

    Parameters
    ----------
    values : mapping of int to float
    window : tuple of int
        First and last epoch, inclusive.

    Returns
    -------
    float or None
    """
    epochs = range(window[0], window[1] + 1)
    if any(epoch not in values for epoch in epochs):
        return None
    return sum(values[epoch] for epoch in epochs) / len(epochs)


def load_curves(
    client: Any,
    reference: tuple[str, str],
    variants: Sequence[str],
    variant_experiment: str,
    metric: str,
) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    """Read the reference curve and each variant's from MLflow.

    Parameters
    ----------
    client : mlflow.tracking.MlflowClient
    reference : tuple of str
        Run name and experiment of the reference.
    variants : sequence of str
        Run names in ``variant_experiment``.
    variant_experiment : str
    metric : str

    Returns
    -------
    tuple
        The reference curve, and the variants' keyed by run name.

    Raises
    ------
    ValueError
        If a run is absent or has logged nothing under ``metric``.
    """
    reference_name, reference_experiment = reference
    reference_runs = _runs_in(client, reference_experiment)
    if reference_name not in reference_runs:
        raise ValueError(
            f"No finished run {reference_name!r} in {reference_experiment!r}."
        )
    reference_curve = _metric_by_epoch(
        client, reference_runs[reference_name], metric
    )
    if not reference_curve:
        raise ValueError(f"{reference_name} has no {metric!r}.")

    runs = _runs_in(client, variant_experiment)
    curves: dict[str, dict[int, float]] = {}
    for name in variants:
        if name not in runs:
            raise ValueError(
                f"No finished run {name!r} in {variant_experiment!r}."
            )
        curve = _metric_by_epoch(client, runs[name], metric)
        if not curve:
            raise ValueError(f"{name} has no {metric!r}.")
        missing = sorted(set(reference_curve) - set(curve))
        if missing:
            logger.warning("%s lacks epoch(s) %s; drawn without them.",
                           name, missing)
        curves[name] = curve
    return reference_curve, curves


def _style_axes(axes: Any) -> None:
    """Recessive hairline grid and axes, muted tick labels."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY,
                     length=0, labelsize=9)


def _shade_window(axes: Any, window: tuple[int, int]) -> None:
    """Mark the epochs the paired comparison is judged on."""
    axes.axvspan(window[0] - 0.5, window[1] + 0.5, color=WINDOW_FILL,
                 zorder=0, linewidth=0)


def _plot_line(axes: Any, curve: Mapping[int, float], colour: str,
               label: str, width: float = LINE_WIDTH) -> None:
    """One series: a line with ringed markers at every epoch."""
    epochs = sorted(curve)
    axes.plot(epochs, [curve[e] for e in epochs], color=colour,
              linewidth=width, marker="o", markersize=MARKER_SIZE,
              markeredgecolor=SURFACE, markeredgewidth=1.2,
              solid_capstyle="round", solid_joinstyle="round",
              label=label, zorder=3)


def _variant_label(label: str, mean_delta: Optional[float],
                   window: tuple[int, int]) -> str:
    """Legend entry carrying the window mean, so it is read, not guessed."""
    if mean_delta is None:
        return f"{label}  (średnia Δ {window[0]}-{window[1]}: brak epok)"
    signed = f"{mean_delta:+.4f}".replace("-", "−")
    return f"{label}  (średnia Δ {window[0]}-{window[1]}: {signed})"


def draw(
    reference: Mapping[int, float],
    curves: Mapping[str, Mapping[int, float]],
    labels: Mapping[str, str],
    window: tuple[int, int],
    out: Path,
) -> None:
    """Draw both panels and write the figure.

    Parameters
    ----------
    reference : mapping of int to float
    curves : mapping of str to mapping of int to float
        Variant curves keyed by run name, in colour order.
    labels : mapping of str to str
        Display label per run name.
    window : tuple of int
    out : Path
    """
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7.8), sharex=True,
        gridspec_kw={"height_ratios": (3, 2), "hspace": 0.2},
    )
    figure.patch.set_facecolor(SURFACE)
    for axes in (top, bottom):
        _style_axes(axes)
        _shade_window(axes, window)

    _plot_line(top, reference, INK, "referencja full_b_seed20260907 "
               "(LoRA q,v)", width=LINE_WIDTH + 0.6)
    bottom.axhline(0.0, color=INK, linewidth=LINE_WIDTH + 0.6, zorder=2)
    for (name, curve), colour in zip(curves.items(), SERIES):
        delta = differences(reference, curve)
        label = _variant_label(labels[name], window_mean(delta, window),
                               window)
        _plot_line(top, curve, colour, label)
        _plot_line(bottom, delta, colour, label)

    epochs = sorted(reference)
    bottom.set_xticks(epochs)
    bottom.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    bottom.set_xlabel("epoka", color=INK_SECONDARY, fontsize=10)
    top.set_ylabel("val_instance_loss", color=INK_SECONDARY, fontsize=10)
    bottom.set_ylabel("wariant − referencja", color=INK_SECONDARY,
                      fontsize=10)
    top.text(window[0] - 0.4, 1.0, f"okno porównania (epoki {window[0]}-"
             f"{window[1]})", transform=top.get_xaxis_transform(),
             va="top", ha="left", color=INK_MUTED, fontsize=9)
    bottom.set_title("Różnica względem referencji w każdej epoce "
                     "(poniżej zera: niższa strata niż referencja)",
                     loc="left", color=INK_SECONDARY, fontsize=9.5)

    figure.suptitle(
        "val_instance_loss: referencja i warianty Etapu 1",
        x=0.06, ha="left", color=INK, fontsize=13, fontweight="semibold",
    )
    figure.text(
        0.06, 0.935, "VAL, 107 obrazów; wszystkie runy: full_b, 12 epok, "
        "seed 20260907 (sparowane z referencją)",
        ha="left", color=INK_SECONDARY, fontsize=9.5,
    )
    handles, legend_labels = top.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower left",
                  bbox_to_anchor=(0.06, 0.0), ncol=2, frameon=False,
                  fontsize=8.8, labelcolor=INK_SECONDARY,
                  handlelength=2.2, columnspacing=1.6)
    figure.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.165)

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def write_table(
    reference: Mapping[int, float],
    curves: Mapping[str, Mapping[int, float]],
    labels: Mapping[str, str],
    out: Path,
) -> None:
    """Write the plotted numbers as a CSV, one row per epoch.

    Parameters
    ----------
    reference : mapping of int to float
    curves : mapping of str to mapping of int to float
    labels : mapping of str to str
    out : Path
    """
    names = list(curves)
    header = ["epoch", "reference"]
    header += [labels[name] for name in names]
    header += [f"delta {labels[name]}" for name in names]
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for epoch in sorted(reference):
            row: list[Any] = [epoch, reference[epoch]]
            row += [curves[name].get(epoch, "") for name in names]
            row += [
                differences(reference, curves[name]).get(epoch, "")
                for name in names
            ]
            writer.writerow(row)


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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Figure path; the CSV is written beside it.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Read the curves, draw them and write the table.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()
    from mlflow.tracking import MlflowClient

    from materials_vision.tracking import configure_tracking

    configure_tracking()
    labels = dict(VARIANTS)
    try:
        reference, curves = load_curves(
            MlflowClient(), (REFERENCE_RUN, REFERENCE_EXPERIMENT),
            [name for name, _ in VARIANTS], VARIANT_EXPERIMENT, METRIC,
        )
    except ValueError as error:
        logger.error("%s", error)
        return EXIT_FAILED

    draw(reference, curves, labels, WINDOW, args.out)
    table = args.out.with_suffix(".csv")
    write_table(reference, curves, labels, table)
    logger.info("Wrote %s and %s.", args.out, table)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
