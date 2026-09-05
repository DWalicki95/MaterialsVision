"""
One reviewable figure per image, family and strength setting.

What a panel has to answer is whether the result could still be
annotated by hand, and that question is only answerable if the panel
shows the things it could go wrong in: the image before and after, the
annotation before and after, the two together, and the image at the
resolution the model works in. Everything the transformation drew is
printed beside them, so a reviewer who sees something odd can name the
value that produced it rather than describing the picture.

**Why the model's resolution gets its own file.** The composed figure
is an overview and is necessarily reduced; a wall three pixels across
does not survive that reduction, so judging it there would be judging
the rendering rather than the augmentation. The encoder's input is
therefore also written on its own, at exactly 1024 by 1024, so the
viewer can show it pixel for pixel.

**Why some panels have a close-up row and others do not.** A crop or a
tonal change touches every pixel, and a close-up of "everywhere" says
nothing. A dark patch and a synthetic wall touch a small region, and
that region is where their two failure modes live - an edge hard enough
to read as a boundary, and a wall too faint to survive the downscaling.
The close-up appears exactly when the change is local, decided by
measuring what changed rather than by listing families.

**Why the seed does not include the strength setting.** The three
settings of a family are meant to differ in strength and in nothing
else: the same window, the same pore, the same wall drawn across it.
Seeding on the image and the repeat, but not the level, gives the
draws the best chance of lining up, which is what makes a weak and a
strong panel comparable at a glance.
"""
import hashlib
import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

from materials_vision.augmentation.config import (FAMILY_MASK_AWARE,
                                                  FAMILY_SEPTUM)
from materials_vision.augmentation.policy import AugmentationPolicy
from materials_vision.data.samples import PreparedSample
from materials_vision.phase0.levels import ReviewLevel
from materials_vision.phase0.preview import (MODE_ISOTROPIC, ModelInput,
                                             place_mask_on_canvas,
                                             to_model_coordinates,
                                             to_model_input)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Above this share of changed pixels a close-up of the change is a
# close-up of the whole frame, so it is left out.
LOCAL_CHANGE_MAX_SHARE = 0.25

# Context kept around a local change, in source pixels. Enough to show
# the pore the change sits in rather than the change alone.
ROI_MARGIN_PX = 40

# Smallest close-up worth rendering, in source pixels per side.
ROI_MIN_SIDE_PX = 60

FIGURE_DPI = 110

# Characters per line of the caption, at the figure width and font
# size below.
CAPTION_WIDTH = 200


@dataclass
class PanelRecord:
    """One rendered panel and everything a verdict needs to cite.

    Parameters
    ----------
    panel_id : str
        ``family__level__image__repeat``; also the file stem.
    family, level, kind : str
        Which family and setting, and whether the setting gates the
        family's admission or is diagnostic evidence.
    fingerprint : str
        Hash of the parameters the level was rendered with. A verdict
        is stored against this, so widening a range later returns the
        panel to the queue instead of inheriting an old decision.
    image_id, formulation, material, microscope, scale_bin : str
        Where the image sits in the dataset, so a pattern in the
        verdicts can be traced to a part of it.
    repeat : int
    seed : int
    note : str
        What this setting is meant to show.
    applied : bool
        Whether the family fired. False means it drew nothing to do -
        the fallback case, which is itself worth seeing.
    params : dict
        What the transformation reported drawing.
    measurements : dict
        Quantities measured on the result, listed in the panel.
    files : dict
        Paths of the figure, the model input and the close-up,
        relative to the review directory.
    """

    panel_id: str
    family: str
    level: str
    kind: str
    fingerprint: str
    image_id: str
    formulation: str
    material: str
    microscope: str
    scale_bin: str
    repeat: int
    seed: int
    note: str
    applied: bool
    params: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the record as plain data for the index.

        Returns
        -------
        dict
        """
        return asdict(self)


def panel_seed(run_seed: int, image_id: str, repeat: int) -> int:
    """Derive the draw for one image and repeat.

    Built the way the sampler's seeds are - a hash rather than
    arithmetic mixing, so neighbouring inputs do not give correlated
    draws - but in its own namespace, so rendering panels can never
    perturb the stream a training run uses.

    The level is deliberately not part of the key; see the module
    docstring.

    Parameters
    ----------
    run_seed : int
    image_id : str
    repeat : int

    Returns
    -------
    int
    """
    digest = hashlib.blake2b(
        f"phase0:{run_seed}:{image_id}:{repeat}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def render_panel(
    sample: PreparedSample,
    level: ReviewLevel,
    *,
    run_seed: int,
    repeat: int,
    output_dir: Path,
    preprocess_mode: str = MODE_ISOTROPIC,
) -> PanelRecord:
    """Render one panel and write its files.

    Parameters
    ----------
    sample : PreparedSample
        The image as the dataloader hands it over, i.e. cropped to the
        content region and reduced to the working channel.
    level : ReviewLevel
    run_seed : int
    repeat : int
    output_dir : Path
        Review directory; the three subdirectories are created here.
    preprocess_mode : str, optional
        Which preprocessing geometry the model-input panel is rendered
        at. The corrected one by default.

    Returns
    -------
    PanelRecord
    """
    record = sample.record
    seed = panel_seed(run_seed, record.image_id, repeat)
    policy = AugmentationPolicy(level.config)
    augmented = policy.apply(
        sample.image, sample.labels, record=record, seed=seed
    )
    transform = next(
        entry for entry in augmented.record.transforms
        if entry.family == level.family
    )

    model_input = to_model_input(
        augmented.image, mode=preprocess_mode
    )
    roi = _region_of_interest(
        sample.image, augmented.image, sample.labels,
        transform.params,
    )
    measurements = _measure(
        sample, augmented.image, augmented.labels, model_input, roi,
        level.family, transform.params,
    )

    panel_id = (
        f"{level.family}__{level.level}__{record.image_id}"
        f"__r{repeat}"
    )
    files = _write_files(
        panel_id, sample, augmented, model_input, roi, output_dir,
        level, measurements,
    )
    return PanelRecord(
        panel_id=panel_id,
        family=level.family,
        level=level.level,
        kind=level.kind,
        fingerprint=level.fingerprint,
        image_id=record.image_id,
        formulation=record.formulation,
        material=record.material,
        microscope=record.microscope,
        scale_bin=record.scale_bin,
        repeat=repeat,
        seed=seed,
        note=level.note,
        applied=transform.applied,
        params=_plain(transform.params),
        measurements=measurements,
        files=files,
    )


def write_index(
    records: list[PanelRecord], output_dir: Path
) -> Path:
    """Write the list of panels the viewer reads.

    Merged with whatever is already there rather than replacing it.
    The revision loop re-renders one family at a time - a range is
    widened, its panels are looked at again - and replacing the index
    would drop every other family's panels from the review while their
    files sat on disk beside it.

    A re-rendered panel keeps its identity, so its entry is replaced
    and, if its parameters changed, arrives with a new fingerprint
    that sets the verdicts on it aside.

    Parameters
    ----------
    records : list of PanelRecord
    output_dir : Path

    Returns
    -------
    Path
        The index file.
    """
    path = output_dir / "panels.json"
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for entry in json.load(handle)["panels"]:
                merged[entry["panel_id"]] = entry
    for record in records:
        merged[record.panel_id] = record.as_dict()

    payload = {
        "n_panels": len(merged),
        "panels": list(merged.values()),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    logger.info(
        "Index holds %d panel(s): %d rendered now, %d kept from "
        "earlier runs.",
        len(merged), len(records), len(merged) - len(records),
    )
    return path


def _measure(
    sample: PreparedSample,
    augmented_image: np.ndarray,
    augmented_labels: np.ndarray,
    model_input: ModelInput,
    roi: Optional[tuple[int, int, int, int]],
    family: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the result on the things the criteria ask about."""
    changed = _changed_pixels(sample.image, augmented_image)
    measurements: dict[str, Any] = {
        "changed_pixel_share": (
            1.0 if changed is None
            else round(float(changed.mean()), 4)
        ),
        "n_instances_before": int(sample.labels.max()),
        "n_instances_after": int(augmented_labels.max()),
        "content_shape_model": list(model_input.content_shape),
        "scale_y": round(model_input.scale_y, 4),
        "scale_x": round(model_input.scale_x, 4),
        "padding_share": round(model_input.padding_share, 4),
    }
    if family == FAMILY_MASK_AWARE and changed is not None and (
        changed.any()
    ):
        measurements["clearance_px"] = _clearance_px(
            sample.labels, changed
        )
    if family == FAMILY_SEPTUM and roi is not None:
        measurements["septum_peak_contrast_after_preprocessing"] = (
            _septum_visibility(
                sample.labels, augmented_labels, model_input,
                params.get("divided_instance"),
            )
        )
    return measurements


def _changed_pixels(
    image: np.ndarray, augmented_image: np.ndarray
) -> Optional[np.ndarray]:
    """Which pixels the transformation touched, if that is a question.

    A quarter turn swaps the sides of the frame, so the two images
    have no pixels in common to compare; ``None`` says so rather than
    letting a broadcast error stand in for the answer.

    Parameters
    ----------
    image, augmented_image : np.ndarray

    Returns
    -------
    np.ndarray or None
    """
    if image.shape != augmented_image.shape:
        return None
    return augmented_image != image


def _clearance_px(
    labels: np.ndarray, changed: np.ndarray
) -> float:
    """How close the change came to a real boundary, in source pixels.

    The shading families must fade to nothing before the annotation's
    edge: a change that reaches it draws a step where the mask says
    there is none, which is the error they exist to suppress. The
    distance is measured from the changed pixels to the nearest pixel
    outside any pore, so zero means the change touched the boundary.
    """
    distance = distance_transform_edt(labels > 0)
    return round(float(np.asarray(distance)[changed].min()), 2)


def _septum_visibility(
    labels: np.ndarray,
    augmented_labels: np.ndarray,
    model_input: ModelInput,
    divided_instance: Optional[int],
) -> Optional[float]:
    """Peak contrast of the drawn wall once the model has resized it.

    The acceptance criterion for the wall is that it is still visible
    after the full preprocessing, and visible means the grey levels
    still separate it from the pore it divides. The wall's core is the
    pixels the annotation lost, and the comparison is against the
    interior of that same pore - not against the pores in general,
    whose brightness has nothing to do with whether this wall can be
    seen.

    Returns ``None`` when the wall left no core to measure, i.e. the
    family did not divide anything on this sample.
    """
    core = (labels > 0) & (augmented_labels == 0)
    if divided_instance is not None:
        core &= labels == int(divided_instance)
    if not core.any():
        return None

    interior = (
        labels == int(divided_instance) if divided_instance is not None
        else labels > 0
    ) & ~core
    canvas_core = place_mask_on_canvas(core, model_input)
    canvas_interior = place_mask_on_canvas(
        interior, model_input, threshold=0.75
    ) & ~canvas_core
    if not canvas_core.any() or not canvas_interior.any():
        return None

    canvas = model_input.image.astype(np.float64)
    return round(float(
        canvas[canvas_core].max() - np.median(canvas[canvas_interior])
    ), 1)


def _region_of_interest(
    image: np.ndarray,
    augmented_image: np.ndarray,
    labels: np.ndarray,
    params: Mapping[str, Any],
) -> Optional[tuple[int, int, int, int]]:
    """Where to look closely, or nothing if the change was everywhere.

    A family that touched a quarter of the frame or more has no close-
    up worth showing. One that touched a patch gets the patch plus the
    pore holding it, because the question asked of it - does this read
    as a boundary - cannot be answered without the surroundings.

    A family that turned the frame has changed everything by
    definition, and its two images cannot even be compared pixel for
    pixel; it gets no close-up either.
    """
    changed = _changed_pixels(image, augmented_image)
    if changed is None:
        return None
    share = float(changed.mean())
    if not changed.any() or share > LOCAL_CHANGE_MAX_SHARE:
        return None

    rows, columns = np.nonzero(changed)
    box = [rows.min(), columns.min(), rows.max() + 1, columns.max() + 1]

    divided = params.get("divided_instance")
    if divided is not None:
        pore_rows, pore_columns = np.nonzero(labels == int(divided))
        if pore_rows.size:
            box = [
                min(box[0], pore_rows.min()),
                min(box[1], pore_columns.min()),
                max(box[2], pore_rows.max() + 1),
                max(box[3], pore_columns.max() + 1),
            ]

    return _padded_box(box, image.shape)


def _padded_box(
    box: list[int], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Grow a box by the context margin and clip it to the frame."""
    y0 = max(0, int(box[0]) - ROI_MARGIN_PX)
    x0 = max(0, int(box[1]) - ROI_MARGIN_PX)
    y1 = min(shape[0], int(box[2]) + ROI_MARGIN_PX)
    x1 = min(shape[1], int(box[3]) + ROI_MARGIN_PX)
    if y1 - y0 < ROI_MIN_SIDE_PX:
        y0, y1 = _widen(y0, y1, shape[0])
    if x1 - x0 < ROI_MIN_SIDE_PX:
        x0, x1 = _widen(x0, x1, shape[1])
    return y0, x0, y1, x1


def _widen(low: int, high: int, limit: int) -> tuple[int, int]:
    """Grow one side of a box to the minimum, staying in the frame."""
    missing = ROI_MIN_SIDE_PX - (high - low)
    low = max(0, low - missing // 2)
    return low, min(limit, low + ROI_MIN_SIDE_PX)


def _write_files(
    panel_id: str,
    sample: PreparedSample,
    augmented,
    model_input: ModelInput,
    roi: Optional[tuple[int, int, int, int]],
    output_dir: Path,
    level: ReviewLevel,
    measurements: dict[str, Any],
) -> dict[str, str]:
    """Write the figure, the model input and the close-up."""
    for name in ("panels", "previews", "zooms"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "panels" / f"{panel_id}.png"
    _compose_figure(
        sample, augmented, model_input, roi, level, measurements,
        figure_path,
    )

    preview_path = output_dir / "previews" / f"{panel_id}.png"
    plt.imsave(
        preview_path, model_input.image, cmap="gray", vmin=0, vmax=255
    )

    files = {
        "figure": f"panels/{panel_id}.png",
        "preview": f"previews/{panel_id}.png",
    }
    if roi is not None:
        zoom_path = output_dir / "zooms" / f"{panel_id}.png"
        _compose_zoom(
            sample, augmented, model_input, roi, zoom_path
        )
        files["zoom"] = f"zooms/{panel_id}.png"
    return files


def _compose_figure(
    sample: PreparedSample,
    augmented,
    model_input: ModelInput,
    roi: Optional[tuple[int, int, int, int]],
    level: ReviewLevel,
    measurements: dict[str, Any],
    path: Path,
) -> None:
    """Draw the six views and the values that produced them."""
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 10.4))
    record = sample.record

    _show_image(axes[0, 0], sample.image, "obraz oryginalny")
    _show_image(axes[0, 1], augmented.image, "obraz po augmentacji")
    _show_overlay(
        axes[0, 2], augmented.image, augmented.labels,
        "maska po augmentacji na obrazie",
    )
    _show_labels(axes[1, 0], sample.labels, "maska oryginalna")
    _show_labels(axes[1, 1], augmented.labels, "maska po augmentacji")
    _show_model_input(axes[1, 2], model_input)

    if roi is not None:
        _mark_roi(axes[0, 1], roi)

    figure.suptitle(
        f"{level.family} / {level.level} ({level.kind})   "
        f"{record.image_id}   {record.material} "
        f"{record.microscope} {record.scale_bin}",
        fontsize=13, y=0.985,
    )
    figure.text(
        0.008, 0.005, _caption(level, augmented.record, measurements),
        fontsize=8.5, family="monospace", va="bottom",
    )
    # Set by hand rather than by tight_layout: the panels hold images
    # of two different aspect ratios, so their axes are taller than
    # what they draw and an automatic layout puts the second row's
    # titles on top of the first row's images.
    figure.subplots_adjust(
        top=0.94, bottom=0.16, left=0.01, right=0.99,
        hspace=0.10, wspace=0.04,
    )
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def _compose_zoom(
    sample: PreparedSample,
    augmented,
    model_input: ModelInput,
    roi: tuple[int, int, int, int],
    path: Path,
) -> None:
    """Draw the changed region before, after, and as the model sees it."""
    y0, x0, y1, x1 = roi
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    _show_image(axes[0], sample.image[y0:y1, x0:x1], "przed (zrodlowa)")
    _show_overlay(
        axes[1], augmented.image[y0:y1, x0:x1],
        augmented.labels[y0:y1, x0:x1], "po (zrodlowa, z maska)",
    )
    my0, mx0, my1, mx1 = to_model_coordinates(roi, model_input)
    _show_image(
        axes[2], model_input.image[my0:my1, mx0:mx1],
        f"po preprocessingu ({model_input.mode})",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def _caption(
    level: ReviewLevel, augmentation, measurements: dict[str, Any]
) -> str:
    """Compose the text block under the figure.

    Wrapped rather than left to run off the page: the drawn parameters
    of the wall family alone are longer than the figure is wide, and a
    value a reviewer cannot read is a value that was not recorded.
    """
    lines = [
        f"na co patrzec: {level.note}",
        f"seed: {augmentation.seed}   fingerprint: "
        f"{level.fingerprint}",
        f"wylosowane: {_format_mapping(_plain(_drawn(augmentation)))}",
        f"zmierzone: {_format_mapping(measurements)}",
    ]
    wrapped = [
        textwrap.fill(
            line, width=CAPTION_WIDTH, subsequent_indent="    "
        )
        for line in lines
    ]
    return "\n".join(wrapped)


def _drawn(augmentation) -> dict[str, Any]:
    """Flatten what every family reported into one mapping."""
    drawn: dict[str, Any] = {}
    for entry in augmentation.transforms:
        if not entry.applied:
            drawn[entry.family] = "nie zadzialala (fallback)"
            continue
        drawn.update({"transformacja": entry.name, **entry.params})
        if entry.attempts > 1:
            drawn["proby"] = entry.attempts
        if entry.fallback:
            drawn["fallback"] = entry.fallback
    return drawn


def _format_mapping(values: dict[str, Any]) -> str:
    """Print a mapping in a line a person can scan."""
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        elif isinstance(value, (list, tuple)) and len(value) > 6:
            parts.append(f"{key}=[{len(value)} wartosci]")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


def _show_image(axis, image: np.ndarray, title: str) -> None:
    """Draw one grey image with its own frame of reference."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255,
                interpolation="nearest")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _show_labels(axis, labels: np.ndarray, title: str) -> None:
    """Draw an instance mask in colours that do not move between
    panels."""
    axis.imshow(_colorize(labels), interpolation="nearest")
    axis.set_title(f"{title} ({int(labels.max())} instancji)",
                   fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _show_overlay(
    axis, image: np.ndarray, labels: np.ndarray, title: str
) -> None:
    """Draw the annotation's outlines over the image behind them."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255,
                interpolation="nearest")
    outlines = find_boundaries(labels, mode="outer")
    overlay = np.zeros((*labels.shape, 4))
    overlay[outlines] = (1.0, 0.25, 0.0, 1.0)
    axis.imshow(overlay, interpolation="nearest")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _show_model_input(axis, model_input: ModelInput) -> None:
    """Draw the encoder's canvas, with the content's extent marked."""
    axis.imshow(model_input.image, cmap="gray", vmin=0, vmax=255,
                interpolation="nearest")
    height, width = model_input.content_shape
    axis.add_patch(plt.Rectangle(
        (-0.5, -0.5), width, height, fill=False, edgecolor="#00b0ff",
        linewidth=1.2,
    ))
    axis.set_title(
        f"po preprocessingu modelu ({model_input.mode}): "
        f"{height}x{width}, padding "
        f"{model_input.padding_share:.0%}",
        fontsize=10,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _mark_roi(axis, roi: tuple[int, int, int, int]) -> None:
    """Mark on the full frame where the close-up was taken."""
    y0, x0, y1, x1 = roi
    axis.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ffd400",
        linewidth=1.4,
    ))


def _colorize(labels: np.ndarray) -> np.ndarray:
    """Give every instance id a fixed colour.

    Fixed, because the eye compares the two mask panels side by side:
    colours assigned in order of appearance would shift as soon as a
    crop renumbered the instances, and every pore would then look
    changed.
    """
    rng = np.random.default_rng(0)
    colors = rng.random((int(labels.max()) + 1, 3)) * 0.75 + 0.25
    colors[0] = 0.0
    return colors[labels]


def _plain(values: Any) -> Any:
    """Convert numpy scalars so the record serializes as JSON."""
    if isinstance(values, dict):
        return {key: _plain(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [_plain(value) for value in values]
    if isinstance(values, np.generic):
        return values.item()
    return values
