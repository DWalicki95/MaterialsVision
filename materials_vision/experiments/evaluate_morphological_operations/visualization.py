"""Instance-mask visualization overlaid on the original image.

Each instance receives a unique color. Masks are drawn with high
transparency so the underlying microscopy image stays visible.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_VISUALIZATIONS_SUBDIR = "visualizations"


def mask_to_rgb(mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Color an instance mask so each instance gets a distinct color.

    Hues are spread over the HSV circle and shuffled so neighboring
    instances are visually different.

    Parameters
    ----------
    mask : np.ndarray
        Labeled instance mask (background is 0).
    seed : int, optional
        Seed for the hue shuffle, by default 0.

    Returns
    -------
    np.ndarray
        Float RGB image of shape ``(H, W, 3)`` in range ``[0, 1]``.
    """
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    colored = np.zeros((*mask.shape, 3), dtype=np.float32)
    n_instances = len(unique_ids)
    if n_instances == 0:
        return colored

    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, n_instances, endpoint=False)
    rng.shuffle(hues)
    for i, instance_id in enumerate(unique_ids):
        rgb = plt.cm.colors.hsv_to_rgb([hues[i], 0.85, 0.95])
        colored[mask == instance_id] = rgb
    return colored


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Return a float grayscale image in range [0, 1]."""
    gray = image
    if image.ndim == 3:
        gray = image[..., :3].mean(axis=2)
    gray = gray.astype(np.float32)
    max_value = gray.max()
    if max_value > 0:
        gray = gray / max_value
    return gray


def overlay_mask_on_image(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a colored instance mask on a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Original image (grayscale or RGB).
    mask : np.ndarray
        Labeled instance mask.
    alpha : float, optional
        Opacity of the mask overlay, by default 0.4 (high transparency).

    Returns
    -------
    np.ndarray
        Float RGB overlay of shape ``(H, W, 3)`` in range ``[0, 1]``.
    """
    gray = _to_grayscale_float(image)
    background = np.stack([gray] * 3, axis=2)
    colored = mask_to_rgb(mask)
    has_instance = (mask != 0)[..., None]
    blended = np.where(
        has_instance,
        (1.0 - alpha) * background + alpha * colored,
        background,
    )
    return blended


def render_sample_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: Dict[str, np.ndarray],
    titles: List[str],
) -> plt.Figure:
    """
    Render ground-truth and prediction overlays side by side.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    gt_mask : np.ndarray
        Ground-truth instance mask.
    pred_masks : Dict[str, np.ndarray]
        Mapping ``variant -> mask`` for predictions.
    titles : List[str]
        Panel titles, first for ground-truth then one per prediction.

    Returns
    -------
    plt.Figure
        The composed figure.
    """
    masks = [gt_mask] + list(pred_masks.values())
    n_panels = len(masks)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    for axis, mask, title in zip(axes, masks, titles):
        axis.imshow(overlay_mask_on_image(image, mask))
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.tight_layout()
    return fig


def save_sample_figure(
    fig: plt.Figure, output_dir: Path, stem: str
) -> Path:
    """
    Save a sample figure as a PNG and close it.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_dir : Path
        Base output directory.
    stem : str
        Image stem used as the file name.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    target_dir = output_dir / _VISUALIZATIONS_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{stem}.png"
    fig.savefig(target_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return target_path
