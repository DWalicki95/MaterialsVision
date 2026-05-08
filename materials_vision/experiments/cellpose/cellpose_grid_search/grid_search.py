"""
Core logic for the Cellpose-SAM grid search pipeline.

Provides training/evaluation grid search orchestration,
MLflow logging, metric computation, and resume logic.
"""
import gc
import itertools
import logging
import re
import time
import traceback
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from cellpose import io, models, train
from cellpose import metrics as cp_metrics

from materials_vision.experiments.cellpose.utils import (
    filter_by_min_masks,
    get_train_test_file_paths,
    patch_cellpose_get_batch,
    precompute_flows_batched,
)
from materials_vision.experiments.plots import plot_loss
from materials_vision.metrics import (
    boundary_scores_batched,
    iou_scores_batch,
    summarize_evaluation_boundary_score,
    summarize_evaluation_iou,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
BOUNDARY_SCALES = [1.0, 2.0, 3.0]
DEFAULT_EVAL_BATCH_SIZE = 8


# =========================================================
# Helpers
# =========================================================
def load_config(path: str) -> Dict:
    """Load YAML configuration file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _ensure_list(value: Any) -> list:
    """Wrap scalar values in a list."""
    if isinstance(value, list):
        return value
    return [value]


_MLFLOW_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_\-. :/]")


def sanitize_metric_name(name: str) -> str:
    """
    Replace MLflow-illegal characters in a metric name with ``_``.

    MLflow allows only alphanumerics, ``_``, ``-``, ``.``, space,
    ``:`` and ``/``.

    Parameters
    ----------
    name : str
        Original metric name.

    Returns
    -------
    str
        Sanitized name safe for ``mlflow.log_metric``.
    """
    return _MLFLOW_METRIC_NAME_RE.sub("_", name)


def safe_log_metric(name: str, value: float) -> None:
    """
    Log a metric to MLflow without aborting the run on failure.

    Sanitizes the name and swallows any MLflow exception, logging
    a warning instead, so a single bad metric cannot kill an
    expensive training run.

    Parameters
    ----------
    name : str
        Metric name (will be sanitized).
    value : float
        Metric value.
    """
    safe_name = sanitize_metric_name(name)
    if safe_name != name:
        logger.warning(
            "Sanitized metric name '%s' -> '%s'", name, safe_name
        )
    try:
        mlflow.log_metric(safe_name, float(value))
    except Exception as exc:
        logger.warning(
            "Failed to log metric '%s'=%s: %s",
            safe_name, value, exc,
        )


def generate_training_combinations(
    grid: Dict,
) -> List[Dict[str, Any]]:
    """
    Generate all training hyperparameter combinations.

    Parameters
    ----------
    grid : Dict
        The ``training_grid`` section from config.

    Returns
    -------
    List[Dict[str, Any]]
        List of flat parameter dictionaries.
    """
    searched_keys = [
        "learning_rate",
        "weight_decay",
        "n_epochs",
        "batch_size",
        "normalize",
    ]
    constant_keys = [
        "min_train_masks",
        "rescale",
        "save_every",
        "save_each",
    ]

    lists = [_ensure_list(grid[k]) for k in searched_keys]
    constants = {
        k: grid.get(k) for k in constant_keys if k in grid
    }

    combos = []
    for vals in itertools.product(*lists):
        combo = dict(zip(searched_keys, vals))
        combo.update(constants)
        combos.append(combo)
    return combos


def generate_eval_combinations(
    grid: Dict,
) -> List[Dict[str, Any]]:
    """
    Generate all evaluation hyperparameter combinations.

    Parameters
    ----------
    grid : Dict
        The ``eval_grid`` section from config.

    Returns
    -------
    List[Dict[str, Any]]
        List of flat parameter dictionaries.
    """
    keys = list(grid.keys())
    lists = [_ensure_list(grid[k]) for k in keys]
    combos = []
    for vals in itertools.product(*lists):
        combos.append(dict(zip(keys, vals)))
    return combos


def build_model_name(params: Dict[str, Any]) -> str:
    """
    Build a unique, descriptive model name from params.

    Parameters
    ----------
    params : Dict[str, Any]
        Training hyperparameters.

    Returns
    -------
    str
        Model name encoding key hyperparameters.
    """
    norm = params.get("normalize", True)
    if isinstance(norm, dict):
        norm_str = next(iter(norm.keys()))
    else:
        norm_str = str(norm)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr = params["learning_rate"]
    wd = params["weight_decay"]
    ep = params["n_epochs"]
    bs = params["batch_size"]
    return (
        f"cpsam_lr{lr}_wd{wd}_ep{ep}_bs{bs}"
        f"_norm{norm_str}_{ts}"
    )


def _model_name_prefix(
    params: Dict[str, Any],
) -> str:
    """
    Deterministic model name prefix (no timestamp).

    Parameters
    ----------
    params : Dict[str, Any]
        Training hyperparameters.

    Returns
    -------
    str
        Prefix for matching checkpoint directories.
    """
    norm = params.get("normalize", True)
    if isinstance(norm, dict):
        norm_str = next(iter(norm.keys()))
    else:
        norm_str = str(norm)
    lr = params["learning_rate"]
    wd = params["weight_decay"]
    ep = params["n_epochs"]
    bs = params["batch_size"]
    return (
        f"cpsam_lr{lr}_wd{wd}_ep{ep}_bs{bs}"
        f"_norm{norm_str}_"
    )


def _find_resume_checkpoint(
    output_dir: Path,
    params: Dict[str, Any],
) -> Optional[Tuple[Path, int, Path, bool]]:
    """
    Find latest checkpoint or completed model for given params.

    Searches output_dir for directories matching the
    hyperparameter prefix. If a directory contains a final
    model file (named like the directory itself, without
    ``_epoch_NNNN`` suffix), training is treated as complete
    and the function returns ``is_final=True``. Otherwise it
    returns the checkpoint with the highest epoch number.

    Parameters
    ----------
    output_dir : Path
        Base grid search output directory.
    params : Dict[str, Any]
        Training hyperparameters.

    Returns
    -------
    Optional[Tuple[Path, int, Path, bool]]
        ``(model_path, epoch, run_dir, is_final)`` or None
        if no checkpoint or completed model is found.
        When ``is_final`` is True the epoch value is 0 and
        the caller should skip training entirely.
    """
    prefix = _model_name_prefix(params)
    if not output_dir.exists():
        return None

    matching_dirs = sorted(
        [
            d
            for d in output_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(prefix)
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    epoch_re = re.compile(r"_epoch_(\d+)$")
    for run_dir in matching_dirs:
        models_dir = run_dir / "models"
        if not models_dir.exists():
            continue

        # Final model file is saved by Cellpose at the end of
        # training and is named exactly like the run directory.
        final_model = models_dir / run_dir.name
        if final_model.is_file():
            return final_model, 0, run_dir, True

        checkpoints = []
        for f in models_dir.iterdir():
            if f.is_file():
                m = epoch_re.search(f.name)
                if m:
                    checkpoints.append(
                        (int(m.group(1)), f)
                    )

        if checkpoints:
            checkpoints.sort(reverse=True)
            epoch, ckpt = checkpoints[0]
            return ckpt, epoch, run_dir, False

    return None


def _is_cuda_fatal(exc: Exception) -> bool:
    """Check if exception is unrecoverable CUDA error."""
    msg = str(exc).lower()
    return (
        "cuda error:" in msg
        or "cudaerrorunknown" in msg
    )


def _normalize_label(norm_value: Any) -> str:
    """Human-readable label for normalize parameter."""
    if isinstance(norm_value, dict):
        return str(norm_value)
    return str(norm_value)


def _params_signature(params: Dict[str, Any]) -> str:
    """
    Deterministic string signature of parameters.

    Used for per-run tagging (unique per full param set).
    """
    parts = []
    for k in sorted(params.keys()):
        parts.append(f"{k}={params[k]}")
    return "|".join(parts)


# Keys used for matching "already trained" runs.
# n_epochs is deliberately excluded so that reducing
# n_epochs is treated as the same run.
_MATCH_KEYS = [
    "learning_rate",
    "weight_decay",
    "batch_size",
    "normalize",
    "min_train_masks",
    "rescale",
]


def _build_match_signature(
    values: Dict[str, Any],
) -> str:
    """
    Build signature for matching completed runs.

    Excludes ``n_epochs`` so that changing only the epoch
    budget does not produce a different signature.

    Parameters
    ----------
    values : Dict[str, Any]
        Parameter dictionary (Python params or MLflow
        string params — both work because values are
        stringified via f-string).

    Returns
    -------
    str
        Signature string, or "" if any match key missing.
    """
    parts = []
    for k in sorted(_MATCH_KEYS):
        if k not in values:
            return ""
        parts.append(f"{k}={values[k]}")
    return "|".join(parts)


# ---------------------------------------------------------
# LR schedule reconstruction
# ---------------------------------------------------------
def reconstruct_lr_schedule(
    learning_rate: float,
    n_epochs: int,
) -> np.ndarray:
    """
    Reconstruct the built-in Cellpose LR schedule.

    Parameters
    ----------
    learning_rate : float
        Base learning rate.
    n_epochs : int
        Total number of epochs.

    Returns
    -------
    np.ndarray
        LR value for each epoch.
    """
    lrs = np.zeros(n_epochs, dtype=np.float64)
    # Phase 1: linear warmup (epochs 0-9)
    warmup = min(10, n_epochs)
    for i in range(warmup):
        lrs[i] = learning_rate * (i / max(warmup, 1))

    # Phase 2: constant LR
    for i in range(warmup, n_epochs):
        lrs[i] = learning_rate

    # Phase 3: step decay in final epochs
    if n_epochs > 300:
        decay_start = n_epochs - 100
        decay_step = 10
    elif n_epochs > 99:
        decay_start = n_epochs - 50
        decay_step = 5
    else:
        decay_start = n_epochs  # no decay
        decay_step = 1

    current_lr = learning_rate
    for i in range(decay_start, n_epochs):
        if (i - decay_start) % decay_step == 0:
            current_lr /= 2.0
        lrs[i] = current_lr

    return lrs


def plot_lr_schedule(
    lrs: np.ndarray,
    save_path: Path,
) -> plt.Figure:
    """
    Plot the reconstructed LR schedule.

    Parameters
    ----------
    lrs : np.ndarray
        LR per epoch.
    save_path : Path
        Where to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lrs, linewidth=2, color="#2ca02c")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Reconstructed LR Schedule", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------
# Segmentation overlay visualisation
# ---------------------------------------------------------
def plot_segmentation_overlay(
    image: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    save_path: Path,
    alpha: float = 0.35,
) -> None:
    """
    Draw 3-panel overlay: original, prediction, GT.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SEM image.
    mask_pred : np.ndarray
        Predicted instance mask.
    mask_gt : np.ndarray
        Ground-truth instance mask.
    save_path : Path
        Output file path.
    alpha : float
        Overlay transparency.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Obraz SEM")

    axes[1].imshow(image, cmap="gray")
    masked_pred = np.ma.masked_where(
        mask_pred == 0, mask_pred
    )
    axes[1].imshow(
        masked_pred,
        cmap="tab20",
        alpha=alpha,
        interpolation="none",
    )
    n_pred = int(mask_pred.max())
    axes[1].set_title(f"Predykcja ({n_pred} instancji)")

    axes[2].imshow(image, cmap="gray")
    masked_gt = np.ma.masked_where(mask_gt == 0, mask_gt)
    axes[2].imshow(
        masked_gt,
        cmap="tab20",
        alpha=alpha,
        interpolation="none",
    )
    n_gt = int(mask_gt.max())
    axes[2].set_title(f"Ground Truth ({n_gt} instancji)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# MLflow resume helpers
# ---------------------------------------------------------
def _get_completed_signatures(
    experiment_name: str,
) -> set:
    """
    Query MLflow for already-completed run signatures.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    set
        Set of parameter signature strings.
    """
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return set()
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            output_format="list",
        )
        sigs = set()
        for run in runs:
            if run.data.tags.get("status") == "FAILED":
                continue
            sig = run.data.tags.get(
                "param_signature", ""
            )
            if sig:
                sigs.add(sig)
        return sigs
    except Exception:
        return set()


def _get_completed_runs(
    experiment_name: str,
) -> Dict[str, int]:
    """
    Query MLflow for runs that finished training and eval.

    A run counts as completed only when it has clear evidence
    of a finished evaluation: either the explicit
    ``eval_complete=True`` tag, or the ``cp_AJI`` metric
    (logged late in ``run_evaluation``, after the per-AP-
    threshold loop, so its presence means eval got past the
    cellpose built-in metrics block).

    Runs that merely logged params and crashed mid-eval (or
    were killed by the user) do **not** count, even if MLflow
    marked them FINISHED — nested ``with mlflow.start_run``
    blocks plus a swallowed ``except Exception`` produce a
    FINISHED status for failed runs.

    For every signature (hyperparams excluding ``n_epochs``)
    we keep the highest ``n_epochs`` value already completed.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    Dict[str, int]
        Mapping: match_signature -> max n_epochs completed.
    """
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return {}
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            output_format="list",
        )
        completed: Dict[str, int] = {}
        for run in runs:
            tags = run.data.tags or {}
            metrics = run.data.metrics or {}
            if tags.get("status") == "FAILED":
                continue
            eval_done = (
                tags.get("eval_complete") == "True"
                or "cp_AJI" in metrics
            )
            if not eval_done:
                continue
            p = run.data.params or {}
            match_sig = _build_match_signature(p)
            if not match_sig:
                continue
            try:
                n_ep = int(p.get("n_epochs", 0))
            except (ValueError, TypeError):
                continue
            prev = completed.get(match_sig)
            if prev is None or n_ep > prev:
                completed[match_sig] = n_ep
        return completed
    except Exception:
        return {}


# ---------------------------------------------------------
# Filter test losses (same logic as training.py)
# ---------------------------------------------------------
def filter_test_losses(
    train_losses: np.ndarray,
    test_losses: np.ndarray,
) -> Tuple[list, list, list]:
    """
    Filter out zero test losses.

    Parameters
    ----------
    train_losses : np.ndarray
        Training losses for all epochs.
    test_losses : np.ndarray
        Test losses (zeros where not evaluated).

    Returns
    -------
    Tuple[list, list, list]
        (epochs, test_values, train_at_test_epochs)
    """
    epochs = [
        i for i, v in enumerate(test_losses) if v > 0
    ]
    test_vals = [test_losses[i] for i in epochs]
    train_vals = [train_losses[i] for i in epochs]
    return epochs, test_vals, train_vals


# =========================================================
# Core: single training run
# =========================================================
def run_single_training(
    params: Dict[str, Any],
    train_img_files: List[str],
    train_flow_files: List[str],
    test_img_files: List[str],
    test_flow_files: List[str],
    test_label_files: List[str],
    output_dir: Path,
    eval_params: Dict[str, Any],
    vis_cfg: Dict[str, Any],
    gpu: bool = True,
) -> Dict[str, Any]:
    """
    Execute one training + evaluation run.

    Parameters
    ----------
    params : Dict[str, Any]
        Training hyperparameters.
    train_img_files : list of str
        Training image file paths.
    train_flow_files : list of str
        Training flow file paths.
    test_img_files : list of str
        Test image file paths.
    test_flow_files : list of str
        Test flow file paths.
    test_label_files : list of str
        Test mask file paths (for evaluation).
    output_dir : Path
        Base output directory.
    eval_params : Dict[str, Any]
        Evaluation parameters.
    vis_cfg : Dict[str, Any]
        Visualization config.
    gpu : bool
        Whether to use GPU.

    Returns
    -------
    Dict[str, Any]
        Dictionary with metrics and model_path.
    """
    lr = float(params["learning_rate"])
    wd = float(params["weight_decay"])
    n_epochs = int(params["n_epochs"])
    bs = int(params["batch_size"])
    normalize = params["normalize"]
    # min_train_masks filtering is done before
    # train_seg to avoid Cellpose file-mode bug
    rescale = params.get("rescale", False)
    save_every = int(params.get("save_every", 25))
    save_each = params.get("save_each", True)

    # ---- Resume, reuse completed, or fresh start ----
    start_epoch = 0
    training_complete = False
    model_path = None
    train_time = 0.0
    resume = _find_resume_checkpoint(
        output_dir, params
    )
    if resume:
        ckpt_path, start_epoch, run_dir, is_final = resume
        model_name = run_dir.name
        if is_final:
            training_complete = True
            model_path = str(ckpt_path)
            logger.info(
                "Found completed model, skipping training: %s",
                model_path,
            )
            mlflow.set_tag("skipped_training", "True")
        else:
            logger.info(
                "Resuming from epoch %d (%s)",
                start_epoch,
                ckpt_path,
            )
            model = models.CellposeModel(
                gpu=gpu,
                pretrained_model=str(ckpt_path),
            )
    else:
        model_name = build_model_name(params)
        run_dir = output_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Initializing fresh CPSAM model"
        )
        model = models.CellposeModel(gpu=gpu)

    if not training_complete:
        train_epochs = n_epochs - start_epoch
        if start_epoch > 0:
            orig_schedule = reconstruct_lr_schedule(
                lr, n_epochs
            )
            lr = float(orig_schedule[start_epoch])
            logger.info(
                "Adjusted LR for resume: %.6e "
                "(original schedule at epoch %d)",
                lr,
                start_epoch,
            )
            mlflow.set_tag(
                "resumed_from_epoch",
                str(start_epoch),
            )

        # ---- Train (single cellpose call, full schedule) ----
        t0 = time.time()
        patch_cellpose_get_batch()
        model_path, train_losses, test_losses = (
            train.train_seg(
                model.net,
                train_files=train_img_files,
                train_labels_files=train_flow_files,
                test_files=test_img_files,
                test_labels_files=test_flow_files,
                load_files=False,
                learning_rate=lr,
                weight_decay=wd,
                n_epochs=train_epochs,
                batch_size=bs,
                normalize=normalize,
                min_train_masks=0,
                rescale=rescale,
                save_path=str(run_dir),
                save_every=save_every,
                save_each=save_each,
                model_name=model_name,
            )
        )
        train_time = time.time() - t0
        logger.info(
            "Training done in %.1f s. Model: %s",
            train_time,
            model_path,
        )

        # ---- Log training losses as time series ----
        for epoch, tl in enumerate(train_losses):
            mlflow.log_metric(
                "training/train_loss",
                tl,
                step=epoch + start_epoch,
            )

        test_epochs, test_vals, train_at_test = (
            filter_test_losses(
                train_losses, test_losses
            )
        )
        for idx, ep in enumerate(test_epochs):
            mlflow.log_metric(
                "training/test_loss",
                test_vals[idx],
                step=ep + start_epoch,
            )

        # ---- Loss plots via plot_loss ----
        train_fig = plot_loss(
            train_epochs, train_losses, "Train"
        )
        train_plot_path = run_dir / "train_losses.png"
        train_fig.savefig(
            train_plot_path, dpi=150, bbox_inches="tight"
        )
        mlflow.log_figure(
            train_fig, "plots/train_losses.png"
        )
        plt.close(train_fig)

        if test_epochs:
            test_fig = plot_loss(
                test_epochs,
                test_vals,
                "Test",
                x_label="Epoch",
            )
            test_plot_path = run_dir / "test_losses.png"
            test_fig.savefig(
                test_plot_path,
                dpi=150,
                bbox_inches="tight",
            )
            mlflow.log_figure(
                test_fig, "plots/test_losses.png"
            )
            plt.close(test_fig)

        # ---- LR schedule plot ----
        lrs = reconstruct_lr_schedule(lr, train_epochs)
        lr_plot_path = run_dir / "lr_schedule.png"
        lr_fig = plot_lr_schedule(lrs, lr_plot_path)
        mlflow.log_figure(lr_fig, "plots/lr_schedule.png")
        plt.close(lr_fig)

        # ---- Summary loss metrics ----
        mlflow.log_metrics(
            {
                "summary/final_train_loss": float(
                    train_losses[-1]
                ),
                "summary/best_train_loss": float(
                    min(train_losses)
                ),
                "summary/train_time_s": train_time,
            }
        )
        if test_vals:
            mlflow.log_metrics(
                {
                    "summary/final_test_loss": float(
                        test_vals[-1]
                    ),
                    "summary/best_test_loss": float(
                        min(test_vals)
                    ),
                }
            )

    # ---- Evaluate ----
    results = run_evaluation(
        model_path=model_path,
        test_img_files=test_img_files,
        test_label_files=test_label_files,
        eval_params=eval_params,
        normalize=normalize,
        run_dir=run_dir,
        vis_cfg=vis_cfg,
        gpu=gpu,
    )
    results["model_path"] = model_path
    results["train_time_s"] = train_time

    # ---- Log model artifact ----
    mp = Path(model_path)
    if mp.exists():
        mlflow.log_artifact(str(mp), artifact_path="models")

    return results


# =========================================================
# Core: evaluation
# =========================================================
def run_evaluation(
    model_path: str,
    test_img_files: List[str],
    test_label_files: List[str],
    eval_params: Dict[str, Any],
    normalize: Any,
    run_dir: Path,
    vis_cfg: Dict[str, Any],
    gpu: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained model and log metrics.

    Parameters
    ----------
    model_path : str
        Path to the trained model.
    test_img_files : list of str
        Test image file paths.
    test_label_files : list of str
        Test mask file paths.
    eval_params : Dict[str, Any]
        Evaluation hyperparameters.
    normalize : Any
        Normalization setting (same as training).
    run_dir : Path
        Directory for saving artifacts.
    vis_cfg : Dict[str, Any]
        Visualization settings.
    gpu : bool
        Whether to use GPU.

    Returns
    -------
    Dict[str, float]
        All computed metrics.
    """
    logger.info(
        "Loading model for evaluation: %s", model_path
    )
    eval_model = models.CellposeModel(
        gpu=gpu, pretrained_model=model_path
    )

    # Load test data (test set is small)
    test_images = [
        io.imread(f) for f in test_img_files
    ]
    test_labels = [
        io.imread(f) for f in test_label_files
    ]

    ft = eval_params.get("flow_threshold", 0.4)
    cpt = eval_params.get(
        "cellprob_threshold", 0.0
    )
    msz = eval_params.get("min_size", 15)
    diam = eval_params.get("diameter", None)
    inv = eval_params.get("invert", False)
    eval_bs = eval_params.get(
        "batch_size", DEFAULT_EVAL_BATCH_SIZE
    )

    masks_pred, flows, styles = eval_model.eval(
        test_images,
        diameter=diam,
        flow_threshold=ft,
        cellprob_threshold=cpt,
        min_size=msz,
        normalize=normalize,
        batch_size=eval_bs,
        invert=inv,
    )

    # Free memory: flows and styles are not used downstream.
    # Boundary scores convolve large filters and can OOM if
    # these multi-GB tensors stay resident.
    del flows, styles
    gc.collect()

    metrics_out = {}

    # -- Cellpose built-in metrics --
    thresholds = [0.5, 0.75, 0.9]
    ap, tp, fp, fn = cp_metrics.average_precision(
        test_labels, masks_pred, threshold=thresholds
    )
    aji = cp_metrics.aggregated_jaccard_index(
        test_labels, masks_pred
    )

    mean_ap = np.mean(ap, axis=0)
    for i, th in enumerate(thresholds):
        key = f"cp_AP_{th}"
        metrics_out[key] = float(mean_ap[i])
        safe_log_metric(key, float(mean_ap[i]))

    metrics_out["cp_AJI"] = float(np.mean(aji))
    safe_log_metric("cp_AJI", float(np.mean(aji)))

    # -- Our IoU metrics --
    iou_results = iou_scores_batch(
        test_labels, masks_pred, greedy_match_threshold=0.5
    )
    iou_report = summarize_evaluation_iou(iou_results)
    for metric_name, row in iou_report.iterrows():
        val = float(row["value"])
        metrics_out[metric_name] = val
        safe_log_metric(metric_name, val)

    # -- Boundary scores --
    # Cellpose's boundary_scores convolves the outline image
    # with a circular filter sized by object diameter; large
    # diameters or many instances can trigger MemoryError.
    # Treat boundary metrics as best-effort so a single OOM
    # does not lose all the other metrics already logged.
    try:
        prec_all, rec_all, f_all = boundary_scores_batched(
            test_labels, masks_pred, BOUNDARY_SCALES
        )
        bs_report = summarize_evaluation_boundary_score(
            prec_all, rec_all, f_all
        )
        for metric_name, row in bs_report.iterrows():
            val = float(row["value"])
            metrics_out[metric_name] = val
            safe_log_metric(metric_name, val)
    except MemoryError as exc:
        logger.warning(
            "Skipping boundary_scores due to MemoryError: %s",
            exc,
        )
        mlflow.set_tag(
            "boundary_scores_skipped", "MemoryError"
        )
        gc.collect()

    # -- Object count stats --
    n_pred_list = [int(m.max()) for m in masks_pred]
    n_gt_list = [int(m.max()) for m in test_labels]
    avg_pred = float(np.mean(n_pred_list))
    avg_gt = float(np.mean(n_gt_list))
    safe_log_metric("avg_n_objects_pred", avg_pred)
    safe_log_metric("avg_n_objects_gt", avg_gt)
    metrics_out["avg_n_objects_pred"] = avg_pred
    metrics_out["avg_n_objects_gt"] = avg_gt

    # -- Segmentation overlay visualisations --
    _log_overlay_visualisations(
        test_images,
        masks_pred,
        test_labels,
        run_dir,
        vis_cfg,
    )

    return metrics_out


def _log_overlay_visualisations(
    test_images: list,
    masks_pred: list,
    test_labels: list,
    run_dir: Path,
    vis_cfg: Dict[str, Any],
) -> None:
    """Save and log overlay visualisations to MLflow."""
    n_samples = vis_cfg.get("n_samples", 8)
    seed = vis_cfg.get("seed", 42)
    alpha = vis_cfg.get("alpha", 0.35)

    n_test = len(test_images)
    n_samples = min(n_samples, n_test)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_test, n_samples, replace=False)

    vis_dir = run_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        img = test_images[idx]
        if img.ndim == 3:
            img = img.mean(axis=-1)
        save_p = vis_dir / f"overlay_{idx:04d}.png"
        plot_segmentation_overlay(
            img,
            masks_pred[idx],
            test_labels[idx],
            save_p,
            alpha=alpha,
        )
        mlflow.log_artifact(
            str(save_p),
            artifact_path="visualisations",
        )


# =========================================================
# Dry run
# =========================================================
def dry_run(
    combos: List[Dict[str, Any]],
    mode: str,
    config: Dict,
) -> None:
    """
    Print all combinations without running anything.

    Parameters
    ----------
    combos : List[Dict[str, Any]]
        Hyperparameter combinations.
    mode : str
        Grid search mode.
    config : Dict
        Full config dict.
    """
    min_per_epoch = config.get(
        "time_estimate_min_per_epoch", 0.5
    )
    total_epochs = 0
    logger.info("=" * 60)
    logger.info("DRY RUN  --  mode: %s", mode)
    logger.info("Total combinations: %d", len(combos))
    logger.info("=" * 60)

    for i, combo in enumerate(combos, 1):
        logger.info(
            "[%d/%d] %s", i, len(combos), combo
        )
        ep = combo.get("n_epochs", 0)
        total_epochs += ep

    est_min = total_epochs * min_per_epoch
    est_h = est_min / 60.0
    logger.info("-" * 60)
    logger.info(
        "Total epochs across all runs: %d", total_epochs
    )
    logger.info(
        "Estimated time: %.0f min (%.1f h)  "
        "[heuristic: %.2f min/epoch]",
        est_min,
        est_h,
        min_per_epoch,
    )
    logger.info("=" * 60)


# =========================================================
# Grid search orchestrators
# =========================================================
def run_training_grid(
    config: Dict,
    gpu: bool,
    force_rerun: bool,
) -> pd.DataFrame:
    """
    Run grid search over training hyperparameters.

    Parameters
    ----------
    config : Dict
        Full config dict.
    gpu : bool
        Whether to use GPU.
    force_rerun : bool
        Re-run even completed runs.

    Returns
    -------
    pd.DataFrame
        Results table for all runs.
    """
    exp_name = config["experiment"]["name"]
    mlflow.set_experiment(exp_name)
    data_cfg = config["data"]
    output_base = Path(
        config.get("output", {}).get(
            "base_dir", "outputs/grid_search"
        )
    )
    output_base.mkdir(parents=True, exist_ok=True)
    vis_cfg = config.get("visualization", {})
    eval_defaults = config.get("eval_defaults", {})

    # Gather file paths (no bulk loading)
    logger.info("Gathering file paths...")
    (
        train_img_files,
        train_label_files,
        test_img_files,
        test_label_files,
    ) = get_train_test_file_paths(
        data_cfg["train_dir"],
        data_cfg["test_dir"],
        image_filter=data_cfg.get("img_filter"),
        mask_filter=data_cfg.get("mask_filter"),
        look_one_level_down=False,
    )
    n_train = len(train_img_files)
    n_test = (
        len(test_img_files)
        if test_img_files else 0
    )
    logger.info(
        "Found %d train, %d test images",
        n_train,
        n_test,
    )

    # Pre-compute flows once (cached on disk)
    logger.info("Pre-computing flows in batches...")
    train_flow_files = precompute_flows_batched(
        train_label_files,
        train_img_files,
        batch_size=10,
    )
    test_flow_files = None
    if test_img_files and test_label_files:
        test_flow_files = precompute_flows_batched(
            test_label_files,
            test_img_files,
            batch_size=10,
        )

    # Filter training images by min_train_masks
    # (done here to work around Cellpose bug in
    #  file-based mode)
    default_min_masks = config["training_grid"].get(
        "min_train_masks", 5
    )
    if default_min_masks > 0:
        (
            train_img_files,
            train_label_files,
            train_flow_files,
        ) = filter_by_min_masks(
            train_img_files,
            train_label_files,
            train_flow_files,
            min_train_masks=default_min_masks,
        )
        n_train = len(train_img_files)

    combos = generate_training_combinations(
        config["training_grid"]
    )
    logger.info(
        "Training grid: %d combinations", len(combos)
    )

    # Resume logic: match on hyperparams excluding n_epochs.
    # A previous run is "good enough" to skip current combo
    # when its n_epochs >= current combo's n_epochs.
    completed_runs: Dict[str, int] = {}
    if not force_rerun:
        completed_runs = _get_completed_runs(exp_name)
        logger.info(
            "Found %d completed parameter sets in MLflow",
            len(completed_runs),
        )

    all_results = []
    n_success = 0
    n_failed = 0
    n_skipped = 0

    # Parent run
    with mlflow.start_run(run_name="grid_search_parent"):
        mlflow.log_params(
            {
                "grid_mode": "training",
                "n_combinations": len(combos),
                "n_train_images": n_train,
                "n_test_images": n_test,
                "train_dir": data_cfg["train_dir"],
                "test_dir": data_cfg["test_dir"],
                "cellpose_version": pkg_version("cellpose"),
            }
        )

        # Save train/test file list
        filelist_path = output_base / "file_lists.txt"
        with open(filelist_path, "w") as fh:
            fh.write("=== TRAIN ===\n")
            for n in train_img_files:
                fh.write(f"{n}\n")
            fh.write("\n=== TEST ===\n")
            if test_img_files:
                for n in test_img_files:
                    fh.write(f"{n}\n")
        mlflow.log_artifact(
            str(filelist_path), artifact_path="info"
        )

        for i, params in enumerate(combos, 1):
            sig = _params_signature(params)
            match_sig = _build_match_signature(params)
            run_label = (
                f"[{i}/{len(combos)}] "
                f"lr={params['learning_rate']} "
                f"wd={params['weight_decay']} "
                f"ep={params['n_epochs']} "
                f"bs={params['batch_size']} "
                f"norm={_normalize_label(params['normalize'])}"
            )

            prev_n_epochs = completed_runs.get(match_sig)
            if (
                not force_rerun
                and prev_n_epochs is not None
                and int(params["n_epochs"]) <= prev_n_epochs
            ):
                logger.info(
                    "Skipping: same hyperparams already "
                    "trained for %d epochs (request %d). %s",
                    prev_n_epochs,
                    int(params["n_epochs"]),
                    run_label,
                )
                n_skipped += 1
                continue

            logger.info("Starting run %s", run_label)

            with mlflow.start_run(
                run_name=f"train_{i:03d}",
                nested=True,
            ):
                try:
                    # Log all hyperparams
                    mlflow.log_params(
                        {
                            "learning_rate": params[
                                "learning_rate"
                            ],
                            "weight_decay": params[
                                "weight_decay"
                            ],
                            "n_epochs": params["n_epochs"],
                            "batch_size": params[
                                "batch_size"
                            ],
                            "normalize": str(
                                params["normalize"]
                            ),
                            "min_train_masks": params.get(
                                "min_train_masks", 5
                            ),
                            "rescale": params.get(
                                "rescale", False
                            ),
                            "save_every": params.get(
                                "save_every", 25
                            ),
                            "n_train_images": n_train,
                            "n_test_images": n_test,
                            "train_dir": data_cfg[
                                "train_dir"
                            ],
                            "test_dir": data_cfg[
                                "test_dir"
                            ],
                            "cellpose_version": (
                                pkg_version("cellpose")
                            ),
                        }
                    )
                    # Log eval defaults
                    for ek, ev in eval_defaults.items():
                        mlflow.log_param(
                            f"eval_{ek}", ev
                        )
                    mlflow.set_tag(
                        "param_signature", sig
                    )
                    mlflow.set_tag(
                        "match_signature", match_sig
                    )

                    results = run_single_training(
                        params=params,
                        train_img_files=train_img_files,
                        train_flow_files=train_flow_files,
                        test_img_files=test_img_files,
                        test_flow_files=test_flow_files,
                        test_label_files=test_label_files,
                        output_dir=output_base,
                        eval_params=eval_defaults,
                        vis_cfg=vis_cfg,
                        gpu=gpu,
                    )
                    results["params"] = params
                    results["run_index"] = i
                    all_results.append(results)
                    n_success += 1
                    mlflow.set_tag("eval_complete", "True")
                    logger.info(
                        "Run %d finished. "
                        "AP@0.5=%.4f",
                        i,
                        results.get("cp_AP_0.5", 0),
                    )

                except Exception as exc:
                    n_failed += 1
                    tb = traceback.format_exc()
                    logger.error(
                        "Run %d FAILED:\n%s", i, tb
                    )
                    mlflow.set_tag("error", tb[:1000])
                    mlflow.set_tag("status", "FAILED")
                    if _is_cuda_fatal(exc):
                        logger.critical(
                            "Unrecoverable CUDA error."
                            " Stopping grid search."
                            " Restart process after"
                            " fixing GPU state"
                            " (nvidia-smi --gpu-reset"
                            " or reboot)."
                        )
                        break

        # ---- Final report ----
        logger.info("=" * 60)
        logger.info(
            "Grid search complete: "
            "%d success, %d failed, %d skipped",
            n_success,
            n_failed,
            n_skipped,
        )

        df = _build_report(all_results, output_base)
        if df is not None and not df.empty:
            csv_path = output_base / "grid_search_results.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(
                str(csv_path), artifact_path="reports"
            )
            _print_top3(df)

    return df if df is not None else pd.DataFrame()


def run_eval_grid(
    config: Dict,
    model_path: str,
    gpu: bool,
    force_rerun: bool,
) -> pd.DataFrame:
    """
    Run grid search over evaluation parameters.

    Parameters
    ----------
    config : Dict
        Full config dict.
    model_path : str
        Path to the pre-trained model.
    gpu : bool
        Whether to use GPU.
    force_rerun : bool
        Re-run even completed runs.

    Returns
    -------
    pd.DataFrame
        Results table for all runs.
    """
    exp_name = config["experiment"]["name"] + "_eval"
    mlflow.set_experiment(exp_name)
    data_cfg = config["data"]
    output_base = Path(
        config.get("output", {}).get(
            "base_dir", "outputs/grid_search"
        )
    )
    output_base.mkdir(parents=True, exist_ok=True)
    vis_cfg = config.get("visualization", {})
    # Use True as default normalize for eval-only mode
    normalize = config.get(
        "eval_defaults", {}
    ).get("normalize", True)

    logger.info("Gathering file paths...")
    (
        _train_img,
        _train_lbl,
        test_img_files,
        test_label_files,
    ) = get_train_test_file_paths(
        data_cfg["train_dir"],
        data_cfg["test_dir"],
        image_filter=data_cfg.get("img_filter"),
        mask_filter=data_cfg.get("mask_filter"),
        look_one_level_down=False,
    )
    n_test = (
        len(test_img_files)
        if test_img_files else 0
    )
    logger.info(
        "Found %d test images for eval", n_test
    )

    combos = generate_eval_combinations(
        config["eval_grid"]
    )
    logger.info(
        "Eval grid: %d combinations", len(combos)
    )

    completed_sigs = set()
    if not force_rerun:
        completed_sigs = _get_completed_signatures(
            exp_name
        )

    all_results = []
    n_success = 0
    n_failed = 0
    n_skipped = 0

    with mlflow.start_run(
        run_name="eval_grid_parent"
    ):
        mlflow.log_params(
            {
                "grid_mode": "evaluation",
                "n_combinations": len(combos),
                "model_path": model_path,
                "n_test_images": n_test,
                "cellpose_version": pkg_version("cellpose"),
            }
        )

        for i, ep in enumerate(combos, 1):
            sig = _params_signature(ep)
            run_label = (
                f"[{i}/{len(combos)}] "
                f"ft={ep.get('flow_threshold')} "
                f"cpt={ep.get('cellprob_threshold')} "
                f"ms={ep.get('min_size')} "
                f"inv={ep.get('invert')}"
            )

            if sig in completed_sigs and not force_rerun:
                logger.info(
                    "Skipping completed eval: %s",
                    run_label,
                )
                n_skipped += 1
                continue

            logger.info("Starting eval %s", run_label)

            with mlflow.start_run(
                run_name=f"eval_{i:03d}",
                nested=True,
            ):
                try:
                    for ek, ev in ep.items():
                        mlflow.log_param(ek, ev)
                    mlflow.log_param(
                        "model_path", model_path
                    )
                    mlflow.set_tag(
                        "param_signature", sig
                    )

                    run_dir = (
                        output_base / f"eval_{i:03d}"
                    )
                    run_dir.mkdir(
                        parents=True, exist_ok=True
                    )

                    results = run_evaluation(
                        model_path=model_path,
                        test_img_files=test_img_files,
                        test_label_files=test_label_files,
                        eval_params=ep,
                        normalize=normalize,
                        run_dir=run_dir,
                        vis_cfg=vis_cfg,
                        gpu=gpu,
                    )
                    results["params"] = ep
                    results["run_index"] = i
                    all_results.append(results)
                    n_success += 1

                except Exception:
                    n_failed += 1
                    tb = traceback.format_exc()
                    logger.error(
                        "Eval %d FAILED:\n%s", i, tb
                    )
                    mlflow.set_tag("error", tb[:1000])

        logger.info(
            "Eval grid complete: "
            "%d success, %d failed, %d skipped",
            n_success,
            n_failed,
            n_skipped,
        )

        df = _build_report(all_results, output_base)
        if df is not None and not df.empty:
            csv_path = (
                output_base / "eval_grid_results.csv"
            )
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(
                str(csv_path), artifact_path="reports"
            )
            _print_top3(df)

    return df if df is not None else pd.DataFrame()


def run_full_grid(
    config: Dict,
    gpu: bool,
    force_rerun: bool,
) -> None:
    """
    Run training grid, then eval grid for top N.

    Parameters
    ----------
    config : Dict
        Full config dict.
    gpu : bool
        Whether to use GPU.
    force_rerun : bool
        Re-run even completed runs.
    """
    train_df = run_training_grid(
        config, gpu, force_rerun
    )
    if train_df is None or train_df.empty:
        logger.warning(
            "No training results; skipping eval grid."
        )
        return

    top_n = config.get("top_n_models", 3)
    sort_col = (
        "cp_AP@0.5"
        if "cp_AP@0.5" in train_df.columns
        else train_df.columns[0]
    )
    top_df = train_df.nlargest(top_n, sort_col)
    logger.info(
        "Running eval grid for top %d models", top_n
    )

    for _, row in top_df.iterrows():
        mp = row.get("model_path", "")
        if not mp or not Path(mp).exists():
            logger.warning(
                "Model not found: %s, skipping", mp
            )
            continue
        logger.info("Eval grid for model: %s", mp)
        run_eval_grid(config, mp, gpu, force_rerun)


# =========================================================
# Report helpers
# =========================================================
def _build_report(
    results: List[Dict],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """Build a DataFrame report from run results."""
    if not results:
        return None

    rows = []
    for r in results:
        row = {}
        params = r.pop("params", {})
        for k, v in params.items():
            row[f"param_{k}"] = v
        row.update(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = (
        "cp_AP@0.5"
        if "cp_AP@0.5" in df.columns
        else df.columns[0]
    )
    df = df.sort_values(sort_col, ascending=False)
    return df


def _print_top3(df: pd.DataFrame) -> None:
    """Log top-3 configurations."""
    sort_col = (
        "cp_AP@0.5"
        if "cp_AP@0.5" in df.columns
        else df.columns[0]
    )
    top3 = df.nlargest(3, sort_col)
    logger.info("=" * 60)
    logger.info(
        "TOP-3 CONFIGURATIONS (by %s):", sort_col
    )
    logger.info("=" * 60)
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        param_cols = [
            c for c in row.index if c.startswith("param_")
        ]
        logger.info(
            "  #%d  %s=%.4f", rank, sort_col, row[sort_col]
        )
        for pc in param_cols:
            logger.info("       %s = %s", pc, row[pc])
    logger.info("=" * 60)
