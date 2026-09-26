"""
The trainer's logger, extended to report into MLflow as well.

The library's logger writes to TensorBoard, and that is kept: it is
what every run so far was recorded in, and dropping it would make new
runs unreadable next to old ones in the tool that already holds them.
Everything it logs is additionally written to the active MLflow run,
together with three things it does not record at all.

**Both learning rates.** The library reports the rate of the first
parameter group only, which here is the low-rank correction. The
decoder runs at a rate thirty times lower, and part II of this study
varies one of the two while holding the other, so a chart showing one
of them would hide exactly the difference being tested.

**Per-epoch figures on the epoch axis.** Instance metrics are computed
afterwards, one point per epoch snapshot. Losses averaged over each
epoch are logged on the same axis, so a loss and an F1 curve can be
laid over one another directly instead of being matched by step count.

**What the decoder predicts, raw.** Once per epoch, on the last
validation image - the same image every epoch, because validation is
never shuffled - the three maps the decoder outputs are drawn before
any post-processing, with the distribution of foreground confidences
beside them. No instances are formed and no metric is computed, so the
picture cannot be read as a score under a setting that is still being
calibrated. What it does show is the block pattern the upsampling
leaves on the foreground map and how far the confidences are from two
clean modes, the two things the post-processing calibration reasons
about.

**Tracking never stops training.** A run costs over two hours of GPU
and the record of it is secondary to the run itself, so a failed write
to MLflow is logged as a warning and training continues. The failures
are counted and the count reported at the end of every epoch, so a
broken record cannot pass unnoticed either.
"""
import logging
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from micro_sam.training.joint_sam_trainer import JointSamLogger

logger = logging.getLogger(__name__)

# Names of the optimizer's parameter groups, in the order the training
# module builds them.
PARAMETER_GROUP_NAMES = ("lora", "decoder")

# Order of the decoder's output channels.
DECODER_MAP_NAMES = (
    "foreground", "centre distance", "boundary distance",
)

FOREGROUND_HISTOGRAM_BINS = 50

BYTES_PER_GIB = 1024 ** 3


def _to_float(value: Any) -> float:
    """Read a scalar that may still be a tensor."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Current rate of every parameter group, by name.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    Returns
    -------
    dict of str to float
        Groups beyond the named ones are numbered, so an optimizer built
        differently from the training module's still reports every rate.
    """
    rates = {}
    for index, group in enumerate(optimizer.param_groups):
        name = (
            PARAMETER_GROUP_NAMES[index]
            if index < len(PARAMETER_GROUP_NAMES) else f"group{index}"
        )
        rates[name] = float(group["lr"])
    return rates


class EpochAccumulator:
    """Running means of the per-step training figures within one epoch."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0

    def add(self, figures: dict[str, float]) -> None:
        """Take in one step's figures.

        Parameters
        ----------
        figures : dict of str to float
        """
        for name, value in figures.items():
            self._sums[name] += value
        self._count += 1

    def means(self) -> dict[str, float]:
        """Mean of every figure over the steps taken in so far.

        Returns
        -------
        dict of str to float
            Empty when no step was taken.
        """
        if self._count == 0:
            return {}
        return {
            name: total / self._count for name, total in self._sums.items()
        }

    def reset(self) -> None:
        """Start a new epoch."""
        self._sums.clear()
        self._count = 0


def decoder_maps_figure(
    image: np.ndarray,
    instances: np.ndarray,
    maps: np.ndarray,
    title: str,
) -> np.ndarray:
    """Draw an input, its annotation and the decoder's raw output.

    Parameters
    ----------
    image : np.ndarray
        ``(C, H, W)`` or ``(H, W)``, any range.
    instances : np.ndarray
        ``(H, W)`` instance labels, background zero.
    maps : np.ndarray
        ``(3, H, W)``: foreground probability, then the two distances.
    title : str

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    gray = image[0] if image.ndim == 3 else image
    figure, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(gray, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(
            ~find_boundaries(instances, mode="inner"), instances > 0
        ),
        cmap="autumn", interpolation="nearest",
    )
    axes[1].set_title(f"annotation ({len(np.unique(instances)) - 1} pores)")
    for axis, channel, name in zip(axes[2:5], maps, DECODER_MAP_NAMES):
        shown = axis.imshow(channel, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(name)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    for axis in axes[:5]:
        axis.set_axis_off()
    axes[5].hist(
        maps[0].ravel(), bins=FOREGROUND_HISTOGRAM_BINS, range=(0.0, 1.0),
        log=True, color="tab:blue",
    )
    axes[5].set_title("foreground confidence (log count)")
    axes[5].set_xlabel("probability")
    figure.suptitle(title)
    figure.tight_layout()
    figure.canvas.draw()
    rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return rgb


class MlflowJointSamLogger(JointSamLogger):
    """TensorBoard as before, and the active MLflow run besides.

    Parameters
    ----------
    trainer : JointSamTrainer
    save_root : str
    """

    def __init__(self, trainer: Any, save_root: Any, **unused: Any):
        super().__init__(trainer, save_root, **unused)
        self._epoch_train = EpochAccumulator()
        self._epoch_started_s = time.perf_counter()
        self._failures = 0
        self._reset_peak_memory()

    def log_train(  # type: ignore[override]
        self, step, loss, lr, x, y, samples, mask_loss,
        iou_regression_loss, model_iou, instance_loss,
    ):
        """Record one optimizer step.

        The arguments are the trainer's; see ``JointSamLogger``.
        """
        super().log_train(
            step, loss, lr, x, y, samples, mask_loss, iou_regression_loss,
            model_iou, instance_loss,
        )
        figures = {
            "loss": _to_float(loss),
            "mask_loss": _to_float(mask_loss),
            "iou_loss": _to_float(iou_regression_loss),
            "model_iou": _to_float(model_iou),
            "instance_loss": _to_float(instance_loss),
        }
        self._epoch_train.add(figures)
        metrics = {f"train/{name}": value for name, value in figures.items()}
        metrics.update({
            f"train/lr_{group}": rate
            for group, rate in learning_rates(self.trainer.optimizer).items()
        })
        self._safely(
            "step metrics", _log_metrics, metrics, step, synchronous=False
        )

    def log_validation(  # type: ignore[override]
        self, step, metric, loss, x, y, samples, mask_loss,
        iou_regression_loss, model_iou, instance_loss,
    ):
        """Record the end of an epoch.

        Called once per epoch, after validation and before the schedule
        advances, so the rates read here are the ones the epoch ran at.
        The arguments are the trainer's; see ``JointSamLogger``.
        """
        # Copied before the library sees it: its TensorBoard writer
        # rescales the image to [0, 1] in place, and the decoder rejects
        # an image in that range as wrongly scaled.
        image = x[:1].detach().clone()
        super().log_validation(
            step, metric, loss, x, y, samples, mask_loss,
            iou_regression_loss, model_iou, instance_loss,
        )
        epoch = int(self.trainer._epoch) + 1
        metrics = self.epoch_metrics(
            loss=loss, metric=metric, mask_loss=mask_loss,
            iou_loss=iou_regression_loss, model_iou=model_iou,
            instance_loss=instance_loss,
        )
        self._safely("epoch metrics", _log_metrics, metrics, epoch)
        self._safely(
            "decoder maps", self._log_decoder_maps, image, y, epoch, step
        )
        if self._failures:
            logger.warning(
                "%d write(s) to MLflow have failed so far in this run.",
                self._failures,
            )
        self._epoch_train.reset()
        self._epoch_started_s = time.perf_counter()
        self._reset_peak_memory()

    def epoch_metrics(self, **validation: Any) -> dict[str, float]:
        """Assemble the figures logged once per epoch.

        Parameters
        ----------
        **validation
            The trainer's validation figures, by name.

        Returns
        -------
        dict of str to float
        """
        metrics = {
            f"epoch/train_{name}": value
            for name, value in self._epoch_train.means().items()
        }
        metrics.update({
            f"epoch/val_{name}": _to_float(value)
            for name, value in validation.items()
        })
        metrics.update({
            f"epoch/lr_{group}": rate
            for group, rate in learning_rates(self.trainer.optimizer).items()
        })
        metrics["epoch/duration_min"] = (
            (time.perf_counter() - self._epoch_started_s) / 60.0
        )
        if torch.cuda.is_available():
            metrics["epoch/gpu_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / BYTES_PER_GIB
            )
            metrics["epoch/gpu_peak_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / BYTES_PER_GIB
            )
        return metrics

    def _log_decoder_maps(
        self, x: torch.Tensor, y: torch.Tensor, epoch: int, step: int
    ) -> None:
        """Run the decoder on one validation image and log the picture."""
        import mlflow

        unetr = self.trainer.unetr
        device = self.trainer.device
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            enabled=torch.device(device).type == "cuda",
        ):
            maps = unetr(x[:1].to(device))
        figure = decoder_maps_figure(
            x[0].detach().float().cpu().numpy(),
            y[0, 0].detach().cpu().numpy().astype(np.int64),
            maps[0].detach().float().clamp(0, 1).cpu().numpy(),
            title=f"epoch {epoch}, step {step}",
        )
        mlflow.log_image(figure, key="val_decoder_maps", step=epoch)

    def _safely(self, what: str, action: Any, *args: Any, **kwargs: Any):
        """Run a tracking action, and count rather than raise a failure."""
        try:
            action(*args, **kwargs)
        except Exception:  # noqa: BLE001 - see the module docstring.
            self._failures += 1
            if self._failures == 1:
                logger.warning(
                    "Writing %s to MLflow failed; training continues.",
                    what, exc_info=True,
                )

    @staticmethod
    def _reset_peak_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _log_metrics(
    metrics: dict[str, float],
    step: int,
    synchronous: Optional[bool] = None,
) -> None:
    """Write to the active run, or fail loudly if there is none."""
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run to log to.")
    mlflow.log_metrics(metrics, step=step, synchronous=synchronous)
