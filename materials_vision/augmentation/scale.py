"""
Multi-scale crop: the family that varies how large a pore looks.

The dataset holds two scales about 1.3 apart, and the coarser one is
88% of it. A model trained on that mixture learns the pore size of the
majority. Cutting a smaller window out of the frame and magnifying it
back to the frame's own size shows the same foam as the finer scale
would have recorded it, without inventing anything: every pixel of the
window was photographed, the magnification only spreads them further
apart.

**Why the whole decision is made in one place.** The library calls
``apply`` and ``apply_to_mask`` separately, with the same parameters,
so anything drawn independently in the two would put the image and the
annotation in different places. The window is therefore chosen, cut,
validated and magnified once, while the parameters are being drawn,
and the two ``apply`` methods only hand out what was already computed.
That also means the connected-components pass needed to validate a
window is the same one whose result is kept, rather than a second one.

**Why a rejected window re-draws only its position.** A window is
rejected when it holds too few instances. Drawing ``q`` again as well
would let rejections push the sample towards weaker magnifications,
and the distribution actually applied would then differ from the
frozen one by an amount nobody chose.

**What this transformation may legitimately change.** Instances leave
the frame, instances are cut by the window and lose area, and a cut
fragment too small to be anything an annotator would have drawn is
removed. Ids are renumbered densely afterwards. That is the whole list;
the count of instances may fall but no instance may end up in two
pieces, which the policy verifies after every sample that fires.
"""
import logging
from typing import Any, Mapping, Optional

import albumentations as A
import numpy as np
from skimage.transform import resize

from materials_vision.augmentation.config import ScaleConfig
from materials_vision.data.instances import CroppedSample, apply_content_crop

logger = logging.getLogger(__name__)

# What a crop reports, in the order it is worth reading. Everything
# else the transformation carries - the magnified arrays themselves,
# the frame size the library injects - is dropped before the record is
# written, because a record goes to the log.
RECORD_KEYS = (
    "q",
    "window",
    "changed_mask",
    "n_instances_before",
    "n_instances_after",
    "n_cut_by_crop",
    "n_dropped_outside",
    "n_dropped_below_min_area",
    "n_dropped_disconnected",
    "n_border_instances",
    "q_above_image_headroom",
    "attempts",
    "fallback",
)


class MultiScaleCrop(A.DualTransform):
    """Cut a window of the frame and magnify it back to frame size.

    Parameters
    ----------
    config : ScaleConfig
        The frozen distribution of magnifications and the rules a
        window must satisfy.

    Notes
    -----
    Requires ``scale_bin`` alongside the image and the mask, because
    the policy is defined per scale bin: only the coarse bin has
    anywhere finer to be magnified towards. ``q_max_i``, the headroom
    of the individual image, is read when present and recorded when
    the drawn magnification exceeds it; it does not alter the draw.
    """

    def __init__(self, config: ScaleConfig) -> None:
        super().__init__(p=config.p)
        self._config = config
        self._weights = tuple(weight for weight, _, _ in config.bands)

    @property
    def targets_as_params(self) -> list[str]:
        """Inputs the drawn parameters depend on.

        Returns
        -------
        list of str
        """
        return ["image", "mask", "scale_bin"]

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw a window, validate it, and magnify the sample.

        Parameters
        ----------
        params : dict
            Parameters drawn so far; unused.
        data : dict
            The sample being transformed, holding at least ``image``,
            ``mask`` and ``scale_bin``.

        Returns
        -------
        dict
            The magnified pair under ``scaled_image`` and
            ``scaled_labels`` - both ``None`` when the sample comes
            through untouched - plus everything worth recording.
        """
        image = data["image"]
        labels = data["mask"]
        config = self._config
        n_before = int(labels.max())

        if data["scale_bin"] not in config.magnified_bins:
            return _untouched(n_before)
        if n_before < config.min_instances:
            return _untouched(
                n_before, fallback="frame_holds_too_few_instances"
            )

        q = self._draw_q()
        if q == 1.0:
            return _untouched(n_before, q=q)

        height, width = labels.shape
        window_h = max(1, min(height, int(round(height / q))))
        window_w = max(1, min(width, int(round(width / q))))

        for attempt in range(1, config.max_retries + 2):
            window = self._draw_window(
                height, width, window_h, window_w
            )
            cropped = apply_content_crop(
                image,
                labels,
                window,
                min_fragment_area_px2=config.min_fragment_area_px2,
            )
            if cropped.n_instances >= config.min_instances:
                return self._magnified(
                    cropped,
                    q=q,
                    window=window,
                    target_shape=(height, width),
                    n_before=n_before,
                    attempts=attempt,
                    q_max_i=data.get("q_max_i"),
                )

        return _untouched(
            n_before,
            attempts=config.max_retries + 1,
            fallback="no_window_held_enough_instances",
        )

    def apply(
        self, img: np.ndarray, **params: Any
    ) -> np.ndarray:
        """Return the magnified image, or the original one.

        Parameters
        ----------
        img : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        scaled = params["scaled_image"]
        return img if scaled is None else scaled

    def apply_to_mask(
        self, mask: np.ndarray, **params: Any
    ) -> np.ndarray:
        """Return the magnified labels, or the original ones.

        Parameters
        ----------
        mask : np.ndarray
        **params : Any

        Returns
        -------
        np.ndarray
        """
        scaled = params["scaled_labels"]
        return mask if scaled is None else scaled

    def _draw_q(self) -> float:
        """Draw a magnification from the frozen distribution."""
        _, low, high = self.py_random.choices(
            self._config.bands, weights=self._weights
        )[0]
        if high == low:
            return float(low)
        return float(self.py_random.uniform(low, high))

    def _draw_window(
        self, height: int, width: int, window_h: int, window_w: int
    ) -> tuple[int, int, int, int]:
        """Place the window uniformly inside the frame.

        Uniform rather than biased towards dense regions: with whole
        images, magnifications up to 1.3 and foams this dense, almost
        every window already holds dozens of instances, so steering
        the draw would buy nothing and would quietly make the training
        distribution denser than the real one.
        """
        x0 = self.py_random.randint(0, width - window_w)
        y0 = self.py_random.randint(0, height - window_h)
        return (x0, y0, x0 + window_w, y0 + window_h)

    def _magnified(
        self,
        cropped: CroppedSample,
        *,
        q: float,
        window: tuple[int, int, int, int],
        target_shape: tuple[int, int],
        n_before: int,
        attempts: int,
        q_max_i: Optional[float],
    ) -> dict[str, Any]:
        """Magnify an accepted window and describe what it cost."""
        above_headroom = False
        if q_max_i is not None and q > float(q_max_i) + 1e-9:
            above_headroom = True
            logger.warning(
                "scale crop drew q=%.4f, past this image's own "
                "headroom of %.4f: the sample now claims detail finer "
                "than anything that was photographed",
                q, float(q_max_i),
            )
        return {
            "q": q,
            "window": window,
            "changed_mask": True,
            "n_instances_before": n_before,
            "n_instances_after": cropped.n_instances,
            "n_cut_by_crop": cropped.n_cut_by_crop,
            "n_dropped_outside": cropped.n_dropped_outside,
            "n_dropped_below_min_area": (
                cropped.n_dropped_below_min_area
            ),
            "n_dropped_disconnected": cropped.n_dropped_disconnected,
            "n_border_instances": cropped.n_border_instances,
            "q_above_image_headroom": above_headroom,
            "attempts": attempts,
            "fallback": None,
            "scaled_image": _magnify_image(
                cropped.image, target_shape
            ),
            "scaled_labels": _magnify_labels(
                cropped.labels, target_shape
            ),
        }


def build_scale(config: ScaleConfig) -> MultiScaleCrop:
    """Build the multi-scale crop transformation.

    Parameters
    ----------
    config : ScaleConfig

    Returns
    -------
    MultiScaleCrop
    """
    return MultiScaleCrop(config)


def summarize_scale_params(
    params: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce a crop's parameters to what belongs in a record.

    The transformation hands its result to ``apply`` through the same
    parameters the pipeline reports, so the magnified arrays travel in
    there too. A record is written to the log, so they must not.

    Parameters
    ----------
    params : Mapping

    Returns
    -------
    dict
    """
    return {key: params[key] for key in RECORD_KEYS if key in params}


def _untouched(
    n_before: int,
    *,
    q: float = 1.0,
    attempts: int = 1,
    fallback: Optional[str] = None,
) -> dict[str, Any]:
    """Parameters for a sample the family leaves as it found it.

    Reported rather than skipped. Half of the draws are the identity
    by design, and a fallback is a draw that failed - the two look the
    same in the sample and must not look the same in the record.
    """
    return {
        "q": q,
        "window": None,
        "changed_mask": False,
        "n_instances_before": n_before,
        "n_instances_after": n_before,
        "n_cut_by_crop": 0,
        "n_dropped_outside": 0,
        "n_dropped_below_min_area": 0,
        "n_dropped_disconnected": 0,
        "n_border_instances": None,
        "q_above_image_headroom": False,
        "attempts": attempts,
        "fallback": fallback,
        "scaled_image": None,
        "scaled_labels": None,
    }


def _magnify_image(
    image: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Magnify an image bilinearly, back to its own dtype."""
    magnified = resize(
        image, shape, order=1, preserve_range=True, anti_aliasing=False
    )
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        magnified = np.clip(
            np.rint(magnified), limits.min, limits.max
        )
    return magnified.astype(image.dtype, copy=False)


def _magnify_labels(
    labels: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Magnify a label image by nearest neighbour.

    Nearest neighbour is not a quality trade-off here but the only
    admissible choice: an interpolated label image holds values
    between two ids, which are instances nobody annotated. Rounding
    afterwards costs nothing and removes any doubt about a value like
    2.9999999 becoming a 2.
    """
    magnified = resize(
        labels, shape, order=0, preserve_range=True,
        anti_aliasing=False,
    )
    return np.rint(magnified).astype(labels.dtype, copy=False)
