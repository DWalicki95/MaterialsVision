"""
The object the dataset calls once per sample.

It holds the enabled families in one pipeline, applies them in a fixed
order, verifies the result and reports what it drew.

**Why one pipeline rather than a chain of separate calls.** The
composition object carries its own random state, seeded per sample and
entirely separate from every other source of randomness in the process.
That separation is what makes two policies comparable: the image order
a run sees must not depend on how much randomness the augmentation
consumed, or part of the difference between two policies would be the
order of the images rather than the augmentation. Building one
composition means that guarantee is inherited rather than reimplemented
- but only for transformations that draw from the composition's own
generators, never from the global ones.

**Why the order is fixed rather than drawn.** Cutting a window out of
the frame, turning it, shading pores and finally adjusting the whole
image are not commutative. Adjusting brightness before cutting would
measure the statistics of a frame the model never sees; drawing a wall
into a pore before cutting could put the wall outside the window. Only
one order is defensible and it is applied always.

**Why an empty policy is refused.** A run with no augmentation is
expressed by giving the dataset no policy, not by a policy that does
nothing. An object that stands for "nothing happens" would appear in a
run's configuration as though something did.
"""
import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import albumentations as A
import numpy as np

from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_TONAL,
                                                  MASK_CHANGING_FAMILIES,
                                                  PolicyConfig)
from materials_vision.augmentation.geometric import build_orientation
from materials_vision.augmentation.integrity import (check_connectivity,
                                                     check_labels_preserved,
                                                     check_mask_untouched,
                                                     check_sample)
from materials_vision.augmentation.mask_aware import (
    build_mask_aware, summarize_darkening_params, summarize_field_params)
from materials_vision.augmentation.photometric import (build_blur, build_tonal,
                                                       summarize_blur_params)
from materials_vision.augmentation.records import (AugmentationRecord,
                                                   AugmentedSample,
                                                   TransformRecord, log_record)
from materials_vision.augmentation.scale import (build_scale,
                                                 summarize_scale_params)

if TYPE_CHECKING:
    from materials_vision.data.samples import SampleRecord

logger = logging.getLogger(__name__)

# Transformations whose reported parameters are rewritten before they
# are recorded, because what they report is not what a reader needs.
PARAM_SUMMARIES = {
    "GaussianBlur": summarize_blur_params,
    "MultiScaleCrop": summarize_scale_params,
    "PoreBrightnessField": summarize_field_params,
    "PoreDarkening": summarize_darkening_params,
}

# Keys the library adds to every transformation's parameters describing
# the frame it worked on. They are not draws, and keeping them would
# put the image size in the record once per family.
INJECTED_PARAMS = frozenset({"shape", "fill", "fill_mask"})


class AugmentationPolicy:
    """One augmentation policy, applied to one sample at a time.

    Parameters
    ----------
    config : PolicyConfig
        Which families are enabled and with what parameters.

    Raises
    ------
    ValueError
        If no family is enabled.
    """

    def __init__(self, config: PolicyConfig) -> None:
        if not config.families:
            raise ValueError(
                "An augmentation policy must enable at least one "
                "family; a run without augmentation is expressed by "
                "passing no policy to the dataset"
            )
        self._config = config
        self._steps = _build_steps(config)
        self._compose = A.Compose(
            [transform for _, _, transform in self._steps],
            save_applied_params=True,
        )
        self._moves_pixels = config.orientation is not None

    @property
    def config(self) -> PolicyConfig:
        """The configuration this policy was built from.

        Returns
        -------
        PolicyConfig
        """
        return self._config

    @property
    def families(self) -> tuple[str, ...]:
        """Enabled families, in the order they are applied.

        Returns
        -------
        tuple of str
        """
        return tuple(family for family, _, _ in self._steps)

    def __call__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        record: "SampleRecord",
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment one sample and return just the arrays.

        This is what the dataset calls. The record of what was drawn is
        dropped here on purpose: in a run with dataloader workers the
        sample is built in another process, so an object returned along
        this path could not reach the trainer anyway. What training
        needs to know is written to the log instead.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` working channel.
        labels : np.ndarray
            ``(H, W)`` instance labels, densely numbered from 1.
        record : SampleRecord
            Description of the sample, including the identifier used in
            log lines and the calibration the scale family needs.
        seed : int
            Seed for this sample.

        Returns
        -------
        tuple of np.ndarray
            The augmented ``(image, labels)``.
        """
        augmented = self.apply(
            image, labels, record=record, seed=seed
        )
        return augmented.image, augmented.labels

    def apply(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        record: "SampleRecord",
        seed: int,
    ) -> AugmentedSample:
        """Augment one sample and return the record with it.

        For callers running in the main process - tests, the panels
        used for judging a transformation by eye, the benchmark - which
        need the values that were drawn rather than only their effect.

        Parameters
        ----------
        image : np.ndarray
        labels : np.ndarray
        record : SampleRecord
            Three of its fields reach the pipeline: the identifier,
            which names the sample in log lines and error messages,
            and the two describing the sample's physical scale, which
            the crop needs because how far an image may be magnified
            is a property of the image and not of the policy.
        seed : int

        Returns
        -------
        AugmentedSample

        Raises
        ------
        IntegrityError
            If the augmented sample violates a property the rest of the
            pipeline is entitled to assume.
        """
        image_id = record.image_id
        self._compose.set_random_seed(int(seed))
        result = self._compose(
            image=image,
            mask=labels,
            scale_bin=record.scale_bin,
            q_max_i=record.q_max_i,
        )
        augmented_image = result["image"]
        augmented_labels = result["mask"]

        transforms = self._transform_records(
            result.get("applied_transforms", ())
        )
        context = f"{image_id} after {'+'.join(self.families)}"
        self._verify(
            labels, augmented_image, augmented_labels, context,
            transforms,
        )

        augmentation = AugmentationRecord(
            image_id=str(image_id),
            seed=int(seed),
            transforms=transforms,
        )
        log_record(augmentation)
        return AugmentedSample(
            image=augmented_image,
            labels=augmented_labels,
            record=augmentation,
        )

    def _verify(
        self,
        labels: np.ndarray,
        augmented_image: np.ndarray,
        augmented_labels: np.ndarray,
        context: str,
        transforms: tuple[TransformRecord, ...],
    ) -> None:
        """Check the augmented sample against what actually happened.

        The strongest applicable statement is used, and which one that
        is depends on the sample rather than on the policy. A policy
        able to cut or divide instances does so on only some of its
        samples; on the rest the mask is either untouched or merely
        rearranged, and holding those to the weaker check would give
        up the stronger one for nothing.

        Where the mask really did change, the requirement that remains
        is that no instance ended up in two pieces. That is not
        cosmetic: the training targets are built per instance from a
        distance transform, so two pieces become two basins and teach
        the model to divide a pore the annotation regards as whole.
        The check costs a labelling pass over the frame, which is why
        it runs only on the samples that need it.
        """
        check_sample(
            augmented_image,
            augmented_labels,
            context=context,
            expect_instances=int(labels.max()) > 0,
        )
        if _mask_was_changed(transforms):
            check_connectivity(augmented_labels, context=context)
        elif self._moves_pixels:
            check_labels_preserved(
                labels, augmented_labels, context=context
            )
        else:
            check_mask_untouched(
                labels, augmented_labels, context=context
            )

    def _transform_records(
        self, applied: Iterable[tuple[str, Mapping[str, Any]]]
    ) -> tuple[TransformRecord, ...]:
        """Turn what the pipeline reported into one record per family.

        Families that did not fire are recorded too, with
        ``applied=False``: how often a family with a probability below
        one actually fires is part of what a comparison against it
        means, and it cannot be recovered from the samples that did
        fire.

        A transformation that validates its own draw reports how many
        draws it took and why it gave up, if it did. Those two are
        lifted out of the reported values into their own fields, since
        they describe the drawing rather than what was drawn, and one
        of them has to be visible in the log without reading through
        a family's parameters.
        """
        by_class = {name: params for name, params in applied}
        records = []
        for family, class_names, _ in self._steps:
            drawn = next(
                (name for name in class_names if name in by_class),
                None,
            )
            if drawn is None:
                records.append(
                    TransformRecord(family=family, applied=False)
                )
                continue
            params = _readable_params(drawn, by_class[drawn])
            attempts = int(params.pop("attempts", 1))
            fallback = params.pop("fallback", None)
            records.append(
                TransformRecord(
                    family=family,
                    applied=True,
                    name=drawn,
                    params=params,
                    attempts=attempts,
                    fallback=fallback,
                )
            )
        return tuple(records)


def _build_steps(
    config: PolicyConfig,
) -> tuple[tuple[str, tuple[str, ...], Any], ...]:
    """Build the enabled transformations in the order they apply.

    Each step carries the class names its family can produce, which is
    how a reported transformation is later attributed to the family it
    belongs to. A family offering alternatives contributes more than
    one name; no name belongs to two families.
    """
    steps: list[tuple[str, tuple[str, ...], Any]] = []
    if config.scale is not None:
        steps.append((
            FAMILY_SCALE,
            ("MultiScaleCrop",),
            build_scale(config.scale),
        ))
    if config.orientation is not None:
        steps.append((
            FAMILY_ORIENTATION,
            ("D4",),
            build_orientation(config.orientation),
        ))
    if config.mask_aware is not None:
        steps.append((
            FAMILY_MASK_AWARE,
            ("PoreBrightnessField", "PoreDarkening"),
            build_mask_aware(config.mask_aware),
        ))
    if config.tonal is not None:
        steps.append((
            FAMILY_TONAL,
            ("RandomBrightnessContrast", "RandomGamma"),
            build_tonal(config.tonal),
        ))
    if config.blur is not None:
        steps.append((
            FAMILY_BLUR,
            ("GaussianBlur",),
            build_blur(config.blur),
        ))
    return tuple(steps)


def _mask_was_changed(
    transforms: Iterable[TransformRecord],
) -> bool:
    """Whether a family entitled to change the mask actually did.

    Three conditions, all necessary: the family must be one of those
    allowed to cut or divide instances, it must have fired, and it
    must report having changed anything. The third matters because
    such a family can fire and still leave the mask alone - a crop
    that draws the identity magnification is the ordinary case, not a
    corner one, and paying for a labelling pass on those samples would
    be paying for nothing.

    A family that reports nothing is treated as having changed the
    mask, so a transformation that forgets to say gets the safe answer
    rather than the cheap one.

    Parameters
    ----------
    transforms : Iterable of TransformRecord

    Returns
    -------
    bool
    """
    return any(
        entry.applied
        and entry.family in MASK_CHANGING_FAMILIES
        and bool(entry.params.get("changed_mask", True))
        for entry in transforms
    )


def _readable_params(
    name: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce reported parameters to values worth writing down."""
    summarize = PARAM_SUMMARIES.get(name)
    if summarize is not None:
        return summarize(params)
    return {
        key: _plain(value)
        for key, value in params.items()
        if key not in INJECTED_PARAMS
    }


def _plain(value: Any) -> Any:
    """Convert a numpy scalar to its Python equivalent.

    Records end up in log lines and metadata files, where a numpy
    scalar prints with its type spelled out and does not serialize.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value
