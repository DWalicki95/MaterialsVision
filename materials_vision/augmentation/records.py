"""
What a policy actually drew for one sample, and where it goes.

Three quantities matter and none can travel inside the transformed
arrays: the values actually drawn, how often a transformation that may
decline to fire really fired, and a trace of every controlled fallback.
Two kinds of consumer need them, and they need different channels.

**In-process consumers** - unit tests, the panels used for judging a
transformation by eye, the dataloader benchmark - run in the main
process and want the values as objects. They call the policy's
``apply``, which hands back the record alongside the sample.

**A training run** cannot use that. With more than one dataloader
worker the samples are built in separate processes, and an object left
on a policy instance there never reaches the trainer. Anything training
has to report is therefore written to the log, which is the only honest
source for it. Hence two channels, neither replacing the other.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformRecord:
    """One transformation's contribution to one sample.

    Parameters
    ----------
    family : str
        Family code, e.g. ``"F3b_blur"``.
    applied : bool
        Whether it fired. Samples it skipped are recorded too: the
        measured firing rate is the denominator of any comparison that
        treats the family as the variable under test, and it can differ
        from the nominal probability.
    name : str, optional
        Transformation that ran, which for a family holding
        alternatives says which alternative was drawn. ``None`` when
        the family did not fire, because then nothing ran.
    params : Mapping
        Values drawn, as the transformation reported them.
    attempts : int
        Draws made before one was accepted. Above one only for
        transformations that validate their own result and retry.
    fallback : str, optional
        Why the transformation gave up and returned the sample
        untouched. ``None`` when it did not give up.
    """

    family: str
    applied: bool
    name: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 1
    fallback: Optional[str] = None


@dataclass(frozen=True)
class AugmentationRecord:
    """Everything one policy did to one sample.

    Parameters
    ----------
    image_id : str
    seed : int
        Seed the sample was drawn with. Together with the policy
        configuration it reproduces the sample exactly.
    transforms : tuple of TransformRecord
        In the order the pipeline applied them.
    """

    image_id: str
    seed: int
    transforms: tuple[TransformRecord, ...] = ()

    @property
    def applied_families(self) -> tuple[str, ...]:
        """Families that actually fired on this sample.

        Returns
        -------
        tuple of str
        """
        seen: list[str] = []
        for record in self.transforms:
            if record.applied and record.family not in seen:
                seen.append(record.family)
        return tuple(seen)

    @property
    def fallbacks(self) -> tuple[TransformRecord, ...]:
        """Transformations that gave up and left the sample alone.

        Returns
        -------
        tuple of TransformRecord
        """
        return tuple(
            record for record in self.transforms
            if record.fallback is not None
        )


@dataclass(frozen=True)
class AugmentedSample:
    """An augmented pair and the record of how it got that way.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)``, one working channel.
    labels : np.ndarray
        ``(H, W)`` instance labels, densely numbered from 1.
    record : AugmentationRecord
    """

    image: np.ndarray
    labels: np.ndarray
    record: AugmentationRecord


def log_record(record: AugmentationRecord) -> None:
    """Write one sample's record to the run log.

    Applied transformations go to DEBUG. One line per sample is far too
    much to read through, but it is what allows a particular sample -
    the one behind a suspicious loss spike, say - to be reconstructed
    after the fact.

    Fallbacks go to INFO instead. A controlled fallback is expected and
    must not stop the run, but it has to stay visible: a sample that
    quietly skips a transformation still counts as that family's sample
    when the results are compared, so the rate at which it happens is
    part of what the comparison means.

    Parameters
    ----------
    record : AugmentationRecord
    """
    for entry in record.transforms:
        if entry.fallback is not None:
            logger.info(
                "augmentation fallback: image=%s family=%s name=%s "
                "reason=%s attempts=%d",
                record.image_id, entry.family, entry.name,
                entry.fallback, entry.attempts,
            )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "augmentation: image=%s seed=%d applied=%s params=%s",
            record.image_id, record.seed,
            list(record.applied_families),
            [
                (entry.name, dict(entry.params))
                for entry in record.transforms if entry.applied
            ],
        )
