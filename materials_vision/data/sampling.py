"""
Order in which TRAIN images reach the optimizer.

The policy is frozen for every run of the experiment: draw images
**proportionally to their counts, with no oversampling**. At
``batch_size = 1`` one optimizer step is one image, so this ordering
*is* the effective training distribution - there is no batch averaging
to soften it. On ``split_v1`` that means AS receives about 85% of
steps, K about 9%, VAB about 5.5%, and the ``fine`` scale bin about
10%.

That imbalance is a condition of the experiment, not a defect to
repair. One of the hypotheses under test is precisely that a model
trained without
scale augmentation learns mostly the ``coarse`` scale; oversampling
``fine`` would answer that question with the sampler instead of with
the augmentation, and the same argument applies to K/VAB and
cross-microscope transfer. The rare cross-sections are watched through
per-``scale_bin``, per-material and per-formulation metrics instead.

**The sampler owns its own random stream.** The experiment compares
augmentation policies in pairs: the same seed is run under two
policies and the difference in the resulting metric is attributed to
the augmentation. That attribution is only valid if the two runs saw
the same images in the same order, which is what the private stream
guarantees. A
``DataLoader`` built with ``shuffle=True`` draws its permutation from
the global torch generator, whose state depends on how much randomness
everything else consumed first - and that differs between augmentation
policies. Two policies at the same seed would then see *different image
orders*, and part of the measured difference would be ordering noise
wearing the costume of an augmentation effect. Seeding a local
generator from ``(run_seed, epoch)`` alone makes the order immune to
whatever the augmentation draws.

The dependency only has to hold in one direction: augmentation must
not perturb the image order. The reverse is fine, since augmentation
is supposed to differ between policies.
"""
import hashlib
import logging
from typing import Iterator, Mapping

import torch
from torch.utils.data import Sampler

from materials_vision.data.split_io import SplitSubset

logger = logging.getLogger(__name__)

STRATEGY = "proportional_no_oversampling"

ORDERING = "epoch_permutation"


class ProportionalImageSampler(Sampler[int]):
    """Yield a fresh permutation of every TRAIN image, once per epoch.

    Proportional sampling falls out of the construction: drawing each
    image exactly once per epoch gives every group a share of steps
    equal to its share of images, exactly rather than in expectation.
    A permutation is preferred over independent draws with replacement
    for that reason, and because it keeps "epoch" a well-defined unit
    for reporting - runs are compared by optimizer steps, but epochs
    remain a readable secondary axis.

    Parameters
    ----------
    n_images : int
        Number of images in the subset being sampled.
    run_seed : int
        Seed of this run. Two runs sharing it see identical image
        orders regardless of their augmentation policies.

    Raises
    ------
    ValueError
        If ``n_images`` is not positive.
    """

    def __init__(self, n_images: int, run_seed: int) -> None:
        if n_images < 1:
            raise ValueError(
                f"n_images must be positive, got {n_images}"
            )
        self._n_images = int(n_images)
        self._run_seed = int(run_seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._n_images

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            derive_seed(self._run_seed, self._epoch)
        )
        order = torch.randperm(self._n_images, generator=generator)
        yield from (int(i) for i in order)

    @property
    def epoch(self) -> int:
        """Epoch the next permutation will be drawn for.

        Returns
        -------
        int
        """
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch, changing the permutation deterministically.

        Must be called before each epoch, exactly as with
        ``DistributedSampler``: without it every epoch would repeat
        the same order.

        Parameters
        ----------
        epoch : int

        Raises
        ------
        ValueError
            If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self._epoch = int(epoch)

    def to_metadata(self) -> dict[str, object]:
        """Describe this sampler for the run metadata.

        Returns
        -------
        dict
        """
        return {
            "strategy": STRATEGY,
            "ordering": ORDERING,
            "n_images": self._n_images,
            "run_seed": self._run_seed,
            "seed_derivation": "blake2b(f'{run_seed}:{epoch}')",
            "oversampling": None,
        }


def derive_seed(run_seed: int, epoch: int) -> int:
    """Derive the permutation seed for one epoch.

    Hashing rather than arithmetic mixing keeps neighbouring epochs
    from producing correlated permutations, and keeps the value
    reproducible across platforms and Python versions - unlike
    ``hash()``, which is randomized per process.

    Parameters
    ----------
    run_seed : int
    epoch : int

    Returns
    -------
    int
        A value in ``[0, 2**64)``, suitable for
        ``torch.Generator.manual_seed``.
    """
    digest = hashlib.blake2b(
        f"{run_seed}:{epoch}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def sampler_run_metadata(
    subset: SplitSubset,
    sampler: ProportionalImageSampler,
    exposure_columns: tuple[str, ...] = ("material", "scale_bin"),
) -> dict[str, object]:
    """Join the sampler's configuration with what it will expose.

    The exposure shares are what makes this record worth writing: a
    per-material metric is not interpretable without knowing how many
    optimizer steps that material actually received.

    Parameters
    ----------
    subset : SplitSubset
        The subset the sampler runs over.
    sampler : ProportionalImageSampler
    exposure_columns : tuple of str, optional
        Columns to report exposure for.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If the sampler's length does not match the subset's, which
        would mean they describe different data.
    """
    if len(sampler) != len(subset):
        raise ValueError(
            f"Sampler covers {len(sampler)} image(s) but subset "
            f"{subset.subset!r} of {subset.split_id} holds "
            f"{len(subset)}"
        )
    metadata: dict[str, object] = {
        "split_id": subset.split_id,
        "subset": subset.subset,
        **sampler.to_metadata(),
        "steps_per_epoch": len(sampler),
        "exposure": {
            column: subset.exposure(column)
            for column in exposure_columns
        },
    }
    _log_exposure(subset, exposure_columns)
    return metadata


def _log_exposure(
    subset: SplitSubset, columns: tuple[str, ...]
) -> None:
    """Log the share of optimizer steps each group will receive."""
    for column in columns:
        shares: Mapping[str, float] = subset.exposure(column)
        rendered = ", ".join(
            f"{value} {100 * share:.1f}%"
            for value, share in shares.items()
        )
        logger.info(
            "Step exposure by %s (%s): %s",
            column, subset.subset, rendered,
        )
