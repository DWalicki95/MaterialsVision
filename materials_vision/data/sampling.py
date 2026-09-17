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

**The epoch travels inside the index.** What this sampler yields is not
a position in the dataset but ``epoch * n_images + position``, and the
dataset divides it back apart. The obvious alternative - telling the
dataset which epoch it is, once per epoch - cannot work here: the
loader prepares samples in worker processes, each holding its own copy
of the dataset, so an epoch set on the copy in this process reaches
none of them. Carrying the epoch in the index instead makes it part of
the data the workers are handed, which is correct for any number of
workers and for workers that outlive an epoch. It is also the property
that a silent regression would have to break loudly: an index that
ignored the epoch would repeat an epoch's augmentation exactly, and the
test that compares two epochs of the same image would fail.

The cost is that indices exceed the dataset's length, which is unusual
enough to be worth stating wherever it surfaces. It is safe because a
map-style dataset is only ever indexed through its sampler, and this is
that sampler.
"""
import hashlib
import logging
from typing import Iterator, Mapping, Optional, Sequence

import torch
from torch.utils.data import Sampler

from materials_vision.data.split_io import SplitSubset

logger = logging.getLogger(__name__)

STRATEGY = "proportional_no_oversampling"

ORDERING = "epoch_permutation"

# Namespaces keeping two uses of the same run seed apart. Without them
# the permutation of epoch 0 and the augmentation of sample 0 would be
# drawn from one value, which is harmless today only because the two
# feed different generators - the kind of coincidence that stops being
# harmless the moment either side changes.
ORDER_DOMAIN = "order"

AUGMENT_DOMAIN = "augment"


def encode_index(epoch: int, position: int, n_images: int) -> int:
    """Combine an epoch and a dataset position into one index.

    Parameters
    ----------
    epoch : int
    position : int
        Position within the dataset, in ``[0, n_images)``.
    n_images : int
        Length of the dataset being indexed, which is the base the two
        parts are packed against.

    Returns
    -------
    int
    """
    return epoch * n_images + position


def decode_index(index: int, n_images: int) -> tuple[int, int]:
    """Split a combined index back into its epoch and its position.

    Parameters
    ----------
    index : int
    n_images : int
        Length of the dataset being indexed. Must match the value the
        index was built against, or both parts come out wrong.

    Returns
    -------
    tuple of int
        ``(epoch, position)``.

    Raises
    ------
    ValueError
        If ``n_images`` is not positive, which would make the split
        meaningless rather than merely wrong.
    """
    if n_images < 1:
        raise ValueError(f"n_images must be positive, got {n_images}")
    return divmod(int(index), int(n_images))


class ProportionalImageSampler(Sampler[int]):
    """Yield a fresh permutation of every TRAIN image, once per epoch.

    Proportional sampling falls out of the construction: drawing each
    image exactly once per epoch gives every group a share of steps
    equal to its share of images, exactly rather than in expectation.
    A permutation is preferred over independent draws with replacement
    for that reason, and because it keeps "epoch" a well-defined unit
    for reporting - runs are compared by optimizer steps, but epochs
    remain a readable secondary axis.

    What is yielded is ``epoch * n_images + position``, not a bare
    position; see the module docstring for why the epoch has to travel
    with the index rather than be set on the dataset.

    Parameters
    ----------
    n_images : int
        Length of the dataset being sampled. This is the base the
        epoch is packed against, so it stays the whole dataset even
        when ``positions`` restricts which of it is drawn.
    run_seed : int
        Seed of this run. Two runs sharing it see identical image
        orders regardless of their augmentation policies.
    positions : sequence of int, optional
        Restrict sampling to these positions, for short rehearsals
        that do not need the whole subset. ``None`` draws all of them.
    shuffle : bool, optional
        Permute each epoch. Off yields the positions in their given
        order, which is what validation and any reproducible pass over
        the data need.

    Raises
    ------
    ValueError
        If ``n_images`` is not positive, if ``positions`` is empty, or
        if it names a position outside the dataset.
    """

    def __init__(
        self,
        n_images: int,
        run_seed: int,
        *,
        positions: Optional[Sequence[int]] = None,
        shuffle: bool = True,
    ) -> None:
        if n_images < 1:
            raise ValueError(
                f"n_images must be positive, got {n_images}"
            )
        self._n_images = int(n_images)
        self._run_seed = int(run_seed)
        self._shuffle = bool(shuffle)
        self._positions = _check_positions(positions, self._n_images)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self) -> Iterator[int]:
        if self._shuffle:
            generator = torch.Generator()
            generator.manual_seed(
                derive_seed(self._run_seed, self._epoch, ORDER_DOMAIN)
            )
            order = torch.randperm(
                len(self._positions), generator=generator
            )
            positions = [self._positions[int(i)] for i in order]
        else:
            positions = list(self._positions)
        yield from (
            encode_index(self._epoch, position, self._n_images)
            for position in positions
        )

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
            "ordering": ORDERING if self._shuffle else "fixed",
            "n_images": self._n_images,
            "n_sampled": len(self._positions),
            "shuffle": self._shuffle,
            "run_seed": self._run_seed,
            "seed_derivation": "blake2b(f'{domain}:{run_seed}:{counter}')",
            "oversampling": None,
        }


def _check_positions(
    positions: Optional[Sequence[int]], n_images: int
) -> tuple[int, ...]:
    """Validate the restricted position list, or build the full one."""
    if positions is None:
        return tuple(range(n_images))
    checked = tuple(int(position) for position in positions)
    if not checked:
        raise ValueError(
            "positions must name at least one image; pass None to "
            "sample the whole dataset"
        )
    out_of_range = [p for p in checked if not 0 <= p < n_images]
    if out_of_range:
        raise ValueError(
            f"positions must lie in [0, {n_images}), got "
            f"{out_of_range[:5]}"
        )
    return checked


def derive_seed(run_seed: int, counter: int, domain: str) -> int:
    """Derive one reproducible seed from a run seed and a counter.

    Hashing rather than arithmetic mixing keeps neighbouring counters
    from producing correlated draws, and keeps the value reproducible
    across platforms and Python versions - unlike ``hash()``, which is
    randomized per process.

    Parameters
    ----------
    run_seed : int
    counter : int
        What varies within the run: the epoch for image order, the
        combined index for augmentation.
    domain : str
        Which use this seed is for, normally ``ORDER_DOMAIN`` or
        ``AUGMENT_DOMAIN``. Stated rather than defaulted because two
        uses that collide would do so silently, and the caller is the
        only one who knows which it means.

    Returns
    -------
    int
        A value in ``[0, 2**64)``, suitable for
        ``torch.Generator.manual_seed``.
    """
    digest = hashlib.blake2b(
        f"{domain}:{run_seed}:{counter}".encode("utf-8"), digest_size=8
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
