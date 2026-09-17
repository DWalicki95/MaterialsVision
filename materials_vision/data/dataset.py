"""
The torch dataset the trainer sees.

It is a thin wrapper on purpose. Everything with a domain rule in it -
reading the split, cropping the information panel, rebuilding the
instances the cut passed through - lives in ``samples.py`` and is
testable without torch. What remains here is the last mile: run the
augmentation policy, turn the label image into the decoder's targets,
and hand back tensors.

**What this deliberately does not do.** It does not resize, pad or
normalize. SAM performs all three inside its own forward pass: the
longest side is scaled to the encoder's input size, pixel values are
normalized with the model's own statistics, and the result is padded
to a square. Doing any of it here would either duplicate the
normalization or, worse, pad before the model scales - which would
shrink the actual content by an extra factor and quietly change the
working resolution every downstream criterion was calibrated against.
The dataset therefore emits images at content resolution, in their
natural aspect ratio.

**Why the shapes differ between samples.** Images from one microscope
keep their full height; those from the other lose the information
panel. Rotations by 90 degrees swap the axes on top of that. Batching
tensors of different shapes would require grouping samples by shape,
which is precisely the complication a batch size of one removes.

**Augmentation randomness is keyed to position, not to arrival
order.** The seed for a sample is derived from the run's seed and the
index it arrived under, so a run reproduces exactly regardless of how
many worker processes load it or in what order they finish. The
alternative - seeding each worker once and drawing from a shared
stream - makes the augmentation of a given image depend on scheduling,
which is reproducible only by accident.

**Indices carry the epoch and exceed this dataset's length.** An index
is ``epoch * len(self) + position``: the position says which image to
read, the epoch changes what the augmentation draws for it. Both are
needed, and the epoch cannot be held as state here, because samples
are prepared in worker processes holding their own copies of this
object - state set on the copy in the training process would never
reach them, and the augmentation would repeat one epoch's draws for
the length of the run. Packing the epoch into the index hands it to
the workers as data instead. The dataset is therefore stateless, and
it is indexed only through the sampler that builds those indices.
"""
import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from materials_vision.data.samples import PreparedSample, SampleSource
from materials_vision.data.sampling import (AUGMENT_DOMAIN, decode_index,
                                            derive_seed)

logger = logging.getLogger(__name__)

RGB_CHANNELS = 3

# Frozen configuration of the decoder's target transform. Both values
# differ from the library defaults and the reasons are specific:
#
# apply_label=False - the default re-runs connected components on the
#   labels, which would split an instance left in two pieces into two
#   ids. The masks already guarantee one connected piece per instance
#   (measured: no violation across the training set), so the only
#   thing that re-derivation could do is absorb a future violation
#   silently instead of letting it surface.
#
# min_size=0 - the default is already 0; it is named here to record
#   that filtering was considered and rejected. Dropping small objects
#   from the targets while the metrics still count them as ground
#   truth would penalize the model at evaluation for pores it was
#   never trained to find. The smallest instance in the training set
#   is 51 px^2, far above anything the distance transform struggles
#   with, so there is nothing to guard against either.
LABEL_TRANSFORM_KWARGS = {
    "distances": True,
    "boundary_distances": True,
    "directed_distances": False,
    "foreground": True,
    "instances": True,
    "apply_label": False,
    "min_size": 0,
}


def build_label_transform():
    """Return the frozen per-object distance transform.

    Returns
    -------
    torch_em.transform.label.PerObjectDistanceTransform

    Notes
    -----
    Imported lazily so that importing this module does not pull in the
    training stack; the pure loading path stays usable without it.
    """
    from torch_em.transform.label import PerObjectDistanceTransform

    return PerObjectDistanceTransform(**LABEL_TRANSFORM_KWARGS)


class InstanceSegmentationDataset(Dataset):
    """Serve prepared samples as (image, target) tensor pairs.

    Parameters
    ----------
    source : SampleSource
        Supplies cropped, single-channel images and their instance
        labels.
    label_transform : Callable
        Turns a label image into the decoder's targets; normally
        ``build_label_transform()``.
    transform : Callable, optional
        Augmentation policy, called as ``transform(image, labels,
        record=..., seed=...)`` and returning a new ``(image, labels)``
        pair. ``None`` means no augmentation, which is both the
        baseline condition and the correct setting for validation.
        The record is passed because scale augmentation is conditioned
        on the sample's own calibration: the magnification range is
        allowed on the coarse scale bin and pinned to 1.0 on the fine
        one, so a policy that could not see ``scale_bin`` would have to
        guess.
    run_seed : int, optional
        Seed of this run, mixed with the index a sample arrives under
        to seed its augmentation.

    Raises
    ------
    ValueError
        If the source is empty.
    """

    def __init__(
        self,
        source: SampleSource,
        *,
        label_transform: Callable,
        transform: Optional[Callable] = None,
        run_seed: int = 0,
    ) -> None:
        if len(source) == 0:
            raise ValueError("Cannot build a dataset over an empty source")
        self._source = source
        self._label_transform = label_transform
        self._transform = transform
        self._run_seed = int(run_seed)

    def __len__(self) -> int:
        return len(self._source)

    @property
    def source(self) -> SampleSource:
        """The underlying sample source.

        Exposed so evaluation can reach the records - image ids,
        formulations, border flags - that the ``(image, target)`` pair
        has no room to carry.

        Returns
        -------
        SampleSource
        """
        return self._source

    def sample_seed(self, index: int) -> int:
        """Seed used to augment the sample arriving under this index.

        Parameters
        ----------
        index : int
            The combined index, epoch included. Two epochs of the same
            image differ here, which is the whole point: the same seed
            twice would show the model one fixed augmented copy of the
            dataset rather than a fresh draw each pass.

        Returns
        -------
        int
        """
        return derive_seed(self._run_seed, index, AUGMENT_DOMAIN)

    def __getitem__(self, index: int):
        _, position = decode_index(index, len(self))
        prepared = self._source.load(position)
        image, labels = self._augment(prepared, index)
        return (
            _to_rgb_tensor(image),
            _to_target_tensor(self._label_transform(labels)),
        )

    def _augment(
        self, prepared: PreparedSample, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the augmentation policy, if there is one."""
        if self._transform is None:
            return prepared.image, prepared.labels
        return self._transform(
            prepared.image, prepared.labels,
            record=prepared.record,
            seed=self.sample_seed(index),
        )


def _to_rgb_tensor(image: np.ndarray) -> torch.Tensor:
    """Turn one working channel into the model's 3-channel input.

    The channel is repeated rather than converted: the images are
    monochrome, and SAM's encoder expects three channels. Values stay
    on their source scale, since the model normalizes with its own
    statistics.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)``.

    Returns
    -------
    torch.Tensor
        ``(3, H, W)`` float32.

    Raises
    ------
    ValueError
        If the image is not two-dimensional.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected one working channel of shape (H, W), got "
            f"{image.shape}"
        )
    stacked = np.repeat(
        image[None].astype(np.float32), RGB_CHANNELS, axis=0
    )
    return torch.from_numpy(stacked)


def _to_target_tensor(target: np.ndarray) -> torch.Tensor:
    """Convert the decoder's targets to a float32 tensor.

    Parameters
    ----------
    target : np.ndarray

    Returns
    -------
    torch.Tensor
    """
    return torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
