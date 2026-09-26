"""
Which training images the post-processing is calibrated on.

The training split is 85% one material, so a sample drawn in proportion
to it would hold about four images of the second material and three of
the third - too few to say anything per material, and per material is
how the calibration has to be read: a threshold that suits one material
and not another would be a hidden per-material correction if it were
chosen from a pool dominated by the first.

So the sample is stratified rather than proportional. Every image of
the two small materials is taken, and a fixed number of the large one,
drawn so that its formulations and scale bins appear in the proportions
the split holds them in. Close-ups are left out, as the evaluation
leaves them out.

**The sample is fixed before any prediction exists.** It is written to
a file by a script that reads only the split, and the calibration reads
that file. Choosing images after seeing how the model does on them would
let the choice favour a setting.
"""
from collections import defaultdict
from typing import Sequence

import numpy as np

from materials_vision.data.samples import SampleRecord
from materials_vision.evaluation.aggregate import SCALE_OUTLIER_BIN

# The material that dominates the split, subsampled; every other one is
# taken whole.
SUBSAMPLED_MATERIAL = "AS"

N_SUBSAMPLED = 45

SAMPLE_SEED = 20260907


def allocate(counts: dict[tuple, int], n_wanted: int) -> dict[tuple, int]:
    """Share a sample size among strata in proportion to their size.

    Largest remainders, so the shares add up to exactly ``n_wanted``,
    with ties broken by stratum key so the allocation is deterministic.

    Parameters
    ----------
    counts : dict of tuple to int
        Images available per stratum.
    n_wanted : int

    Returns
    -------
    dict of tuple to int
        Never more than a stratum holds.

    Raises
    ------
    ValueError
        If more images are wanted than exist.
    """
    total = sum(counts.values())
    if n_wanted > total:
        raise ValueError(f"{n_wanted} image(s) wanted, {total} available.")
    exact = {key: n_wanted * count / total for key, count in counts.items()}
    shares = {key: int(np.floor(value)) for key, value in exact.items()}
    by_remainder = sorted(
        counts, key=lambda key: (-(exact[key] - shares[key]), key)
    )
    for key in by_remainder[:n_wanted - sum(shares.values())]:
        shares[key] += 1
    return shares


def allocate_by_bin_then_formulation(
    counts: dict[tuple[str, str], int], n_wanted: int
) -> dict[tuple[str, str], int]:
    """Share a sample first among scale bins, then among formulations.

    Allocating over formulation-and-bin strata in one step lets a bin
    that is spread thinly over many formulations vanish: each of its
    strata is owed a fraction of an image, and the leftover places go to
    the larger remainders of the other bin. Seven percent of the images
    of the dominant material are in the fine bin, spread over several
    formulations, and a single-step draw of 45 gave it none. The bin is
    what the minimum instance size is derived per, so it is allocated
    first, and formulations share what their bin receives.

    Parameters
    ----------
    counts : dict of (formulation, scale_bin) to int
    n_wanted : int

    Returns
    -------
    dict of (formulation, scale_bin) to int
    """
    per_bin: dict[str, int] = defaultdict(int)
    for (_, scale_bin), count in counts.items():
        per_bin[scale_bin] += count
    bin_shares = allocate(
        {(scale_bin,): count for scale_bin, count in per_bin.items()},
        n_wanted,
    )
    shares: dict[tuple[str, str], int] = {}
    for (scale_bin,), n_bin in bin_shares.items():
        within = {key: count for key, count in counts.items()
                  if key[1] == scale_bin}
        shares.update(allocate(within, n_bin))
    return shares


def select_calibration_sample(
    records: Sequence[SampleRecord],
    subsampled_material: str = SUBSAMPLED_MATERIAL,
    n_subsampled: int = N_SUBSAMPLED,
    seed: int = SAMPLE_SEED,
) -> list[SampleRecord]:
    """Pick the calibration images from the records of one subset.

    Parameters
    ----------
    records : sequence of SampleRecord
    subsampled_material : str, optional
    n_subsampled : int, optional
    seed : int, optional

    Returns
    -------
    list of SampleRecord
        In the order the records were given.
    """
    scored = [
        record for record in records
        if record.scale_bin != SCALE_OUTLIER_BIN
    ]
    strata: dict[tuple, list[SampleRecord]] = defaultdict(list)
    chosen = []
    for record in scored:
        if record.material == subsampled_material:
            strata[(record.formulation, record.scale_bin)].append(record)
        else:
            chosen.append(record)

    rng = np.random.default_rng(seed)
    shares = allocate_by_bin_then_formulation(
        {key: len(members) for key, members in strata.items()},
        n_subsampled,
    )
    for key in sorted(strata):
        # Sorted so the draw depends on the seed and the split, not on
        # the order a reader happened to list the records in.
        members = sorted(strata[key], key=lambda record: record.image_id)
        picked = rng.choice(len(members), size=shares[key], replace=False)
        chosen.extend(members[index] for index in picked)

    order = {record.index: position for position, record in enumerate(records)}
    return sorted(chosen, key=lambda record: order[record.index])
