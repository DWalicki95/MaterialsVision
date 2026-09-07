#!/usr/bin/env python3
"""
Whether preparing a training sample costs anything in wall-clock time.

Training here runs one image per optimizer step. Each step therefore
needs exactly one prepared sample: an image read from disk, stripped of
the microscope's information panel, augmented, and turned into the
targets the instance decoder regresses against. That preparation runs
on CPU worker processes while the GPU works on the previous step, so it
costs real time only if the workers cannot keep up. If they can, the
augmentation policy is free no matter how elaborate it is, and there is
nothing to trade off against the quality it buys.

The measurement is a comparison of two rates.

**Supply rate** - how long the loader needs per prepared sample,
measured by consuming it as fast as possible and dividing the elapsed
time by the samples delivered. Measured twice, with augmentation and
without, so the augmentation's share is visible rather than inferred.
Averaging over the whole pass rather than taking a typical gap between
arrivals is deliberate: workers run ahead of the consumer, so samples
come out in bursts separated by stalls, and the gap inside a burst says
nothing about the rate the loader can sustain.

**Step cost** - how long the GPU spends on one optimizer step. This is
measured as a *lower bound*, and deliberately so: a lower bound is
enough to settle the question and is far cheaper to obtain than the
real thing. One step runs the image encoder for the interactive branch,
then eight iterative passes of the mask decoder, then the instance
decoder, with a backward pass after each branch. Measured here are only
the two encoder-bearing forward/backward passes; the mask decoder's
iterations and the prompt sampling around them can only add to the
total. So if the supply interval already sits below this bound, it sits
below the true step cost too, and the conclusion is safe. If it does
not, the bound was too crude to decide and the real step has to be
timed inside a training run instead.

**Why the two branches are timed separately and then added.** They are
not two ways of measuring the same forward pass. A step really does run
the image encoder twice - once directly, for the prompt-driven masks,
and once again inside the instance decoder, which uses the same encoder
as its backbone. Adding the two is faithful to the step, not double
counting.

**Why the geometry correction is installed first.** The library that
resizes an image to the encoder's square canvas computes the target
size from the wrong two axes of the tensor, which squeezes every
three-channel image to a fixed 341 by 1024 regardless of its real
shape. Both the canvas and, therefore, the encoder's cost come out the
same either way, because the remainder is padded to the full square -
so this is not a correction that changes the timing. It is installed
because a benchmark should exercise the geometry the training will
actually use, and because reading the geometry back out of the library
on a real model is worth doing once before relying on it.

**Worker count.** Four, fixed, not swept. It cannot change what a run
produces: the seed for a sample is derived from its position in the
epoch rather than from the order in which workers happen to finish, so
the same run reproduces bit for bit at any worker count. It is fixed
rather than tuned so that a supply interval measured today can be
compared with one measured after a change to the policy.

**What this does not measure.** How often a family declines to fire and
falls back to an untransformed sample - that is a property of the
policy rather than of its cost, it needs the per-sample records which
do not survive the trip out of a worker process, and it is already
checked over the whole training set by check_augmentation_layer.py.

Examples
--------
Measure both halves over the whole training set:
    $ python scripts/benchmark_dataloader.py

Measure only the loader, on a machine with no GPU:
    $ python scripts/benchmark_dataloader.py --skip-step

Try it quickly while developing:
    $ python scripts/benchmark_dataloader.py --n-images 60
"""
import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset

from materials_vision.augmentation import (AugmentationPolicy, BlurConfig,
                                           MaskAwareConfig, OrientationConfig,
                                           PolicyConfig, ScaleConfig,
                                           SeptumConfig, TonalConfig)
from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.data.dataset import (InstanceSegmentationDataset,
                                           build_label_transform)
from materials_vision.logging_config import setup_logging
from materials_vision.sam_geometry import (CONTENT_GEOMETRIES,
                                           patch_resize_longest_side,
                                           verify_preprocess_geometry)

logger = logging.getLogger(__name__)

# What the loader is fed: either the training dataset itself or an
# evenly spaced slice of it. Named because torch's base Dataset makes
# no promise about having a length, and both of these do.
SampleFeed = Union[InstanceSegmentationDataset, Subset]

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

A_MIN_FRAGMENT_PX2 = 388.43

# Fixed rather than tuned; see the module docstring.
NUM_WORKERS = 4

# Samples discarded before timing starts. The first batches out of a
# freshly started loader are the ones its workers prepared while the
# main process was still setting up, so they arrive at a rate the
# loader cannot sustain.
WARMUP_SAMPLES = 100

# Repeats per geometry when timing the GPU, after a discarded first
# pass. Two passes differ by well under the margin this benchmark has
# to resolve, so a handful is enough.
STEP_REPEATS = 10

MODEL_TYPE = "vit_l"

# The encoder is adapted by low-rank updates to the query and value
# projections of every attention block; the prompt encoder and the
# mask decoder are left as they are. The instance decoder trains in
# full. This mirrors the training configuration because a step timed
# under a different one would be timing a different model.
PEFT_KWARGS = {
    "rank": 8,
    "update_matrices": ["q", "v"],
    "attention_layers_to_update": [],
    "quantize": False,
}

FREEZE_PARTS = ["prompt_encoder", "mask_decoder"]

# The trainer runs its forward passes in half precision, so a step
# timed in full precision would be timing a slower model than the one
# that will run. The gradient scaler that accompanies it in training is
# left out: it guards the numerics, and the numbers here are discarded.
AUTOCAST_DTYPE = torch.float16

GREY_LEVELS = 255.0

MS_PER_S = 1000.0

BYTES_PER_GB = 1024 ** 3


@dataclass
class Supply:
    """How fast one loader configuration delivered prepared samples."""

    name: str
    intervals_ms: list[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cpu_seconds: float = 0.0
    peak_rss_gb: float = 0.0

    @property
    def n_measured(self) -> int:
        """Number of timed samples, warm-up excluded.

        Returns
        -------
        int
        """
        return len(self.intervals_ms)

    @property
    def sustained_ms(self) -> float:
        """Seconds of wall clock the loader needs, per sample.

        This is the number the comparison rests on: total elapsed time
        divided by samples delivered, which is what "how fast can this
        loader feed a training run" means.

        It is emphatically not the median gap between arrivals. Workers
        prepare several samples ahead of the consumer, so a consumer
        that does no work drains the queue in a burst and then waits
        for it to refill. The gaps are bimodal - near zero inside a
        burst, long between bursts - and their median reports the
        burst, which gets faster the more expensive preparation is,
        because slower samples make longer queues. Averaging over the
        whole pass is immune to that: the bursts and the stalls are
        both inside it.

        Returns
        -------
        float
        """
        return self.elapsed_ms / self.n_measured

    @property
    def slowest_ms(self) -> float:
        """Longest single wait for a sample.

        Reported because a loader that keeps up on average can still
        stall on the occasional image carrying several hundred pores.

        Returns
        -------
        float
        """
        return max(self.intervals_ms)


@dataclass
class StepCost:
    """Lower bound on one optimizer step, per input geometry."""

    geometry_px: tuple[int, int]
    encoder_ms: float
    decoder_ms: float

    @property
    def total_ms(self) -> float:
        """The two encoder-bearing passes a step performs.

        Returns
        -------
        float
        """
        return self.encoder_ms + self.decoder_ms


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="Use a subset of this size; 0 means the whole set.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260907,
        help="Run seed, which fixes the augmentation draws.",
    )
    parser.add_argument(
        "--skip-supply", action="store_true",
        help="Skip the loader measurement.",
    )
    parser.add_argument(
        "--skip-step", action="store_true",
        help="Skip the GPU measurement.",
    )
    return parser.parse_args(argv)


def build_full_policy() -> AugmentationPolicy:
    """The complete augmentation policy, every family enabled.

    The most expensive configuration that will ever run, which is the
    one worth timing: if it is hidden behind the GPU then so is every
    subset of it.

    Returns
    -------
    AugmentationPolicy
    """
    return AugmentationPolicy(PolicyConfig(
        scale=ScaleConfig(),
        orientation=OrientationConfig(),
        mask_aware=MaskAwareConfig(),
        septum=SeptumConfig(),
        tonal=TonalConfig(),
        blur=BlurConfig(),
    ))


def build_dataset(
    source: SampleSource,
    policy: Optional[AugmentationPolicy],
    run_seed: int,
) -> InstanceSegmentationDataset:
    """Wrap a sample source the way training wraps it.

    Parameters
    ----------
    source : SampleSource
    policy : AugmentationPolicy, optional
        ``None`` gives the unaugmented condition.
    run_seed : int

    Returns
    -------
    InstanceSegmentationDataset
    """
    return InstanceSegmentationDataset(
        source,
        label_transform=build_label_transform(),
        transform=policy,
        run_seed=run_seed,
    )


def _limit(dataset: SampleFeed, n_images: int) -> SampleFeed:
    """Take an evenly spaced subset, or the whole set.

    Spaced rather than taken from the front so that a short run still
    sees both microscopes and both scale bins, which sit in different
    parts of the ordering and take different amounts of work.

    Parameters
    ----------
    dataset : InstanceSegmentationDataset or Subset
    n_images : int
        Zero or less means the whole set.

    Returns
    -------
    Dataset
    """
    if n_images <= 0:
        return dataset
    stride = max(1, len(dataset) // n_images)
    return Subset(dataset, np.arange(0, len(dataset), stride).tolist())


def measure_supply(name: str, dataset: SampleFeed) -> Supply:
    """Time how fast the loader delivers prepared samples.

    Consumes the loader as fast as the main process can, which is what
    makes the result a supply rate rather than a measure of whatever
    the consumer happens to be doing.

    Parameters
    ----------
    name : str
        Label for the report.
    dataset : InstanceSegmentationDataset or Subset

    Returns
    -------
    Supply

    Raises
    ------
    RuntimeError
        If the dataset is too small to leave anything after warm-up.
    """
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
    )
    supply = Supply(name)
    parent = psutil.Process()
    window_started_s = 0.0
    previous_s = time.perf_counter()
    for position, _ in enumerate(loader):
        now_s = time.perf_counter()
        if position == WARMUP_SAMPLES:
            window_started_s = now_s
        elif position > WARMUP_SAMPLES:
            supply.intervals_ms.append((now_s - previous_s) * MS_PER_S)
            supply.elapsed_ms = (now_s - window_started_s) * MS_PER_S
        previous_s = now_s
        if position % 50 == 0:
            _sample_resources(parent, supply)
    _sample_resources(parent, supply)

    if not supply.intervals_ms:
        raise RuntimeError(
            f"{name}: every sample fell inside the {WARMUP_SAMPLES}-sample "
            f"warm-up, so nothing was timed. Use more images."
        )
    logger.info(
        "%s: %d sample(s) timed, %.1f ms each, slowest %.1f ms.",
        name, supply.n_measured, supply.sustained_ms, supply.slowest_ms,
    )
    return supply


def _sample_resources(parent: psutil.Process, supply: Supply) -> None:
    """Record CPU and memory across the loader's process tree.

    Sampled during the pass rather than after it: the worker processes
    exist only while the loader is being iterated, and their usage
    disappears with them.

    Parameters
    ----------
    parent : psutil.Process
    supply : Supply
        Updated in place.
    """
    tree = [parent] + parent.children(recursive=True)
    cpu_seconds = 0.0
    rss_bytes = 0
    for process in tree:
        try:
            times = process.cpu_times()
            cpu_seconds += times.user + times.system
            rss_bytes += process.memory_info().rss
        except psutil.Error:
            continue
    supply.cpu_seconds = cpu_seconds
    supply.peak_rss_gb = max(
        supply.peak_rss_gb, rss_bytes / BYTES_PER_GB
    )


def build_model(device: torch.device):
    """Build the model a training run would build.

    The geometry correction is installed before the model exists and
    verified afterwards, because both the correction and the check read
    the library's own resizing function: installing it late would leave
    a model already holding the uncorrected one.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    tuple
        The trainable SAM wrapper and the instance decoder.
    """
    import micro_sam.training as sam_training
    from micro_sam.instance_segmentation import get_unetr

    patch_resize_longest_side()
    measured = verify_preprocess_geometry()
    for geometry_px, content_px in measured.items():
        logger.info(
            "Content %dx%d reaches the encoder as %dx%d.",
            geometry_px[0], geometry_px[1], content_px[0], content_px[1],
        )

    model, state = sam_training.get_trainable_sam_model(
        model_type=MODEL_TYPE,
        device=device,
        freeze=FREEZE_PARTS,
        return_state=True,
        peft_kwargs=dict(PEFT_KWARGS),
    )
    decoder = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device=device,
    )
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) + sum(
        p.numel() for name, p in decoder.named_parameters()
        if p.requires_grad and not name.startswith("encoder")
    )
    logger.info("Trainable parameters: %d.", trainable)
    return model, decoder


def measure_step_cost(
    model, decoder, geometry_px: tuple[int, int], device: torch.device
) -> StepCost:
    """Time the two encoder-bearing passes of one step.

    Parameters
    ----------
    model : torch.nn.Module
        Trainable SAM wrapper.
    decoder : torch.nn.Module
        Instance decoder, which carries the same image encoder.
    geometry_px : tuple of int
        ``(height, width)`` of the image content, before the model
        resizes it to its own canvas.
    device : torch.device

    Returns
    -------
    StepCost
    """
    model.train()
    decoder.train()
    # On the source grey scale, which is what the loader emits and what
    # both branches expect: each normalizes with the model's own
    # statistics, and the instance decoder refuses input that has
    # already been scaled to the unit interval. The content is noise
    # because cost depends on the size of the tensors, not on what is
    # in them.
    image = torch.rand(
        (1, 3, geometry_px[0], geometry_px[1]), device=device
    ) * GREY_LEVELS

    def encoder_loss() -> torch.Tensor:
        embeddings, _ = model.image_embeddings_oft([{"image": image[0]}])
        return embeddings.float().mean()

    def decoder_loss() -> torch.Tensor:
        return decoder(image).float().mean()

    return StepCost(
        geometry_px=geometry_px,
        encoder_ms=_time_pass(encoder_loss, model, decoder),
        decoder_ms=_time_pass(decoder_loss, model, decoder),
    )


def _time_pass(forward, model, decoder) -> float:
    """Median wall-clock cost of one forward and backward pass.

    The value the forward pass returns stands in for the training
    loss. What it is does not matter, only that it depends on every
    trainable parameter the real loss depends on, so that the backward
    pass traverses the same graph and costs the same.

    Parameters
    ----------
    forward : Callable
        Runs one forward pass and returns a scalar to differentiate.
    model, decoder : torch.nn.Module
        Zeroed between repeats so that gradients do not accumulate
        across a measurement and change what the backward pass costs.

    Returns
    -------
    float
        Milliseconds.
    """
    durations_ms = []
    for repeat in range(STEP_REPEATS + 1):
        model.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started_s = time.perf_counter()
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            loss = forward()
        loss.backward()
        torch.cuda.synchronize()
        if repeat > 0:
            durations_ms.append(
                (time.perf_counter() - started_s) * MS_PER_S
            )
    return statistics.median(durations_ms)


def report(supplies: list[Supply], steps: list[StepCost]) -> None:
    """Write both measurements and, if both ran, the comparison.

    Parameters
    ----------
    supplies : list of Supply
    steps : list of StepCost
    """
    for supply in supplies:
        logger.info(
            "SUPPLY %-14s %7.1f ms/sample  slowest %7.1f ms  "
            "cpu %6.1f s  peak rss %.1f GB  (n = %d)",
            supply.name, supply.sustained_ms, supply.slowest_ms,
            supply.cpu_seconds, supply.peak_rss_gb, supply.n_measured,
        )
    for step in steps:
        logger.info(
            "STEP   %-14s encoder %7.1f ms  decoder %7.1f ms  "
            "lower bound %7.1f ms",
            f"{step.geometry_px[0]}x{step.geometry_px[1]}",
            step.encoder_ms, step.decoder_ms, step.total_ms,
        )

    if len(supplies) == 2:
        augmented, plain = supplies[0], supplies[1]
        logger.info(
            "Augmentation adds %.1f ms per sample "
            "(%.1f ms with, %.1f ms without).",
            augmented.sustained_ms - plain.sustained_ms,
            augmented.sustained_ms, plain.sustained_ms,
        )

    if not supplies or not steps:
        logger.info(
            "Only one half was measured, so there is no comparison to "
            "draw. Run both to decide whether preparation is hidden."
        )
        return

    slowest_supply_ms = max(supply.sustained_ms for supply in supplies)
    cheapest_step_ms = min(step.total_ms for step in steps)
    if slowest_supply_ms < cheapest_step_ms:
        logger.info(
            "HIDDEN: a sample arrives every %.1f ms, below the %.1f ms "
            "lower bound on a step, so preparation never keeps the GPU "
            "waiting and augmentation costs no wall-clock time.",
            slowest_supply_ms, cheapest_step_ms,
        )
    else:
        logger.info(
            "NOT SETTLED: a sample arrives every %.1f ms, at or above "
            "the %.1f ms lower bound on a step. The bound is too crude "
            "to decide; time a real step inside a training run, and "
            "split the interval per family to see which one pays for "
            "it.",
            slowest_supply_ms, cheapest_step_ms,
        )


def main(argv: Optional[list[str]] = None) -> int:
    """Measure sample supply against step cost.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    supplies: list[Supply] = []
    steps: list[StepCost] = []

    if not args.skip_supply:
        split = load_split(args.split, subset="train")
        manifest = read_manifest(args.manifest)
        source = SampleSource(
            split, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
        )
        augmented = _limit(
            build_dataset(source, build_full_policy(), args.seed),
            args.n_images,
        )
        plain = _limit(
            build_dataset(source, None, args.seed), args.n_images
        )
        logger.info(
            "Timing the loader over %d TRAIN image(s), %d worker(s).",
            len(augmented), NUM_WORKERS,
        )
        # Augmented first, on purpose. The second pass reads the same
        # files and finds them in the page cache, so whichever runs
        # second looks faster than it is. Putting augmentation first
        # makes the measurement overstate its cost rather than hide it.
        supplies.append(measure_supply("full", augmented))
        supplies.append(measure_supply("no augmentation", plain))

    if not args.skip_step:
        if not torch.cuda.is_available():
            logger.error(
                "No GPU is visible, so a step cannot be timed. Re-run "
                "with --skip-step to measure the loader alone."
            )
            return EXIT_FAILED
        device = torch.device("cuda")
        logger.info("Timing steps on %s.", torch.cuda.get_device_name(0))
        model, decoder = build_model(device)
        for geometry_px in CONTENT_GEOMETRIES:
            steps.append(
                measure_step_cost(model, decoder, geometry_px, device)
            )

    report(supplies, steps)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
