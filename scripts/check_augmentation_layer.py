#!/usr/bin/env python3
"""
Health check for the augmentation layer, on real annotated images.

The unit tests pin each family on small synthetic frames. This script
answers a different question: run every family over the whole training
set and verify that what it does to real foam is what it promises. A
transformation that behaves on a 40x64 test frame and quietly breaks on
a 1280x960 micrograph with five hundred pores would pass the tests and
fail here.

Two kinds of finding come out of it.

**Violations** are properties that must hold on every single sample.
Each family is run on its own so that its own promise can be checked
without another family's changes in the way: photometry must leave the
annotation identical to the bit, an orientation change may rearrange
the annotation but not resize any pore, a crop and a wall may change it
but never leave a pore in two pieces or a gap in the numbering. Any
violation fails the run.

**Rates** are the figures a comparison between two policies is read
against. How often a family with a probability below one actually
fires, how often it tries and gives up, how many draws it takes when it
does - none of that is recoverable from the samples afterwards, and all
of it changes what a difference in a metric means. A family firing on
half the samples it was thought to fire on would make its measured
contribution half of what it should be, with nothing in the training
log to say so.

Both are reported line by line, and the script exits non-zero if any
expectation fails, so it can be run as a gate after a change to the
layer.

Examples
--------
Check every family over the whole training set:
    $ python scripts/check_augmentation_layer.py

Check quickly while developing:
    $ python scripts/check_augmentation_layer.py --n-images 60
"""
import argparse
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from skimage.measure import label as connected_components

from materials_vision.augmentation import (AugmentationPolicy, BlurConfig,
                                           IntegrityError, MaskAwareConfig,
                                           OrientationConfig, PolicyConfig,
                                           ScaleConfig, SeptumConfig,
                                           TonalConfig)
from materials_vision.data import SampleSource, load_split, read_manifest
from materials_vision.data.sampling import derive_seed
from materials_vision.logging_config import setup_logging

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v2/manifest_v2.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

A_MIN_FRAGMENT_PX2 = 432.0

# How far a measured firing rate may sit from the stated probability
# before it counts as a finding: four standard errors of a proportion,
# plus a little for the fact that a family can also decline to fire
# because it found nothing to work on.
RATE_TOLERANCE_SIGMA = 4.0
RATE_TOLERANCE_SLACK = 0.02

# A family that gives up this often is not doing what it is being
# compared for, even though giving up is legitimate.
MAX_FALLBACK_RATE = 0.30


@dataclass
class Tally:
    """What one policy did across every sample it saw."""

    name: str
    n_samples: int = 0
    seconds: float = 0.0
    fired: Counter = field(default_factory=Counter)
    fallbacks: Counter = field(default_factory=Counter)
    attempts: Counter = field(default_factory=Counter)
    violations: list[str] = field(default_factory=list)
    notes: Counter = field(default_factory=Counter)
    q_values: list[float] = field(default_factory=list)

    def violate(self, image_id: str, what: str) -> None:
        """Record a property that failed on one sample."""
        if len(self.violations) < 20:
            self.violations.append(f"{image_id}: {what}")


@dataclass(frozen=True)
class Probe:
    """One policy and the promise it has to keep on every sample."""

    name: str
    policy: AugmentationPolicy
    nominal: dict[str, float]
    observe: Callable[[Tally, Any, Any], None]


@dataclass(frozen=True)
class Check:
    """One expectation and whether the run met it."""

    name: str
    description: str
    passed: bool
    observed: str


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="check an evenly spaced sample; 0 checks all of them",
    )
    parser.add_argument(
        "--seed", type=int, default=20260904,
        help="run seed; every sample's seed is derived from it",
    )
    return parser.parse_args(argv)


def build_probes() -> list[Probe]:
    """Build one probe per family, plus the whole policy together.

    Each family runs on its own because its promise is only checkable
    in isolation: a mask left bitwise identical says nothing once a
    crop has legitimately changed it. The combined policy runs too,
    since the families are only ever used together and an interaction
    between two of them would show up nowhere else.
    """
    return [
        Probe(
            "orientation",
            AugmentationPolicy(
                PolicyConfig(orientation=OrientationConfig())
            ),
            {"F1_orientation": 1.0},
            observe_orientation,
        ),
        Probe(
            "scale",
            AugmentationPolicy(PolicyConfig(scale=ScaleConfig())),
            {"F2_scale": 1.0},
            observe_scale,
        ),
        Probe(
            "mask_aware",
            AugmentationPolicy(
                PolicyConfig(mask_aware=MaskAwareConfig())
            ),
            {"F4_mask_aware": MaskAwareConfig().p},
            observe_mask_aware,
        ),
        Probe(
            "septum",
            AugmentationPolicy(PolicyConfig(septum=SeptumConfig())),
            {"F5_septum": SeptumConfig().p},
            observe_septum,
        ),
        Probe(
            "photometric",
            AugmentationPolicy(
                PolicyConfig(tonal=TonalConfig(), blur=BlurConfig())
            ),
            {"F3a_tonal": TonalConfig().p, "F3b_blur": BlurConfig().p},
            observe_photometric,
        ),
        Probe(
            "full",
            AugmentationPolicy(PolicyConfig(
                scale=ScaleConfig(),
                orientation=OrientationConfig(),
                mask_aware=MaskAwareConfig(),
                septum=SeptumConfig(),
                tonal=TonalConfig(),
                blur=BlurConfig(),
            )),
            {},
            observe_common,
        ),
    ]


def observe_common(tally: Tally, prepared: Any, result: Any) -> None:
    """Check what every policy owes every sample.

    Dense numbering and connectivity are verified here as well as
    inside the policy, on purpose. A gate that trusted the checks it is
    testing would pass whenever they were the thing that broke.
    """
    image, labels = result.image, result.labels
    image_id = prepared.record.image_id
    if image.shape != labels.shape:
        tally.violate(image_id, "image and labels describe different frames")
        return
    if image.dtype != prepared.image.dtype:
        tally.violate(image_id, f"image dtype became {image.dtype}")
    if labels.dtype != prepared.labels.dtype:
        tally.violate(image_id, f"label dtype became {labels.dtype}")

    present = np.unique(labels)
    if not np.array_equal(present, np.arange(present.max() + 1)):
        tally.violate(image_id, "instance numbering has a gap")
    components = connected_components(labels, background=0, connectivity=1)
    if int(components.max()) != int(labels.max()):
        tally.violate(image_id, "an instance occupies more than one region")


def observe_orientation(
    tally: Tally, prepared: Any, result: Any
) -> None:
    """A quarter turn moves pixels; it may not resample them."""
    observe_common(tally, prepared, result)
    before = np.sort(np.bincount(prepared.labels.ravel())[1:])
    after = np.sort(np.bincount(result.labels.ravel())[1:])
    if not np.array_equal(before, after):
        tally.violate(prepared.record.image_id, "instance areas changed")


def observe_scale(tally: Tally, prepared: Any, result: Any) -> None:
    """A window may drop instances but never resample the mask."""
    observe_common(tally, prepared, result)
    image_id = prepared.record.image_id
    params = result.record.transforms[0].params
    q = params["q"]
    tally.q_values.append(q)

    if result.image.shape != prepared.image.shape:
        tally.violate(image_id, "the frame changed size")
    if q < 1.0 or q > ScaleConfig().q_max + 1e-9:
        tally.violate(image_id, f"q={q} is outside the frozen range")
    if prepared.record.scale_bin != "coarse" and q != 1.0:
        tally.violate(
            image_id,
            f"a {prepared.record.scale_bin} image was magnified by {q}",
        )
    annotated = set(np.unique(prepared.labels).tolist())
    if not set(np.unique(result.labels).tolist()) <= annotated:
        tally.violate(image_id, "the mask gained a value nobody annotated")
    if params["changed_mask"]:
        tally.notes["cropped"] += 1


def observe_mask_aware(
    tally: Tally, prepared: Any, result: Any
) -> None:
    """Shading reads the annotation and writes to none of it."""
    observe_common(tally, prepared, result)
    image_id = prepared.record.image_id
    if not np.array_equal(result.labels, prepared.labels):
        tally.violate(image_id, "the annotation changed")
    outside = prepared.labels == 0
    if not np.array_equal(result.image[outside], prepared.image[outside]):
        tally.violate(image_id, "pixels outside every pore changed")
    if result.record.transforms[0].applied:
        tally.notes[result.record.transforms[0].name or "?"] += 1


def observe_septum(tally: Tally, prepared: Any, result: Any) -> None:
    """A wall divides one pore in two and touches nothing else."""
    observe_common(tally, prepared, result)
    image_id = prepared.record.image_id
    entry = result.record.transforms[0]
    if not entry.applied or not entry.params.get("changed_mask"):
        if not np.array_equal(result.labels, prepared.labels):
            tally.violate(image_id, "the annotation changed unannounced")
        return

    tally.notes["divided"] += 1
    if int(result.labels.max()) != int(prepared.labels.max()) + 1:
        tally.violate(image_id, "the division did not add exactly one pore")
    divided = entry.params["divided_instance"]
    elsewhere = prepared.labels != divided
    if not np.array_equal(result.image[elsewhere], prepared.image[elsewhere]):
        tally.violate(image_id, "the wall was painted outside its own pore")
    if not np.array_equal(
        result.labels[elsewhere], prepared.labels[elsewhere]
    ):
        tally.violate(image_id, "another instance was disturbed")


def observe_photometric(
    tally: Tally, prepared: Any, result: Any
) -> None:
    """Brightness changes leave the annotation bitwise identical."""
    observe_common(tally, prepared, result)
    if not np.array_equal(result.labels, prepared.labels):
        tally.violate(prepared.record.image_id, "the annotation changed")


def run_probe(
    probe: Probe, source: SampleSource, indices: np.ndarray, seed: int
) -> Tally:
    """Apply one policy to every chosen sample and tally what happened.

    Parameters
    ----------
    probe : Probe
    source : SampleSource
    indices : np.ndarray
        Positions in the source to check.
    seed : int
        Run seed; each sample's seed is derived from it and its
        position, the same way training derives them.

    Returns
    -------
    Tally
    """
    tally = Tally(probe.name)
    for index in indices:
        prepared = source.load(int(index))
        started = time.perf_counter()
        try:
            result = probe.policy.apply(
                prepared.image, prepared.labels,
                record=prepared.record,
                seed=derive_seed(seed, int(index)),
            )
        except IntegrityError as error:
            tally.violate(
                prepared.record.image_id, f"stopped the run: {error}"
            )
            continue
        finally:
            tally.seconds += time.perf_counter() - started
            tally.n_samples += 1

        for entry in result.record.transforms:
            if entry.applied:
                tally.fired[entry.family] += 1
            if entry.fallback is not None:
                tally.fallbacks[f"{entry.family}: {entry.fallback}"] += 1
            tally.attempts[entry.attempts] += 1
        probe.observe(tally, prepared, result)
    return tally


def build_checks(tallies: list[Tally], probes: list[Probe]) -> list[Check]:
    """Turn the tallies into pass or fail expectations."""
    checks: list[Check] = []
    for probe, tally in zip(probes, tallies):
        checks.append(Check(
            f"{tally.name}: every sample kept every promise",
            "no sample violated a property the pipeline relies on",
            not tally.violations,
            f"{len(tally.violations)} violation(s)",
        ))
        checks.extend(_rate_checks(probe, tally))
        checks.extend(_fallback_checks(tally))
    checks.extend(_scale_checks(tallies))
    return checks


def _rate_checks(probe: Probe, tally: Tally) -> list[Check]:
    """Each family must fire about as often as it says it does."""
    checks = []
    for family, nominal in probe.nominal.items():
        measured = tally.fired[family] / max(tally.n_samples, 1)
        band = RATE_TOLERANCE_SIGMA * float(
            np.sqrt(nominal * (1.0 - nominal) / max(tally.n_samples, 1))
        ) + RATE_TOLERANCE_SLACK
        checks.append(Check(
            f"{tally.name}: {family} fires as often as it states",
            f"measured rate within {band:.3f} of {nominal}",
            abs(measured - nominal) <= band,
            f"{measured:.3f} against {nominal}",
        ))
    return checks


def _fallback_checks(tally: Tally) -> list[Check]:
    """Giving up is legitimate; giving up usually is a finding."""
    total = sum(tally.fallbacks.values())
    rate = total / max(tally.n_samples, 1)
    return [Check(
        f"{tally.name}: controlled give-ups stay rare",
        f"fallback rate at or below {MAX_FALLBACK_RATE}",
        rate <= MAX_FALLBACK_RATE,
        f"{rate:.3f} ({total} sample(s))",
    )]


def _scale_checks(tallies: list[Tally]) -> list[Check]:
    """The magnifications drawn have to match the frozen distribution."""
    scale = next(
        (tally for tally in tallies if tally.name == "scale"), None
    )
    if scale is None or not scale.q_values:
        return []
    drawn = np.array(scale.q_values)
    identity = float(np.mean(drawn == 1.0))
    return [Check(
        "scale: half the draws are the identity",
        "share of unmagnified samples between 0.40 and 0.70",
        0.40 <= identity <= 0.70,
        f"{identity:.3f}",
    )]


def report(tallies: list[Tally], checks: list[Check]) -> int:
    """Write the findings to the log and return an exit code."""
    for tally in tallies:
        per_sample = tally.seconds / max(tally.n_samples, 1) * 1000.0
        logger.info(
            "%s: %d sample(s), %.1f ms each, fired %s",
            tally.name, tally.n_samples, per_sample, dict(tally.fired),
        )
        if tally.notes:
            logger.info("%s: %s", tally.name, dict(tally.notes))
        if tally.fallbacks:
            logger.info(
                "%s: fallbacks %s", tally.name, dict(tally.fallbacks)
            )
        if sum(tally.attempts.values()) > tally.n_samples:
            logger.info("%s: attempts %s", tally.name, dict(tally.attempts))
        for violation in tally.violations:
            logger.error("%s: %s", tally.name, violation)

    failed = 0
    for check in checks:
        if check.passed:
            logger.info("PASS %s (%s)", check.name, check.observed)
        else:
            failed += 1
            logger.error(
                "FAIL %s: expected %s, got %s",
                check.name, check.description, check.observed,
            )
    logger.info(
        "%d of %d expectation(s) met.", len(checks) - failed, len(checks)
    )
    return EXIT_OK if failed == 0 else EXIT_FAILED


def main(argv: Optional[list[str]] = None) -> int:
    """Run every family over the training set and check what it did.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    split = load_split(args.split, subset="train")
    manifest = read_manifest(args.manifest)
    source = SampleSource(
        split, manifest, min_fragment_area_px2=A_MIN_FRAGMENT_PX2
    )
    step = 1 if args.n_images <= 0 else max(
        1, len(source) // args.n_images
    )
    indices = np.arange(0, len(source), step)
    logger.info("Checking %d TRAIN image(s).", indices.size)

    probes = build_probes()
    tallies = [
        run_probe(probe, source, indices, args.seed) for probe in probes
    ]
    return report(tallies, build_checks(tallies, probes))


if __name__ == "__main__":
    sys.exit(main())
