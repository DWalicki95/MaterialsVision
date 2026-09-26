#!/usr/bin/env python3
"""
How well a trained checkpoint segments the images it was held out from.

Training reports a loss, which says how well the model is optimizing
the thing it was given to optimize. That is not the same question as
how well it finds pores, and the two can disagree: a loss can improve
while instances merge, because merging two neighbouring pores costs
little in pixel overlap and everything in a count. So a checkpoint is
judged here by running it the way it will be used and comparing the
instances it produces against the annotation.

**How the instances are produced.** The trained decoder predicts, for
every pixel, whether it lies inside a pore and how far it is from that
pore's centre and from its boundary. Seeds are taken where both
distances are small, and a watershed grows them until they meet. This
is one pass over the image, in contrast to prompting the model on a
grid of points and filtering thousands of candidate masks, and it
separates touching objects far better - which is the whole difficulty
in a foam, where every pore shares its wall with the next.

The thresholds behind that step are frozen, and the frozen value is
passed in rather than inherited, so that a run's report says what it
scored under. They are read against the scale of the decoder's distance
maps, and fine-tuning moves that scale: the same snapshot can score
0.807 at a centre threshold of one half and 0.833 at three tenths, with
identical predictions underneath and only the seeding changed. A
threshold left where the untrained decoder wanted it therefore measures
that drift as well as the model.

Several thresholds can be scored in one pass. Almost all the cost of an
evaluation is the encoder and the decoder, which run once per image;
the watershed on top of them is cheap, so a second threshold costs a
fraction of the first rather than another full pass.

**What decides the winner.** Instance F1 at a mask overlap of one half,
pooled over instances rather than averaged over images. Pooling matters
because the images differ enormously in how many pores they carry, and
averaging per-image scores would give an image with twelve pores the
same weight as one with six hundred. Boundary agreement and the merge
and split counts are reported alongside, as the tie-breakers they are:
two checkpoints can reach the same F1 while one traces walls faithfully
and the other cuts corners.

**Predictions and annotation are compared on the same frame.** Both are
at content resolution, i.e. after the information panel of the second
microscope has been cropped away, so no rescaling stands between the
two sides of the comparison.

Close-up images are excluded from the figures automatically, and the
number excluded is reported: they are three to thirteen times finer
than everything else and are trained on but not scored.

**TEST is scored once, and only snapshots chosen beforehand.** It is
the one set no decision has touched, which is what makes its figures
unbiased, so it stays locked until the policy is frozen and has to be
unlocked by an explicit flag. Even then it accepts a single snapshot
per run: the snapshot is picked on VALIDATION, before TEST is opened.
Scoring several epochs of one run on TEST would let TEST pick among
them, and the figure reported would then be the best of several draws
on the set meant to measure generalization - optimistic in exactly the
way the lock exists to prevent.

Examples
--------
Score both checkpoints of the base-model pilot (the trainer nests its
output one directory deeper than the root it is given):
    $ python scripts/evaluate_checkpoint.py \\
        --checkpoint checkpoints/p0/checkpoints/vit_l_em_organelles/best.pt \\
        --checkpoint checkpoints/p0/checkpoints/vit_l_lm/best.pt

Score one checkpoint quickly, on a fraction of the images:
    $ python scripts/evaluate_checkpoint.py --checkpoint <path> --n-images 12

Score the snapshots chosen on VALIDATION on TEST, once, at the end:
    $ python scripts/evaluate_checkpoint.py --subset test --unlock-test \\
        --checkpoint checkpoints/e1/checkpoints/d4_seed20260907/epoch-10.pt
"""
import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from materials_vision.evaluation import (DECISION_SCALE, AggregateResult,
                                         WatershedParams, aggregate,
                                         cross_sections, load_size_bins,
                                         robustness_series)
from materials_vision.evaluation.inference import (build_segmenter,
                                                   score_settings)
from materials_vision.evaluation.watershed import POSTPROCESSING_CONFIGS
from materials_vision.logging_config import setup_logging
from materials_vision.tracking import (configure_tracking,
                                       find_run_for_checkpoint_dir,
                                       log_evaluation_to_run,
                                       postprocessing_id, snapshot_epoch,
                                       summarize_curve)
from materials_vision.training import (LORA_RANK, PEFT_KWARGS, build_source,
                                       prepare_geometry)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MANIFEST = Path("/home/dwalicki/dane/manifests/v3/manifest_v3.csv")

DEFAULT_SPLIT = Path("/home/dwalicki/dane/splits/split_v1.csv")

DEFAULT_SIZE_BINS = Path("/home/dwalicki/dane/splits/size_bins_v1.json")

MODEL_TYPE = "vit_l"

# Reported per checkpoint so the comparison can be read at a glance.
# Everything else the evaluation produces goes to the JSON file.
HEADLINE_FIELDS = (
    "f1", "precision", "recall", "mean_pair_iou",
    "merges_per_100_gt", "splits_per_100_gt", "pore_count_error",
)


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
    parser.add_argument(
        "--checkpoint", type=Path, action="append", default=[],
        help="Checkpoint to score; repeat to compare several.",
    )
    parser.add_argument(
        "--checkpoint-glob", action="append", default=[],
        help=(
            "Pattern selecting checkpoints, e.g. "
            "'checkpoints/e0/checkpoints/b0_*/epoch-*.pt'. Repeatable, "
            "and combinable with --checkpoint. Scoring a whole run's "
            "epochs is the usual case, and naming sixty paths by hand "
            "is both unreadable and easy to get wrong."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--size-bins", type=Path, default=DEFAULT_SIZE_BINS)
    parser.add_argument(
        "--subset", default="val", choices=["train", "val", "test"],
        help="Which split subset to score on. TEST needs --unlock-test.",
    )
    parser.add_argument(
        "--unlock-test", action="store_true",
        help="Open TEST. For the single, final evaluation only, after "
             "the policy and each run's snapshot have been fixed on "
             "VALIDATION.",
    )
    parser.add_argument(
        "--n-images", type=int, default=0,
        help="Score this many images; 0 means all of them.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the full figures as JSON.",
    )
    parser.add_argument(
        "--postprocessing", default="frozen",
        choices=sorted(POSTPROCESSING_CONFIGS),
        help=(
            "Named post-processing to score under. 'frozen' is what the "
            "augmentation study was scored under; "
            "'calibrated_2026_09_24' is part I of the optimization "
            "study. The flags below override single values of it."
        ),
    )
    parser.add_argument(
        "--also-postprocessing", action="append", default=[],
        choices=sorted(POSTPROCESSING_CONFIGS),
        help=(
            "Score under this named configuration as well, in the same "
            "pass over the images. Repeatable. Meant for measuring what "
            "a recalibration changed, on the same predictions."
        ),
    )
    parser.add_argument(
        "--center-distance-threshold", type=float,
        default=None,
        help="Seed where the predicted distance to a centre is below "
             "this. Lower seeds more selectively and splits fewer pores.",
    )
    parser.add_argument(
        "--boundary-distance-threshold", type=float,
        default=None,
        help="The same for the predicted distance to a boundary.",
    )
    parser.add_argument(
        "--foreground-threshold", type=float,
        default=None,
        help="Predicted foreground probability above which a pixel can "
             "belong to an instance.",
    )
    parser.add_argument(
        "--foreground-smoothing", type=float,
        default=None,
        help="Blur applied to the foreground map before thresholding.",
    )
    parser.add_argument(
        "--distance-smoothing", type=float,
        default=None,
        help="Blur applied to both distance maps before seeding.",
    )
    parser.add_argument(
        "--min-size", type=int, default=None,
        help="Drop instances smaller than this from the prediction.",
    )
    parser.add_argument(
        "--min-instance-area-um2", type=float,
        default=None,
        help="Drop predicted instances below this physical area, "
             "converted per image from its pixel size. 0 keeps all.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help=(
            "Do not add the figures to the MLflow runs the checkpoints "
            "were trained in. They are added by default, one point per "
            "epoch snapshot, under a name that spells out the watershed "
            "setting they were read under."
        ),
    )
    parser.add_argument(
        "--center-threshold-sweep", type=float, nargs="+", default=None,
        help=(
            "Score at each of these centre thresholds in one pass, for "
            "the robustness cross-section. Almost all the cost is the "
            "encoder and decoder, which run once per image whatever "
            "this holds."
        ),
    )
    return parser.parse_args(argv)


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Expand the patterns and drop duplicates, keeping a stable order.

    Sorted so that a run's epochs are scored in the order they were
    trained, which is the order the resulting curve has to be read in;
    the numeric part is sorted as a number, so that epoch 2 does not
    follow epoch 19.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    list of Path
    """
    found = list(args.checkpoint)
    for pattern in args.checkpoint_glob:
        found.extend(Path().glob(pattern))

    unique = list(dict.fromkeys(found))
    return sorted(unique, key=_training_order)


def _training_order(path: Path) -> tuple[str, int, str]:
    """Sort key placing a run's snapshots in the order trained.

    Parameters
    ----------
    path : Path

    Returns
    -------
    tuple
        Run directory, epoch number (zero when the name carries none),
        then the name itself to keep the order total.
    """
    match = re.search(r"epoch-(\d+)", path.stem)
    return (path.parent.name, int(match.group(1)) if match else 0, path.name)


def locked_test_reason(args: argparse.Namespace) -> Optional[str]:
    """Say why TEST may not be scored as asked, or nothing if it may.

    Checked before any model is loaded, so a refused request costs
    nothing and leaves no partial result behind that could tempt
    anyone to read it.

    Parameters
    ----------
    args : argparse.Namespace
        With ``checkpoint`` already resolved to paths.

    Returns
    -------
    str or None
        The reason for refusing, or ``None`` when scoring may go ahead.
        Always ``None`` for a subset other than TEST.
    """
    if args.subset != "test":
        return None
    if not args.unlock_test:
        return (
            "TEST is locked. Pass --unlock-test, and only for the single "
            "final evaluation of snapshots already chosen on VALIDATION."
        )
    per_run = Counter(path.parent.name for path in args.checkpoint)
    repeated = sorted(run for run, count in per_run.items() if count > 1)
    if repeated:
        return (
            f"TEST scores one snapshot per run, chosen on VALIDATION "
            f"beforehand. Several were given for {', '.join(repeated)}, "
            f"which would let TEST choose among them."
        )
    return None


def resolve_settings(
    args: argparse.Namespace
) -> tuple[WatershedParams, ...]:
    """Work out which watershed settings to score under.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    tuple of WatershedParams
    """
    fields = POSTPROCESSING_CONFIGS[args.postprocessing].to_kwargs()
    for name in fields:
        given = getattr(args, name, None)
        if given is not None:
            fields[name] = given
    base = WatershedParams(**fields)
    settings = (
        list(robustness_series(base, tuple(args.center_threshold_sweep)))
        if args.center_threshold_sweep else [base]
    )
    for name in args.also_postprocessing:
        extra = POSTPROCESSING_CONFIGS[name]
        if extra not in settings:
            settings.append(extra)
    return tuple(settings)


def score_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
    settings: tuple[WatershedParams, ...],
) -> list[tuple[WatershedParams, AggregateResult, list]]:
    """Segment every image of a subset and measure the result.

    Parameters
    ----------
    checkpoint : Path
    args : argparse.Namespace
    settings : tuple of WatershedParams
        Scored together, on one pass over the images.

    Returns
    -------
    list of tuple
        Per setting: the setting, the pooled figures, and the per-image
        evaluations behind them.
    """
    source = build_source(
        args.split, args.manifest, args.subset,
        allow_test=args.unlock_test,
    )
    size_bins = load_size_bins(args.size_bins)
    segmenter = build_segmenter(
        checkpoint, MODEL_TYPE, PEFT_KWARGS, LORA_RANK
    )

    positions = list(range(len(source)))
    if 0 < args.n_images < len(source):
        stride = len(source) / args.n_images
        positions = [int(step * stride) for step in range(args.n_images)]

    started_s = time.perf_counter()
    evaluations = score_settings(
        segmenter, source, positions, size_bins, settings,
        boundary_scales=(DECISION_SCALE,),
    )
    elapsed_s = time.perf_counter() - started_s
    logger.info(
        "%s: %d image(s) x %d setting(s) in %.1f s (%.2f s each).",
        checkpoint, len(positions), len(settings), elapsed_s,
        elapsed_s / max(1, len(positions)),
    )
    return [
        (
            setting,
            aggregate(
                evaluations[setting],
                label=f"{checkpoint}@{setting.label()}",
            ),
            evaluations[setting],
        )
        for setting in settings
    ]


def report(results: dict[str, AggregateResult]) -> None:
    """Write the headline figures for every checkpoint scored.

    One row per checkpoint rather than one column, because the usual
    case is no longer two checkpoints side by side but every epoch of
    a run, read as a curve.

    Parameters
    ----------
    results : dict
        Pooled figures per checkpoint.
    """
    header = "  ".join(f"{field:>10.10}" for field in HEADLINE_FIELDS)
    logger.info("%-44s %s", "checkpoint", header)
    for name, result in results.items():
        values = "  ".join(
            f"{getattr(result, field):10.4f}" for field in HEADLINE_FIELDS
        )
        logger.info("%-44s %s", name, values)

    counts = {
        (result.n_images, result.n_gt, result.n_scale_outliers_excluded)
        for result in results.values()
    }
    if len(counts) == 1:
        n_images, n_gt, n_excluded = counts.pop()
        logger.info(
            "All scored on the same %d image(s), %d annotated "
            "instance(s), %d close-up(s) excluded.",
            n_images, n_gt, n_excluded,
        )
    else:
        # A differing population would make the scores incomparable,
        # so it is worth saying loudly rather than leaving to be
        # noticed in the JSON.
        logger.warning(
            "Checkpoints were scored on differing populations: %s",
            sorted(counts),
        )

    if len(results) < 2:
        return
    ranked = sorted(
        results.items(), key=lambda item: item[1].f1, reverse=True
    )
    best, second = ranked[0], ranked[1]
    logger.info(
        "Best on instance F1: %s at %.4f, ahead of %s at %.4f by %.4f.",
        best[0], best[1].f1, second[0], second[1].f1,
        best[1].f1 - second[1].f1,
    )


def mlflow_refusal(args: argparse.Namespace) -> Optional[str]:
    """Say why figures must stay out of MLflow, or nothing if they may go.

    A partial evaluation covers a different population of images from a
    full one, and a point from it on the same curve would read as the
    model changing when only the sample did.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str or None
    """
    if args.no_mlflow:
        return "disabled on the command line"
    if args.n_images > 0:
        return (
            f"only {args.n_images} image(s) scored, which would put a "
            f"different population on the same curve"
        )
    return None


def record_in_mlflow(
    checkpoint: Path,
    subset: str,
    scored: list[tuple[WatershedParams, AggregateResult, list]],
) -> Optional[str]:
    """Add one snapshot's figures to the run it was trained in.

    Parameters
    ----------
    checkpoint : Path
    subset : str
    scored : list of tuple
        From :func:`score_checkpoint`.

    Returns
    -------
    str or None
        The run written to, or ``None`` when the snapshot has no place
        on a curve or its run is not in the store.
    """
    epoch = snapshot_epoch(checkpoint)
    if epoch is None:
        logger.info(
            "%s carries no epoch in its name; not added to MLflow.",
            checkpoint,
        )
        return None
    run_id = find_run_for_checkpoint_dir(checkpoint.parent)
    if run_id is None:
        logger.warning(
            "No MLflow run is tagged with %s; its figures stay in the "
            "JSON report only.", checkpoint.parent,
        )
        return None
    for setting, result, per_image in scored:
        sections = (
            cross_sections(per_image, "material")
            + cross_sections(per_image, "scale_bin")
        )
        written = log_evaluation_to_run(
            run_id, epoch, subset, postprocessing_id(setting), result,
            sections,
        )
        logger.info(
            "%s: %d figure(s) added to MLflow run %s at epoch %d under "
            "%s.", checkpoint, written, run_id, epoch,
            postprocessing_id(setting),
        )
    return run_id


def write_json(
    destination: Path, results: dict[str, AggregateResult], evaluations: dict
) -> None:
    """Save the full figures, including every cross-section.

    Parameters
    ----------
    destination : Path
    results : dict
    evaluations : dict
        Per-image evaluations per checkpoint, used for the
        cross-sections.
    """
    from dataclasses import asdict

    payload = {}
    for name, result in results.items():
        payload[name] = {
            "overall": asdict(result),
            "per_material": [
                asdict(section)
                for section in cross_sections(evaluations[name], "material")
            ],
            "per_scale_bin": [
                asdict(section)
                for section in cross_sections(evaluations[name], "scale_bin")
            ],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s.", destination)


def main(argv: Optional[list[str]] = None) -> int:
    """Score every checkpoint given and compare them.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    args.checkpoint = resolve_checkpoints(args)
    if not args.checkpoint:
        logger.error(
            "No checkpoint selected. Pass --checkpoint or "
            "--checkpoint-glob."
        )
        return EXIT_FAILED
    missing = [path for path in args.checkpoint if not path.exists()]
    if missing:
        logger.error(
            "No such checkpoint: %s.",
            ", ".join(str(path) for path in missing),
        )
        return EXIT_FAILED
    refusal = locked_test_reason(args)
    if refusal is not None:
        logger.error(refusal)
        return EXIT_FAILED
    logger.info(
        "Scoring %d checkpoint(s) on %s.",
        len(args.checkpoint), args.subset.upper(),
    )

    prepare_geometry()
    settings = resolve_settings(args)
    logger.info(
        "Watershed setting(s): %s.",
        ", ".join(setting.label() for setting in settings),
    )
    refusal = mlflow_refusal(args)
    if refusal is None:
        configure_tracking()
    else:
        logger.info("Figures are not added to MLflow: %s.", refusal)
    results: dict[str, AggregateResult] = {}
    evaluations: dict[str, list] = {}
    tracked_runs: set[str] = set()
    for checkpoint in args.checkpoint:
        # Run and snapshot both, because every epoch of one run lives
        # in a single directory: the directory alone would name them
        # all the same.
        stem = f"{checkpoint.parent.name}/{checkpoint.stem}"
        scored = score_checkpoint(checkpoint, args, settings)
        for setting, result, per_image in scored:
            # The setting only enters the name when there is more than
            # one, so an ordinary run's report reads as it always did.
            name = stem if len(scored) == 1 else f"{stem}@{setting.label()}"
            results[name], evaluations[name] = result, per_image
        if refusal is None:
            # The figures that matter go to the JSON report; the MLflow
            # copy is for viewing. Several evaluations may write to the
            # store at once, and a busy store must not cost the report.
            try:
                run_id = record_in_mlflow(checkpoint, args.subset, scored)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s: writing to MLflow failed; the JSON report is "
                    "unaffected.", checkpoint, exc_info=True,
                )
                run_id = None
            if run_id is not None:
                tracked_runs.add(run_id)

    for run_id in sorted(tracked_runs):
        for setting in settings:
            try:
                summarize_curve(
                    run_id, args.subset, postprocessing_id(setting)
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Summarizing run %s in MLflow failed.", run_id,
                    exc_info=True,
                )
    report(results)
    if args.out is not None:
        write_json(args.out, results, evaluations)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
