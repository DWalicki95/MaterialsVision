#!/usr/bin/env python3
"""
CLI entry point for turning the Phase 0 review into a decision sheet.

The review page records two things: a verdict per panel and a verdict
per family and strength setting. What the experiment needs from them is
one status per family, the parameter ranges that status applies to, and
the evidence behind it - which images failed, on which criterion. That
reduction and the rules behind it are in
``materials_vision.phase0.review``.

Two files come out. The YAML is the decision sheet in the shape the
plan's template uses, and belongs in the repository: it is the record
that a family was admitted to the experiment and on what basis. The
Markdown report is for reading, and states plainly what is still
unreviewed, since a family nobody has finished looking at must not
appear as accepted.

Run it whenever you want to see where the review stands; it reads and
writes nothing the reviewer is using.

Examples
--------
Summarize the default review directory:
    $ python scripts/summarize_phase0_review.py

Check progress without writing anything:
    $ python scripts/summarize_phase0_review.py --dry-run
"""
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from materials_vision.logging_config import setup_logging
from materials_vision.phase0.review import (STATUS_ACCEPTED, STATUS_PENDING,
                                            LevelSummary, decision_sheet,
                                            family_status, load_panels,
                                            load_review, summarize)
from materials_vision.phase0.viewer import CRITERIA

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 2

DEFAULT_REVIEW_DIR = Path("/home/dwalicki/dane/faza0")

DEFAULT_SHEET = Path(".claude/phase0_decisions.yaml")

DEFAULT_REPORT = Path(".claude/phase0_report.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir", type=Path, default=DEFAULT_REVIEW_DIR
    )
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report progress without writing the sheet",
    )
    return parser.parse_args(argv)


def build_report(
    summaries: list[LevelSummary], review_dir: Path
) -> str:
    """Write the review out in the form a person checks it in."""
    by_family: dict[str, list[LevelSummary]] = {}
    for summary in summaries:
        by_family.setdefault(summary.family, []).append(summary)

    decided = sum(s.n_decided for s in summaries)
    panels = sum(s.n_panels for s in summaries)
    lines = [
        "# Faza 0 - stan przegladu",
        "",
        f"Zrodlo: `{review_dir}`. Wygenerowano "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        f"Ocenionych plansz: **{decided}/{panels}**.",
        "",
        "## Status rodzin",
        "",
        "| rodzina | status | plansze | problemy | poziomy bramkujace |",
        "|---|---|---:|---:|---|",
    ]
    for family, group in by_family.items():
        status = family_status(group)
        gated = [s for s in group if s.kind == "gate"]
        lines.append(
            f"| {family} | **{status}** | "
            f"{sum(s.n_decided for s in group)}/"
            f"{sum(s.n_panels for s in group)} | "
            f"{sum(s.n_problems for s in group)} | "
            + ", ".join(f"{s.level}={s.status}" for s in gated)
            + " |"
        )

    lines += [
        "",
        "## Poziomy",
        "",
        "| rodzina / poziom | rodzaj | ocenione | problemy | werdykt | "
        "uzasadnienie |",
        "|---|---|---:|---:|---|---|",
    ]
    for summary in summaries:
        stale = " (nieaktualny)" if summary.stale else ""
        lines.append(
            f"| {summary.family} / {summary.level} | {summary.kind} | "
            f"{summary.n_decided}/{summary.n_panels} | "
            f"{summary.n_problems} | {summary.status}{stale} | "
            f"{summary.reason} |"
        )

    problems = [s for s in summaries if s.n_problems]
    if problems:
        lines += ["", "## Zgloszone problemy", ""]
        for summary in problems:
            named = ", ".join(
                f"{CRITERIA[int(index)]} ({count})"
                for index, count in summary.criteria.most_common()
            )
            lines.append(
                f"- **{summary.family} / {summary.level}**: "
                f"{summary.n_problems} plansz(y) - {named}; obrazy: "
                + ", ".join(sorted(set(summary.problem_images)))
            )

    pending = [
        s for s in summaries
        if s.status == STATUS_PENDING and s.kind == "gate"
    ]
    if pending:
        lines += [
            "",
            "## Czego brakuje do zamkniecia bramki",
            "",
        ]
        for summary in pending:
            missing = summary.n_panels - summary.n_decided
            why = (
                f"{missing} plansz(y) bez oceny" if missing
                else "brak werdyktu rodziny"
            )
            if summary.stale:
                why = "werdykt zapadl przy innych parametrach"
            lines.append(f"- {summary.family} / {summary.level}: {why}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Summarize the review and write the decision sheet.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging()

    index_path = args.review_dir / "panels.json"
    if not index_path.exists():
        logger.error(
            "%s holds no panels.json; nothing to summarize.",
            args.review_dir,
        )
        return EXIT_FATAL

    panels = load_panels(index_path)
    review = load_review(args.review_dir / "review.json")
    summaries = summarize(panels, review)
    sheet = decision_sheet(summaries)

    accepted = [
        family for family, record in sheet.items()
        if record["faza0_wizualna"]["status"] == STATUS_ACCEPTED
    ]
    logger.info(
        "%d/%d panel(s) reviewed; %d of %d family/families accepted "
        "(%s).",
        sum(s.n_decided for s in summaries),
        sum(s.n_panels for s in summaries),
        len(accepted), len(sheet), ", ".join(accepted) or "none",
    )

    report = build_report(summaries, args.review_dir)
    if args.dry_run:
        logger.info("Dry run; nothing written.\n%s", report)
        return EXIT_OK

    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sheet, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            sheet, handle, allow_unicode=True, sort_keys=False,
            default_flow_style=False,
        )
    args.report.write_text(report, encoding="utf-8")
    logger.info(
        "Decision sheet written to %s, report to %s.",
        args.sheet, args.report,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
