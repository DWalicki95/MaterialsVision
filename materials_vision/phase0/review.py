"""
Turning a stack of judged panels into one decision per family.

The reviewer works panel by panel, but the experiment does not: what it
needs is a single status per family, the parameter ranges that status
applies to, and the evidence behind it. This module performs that
reduction, and the rules it reduces by are the ones that keep the
result honest.

**A family is only as good as its worst gated setting.** The
acceptance criteria talk about the maximum strength, so a family whose
strong setting was rejected is rejected, whatever its weak setting
looked like. Accepted requires every gated setting to be accepted;
anything still undecided leaves the family pending, never accepted by
default.

**Diagnostic settings inform, they do not gate.** The punishing patch
and the faintest wall are deliberately outside the frozen ranges. A
problem found there says where the family breaks, which is what a
revision needs to know, but it cannot reject numbers nobody proposed
using.

**A verdict expires when the numbers change.** Every verdict carries
the fingerprint of the parameters it was made against. Re-render with a
widened range and the fingerprints move; the old verdicts are then
reported as stale rather than silently applied to settings nobody
looked at.
"""
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from materials_vision.phase0.levels import KIND_GATE, review_levels
from materials_vision.phase0.viewer import CRITERIA

logger = logging.getLogger(__name__)

STATUS_ACCEPTED = "accepted"
STATUS_REVISE = "revise"
STATUS_REJECTED = "rejected"
STATUS_PENDING = "pending"

# Worst wins: one rejected setting rejects the family, and anything
# unreviewed keeps it pending rather than letting the others carry it.
STATUS_SEVERITY = {
    STATUS_ACCEPTED: 0,
    STATUS_REVISE: 1,
    STATUS_REJECTED: 2,
    STATUS_PENDING: 3,
}


@dataclass
class LevelSummary:
    """One family at one setting, with the evidence for its verdict.

    Parameters
    ----------
    key : str
        ``family__level``.
    family, level, kind : str
    fingerprint : str
        Parameters the panels were rendered with.
    n_panels, n_decided, n_problems : int
    criteria : Counter
        How often each acceptance criterion was marked as failed.
    problem_images : list of str
        Images a problem was recorded on, so the verdict can be
        checked against them.
    verdict : str or None
        What the reviewer recorded, if anything.
    reason : str
    stale : bool
        Whether the verdict was made against different parameters than
        the panels now carry.
    """

    key: str
    family: str
    level: str
    kind: str
    fingerprint: str
    n_panels: int = 0
    n_decided: int = 0
    n_problems: int = 0
    criteria: Counter = field(default_factory=Counter)
    problem_images: list[str] = field(default_factory=list)
    verdict: Optional[str] = None
    reason: str = ""
    stale: bool = False

    @property
    def status(self) -> str:
        """Status of this setting, counting what is still unreviewed.

        Returns
        -------
        str
        """
        if self.verdict is None or self.stale:
            return STATUS_PENDING
        if self.n_decided < self.n_panels:
            return STATUS_PENDING
        return self.verdict

    @property
    def complete(self) -> bool:
        """Whether every panel of this setting has been looked at.

        Returns
        -------
        bool
        """
        return self.n_panels > 0 and self.n_decided == self.n_panels


def load_review(path: Path) -> dict[str, Any]:
    """Read the file the review page writes.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
        ``{"decisions": {...}, "verdicts": {...}}``, empty if the file
        does not exist yet.
    """
    if not path.exists():
        return {"decisions": {}, "verdicts": {}}
    with open(path, encoding="utf-8") as handle:
        review = json.load(handle)
    review.setdefault("decisions", {})
    review.setdefault("verdicts", {})
    return review


def load_panels(path: Path) -> list[dict[str, Any]]:
    """Read the panel index a rendering run wrote.

    Parameters
    ----------
    path : Path

    Returns
    -------
    list of dict
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["panels"]


def summarize(
    panels: Iterable[dict[str, Any]], review: dict[str, Any]
) -> list[LevelSummary]:
    """Reduce panels and verdicts to one summary per setting.

    A decision whose fingerprint no longer matches its panel is
    ignored rather than counted: it describes a picture that is not on
    screen any more.

    Parameters
    ----------
    panels : Iterable of dict
    review : dict

    Returns
    -------
    list of LevelSummary
        In the order the settings are rendered.
    """
    summaries: dict[str, LevelSummary] = {}
    for panel in panels:
        key = f"{panel['family']}__{panel['level']}"
        summary = summaries.setdefault(key, LevelSummary(
            key=key,
            family=panel["family"],
            level=panel["level"],
            kind=panel["kind"],
            fingerprint=panel["fingerprint"],
        ))
        summary.n_panels += 1
        decision = review["decisions"].get(panel["panel_id"])
        if decision is None:
            continue
        if decision.get("fingerprint") != panel["fingerprint"]:
            continue
        summary.n_decided += 1
        if decision.get("status") == "problem":
            summary.n_problems += 1
            summary.problem_images.append(panel["image_id"])
            summary.criteria.update(decision.get("criteria", []))

    for key, summary in summaries.items():
        verdict = review["verdicts"].get(key)
        if verdict is None:
            continue
        summary.verdict = verdict.get("status")
        summary.reason = verdict.get("reason", "")
        summary.stale = (
            verdict.get("fingerprint") != summary.fingerprint
        )
    return list(summaries.values())


def family_status(summaries: Sequence[LevelSummary]) -> str:
    """Status of one family, from the settings that gate it.

    Parameters
    ----------
    summaries : Sequence of LevelSummary
        Every setting of one family; diagnostics are ignored here.

    Returns
    -------
    str
    """
    gated = [s for s in summaries if s.kind == KIND_GATE]
    if not gated:
        return STATUS_PENDING
    return max(
        (s.status for s in gated), key=lambda s: STATUS_SEVERITY[s]
    )


def decision_sheet(
    summaries: Sequence[LevelSummary],
) -> dict[str, Any]:
    """Build the record the experiment's decision table expects.

    One entry per family, in the shape the plan's template uses: the
    visual status, the settings it was judged at, and the parameters
    that status applies to. The approved parameters are the ones the
    accepted gated settings were rendered with - not the family's
    defaults, which may have been edited since.

    Parameters
    ----------
    summaries : Sequence of LevelSummary

    Returns
    -------
    dict
        Family code to its record.
    """
    levels = {level.key: level for level in review_levels()}
    sheet: dict[str, Any] = {}
    for summary in summaries:
        record = sheet.setdefault(summary.family, {
            "family": summary.family,
            "faza0_wizualna": {
                "status": STATUS_PENDING,
                "poziomy_sily": [],
                "rozpoznawalnosc_zachowana": None,
                "approved_parameters": {},
            },
            "evidence": {
                "n_panels": 0,
                "n_decided": 0,
                "n_problems": 0,
                "criteria": {},
            },
        })
        record["faza0_wizualna"]["poziomy_sily"].append({
            "level": summary.level,
            "kind": summary.kind,
            "status": summary.status,
            "reason": summary.reason,
            "fingerprint": summary.fingerprint,
            "stale_verdict": summary.stale,
            "n_panels": summary.n_panels,
            "n_decided": summary.n_decided,
            "n_problems": summary.n_problems,
            "problem_images": sorted(set(summary.problem_images)),
        })
        record["evidence"]["n_panels"] += summary.n_panels
        record["evidence"]["n_decided"] += summary.n_decided
        record["evidence"]["n_problems"] += summary.n_problems
        for index, count in summary.criteria.items():
            name = CRITERIA[int(index)]
            record["evidence"]["criteria"][name] = (
                record["evidence"]["criteria"].get(name, 0) + count
            )
        if (
            summary.kind == KIND_GATE
            and summary.status == STATUS_ACCEPTED
            and summary.key in levels
        ):
            record["faza0_wizualna"]["approved_parameters"][
                summary.level
            ] = levels[summary.key].parameters

    by_family: dict[str, list[LevelSummary]] = {}
    for summary in summaries:
        by_family.setdefault(summary.family, []).append(summary)
    for family, record in sheet.items():
        status = family_status(by_family[family])
        record["faza0_wizualna"]["status"] = status
        record["faza0_wizualna"]["rozpoznawalnosc_zachowana"] = (
            status == STATUS_ACCEPTED
        )
    return sheet
