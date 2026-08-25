"""
Issue taxonomy for the data inventory pipeline.

Every problem encountered while building the manifest is recorded as an
``Issue`` instead of being silently skipped or only logged. Collected issues
back the validation report (see ``reporting.py``) and the fatal/error
short-circuit behaviour in ``manifest.py``.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IssueLevel(Enum):
    """Severity of a data inventory issue.

    FATAL aborts the run before any artifact is written. ERROR drops the
    offending image from the manifest but lets the run continue. WARNING
    keeps the row with a flag. INFO records an expected, non-degrading
    event (e.g. use of fuzzy matching).
    """

    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class Issue:
    """A single recorded problem.

    Parameters
    ----------
    level : IssueLevel
        Severity, see ``IssueLevel``.
    code : str
        Short machine-readable code naming the kind of problem, e.g.
        ``"sidecar_missing"`` or ``"magnification_conflict"``. Codes
        are grouped by this value in the validation report, so reuse
        an existing one rather than inventing a near-synonym.
    image_ref : str
        ``image_id`` if already known, otherwise the raw source filename.
    detail : str
        Human-readable explanation, included verbatim in the report.
    """

    level: IssueLevel
    code: str
    image_ref: str
    detail: str = ""


class IssueCollector:
    """Accumulates ``Issue`` instances during a run.

    Not thread-safe; the inventory pipeline is single-threaded by design
    (one image at a time, deterministic order).
    """

    def __init__(self) -> None:
        self._issues: list[Issue] = []

    def add(
        self, level: IssueLevel, code: str, image_ref: str, detail: str = ""
    ) -> Issue:
        """Record an issue and return it.

        Parameters
        ----------
        level : IssueLevel
        code : str
        image_ref : str
        detail : str, optional

        Returns
        -------
        Issue
            The recorded issue, for convenience when raising afterwards.
        """
        issue = Issue(
            level=level, code=code, image_ref=image_ref, detail=detail
        )
        self._issues.append(issue)
        log_fn = {
            IssueLevel.FATAL: logger.error,
            IssueLevel.ERROR: logger.error,
            IssueLevel.WARNING: logger.warning,
            IssueLevel.INFO: logger.info,
        }[level]
        log_fn("[%s] %s (%s): %s", level.value, code, image_ref, detail)
        return issue

    def all(self) -> list[Issue]:
        """Return all recorded issues, in recording order."""
        return list(self._issues)

    def by_level(self, level: IssueLevel) -> list[Issue]:
        """Return issues filtered to a single severity level."""
        return [i for i in self._issues if i.level == level]

    def has_fatal(self) -> bool:
        """Return True if any FATAL issue was recorded."""
        return any(i.level == IssueLevel.FATAL for i in self._issues)

    def has_error(self) -> bool:
        """Return True if any ERROR issue was recorded."""
        return any(i.level == IssueLevel.ERROR for i in self._issues)


class FilenameParseError(ValueError):
    """Raised when a filename does not match its series' naming convention."""


class CrossSectionError(ValueError):
    """Raised when a cross-section token has no known or fuzzy match."""


class AnnotationSelectionError(ValueError):
    """Raised when a Label Studio task has no usable annotation."""


class PolygonConversionError(ValueError):
    """Raised when a polygon result cannot be converted to pixel coords."""


class ManifestSchemaError(ValueError):
    """Raised when the built manifest does not match the frozen schema."""


class ImageIdCollisionError(ValueError):
    """Raised when two rows would share the same image_id."""


class ManifestBuildAborted(RuntimeError):
    """Raised by build_manifest when any FATAL issue was recorded.

    Per-image FATAL conditions do not stop the run immediately - the
    pipeline keeps processing the remaining images so a single run
    surfaces every fatal problem at once, not just the first. This
    exception is raised only after the full pass (and the global
    validation step) completes, and signals that no artifacts may be
    written for this run.
    """

    def __init__(self, fatal_issues: list[Issue]) -> None:
        self.fatal_issues = fatal_issues
        super().__init__(
            f"{len(fatal_issues)} fatal issue(s) - see log for details"
        )


@dataclass
class RejectionLog:
    """Tracks images dropped from the manifest with their reason.

    Used to enforce the "nothing vanishes silently" invariant: every task
    from the Label Studio export ends up either as a manifest row or as an
    entry here.
    """

    entries: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        """Record a rejection.

        Parameters
        ----------
        issue : Issue
            The ERROR-level issue explaining the rejection.
        """
        self.entries.append(issue)

    def __len__(self) -> int:
        return len(self.entries)
