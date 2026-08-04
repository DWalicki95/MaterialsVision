"""Canonicalization of the VAB/K cross-section token.

Filenames spell the cross-section inconsistently (different Polish
endings, abbreviations, occasional typos). This module resolves any
spelling to one of two canonical values through a closed, deterministic
procedure: an exact dictionary lookup first, a gated fuzzy match only as
a fallback, and a fatal error for anything neither resolves - fuzzy
matching never silently assigns a category.
"""
import difflib
from typing import Optional

from data_prep.inventory.issues import (CrossSectionError, IssueCollector,
                                        IssueLevel)

CANONICAL = ("rownolegly", "prostopadly")

# Every variant observed in the real data (both series), plus the
# abbreviations used in the redundant VAB/K filename segment.
KNOWN_VARIANTS: dict[str, str] = {
    "rownolegle": "rownolegly",
    "rownolegla": "rownolegly",
    "rownolegly": "rownolegly",
    "r": "rownolegly",
    "prostopadle": "prostopadly",
    "prostopadla": "prostopadly",
    "prostopadly": "prostopadly",
    "p": "prostopadly",
}


def canonicalize_cross_section(
    token: str,
    *,
    fuzzy_cutoff: float,
    collector: IssueCollector,
    image_ref: str,
) -> str:
    """Resolve a cross-section token to a canonical value.

    Resolution order: (1) exact match in ``KNOWN_VARIANTS`` after
    lower-casing; (2) ``difflib.get_close_matches`` against the known
    variants, only if it clears ``fuzzy_cutoff`` - every such use is
    recorded as an INFO issue; (3) otherwise a fatal
    ``CrossSectionError`` is raised, requiring a human decision.

    Parameters
    ----------
    token : str
        Raw cross-section token from the filename.
    fuzzy_cutoff : float
        Minimum similarity ratio (0-1) for the fuzzy fallback to accept
        a match.
    collector : IssueCollector
        Collector to record fuzzy-match usage on.
    image_ref : str
        Image identifier or filename, for the issue record.

    Returns
    -------
    str
        One of ``CANONICAL``.

    Raises
    ------
    CrossSectionError
        If neither exact nor fuzzy matching resolves the token.
    """
    lowered = token.lower()
    exact = KNOWN_VARIANTS.get(lowered)
    if exact is not None:
        return exact

    matches = difflib.get_close_matches(
        lowered, KNOWN_VARIANTS.keys(), n=1, cutoff=fuzzy_cutoff
    )
    if matches:
        canonical = KNOWN_VARIANTS[matches[0]]
        collector.add(
            IssueLevel.INFO,
            "fuzzy_cross_section",
            image_ref,
            f"'{token}' -> '{matches[0]}' -> '{canonical}' "
            f"(fuzzy cutoff={fuzzy_cutoff})",
        )
        return canonical

    raise CrossSectionError(
        f"Unknown cross-section token: '{token}' (image: {image_ref}). "
        "Add it to KNOWN_VARIANTS after human review."
    )


def lookup_cross_section(token: str) -> Optional[str]:
    """Exact-only, non-raising cross-section lookup.

    Used to check the redundant second cross-section segment in VAB/K
    filenames against the canonicalized first segment, without
    triggering fuzzy matching or fatal errors for that quality check.

    Parameters
    ----------
    token : str
        Raw cross-section token.

    Returns
    -------
    str or None
        Canonical value if ``token`` is a known exact variant,
        otherwise ``None``.
    """
    return KNOWN_VARIANTS.get(token.lower())
