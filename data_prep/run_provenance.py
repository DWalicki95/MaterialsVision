"""
Provenance helpers shared by the pipelines that freeze an artifact.

Every frozen artifact of this project (the inventory manifest, the
dataset split) records the same three things about the run that
produced it: which commit of this repository it came from, which
library versions were installed, and the hash of the file itself.
They live here so both pipelines answer those questions identically.
"""
import hashlib
import logging
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_TRACKED_LIBRARIES: tuple[str, ...] = (
    "pandas", "numpy", "Pillow", "PyYAML", "scikit-image",
)

_GIT_TIMEOUT_S = 5


def git_commit() -> Optional[str]:
    """Return the current git commit hash, or None outside a repo.

    Returns
    -------
    str or None
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            timeout=_GIT_TIMEOUT_S,
        )
        return out.stdout.strip()
    except Exception:
        return None


def library_versions(
    names: Sequence[str] = DEFAULT_TRACKED_LIBRARIES,
) -> dict[str, Optional[str]]:
    """Return installed versions of the given libraries.

    Parameters
    ----------
    names : Sequence of str, optional
        Distribution names to look up.

    Returns
    -------
    dict of str to str or None
        None for any name that cannot be resolved.
    """
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_version() -> str:
    """Return the running interpreter's version string.

    Returns
    -------
    str
    """
    return platform.python_version()
