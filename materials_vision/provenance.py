"""
What a run needs to record about itself to be reproducible.

A comparison between two augmentation policies is only worth anything
if nothing else differed between them, and the only way to show that
later is to have written down what "everything else" was at the time.
This module collects it: the commit of this repository, the versions
of the libraries that decide numerical behaviour, and the hardware.

Two of these need more than a package version. ``peft_sam`` is not on
PyPI and is installed straight from a git repository, so its version
number is meaningless on its own - ``0.0.1`` has pointed at many
different states of that repository - and the commit is recorded
instead. ``torch`` reports ``2.9.1`` through the packaging metadata
but ``2.9.1+cu128`` through the library itself: the metadata layer
strips the local version tag, and that tag names the CUDA build, which
decides kernel behaviour. The build is therefore read from the library
rather than from its metadata.

Nothing here raises. A run must not die because a version could not be
read, and a missing entry is more honest than a guessed one: the
field is recorded as ``None`` and a warning is logged, so the gap is
visible in the run's own record rather than silently absent.
"""
import json
import logging
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "micro_sam",
    "peft_sam",
    "torch_em",
    "torch",
    "numpy",
    "scipy",
    "scikit-image",
    "albumentations",
    "cellpose",
    "python-elf",
)

VCS_PACKAGES = ("peft_sam",)

GIT_TIMEOUT_S = 10


def run_provenance() -> dict[str, object]:
    """Describe the environment a run is about to execute in.

    Returns
    -------
    dict
        Repository state, library versions, and hardware. Any entry
        that could not be read is ``None``.
    """
    return {
        "git": _repository_state(),
        "python_version": platform.python_version(),
        "packages": {
            name: package_version(name) for name in TRACKED_PACKAGES
        },
        "package_commits": {
            name: package_commit(name) for name in VCS_PACKAGES
        },
        "hardware": _hardware(),
    }


def package_version(name: str) -> Optional[str]:
    """Installed version of one package.

    Parameters
    ----------
    name : str

    Returns
    -------
    str or None
        ``None`` when the package is not installed.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        logger.warning(
            "Package %r is not installed; its version cannot be "
            "recorded in the run metadata.", name,
        )
        return None


def package_commit(name: str) -> Optional[str]:
    """Commit a package was installed from, for git installs.

    A version number identifies a release. A package installed from a
    git URL has no release to identify, so the commit is the only
    thing that pins what the code actually was.

    Parameters
    ----------
    name : str

    Returns
    -------
    str or None
        The commit hash, or ``None`` when the package was not
        installed from a repository or records no such information.
    """
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None

    raw = distribution.read_text("direct_url.json")
    if raw is None:
        logger.warning(
            "Package %r records no installation source; the commit it "
            "was built from cannot be recovered.", name,
        )
        return None
    try:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        logger.warning(
            "Installation record of %r is not readable JSON.", name,
        )
        return None
    if commit is None:
        logger.warning(
            "Package %r was not installed from a repository; there is "
            "no commit to record.", name,
        )
    return commit


def _repository_state() -> dict[str, object]:
    """Commit of this repository, and whether it was modified.

    A run started from a dirty tree cannot be reproduced from its
    commit alone, so that fact is recorded rather than the commit
    being reported as if it described the code that ran.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _git(*arguments: str) -> Optional[str]:
    """Run one git command in this repository, or give up quietly."""
    repository = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository)) + arguments,
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Cannot read the repository state: %s", error)
        return None
    if completed.returncode != 0:
        logger.warning(
            "git %s failed: %s", " ".join(arguments),
            completed.stderr.strip(),
        )
        return None
    return completed.stdout.strip()


def _hardware() -> dict[str, object]:
    """Machine and accelerator the run will use."""
    record: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
    }
    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; no accelerator recorded.")
        record["cuda_available"] = None
        return record

    record["torch_build"] = torch.__version__
    record["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        return record

    properties = torch.cuda.get_device_properties(0)
    record["gpu_name"] = properties.name
    record["gpu_memory_gib"] = round(
        properties.total_memory / 1024 ** 3, 1
    )
    record["cuda_version"] = torch.version.cuda
    return record
