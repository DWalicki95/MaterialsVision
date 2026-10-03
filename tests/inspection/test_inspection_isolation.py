"""The inspection package must stay a leaf of the dependency graph.

Anything in it may be rewritten or removed at will only as long as no
pipeline code imports from it; otherwise a change to a picture could
reach a reported number. Checked on the source text, so that an import
hidden inside a function is caught as well as one at the top.
"""
import re
from pathlib import Path

import materials_vision

PACKAGE_ROOT = Path(materials_vision.__file__).parent

INSPECTION_DIR = PACKAGE_ROOT / "inspection"

INSPECTION_IMPORT = re.compile(
    r"^\s*(from|import)\s+materials_vision\.inspection\b", re.MULTILINE
)


def test_no_module_outside_inspection_imports_it():
    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if INSPECTION_DIR not in path.parents
        and INSPECTION_IMPORT.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
