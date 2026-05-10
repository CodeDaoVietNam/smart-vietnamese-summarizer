from __future__ import annotations

import sys
from pathlib import Path


def add_project_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for path in (root, src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
