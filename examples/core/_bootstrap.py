"""Make examples import the local checkout when run as plain scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_import() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
