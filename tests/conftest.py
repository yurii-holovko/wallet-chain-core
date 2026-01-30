"""Test configuration for module import paths."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    src_value = str(src_path)
    if src_value not in sys.path:
        sys.path.insert(0, src_value)


def _ensure_repo_on_path() -> None:
    repo_path = Path(__file__).resolve().parents[1]
    repo_value = str(repo_path)
    if repo_value not in sys.path:
        sys.path.insert(0, repo_value)


_ensure_src_on_path()
_ensure_repo_on_path()
