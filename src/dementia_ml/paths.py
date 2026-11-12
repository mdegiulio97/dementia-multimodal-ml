"""Filesystem locations. Data always live outside the repository."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path.home() / "data" / "dementia-ml"
KINDS = ("open", "synthetic", "adni")

Kind = Literal["open", "synthetic", "adni"]


def data_root() -> Path:
    """Root of downloaded datasets (env ``DEMENTIA_ML_DATA`` overrides)."""
    return Path(os.environ.get("DEMENTIA_ML_DATA", DEFAULT_DATA_ROOT))


def _kind_dir(base: str, kind: str, root: Path | None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    path = (root or REPO_ROOT) / base / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(kind: Kind, root: Path | None = None) -> Path:
    return _kind_dir("results", kind, root)


def figures_dir(kind: Kind, root: Path | None = None) -> Path:
    return _kind_dir("figures", kind, root)


def models_dir(kind: Kind, root: Path | None = None) -> Path:
    return _kind_dir("models", kind, root)
