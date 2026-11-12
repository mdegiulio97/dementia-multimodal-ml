"""Provenance block attached to every result file."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from dementia_ml import __version__
from dementia_ml.paths import REPO_ROOT


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def block(dataset: str, dataset_version: str, config: dict, seed: int) -> dict:
    return {
        "dataset": dataset,
        "dataset_version": dataset_version,
        "config": config,
        "seed": seed,
        "git_commit": git_commit(),
        "package_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
