"""Cross-validation folds that never split a participant across train and test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def subject_folds(
    subject_ids: Sequence,
    n_splits: int = 5,
    stratify: Sequence | None = None,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) row indices; all rows of a subject share a fold."""
    groups = np.asarray(subject_ids)
    y = np.zeros(len(groups), dtype=int) if stratify is None else np.asarray(stratify)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(groups)), y, groups)]
