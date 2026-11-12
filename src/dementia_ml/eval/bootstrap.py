"""Percentile bootstrap over subjects."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ci(
    fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return (estimate, low, high). ``fn`` receives an index array of length ``n``.

    Resamples that make the statistic undefined (e.g. a single class) are skipped.
    """
    rng = np.random.default_rng(seed)
    estimate = float(fn(np.arange(n)))
    stats = []
    for _ in range(n_boot):
        try:
            value = float(fn(rng.integers(0, n, n)))
        except ValueError:
            continue
        if np.isfinite(value):
            stats.append(value)
    alpha = (1 - level) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return estimate, float(low), float(high)
