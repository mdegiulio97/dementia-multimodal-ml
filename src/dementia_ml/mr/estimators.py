"""Two-sample Mendelian randomisation estimators on summary statistics.

Conventions follow the standard methods (Burgess et al. 2013; Bowden et al.
2015, 2016): first-order ratio standard errors, multiplicative random-effects
IVW, MR-Egger with the residual standard error floored at 1, and the simple
weighted median with a parametric bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class MRResult:
    method: str
    beta: float
    se: float
    ci_low: float
    ci_high: float
    p: float
    n_snps: int
    intercept: float | None = None
    intercept_se: float | None = None
    intercept_p: float | None = None


def _result(method, beta, se, k, dist=stats.norm, **extra) -> MRResult:
    q = dist.ppf(0.975)
    p = 2 * dist.sf(abs(beta / se)) if se > 0 else np.nan
    return MRResult(
        method, float(beta), float(se), beta - q * se, beta + q * se, float(p), k, **extra
    )


def _arrays(*xs):
    return [np.atleast_1d(np.asarray(x, float)) for x in xs]


def wald_ratio(bx, sx, by, sy) -> MRResult:
    bx, sx, by, sy = _arrays(bx, sx, by, sy)
    return _result("wald_ratio", by[0] / bx[0], sy[0] / abs(bx[0]), 1)


def ivw(bx, sx, by, sy) -> MRResult:
    bx, sx, by, sy = _arrays(bx, sx, by, sy)
    w = 1 / sy**2
    beta = np.sum(w * bx * by) / np.sum(w * bx**2)
    se_fixed = 1 / np.sqrt(np.sum(w * bx**2))
    k = len(bx)
    phi = np.sum(w * (by - beta * bx) ** 2) / (k - 1) if k > 1 else 1.0
    return _result("ivw", beta, se_fixed * max(1.0, np.sqrt(phi)), k)


def egger(bx, sx, by, sy) -> MRResult:
    bx, sx, by, sy = _arrays(bx, sx, by, sy)
    sign = np.sign(bx)
    bx, by = bx * sign, by * sign  # orient instruments to positive exposure effect
    k = len(bx)
    w = 1 / sy**2
    X = np.column_stack([np.ones(k), bx])
    XtW = X.T * w
    cov_unscaled = np.linalg.inv(XtW @ X)
    coef = cov_unscaled @ XtW @ by
    resid = by - X @ coef
    sigma = np.sqrt(np.sum(w * resid**2) / (k - 2))
    cov = cov_unscaled * max(1.0, sigma) ** 2
    se = np.sqrt(np.diag(cov))
    t = stats.t(df=k - 2)
    return _result(
        "egger",
        coef[1],
        se[1],
        k,
        dist=t,
        intercept=float(coef[0]),
        intercept_se=float(se[0]),
        intercept_p=float(2 * t.sf(abs(coef[0] / se[0]))),
    )


def _wm(beta_j: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(beta_j)
    b, w = beta_j[order], w[order] / w.sum()
    p = np.cumsum(w) - 0.5 * w
    return float(np.interp(0.5, p, b))


def weighted_median(bx, sx, by, sy, n_boot: int = 1000, seed: int = 0) -> MRResult:
    bx, sx, by, sy = _arrays(bx, sx, by, sy)
    w = (bx / sy) ** 2
    beta = _wm(by / bx, w)
    rng = np.random.default_rng(seed)
    boots = [_wm((by + rng.normal(0, sy)) / (bx + rng.normal(0, sx)), w) for _ in range(n_boot)]
    return _result("weighted_median", beta, float(np.std(boots, ddof=1)), len(bx))
