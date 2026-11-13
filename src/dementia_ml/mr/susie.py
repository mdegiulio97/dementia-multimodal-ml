"""SuSiE fine-mapping from summary statistics and SuSiE-based colocalisation.

Implements the IBSS algorithm of Wang et al. 2020 (JRSS-B, doi:10.1111/rssb.12388) in the
sufficient-statistics form used for z-scores and an LD matrix (Zou et al. 2022, PLoS Genet,
doi:10.1371/journal.pgen.1010299), with standardised genotypes and unit residual variance.
Colocalisation then compares every pair of credible sets across the two traits with the
per-variant log Bayes factors of each single effect (Wallace 2021, PLoS Genet,
doi:10.1371/journal.pgen.1009440; ``coloc.susie``), which drops the one-causal-variant
assumption of ABF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from dementia_ml.mr.coloc import posterior_from_lbf


def single_effect_lbf(betahat: np.ndarray, shat: np.ndarray, V: float) -> np.ndarray:
    """Per-variant log Bayes factor of a single effect with prior variance V."""
    s2 = np.asarray(shat, float) ** 2
    b = np.asarray(betahat, float)
    return 0.5 * np.log(s2 / (V + s2)) + 0.5 * b**2 * V / (s2 * (V + s2))


@dataclass
class SusieFit:
    alpha: np.ndarray  # L x p posterior inclusion probabilities per effect
    mu: np.ndarray
    lbf: np.ndarray  # L x p per-variant log Bayes factors
    lbf_model: np.ndarray  # L
    V: np.ndarray  # L prior variances (0 = effect switched off)
    converged: bool


def _optimise_V(betahat, shat, log_prior):
    def neg(logv):
        return -logsumexp(single_effect_lbf(betahat, shat, np.exp(logv)) + log_prior)

    res = minimize_scalar(neg, bounds=(-30, 0), method="bounded")
    best = float(np.exp(res.x))
    return best if -res.fun > 0 else 0.0


def susie_rss(
    z: np.ndarray, R: np.ndarray, n: int, L: int = 10, max_iter: int = 200, tol: float = 1e-4
) -> SusieFit:
    z, R = np.asarray(z, float), np.asarray(R, float)
    p = len(z)
    XtX = n * R
    Xty = np.sqrt(n) * z
    d = np.diag(XtX)
    log_prior = np.full(p, -np.log(p))
    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    lbf = np.zeros((L, p))
    lbf_model = np.zeros(L)
    V = np.zeros(L)
    XtXr = XtX @ (alpha * mu).sum(axis=0)
    converged = False
    for _ in range(max_iter):
        old = alpha.copy()
        for k in range(L):
            XtXr -= XtX @ (alpha[k] * mu[k])
            r = Xty - XtXr
            betahat, shat = r / d, 1.0 / np.sqrt(d)
            V[k] = _optimise_V(betahat, shat, log_prior)
            if V[k] == 0.0:
                alpha[k], mu[k], lbf[k], lbf_model[k] = 1.0 / p, 0.0, 0.0, 0.0
            else:
                lbf[k] = single_effect_lbf(betahat, shat, V[k])
                w = lbf[k] + log_prior
                lbf_model[k] = logsumexp(w)
                alpha[k] = np.exp(w - lbf_model[k])
                post_var = 1.0 / (1.0 / V[k] + d)
                mu[k] = post_var * r
            XtXr += XtX @ (alpha[k] * mu[k])
        if np.max(np.abs(alpha - old)) < tol:
            converged = True
            break
    return SusieFit(alpha, mu, lbf, lbf_model, V, converged)


def credible_sets(
    fit: SusieFit, R: np.ndarray, coverage: float = 0.95, min_purity: float = 0.5
) -> list[dict]:
    sets, seen = [], set()
    for k in range(len(fit.V)):
        if fit.V[k] == 0.0:
            continue
        order = np.argsort(fit.alpha[k])[::-1]
        size = int(np.searchsorted(np.cumsum(fit.alpha[k][order]), coverage) + 1)
        members = tuple(sorted(order[:size].tolist()))
        sub = np.abs(np.asarray(R)[np.ix_(members, members)])
        purity = float(sub.min()) if len(members) > 1 else 1.0
        if purity < min_purity or members in seen:
            continue
        seen.add(members)
        sets.append(
            {
                "effect": k,
                "variants": list(members),
                "purity": purity,
                "lbf_model": float(fit.lbf_model[k]),
            }
        )
    return sets


def coloc_susie(
    z1: np.ndarray,
    z2: np.ndarray,
    R: np.ndarray,
    n1: int,
    n2: int,
    L: int = 10,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 5e-6,
) -> list[dict]:
    """Colocalisation for every pair of credible sets (one per trait)."""
    f1, f2 = susie_rss(z1, R, n1, L), susie_rss(z2, R, n2, L)
    cs1, cs2 = credible_sets(f1, R), credible_sets(f2, R)
    out = []
    for a in cs1:
        for b in cs2:
            pp = posterior_from_lbf(f1.lbf[a["effect"]], f2.lbf[b["effect"]], p1, p2, p12)
            out.append(
                {
                    **{f"PP.H{i}": float(v) for i, v in enumerate(pp)},
                    "variants1": a["variants"],
                    "variants2": b["variants"],
                    "lead1": int(np.argmax(f1.alpha[a["effect"]])),
                    "lead2": int(np.argmax(f2.alpha[b["effect"]])),
                }
            )
    return out
