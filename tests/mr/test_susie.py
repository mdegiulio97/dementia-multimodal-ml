import numpy as np
import pytest

from dementia_ml.mr import susie


def _ld(p=120, rho=0.6):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _z(R, causal, effects, n, seed=0):
    """z-scores for a trait with the given causal variants under LD matrix R."""
    rng = np.random.default_rng(seed)
    b = np.zeros(len(R))
    b[causal] = effects
    mean = np.sqrt(n) * (R @ b)
    L = np.linalg.cholesky(R + 1e-8 * np.eye(len(R)))
    return mean + L @ rng.normal(size=len(R))


def test_susie_finds_two_independent_signals():
    R = _ld()
    z = _z(R, [30, 90], [0.06, -0.05], n=20000)
    fit = susie.susie_rss(z, R, n=20000, L=5)
    sets = susie.credible_sets(fit, R)
    assert len(sets) == 2
    found = sorted(next(c for c in (30, 90) if c in s["variants"]) for s in sets)
    assert found == [30, 90]


def test_susie_no_signal_gives_no_credible_set():
    R = _ld()
    z = _z(R, [], [], n=20000, seed=3)
    fit = susie.susie_rss(z, R, n=20000, L=5)
    assert susie.credible_sets(fit, R) == []


def test_coloc_susie_pairs_the_shared_signal():
    R = _ld()
    z1 = _z(R, [30, 90], [0.06, -0.05], n=20000, seed=1)  # protein: two signals
    z2 = _z(R, [90], [0.04], n=20000, seed=2)  # disease: one, shared with signal at 90
    res = susie.coloc_susie(z1, z2, R, n1=20000, n2=20000)
    best = max(res, key=lambda r: r["PP.H4"])
    assert best["PP.H4"] > 0.8 and 90 in best["variants1"]
    other = [r for r in res if 30 in r["variants1"]]
    assert all(r["PP.H4"] < 0.2 for r in other)


def test_lbf_matches_wakefield_for_single_effect():
    from dementia_ml.mr import coloc

    z = np.array([4.0])
    n, V = 10000, 0.01
    lbf = susie.single_effect_lbf(z / np.sqrt(n), np.array([1 / np.sqrt(n)]), V)
    assert lbf[0] == pytest.approx(
        coloc.log_abf(z / np.sqrt(n), np.array([1 / np.sqrt(n)]), np.sqrt(V))[0]
    )
