import numpy as np
import pytest

from dementia_ml.mr import estimators


def _sim(k=30, effect=0.3, pleio_mean=0.0, pleio_sd=0.0, invalid_share=0.0, seed=0):
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0.02, 0.3, k)
    sx = np.full(k, 0.01)
    sy = np.full(k, 0.01)
    alpha = rng.normal(pleio_mean, pleio_sd, k) if pleio_sd or pleio_mean else np.zeros(k)
    if invalid_share:
        invalid = rng.random(k) < invalid_share
        alpha = np.where(invalid, rng.normal(0.15, 0.03, k), 0.0)
    by = effect * bx + alpha + rng.normal(0, sy)
    return bx + rng.normal(0, sx), sx, by, sy


def test_ivw_recovers_causal_effect():
    r = estimators.ivw(*_sim())
    assert abs(r.beta - 0.3) < 0.05
    assert r.ci_low < 0.3 < r.ci_high
    assert r.n_snps == 30 and r.method == "ivw"


def test_egger_detects_directional_pleiotropy():
    data = _sim(pleio_mean=0.02, pleio_sd=0.005)
    eg = estimators.egger(*data)
    iv = estimators.ivw(*data)
    assert eg.intercept_p < 0.05
    assert abs(eg.beta - 0.3) < 0.1
    assert abs(iv.beta - 0.3) > abs(eg.beta - 0.3)


def test_weighted_median_robust_to_30pct_invalid():
    data = _sim(invalid_share=0.3, seed=3)
    wm = estimators.weighted_median(*data, n_boot=500, seed=0)
    iv = estimators.ivw(*data)
    assert abs(wm.beta - 0.3) < 0.07
    assert abs(iv.beta - 0.3) > abs(wm.beta - 0.3)
    assert wm.se > 0


def test_wald_ratio_single_instrument():
    r = estimators.wald_ratio(0.2, 0.01, 0.06, 0.01)
    assert r.beta == pytest.approx(0.3)
    assert r.se == pytest.approx(0.05)
    assert r.n_snps == 1
