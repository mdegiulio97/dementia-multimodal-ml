import numpy as np
import pytest

from dementia_ml.eval import bootstrap, metrics


def _survival_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    time = rng.exponential(np.exp(-risk)) * 30
    cens = rng.exponential(60, n)
    return np.minimum(time, cens), time <= cens, risk


def test_c_index_perfect_and_random():
    time, event, _ = _survival_data()
    assert metrics.c_index(time, event, -time) == pytest.approx(1.0)
    assert metrics.c_index(time, event, np.zeros_like(time)) == pytest.approx(0.5)


def test_uno_c_and_td_auc_informative_model():
    time, event, risk = _survival_data(n=800)
    assert metrics.uno_c(time, event, time, event, risk, tau=40) > 0.65
    auc = metrics.td_auc(time, event, time, event, risk, times=[10, 20, 30])
    assert auc.shape == (3,) and (auc > 0.65).all()


def test_bootstrap_ci_contains_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 300)
    est, lo, hi = bootstrap.ci(lambda idx: float(x[idx].mean()), n=len(x), n_boot=500)
    assert lo < est < hi
    assert lo < 2.0 < hi


def test_calibration_perfect_model():
    rng = np.random.default_rng(0)
    n, t = 20000, 36.0
    rate = np.exp(rng.normal(-4, 0.8, n))
    time = rng.exponential(1 / rate)
    cens = rng.uniform(0, 200, n)
    obs_t, event = np.minimum(time, cens), time <= cens
    surv_at_t = np.exp(-rate * t)
    cal = metrics.calibration_at(obs_t, event, surv_at_t, t=t, n_bins=5)
    assert list(cal.columns) == ["bin", "predicted", "observed_km", "n"]
    assert (abs(cal["predicted"] - cal["observed_km"]) < 0.05).all()


def test_classification_metrics():
    y = np.array([0, 0, 1, 1])
    assert metrics.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.balanced_accuracy(y, np.array([0, 1, 1, 1])) == 0.75
