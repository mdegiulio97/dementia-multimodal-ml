"""Survival models for MCI-to-dementia conversion and subject-level cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from dementia_ml.data.conversion import BASELINE_FEATURES, FULL_FEATURES, ConversionData
from dementia_ml.splits import subject_folds
from dementia_ml.survival.preprocess import make_preprocessor


def _cox_baseline(**_) -> CoxPHSurvivalAnalysis:
    return CoxPHSurvivalAnalysis(alpha=1e-4)


def _cox_full(alpha: float = 1.0, **_) -> CoxPHSurvivalAnalysis:
    return CoxPHSurvivalAnalysis(alpha=alpha)  # L2 (ridge) penalty


def _rsf(n_estimators: int = 500, min_samples_leaf: int = 10, seed: int = 0, **_):
    return RandomSurvivalForest(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed
    )


def _gbsa(n_estimators: int = 300, learning_rate: float = 0.05, seed: int = 0, **_):
    return GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=2, random_state=seed
    )


MODELS: dict[str, tuple[list[str], Callable[..., object]]] = {
    "cox_baseline": (BASELINE_FEATURES, _cox_baseline),
    "cox_full": (FULL_FEATURES, _cox_full),
    "rsf": (FULL_FEATURES, _rsf),
    "gbsa": (FULL_FEATURES, _gbsa),
}


def make_pipeline(name: str, seed: int = 0, **params) -> Pipeline:
    features, factory = MODELS[name]
    return Pipeline([("pre", make_preprocessor(features)), ("model", factory(seed=seed, **params))])


def features_of(name: str) -> list[str]:
    return list(MODELS[name][0])


def survival_at(pipe: Pipeline, X, horizons: Sequence[float]) -> np.ndarray:
    fns = pipe.predict_survival_function(X)
    out = np.empty((len(fns), len(horizons)))
    for i, fn in enumerate(fns):
        h = np.clip(np.asarray(horizons, float), fn.domain[0], fn.domain[1])
        out[i] = fn(h)
    return out


def cross_validate(
    name: str,
    data: ConversionData,
    n_splits: int = 5,
    seed: int = 0,
    horizons: Sequence[float] = (24.0, 36.0, 48.0),
    **params,
) -> dict[str, np.ndarray]:
    """Out-of-fold risk scores and survival probabilities at ``horizons``."""
    X = data.X[features_of(name)]
    y = Surv.from_arrays(data.event, data.time)
    risk = np.full(len(X), np.nan)
    surv = np.full((len(X), len(horizons)), np.nan)
    for train, test in subject_folds(X.index.to_numpy(), n_splits, stratify=data.event, seed=seed):
        pipe = make_pipeline(name, seed=seed, **params).fit(X.iloc[train], y[train])
        risk[test] = pipe.predict(X.iloc[test])
        surv[test] = survival_at(pipe, X.iloc[test], horizons)
    return {"risk": risk, "surv": surv}
