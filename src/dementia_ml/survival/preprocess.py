"""Training-fold preprocessing: ICV-adjusted hippocampus, imputation, scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HippocampusResidualizer(BaseEstimator, TransformerMixin):
    """Replace Hippocampus and ICV with the z-scored residual of Hippocampus on ICV."""

    def fit(self, X: pd.DataFrame, y=None):
        self.active_ = {"Hippocampus", "ICV"} <= set(X.columns)
        if self.active_:
            ok = X[["Hippocampus", "ICV"]].notna().all(axis=1)
            icv, hip = X.loc[ok, "ICV"].to_numpy(float), X.loc[ok, "Hippocampus"].to_numpy(float)
            self.slope_, self.intercept_ = np.polyfit(icv, hip, 1)
            resid = hip - (self.intercept_ + self.slope_ * icv)
            self.resid_sd_ = float(resid.std())
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.active_:
            return X.copy()
        out = X.drop(columns=["Hippocampus", "ICV"])
        resid = X["Hippocampus"] - (self.intercept_ + self.slope_ * X["ICV"])
        out["Hippocampus_z"] = resid / self.resid_sd_
        return out

    def get_feature_names_out(self, input_features=None):
        names = [c for c in self.feature_names_in_ if c not in {"Hippocampus", "ICV"}]
        return np.asarray(names + (["Hippocampus_z"] if self.active_ else []))


class ImputeWithIndicators(BaseEstimator, TransformerMixin):
    """Median imputation; a missingness indicator only for features missing in at least
    ``min_missing_frac`` of the training rows.

    A near-constant indicator (a handful of missing values) carries no usable information and
    makes an unpenalised Cox model diverge, so it is not created.
    """

    def __init__(self, min_missing_frac: float = 0.05):
        self.min_missing_frac = min_missing_frac

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.median()
        frac = X.isna().mean()
        self.indicators_ = [c for c in X.columns if frac[c] >= self.min_missing_frac]
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.fillna(self.medians_).astype(float)
        for c in self.indicators_:
            out[f"missingindicator_{c}"] = X[c].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            list(self.feature_names_in_) + [f"missingindicator_{c}" for c in self.indicators_]
        )


def make_preprocessor(features: list[str]) -> Pipeline:
    pipe = Pipeline(
        [
            ("hippocampus", HippocampusResidualizer()),
            ("impute", ImputeWithIndicators(min_missing_frac=0.05)),
            ("scale", StandardScaler()),
        ]
    )
    return pipe.set_output(transform="pandas")
