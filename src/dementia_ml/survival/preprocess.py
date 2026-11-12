"""Training-fold preprocessing: ICV-adjusted hippocampus, imputation, scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
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


def make_preprocessor(features: list[str]) -> Pipeline:
    pipe = Pipeline(
        [
            ("hippocampus", HippocampusResidualizer()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return pipe.set_output(transform="pandas")
