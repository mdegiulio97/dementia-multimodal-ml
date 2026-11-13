"""Omics preparation: name mapping, rank-based inverse normal transform, baseline samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def nmr_column(mr_trait: str) -> str:
    """Karjalainen et al. 2024 trait name -> ADNI Nightingale column name."""
    return mr_trait.upper().replace("-", "_").replace("_PERCENT", "_PCT")


def inverse_normal(x: pd.Series) -> pd.Series:
    """Rank-based inverse normal transform (Blom); NaN stay NaN."""
    ok = x.notna()
    ranks = x[ok].rank(method="average")
    out = pd.Series(np.nan, index=x.index)
    out[ok] = norm.ppf((ranks - 0.375) / (ok.sum() + 0.25))
    return out


def baseline_samples(
    values: pd.DataFrame, features: list[str], anchors: pd.Series, window_days: int = 180
) -> pd.DataFrame:
    """Per participant (RID), the sample closest to the anchor date within the window."""
    v = values.copy()
    v["RID"] = v["RID"].astype(int)
    v["_date"] = pd.to_datetime(v["EXAMDATE"], errors="coerce")
    v = v[v["RID"].isin(anchors.index)].dropna(subset=["_date"])
    v["_gap"] = (v["_date"] - v["RID"].map(anchors)).abs().dt.days
    v = v[v["_gap"] <= window_days].sort_values(["RID", "_gap"]).drop_duplicates("RID")
    return v.set_index("RID")[features]
