"""Linear mixed-effects models of cognitive trajectories (converters vs stable MCI)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class TrajectoryResult:
    outcome: str
    fixed_effects: pd.DataFrame
    slope_difference: tuple[float, float, float]
    n_subjects: int
    n_obs: int
    model: object = None


def fit_trajectory(
    df: pd.DataFrame, outcome: str, group_col: str = "converter"
) -> TrajectoryResult:
    """Random intercept and slope per participant; time in years from baseline.

    ``outcome ~ years * group`` - the ``years:group`` term is the difference in
    annual change between groups.
    """
    data = df[["RID", "M", outcome, group_col]].dropna().copy()
    data["years"] = data["M"] / 12.0
    data = data.rename(columns={outcome: "y", group_col: "group"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from statsmodels
        fit = smf.mixedlm("y ~ years * group", data, groups=data["RID"], re_formula="~years").fit(
            method=["lbfgs", "powell"], reml=True
        )
    ci = fit.conf_int()
    fe_names = fit.fe_params.index
    fe = pd.DataFrame(
        {
            "coef": fit.fe_params,
            "se": fit.bse_fe,
            "ci_low": ci.loc[fe_names, 0],
            "ci_high": ci.loc[fe_names, 1],
            "p": fit.pvalues.loc[fe_names],
        }
    )
    term = "years:group"
    return TrajectoryResult(
        outcome=outcome,
        fixed_effects=fe,
        slope_difference=(
            float(fe.loc[term, "coef"]),
            float(fe.loc[term, "ci_low"]),
            float(fe.loc[term, "ci_high"]),
        ),
        n_subjects=int(data["RID"].nunique()),
        n_obs=len(data),
        model=fit,
    )
