"""Do the MR-selected molecules stand out in ADNI?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


def compare(res: pd.DataFrame, selected: list[str]) -> dict:
    """Are p-values of the selected features smaller than those of all other features?"""
    is_sel = res["feature"].isin(selected)
    sel, other = res.loc[is_sel, "p"], res.loc[~is_sel, "p"]
    out = {
        "n_selected_tested": int(is_sel.sum()),
        "n_other": int((~is_sel).sum()),
        "selected_p05": int((sel < 0.05).sum()),
        "other_p05_fraction": float((other < 0.05).mean()) if len(other) else None,
    }
    if len(sel) and len(other):
        out["mannwhitney_p"] = float(mannwhitneyu(sel, other, alternative="less").pvalue)
        out["binomial_p05_p"] = float(
            binomtest(
                out["selected_p05"],
                len(sel),
                max(out["other_p05_fraction"], 1e-9),
                alternative="greater",
            ).pvalue
        )
    return out


def direction(res: pd.DataFrame, mr_beta: pd.Series) -> dict:
    """Same direction in ADNI (log HR) and in MR (log-odds of AD)?"""
    m = (
        res.set_index("feature")
        .join(mr_beta.rename("mr_beta"), how="inner")
        .dropna(subset=["mr_beta"])
    )
    same = np.sign(np.log(m["hr_per_sd"])) == np.sign(m["mr_beta"])
    k, n = int(same.sum()), len(m)
    return {
        "n": n,
        "same_sign": k,
        "binomial_p": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
    }
