"""Exposure-wide MR screen: instrument selection, harmonisation, estimation, FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from dementia_ml.mr import estimators
from dementia_ml.mr.harmonise import harmonise


def prune_by_distance(exp: pd.DataFrame, window_kb: float = 500) -> pd.DataFrame:
    """Greedy pruning: keep the most significant SNP, drop others within the window.

    Distance-based approximation of LD clumping (no LD reference panel).
    """
    kept: list[int] = []
    window = window_kb * 1000
    for idx, row in exp.sort_values("pval").iterrows():
        same = exp.loc[kept]
        same = same[same["chrom"] == row["chrom"]]
        if (abs(same["pos"] - row["pos"]) > window).all():
            kept.append(idx)
    return exp.loc[kept]


def screen_trait(h: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Primary estimate (Wald ratio or IVW) plus sensitivity estimators."""
    bx, sx, by, sy = (h[c].to_numpy(float) for c in ("beta_exp", "se_exp", "beta_out", "se_out"))
    k = len(h)
    primary = estimators.wald_ratio(bx, sx, by, sy) if k == 1 else estimators.ivw(bx, sx, by, sy)
    row = {
        "method": primary.method,
        "beta": primary.beta,
        "se": primary.se,
        "ci_low": primary.ci_low,
        "ci_high": primary.ci_high,
        "p": primary.p,
        "n_snps": k,
        "egger_beta": np.nan,
        "egger_intercept_p": np.nan,
        "wm_beta": np.nan,
        "wm_se": np.nan,
    }
    if k >= 3:
        eg = estimators.egger(bx, sx, by, sy)
        wm = estimators.weighted_median(bx, sx, by, sy, n_boot=n_boot, seed=seed)
        row.update(
            egger_beta=eg.beta, egger_intercept_p=eg.intercept_p, wm_beta=wm.beta, wm_se=wm.se
        )
    return row


def run_screen(
    exposures: pd.DataFrame,
    outcome: pd.DataFrame,
    kind: str,
    p_threshold: float = 5e-8,
    window_kb: float = 500,
    cis_only: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Return (one row per trait, instruments used per trait, harmonisation log)."""
    sel = exposures[exposures["pval"] < p_threshold]
    if cis_only and "cis" in sel:
        sel = sel[sel["cis"].astype(bool)]
    rows, instruments, logs = [], {}, []
    for trait, exp in sel.groupby("trait", sort=True):
        exp = prune_by_distance(exp, window_kb)
        h = harmonise(exp, outcome)
        logs.append(h.assign(trait=trait, kind=kind))
        usable = h[h["dropped_reason"].isna()]
        if usable.empty:
            continue
        instruments[trait] = usable["snp"].tolist()
        rows.append({"trait": trait, "kind": kind, **screen_trait(usable, n_boot, seed)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["q_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    return res, instruments, log
