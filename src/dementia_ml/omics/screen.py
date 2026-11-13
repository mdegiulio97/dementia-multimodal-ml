"""One Cox model per molecule: conversion ~ molecule (per SD) + covariates."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

from dementia_ml.omics.prep import inverse_normal


def association_screen(
    omics: pd.DataFrame,
    covariates: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    min_n: int = 50,
    penalizer: float = 0.01,
) -> pd.DataFrame:
    """Hazard ratio per SD (after inverse-normal transform) for every column of ``omics``.

    ``omics`` and ``covariates`` share the participant index; ``time``/``event`` are aligned
    to that index. Features observed in fewer than ``min_n`` participants are skipped.
    """
    base = covariates.copy()
    base["time"], base["event"] = np.asarray(time, float), np.asarray(event, bool)
    rows = []
    for feat in omics.columns:
        x = inverse_normal(omics[feat])
        frame = base.assign(feature=x).dropna()
        if len(frame) < min_n or frame["event"].sum() < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = CoxPHFitter(penalizer=penalizer).fit(frame, "time", "event")
            except Exception:  # noqa: BLE001 - convergence failures are reported as skipped
                continue
        s = fit.summary.loc["feature"]
        rows.append(
            {
                "feature": feat,
                "hr_per_sd": float(s["exp(coef)"]),
                "ci_low": float(s["exp(coef) lower 95%"]),
                "ci_high": float(s["exp(coef) upper 95%"]),
                "p": float(s["p"]),
                "n": len(frame),
                "n_events": int(frame["event"].sum()),
            }
        )
    res = pd.DataFrame(rows)
    if not res.empty:
        res["q_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    return res
