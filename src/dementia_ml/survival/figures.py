"""Figures for the conversion analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402

LABELS = {
    "cox_baseline": "Cox, age+MMSE+APOE4",
    "cox_full": "Cox (ridge), all baseline",
    "rsf": "Random survival forest",
    "gbsa": "Gradient-boosted survival",
}


def km_by_risk_tertile(time, event, risk, model: str, path: Path) -> None:
    tert = pd.qcut(risk, 3, labels=["low", "middle", "high"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, color in zip(
        ["low", "middle", "high"], ["#2a9d8f", "#e9c46a", "#e76f51"], strict=True
    ):
        m = np.asarray(tert == label)
        t, s = kaplan_meier_estimator(event[m], time[m])
        t, s = np.r_[0.0, t], np.r_[1.0, s]
        ax.step(t, 1 - s, where="post", color=color, label=f"{label} risk (n={m.sum()})")
    ax.set_xlabel("Months from baseline")
    ax.set_ylabel("Cumulative conversion to dementia")
    ax.set_title(f"Out-of-fold risk tertiles: {LABELS.get(model, model)}", fontsize=10)
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration(cal: pd.DataFrame, horizon: float, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.plot(cal["predicted"], cal["observed_km"], "o-", color="#264653")
    ax.set_xlabel(f"Predicted conversion by {horizon:.0f} months")
    ax.set_ylabel("Observed (Kaplan-Meier)")
    ax.set_title(f"Calibration: {LABELS.get(model, model)}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cindex_forest(results: dict, path: Path) -> None:
    names = list(results)
    est = [results[n]["c_index"]["estimate"] for n in names]
    lo = [results[n]["c_index"]["ci_low"] for n in names]
    hi = [results[n]["c_index"]["ci_high"] for n in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.6 * len(names) + 1.2))
    ax.errorbar(est, y, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o", color="#264653")
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.set_yticks(y, [LABELS.get(n, n) for n in names])
    ax.set_xlabel("Harrell's C-index (out-of-fold, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
