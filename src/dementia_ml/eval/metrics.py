"""Discrimination and calibration metrics for survival and classification models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


def _truncate(time, event, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Administrative censoring at ``tau``: IPCW weights are only needed up to ``tau``.

    Without it, events at the last follow-up visit get a zero censoring
    probability and the IPCW estimators fail.
    """
    time, event = np.asarray(time, float), np.asarray(event, bool)
    return np.minimum(time, tau), event & (time <= tau)


def c_index(time, event, risk) -> float:
    """Harrell's C (higher risk should mean earlier event)."""
    return float(
        concordance_index_censored(np.asarray(event, bool), np.asarray(time, float), risk)[0]
    )


def uno_c(train_time, train_event, time, event, risk, tau: float) -> float:
    """Uno's IPCW concordance truncated at ``tau``."""
    train = Surv.from_arrays(np.asarray(train_event, bool), np.asarray(train_time, float))
    t, e = _truncate(time, event, tau)
    test = Surv.from_arrays(e, t)
    return float(concordance_index_ipcw(train, test, np.asarray(risk), tau=tau)[0])


def td_auc(train_time, train_event, time, event, risk, times: Sequence[float]) -> np.ndarray:
    """Cumulative/dynamic AUC at each time in ``times``."""
    train = Surv.from_arrays(np.asarray(train_event, bool), np.asarray(train_time, float))
    times = np.asarray(times, float)
    tau = float(times.max()) + 1e-6  # controls beyond the last horizon stay controls
    t, e = _truncate(time, event, tau)
    test = Surv.from_arrays(e, t)
    auc, _ = cumulative_dynamic_auc(train, test, np.asarray(risk), times)
    return np.asarray(auc, float)


def _km_event_prob(time, event, t: float) -> float:
    grid, surv = kaplan_meier_estimator(np.asarray(event, bool), np.asarray(time, float))
    idx = np.searchsorted(grid, t, side="right") - 1
    return float(1 - (surv[idx] if idx >= 0 else 1.0))


def calibration_at(time, event, surv_prob_at_t, t: float, n_bins: int = 5) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed event probability by ``t``, per risk bin.

    Both columns are probabilities of the event by ``t`` (1 - survival).
    """
    pred_event = 1 - np.asarray(surv_prob_at_t, float)
    bins = pd.qcut(pred_event, q=n_bins, labels=False, duplicates="drop")
    time, event = np.asarray(time, float), np.asarray(event, bool)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append(
            {
                "bin": int(b),
                "predicted": float(pred_event[m].mean()),
                "observed_km": _km_event_prob(time[m], event[m], t),
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["bin", "predicted", "observed_km", "n"])


def auc(y_true, score) -> float:
    return float(roc_auc_score(y_true, score))


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))
