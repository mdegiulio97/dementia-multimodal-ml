"""Subject-level classification of resting-state EEG features.

Epochs of a participant never cross folds; each participant gets one
prediction (mean probability over their epochs, averaged over CV repeats) and
weighs once in every metric.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dementia_ml.eval import bootstrap, metrics
from dementia_ml.splits import subject_folds

Task = Literal["AD_vs_CN", "FTD_vs_CN", "AD_vs_FTD", "3class"]
TASKS: dict[str, tuple[list[str], str | None]] = {
    "AD_vs_CN": (["CN", "AD"], "AD"),
    "FTD_vs_CN": (["CN", "FTD"], "FTD"),
    "AD_vs_FTD": (["FTD", "AD"], "AD"),
    "3class": (["CN", "AD", "FTD"], None),
}


def _model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(name)


def folds(subjects: pd.Series, y: pd.Series, seed: int = 0, n_splits: int = 5):
    return subject_folds(subjects.to_numpy(), n_splits=n_splits, stratify=y.to_numpy(), seed=seed)


def _subject_metric(y_true: np.ndarray, prob: np.ndarray, classes: list[str]):
    binary = prob.ndim == 1

    def auc(i: np.ndarray) -> float:
        if binary:
            return metrics.auc(y_true[i], prob[i])
        labels = range(len(classes))
        return float(roc_auc_score(y_true[i], prob[i], multi_class="ovr", labels=labels))

    def bacc(i: np.ndarray) -> float:
        pred = (prob[i] >= 0.5).astype(int) if binary else prob[i].argmax(axis=1)
        return metrics.balanced_accuracy(y_true[i], pred)

    return auc, bacc


def run(
    features: pd.DataFrame,
    labels: pd.Series,
    subjects: pd.Series,
    task: Task,
    model: Literal["logreg", "lgbm"],
    seed: int = 0,
    n_repeats: int = 5,
    n_boot: int = 1000,
) -> dict:
    classes, positive = TASKS[task]
    keep = labels.isin(classes).to_numpy()
    X = features.loc[keep].reset_index(drop=True)
    lab = labels.loc[keep].reset_index(drop=True)
    sub = subjects.loc[keep].reset_index(drop=True)
    y = (
        (lab == positive).astype(int)
        if positive
        else lab.map({c: i for i, c in enumerate(classes)})
    )

    subj_y = y.groupby(sub).first()
    sum_prob = None
    for r in range(n_repeats):
        epoch_prob = np.zeros((len(X), 1 if positive else len(classes)))
        for train, test in folds(sub, y, seed=seed + r):
            clf = _model(model, seed + r).fit(X.iloc[train], y.iloc[train])
            p = clf.predict_proba(X.iloc[test])
            epoch_prob[test] = p[:, [1]] if positive else p
        per_subject = pd.DataFrame(epoch_prob, index=sub).groupby(level=0).mean()
        sum_prob = per_subject if sum_prob is None else sum_prob + per_subject
    subj_prob = (sum_prob / n_repeats).loc[subj_y.index]
    prob = subj_prob.iloc[:, 0].to_numpy() if positive else subj_prob.to_numpy()
    yt = subj_y.to_numpy()
    auc_fn, bacc_fn = _subject_metric(yt, prob, classes)

    def with_ci(fn):
        est, lo, hi = bootstrap.ci(fn, n=len(yt), n_boot=n_boot, seed=seed)
        return {"estimate": est, "ci_low": lo, "ci_high": hi}

    return {
        "task": task,
        "model": model,
        "classes": classes,
        "n_subjects": int(len(yt)),
        "n_epochs": int(len(X)),
        "class_counts": {c: int((lab.groupby(sub).first() == c).sum()) for c in classes},
        "auc": with_ci(auc_fn),
        "balanced_accuracy": with_ci(bacc_fn),
        "subject_predictions": pd.DataFrame(
            subj_prob.to_numpy(),
            index=subj_prob.index,
            columns=["prob"] if positive else [f"prob_{c}" for c in classes],
        ).assign(label=lab.groupby(sub).first().loc[subj_prob.index].to_numpy()),
    }
