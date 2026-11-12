import numpy as np
import pandas as pd

from dementia_ml.eeg import classify


def _features(n_per_group=30, shift=3.0, epochs=20, seed=0, uneven=False):
    rng = np.random.default_rng(seed)
    rows, labels, subjects = [], [], []
    for g, group in enumerate(["CN", "AD", "FTD"]):
        for s in range(n_per_group):
            sid = f"{group}{s}"
            n_ep = 1000 if (uneven and s == 0 and group == "AD") else epochs
            subject_mean = rng.normal(g * shift, 1.0)
            for _ in range(n_ep):
                rows.append({"f1": subject_mean + rng.normal(0, 1.0), "f2": rng.normal()})
                labels.append(group)
                subjects.append(sid)
    return pd.DataFrame(rows), pd.Series(labels), pd.Series(subjects)


def test_separable_synthetic_groups_auc_high():
    X, y, s = _features()
    res = classify.run(X, y, s, task="AD_vs_CN", model="logreg", seed=0, n_repeats=2)
    assert res["auc"]["estimate"] > 0.9
    assert res["auc"]["ci_low"] < res["auc"]["estimate"] < res["auc"]["ci_high"]


def test_epochs_of_subject_never_split():
    X, y, s = _features(n_per_group=10)
    for train, test in classify.folds(s, y, seed=0):
        assert not set(s.iloc[train]) & set(s.iloc[test])


def test_subjects_weigh_equally():
    X, y, s = _features(uneven=True)
    res = classify.run(X, y, s, task="AD_vs_CN", model="logreg", seed=0, n_repeats=1)
    preds = res["subject_predictions"]
    assert len(preds) == 60  # 30 AD + 30 CN subjects, regardless of epoch counts
    assert res["n_subjects"] == 60


def test_three_class_and_lgbm_run():
    X, y, s = _features(n_per_group=12)
    res = classify.run(X, y, s, task="3class", model="lgbm", seed=0, n_repeats=1)
    assert 0 <= res["balanced_accuracy"]["estimate"] <= 1
    assert res["n_subjects"] == 36
