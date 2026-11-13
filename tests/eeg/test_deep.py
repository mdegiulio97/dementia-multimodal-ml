import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("braindecode")

from dementia_ml.eeg import deep  # noqa: E402


def _windows(n_per_group=12, per_subject=8, n_ch=4, sfreq=125, seconds=2, seed=0):
    """Two groups that differ in 10 Hz amplitude; one row of windows per subject."""
    rng = np.random.default_rng(seed)
    t = np.arange(sfreq * seconds) / sfreq
    X, labels, subjects = [], [], []
    for group in ["CN", "AD"]:
        for s in range(n_per_group):
            amp = 2.0 if group == "CN" else 0.5
            for _ in range(per_subject):
                x = amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6)) + rng.normal(
                    0, 1, (n_ch, len(t))
                )
                X.append(x)
                labels.append(group)
                subjects.append(f"{group}{s}")
    return np.asarray(X, dtype=np.float32), pd.Series(labels), pd.Series(subjects)


def test_eegnet_windows_never_cross_subjects():
    _, y, s = _windows()
    for train, val, test in deep.fold_indices(s, y, n_splits=3, seed=0):
        groups = [set(s.iloc[i]) for i in (train, val, test)]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        assert groups[1], "inner validation set must not be empty"


def test_eegnet_learns_synthetic_alpha_difference():
    X, y, s = _windows()
    res = deep.run(X, y, s, task="AD_vs_CN", seed=0, max_epochs=4, n_splits=3, sfreq=125)
    assert {"auc", "balanced_accuracy", "subject_predictions", "n_subjects", "classes"} <= set(res)
    assert res["n_subjects"] == 24 and len(res["subject_predictions"]) == 24
    assert res["train_loss_first"] > res["train_loss_last"]
