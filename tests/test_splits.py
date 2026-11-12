import numpy as np

from dementia_ml.splits import subject_folds


def test_subject_folds_no_overlap():
    subjects = np.repeat(np.arange(50), 4)
    folds = subject_folds(subjects, n_splits=5, seed=0)
    assert len(folds) == 5
    seen_in_test = []
    for train, test in folds:
        assert not set(subjects[train]) & set(subjects[test])
        seen_in_test.extend(np.unique(subjects[test]))
    assert sorted(seen_in_test) == list(range(50))


def test_subject_folds_stratified_keeps_event_share():
    subjects = np.arange(200)
    y = np.r_[np.ones(60), np.zeros(140)]
    for _, test in subject_folds(subjects, n_splits=5, stratify=y, seed=1):
        assert abs(y[test].mean() - 0.3) < 0.06


def test_subject_folds_deterministic():
    s = np.repeat(np.arange(30), 2)
    a = subject_folds(s, seed=4)
    b = subject_folds(s, seed=4)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b, strict=True))
