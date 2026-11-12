import numpy as np
import pandas as pd
import pytest

from dementia_ml.data import conversion, synthetic
from dementia_ml.survival import models, preprocess


def test_imputer_fit_on_train_only():
    train = pd.DataFrame(
        {
            "AGE": [70.0, 72, 74, 76, 78],
            "MMSE": [26.0, 27, 27, 28, np.nan],
            "APOE4": [0.0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame({"AGE": [71.0, 73, 75], "MMSE": [np.nan, 20, 20], "APOE4": [0.0, 1, 1]})
    pre = preprocess.make_preprocessor(["AGE", "MMSE", "APOE4"]).fit(train)
    out = pre.transform(test)
    ref = pre.transform(test.assign(MMSE=[27.0, 20, 20]))
    assert out.loc[0, "MMSE"] == pytest.approx(ref.loc[0, "MMSE"])
    assert "missingindicator_MMSE" in out.columns


def test_preprocessor_derives_hippocampus_z_from_train():
    rng = np.random.default_rng(0)
    icv = rng.normal(1.5e6, 1.5e5, 200)
    hip = 3000 + 0.0027 * icv + rng.normal(0, 850, 200)
    x = pd.DataFrame({"AGE": rng.normal(73, 7, 200), "Hippocampus": hip, "ICV": icv})
    out = preprocess.make_preprocessor(list(x.columns)).fit(x).transform(x)
    assert "Hippocampus_z" in out.columns
    assert "Hippocampus" not in out.columns and "ICV" not in out.columns
    assert abs(np.corrcoef(out["Hippocampus_z"], icv)[0, 1]) < 0.05


def _oof_cindex(name, data):
    res = models.cross_validate(name, data, n_splits=5, seed=0)
    from dementia_ml.eval import metrics

    return metrics.c_index(data.time, data.event, res["risk"])


def test_full_beats_baseline_on_synthetic():
    df = synthetic.make_cohort(n_subjects=1500, seed=1)
    data = conversion.build(df, features=conversion.FULL_FEATURES)
    base = _oof_cindex("cox_baseline", data)
    full = _oof_cindex("cox_full", data)
    assert full - base >= 0.03, (base, full)


def test_cross_validate_returns_survival_at_horizons():
    df = synthetic.make_cohort(n_subjects=400, seed=2)
    data = conversion.build(df, features=conversion.FULL_FEATURES)
    res = models.cross_validate("cox_full", data, n_splits=3, seed=0, horizons=(24.0, 36.0))
    assert res["risk"].shape == (len(data.X),)
    assert res["surv"].shape == (len(data.X), 2)
    assert ((res["surv"] >= 0) & (res["surv"] <= 1)).all()
