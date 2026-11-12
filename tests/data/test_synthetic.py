import numpy as np
import pandas as pd
import pytest

from dementia_ml.data import schema, synthetic


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return synthetic.make_cohort(n_subjects=800, seed=0)


def test_cohort_is_deterministic(cohort):
    assert synthetic.make_cohort(n_subjects=800, seed=0).equals(cohort)


def test_cohort_passes_schema(cohort):
    schema.validate(cohort)


def test_schema_reports_missing_column(cohort):
    with pytest.raises(schema.SchemaError, match="MMSE"):
        schema.validate(cohort.drop(columns="MMSE"))


def test_baseline_rows_unique(cohort):
    bl = cohort[cohort["VISCODE"] == "bl"]
    assert bl["RID"].is_unique
    assert set(bl["RID"]) == set(cohort["RID"])


def test_conversion_rate_in_range(cohort):
    mci = cohort[cohort["DX_bl"].isin(schema.DX_BL_MCI)]
    converted = mci[(mci["DX"] == schema.DX_DEMENTIA) & (mci["M"] <= 48)]["RID"].nunique()
    rate = converted / mci["RID"].nunique()
    assert 0.25 <= rate <= 0.45, rate


def test_cohort_contains_messy_adni_features(cohort):
    bl = cohort[cohort["VISCODE"] == "bl"]
    assert bl["ABETA"].isna().mean() == pytest.approx(0.5, abs=0.06)
    assert (bl["ABETA"].astype(str) == ">1700").any()


def _hippocampus_z(bl: pd.DataFrame) -> pd.Series:
    x = np.column_stack([np.ones(len(bl)), bl["ICV"].to_numpy(float)])
    y = bl["Hippocampus"].to_numpy(float)
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return pd.Series((resid - resid.mean()) / resid.std(), index=bl.index)


@pytest.mark.xfail(reason="needs conversion.build (Task 3)", strict=True)
def test_cox_recovers_true_log_hr():
    from lifelines import CoxPHFitter

    from dementia_ml.data import conversion

    df = synthetic.make_cohort(n_subjects=3000, seed=0)
    data = conversion.build(df, features=["APOE4", "MMSE", "ADAS13", "Hippocampus", "ICV"])
    x = data.X.copy()
    x["Hippocampus_z"] = _hippocampus_z(x)
    frame = x[list(synthetic.TRUE_LOG_HR)].astype(float)
    frame["time"], frame["event"] = data.time, data.event
    fit = CoxPHFitter().fit(frame, "time", "event")
    for name, truth in synthetic.TRUE_LOG_HR.items():
        assert abs(fit.params_[name] - truth) < 0.15, (name, fit.params_[name], truth)
