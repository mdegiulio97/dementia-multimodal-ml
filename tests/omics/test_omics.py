import numpy as np
import pandas as pd
import pytest

from dementia_ml.omics import enrichment, prep, screen


def test_mr_trait_names_map_to_nightingale_columns():
    assert prep.nmr_column("L-VLDL-TG_percent") == "L_VLDL_TG_PCT"
    assert prep.nmr_column("Gln") == "GLN"
    assert prep.nmr_column("XS-VLDL-FC") == "XS_VLDL_FC"


def test_inverse_normal_transform_is_rank_based_and_keeps_nan():
    x = pd.Series([5.0, 1.0, np.nan, 1000.0, 3.0])
    z = prep.inverse_normal(x)
    assert np.isnan(z.iloc[2])
    assert list(z.dropna().rank()) == list(x.dropna().rank())
    assert abs(z.dropna().mean()) < 1e-9


def test_baseline_sample_is_nearest_within_window():
    values = pd.DataFrame(
        {
            "RID": [1, 1, 2],
            "EXAMDATE": ["2010-02-01", "2011-01-01", "2014-01-01"],
            "A": [1.0, 2.0, 3.0],
        }
    )
    anchors = pd.Series(pd.to_datetime(["2010-01-01", "2012-01-01"]), index=[1, 2])
    out = prep.baseline_samples(values, ["A"], anchors, window_days=180)
    assert out.loc[1, "A"] == 1.0 and 2 not in out.index


def _cohort(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cov = pd.DataFrame(
        {
            "AGE": rng.normal(73, 7, n),
            "MALE": rng.integers(0, 2, n).astype(float),
            "APOE4": rng.integers(0, 3, n).astype(float),
        },
        index=pd.Index(np.arange(n) + 1, name="RID"),
    )
    signal = rng.normal(size=n)
    omics = pd.DataFrame({f"P{i}": rng.normal(size=n) for i in range(30)}, index=cov.index)
    omics["P0"] = signal + rng.normal(0, 0.3, n)
    hazard = np.exp(0.7 * signal + 0.03 * (cov["AGE"] - 73))
    t = rng.exponential(60 / hazard)
    c = rng.uniform(12, 96, n)
    return cov, omics, np.minimum(t, c), t <= c


def test_screen_recovers_a_planted_protein():
    cov, omics, time, event = _cohort()
    res = screen.association_screen(omics, cov, time, event)
    top = res.sort_values("p").iloc[0]
    assert top["feature"] == "P0" and top["hr_per_sd"] > 1.5
    assert {"hr_per_sd", "ci_low", "ci_high", "p", "q_fdr", "n", "n_events"} <= set(res.columns)
    assert (res.loc[res.feature != "P0", "q_fdr"] > 0.05).mean() > 0.9


def test_screen_skips_features_with_too_few_values():
    cov, omics, time, event = _cohort(n=200)
    omics.loc[omics.index[20:], "P1"] = np.nan
    res = screen.association_screen(omics, cov, time, event, min_n=50)
    assert "P1" not in set(res["feature"])


def test_enrichment_detects_signal_in_selected_set():
    rng = np.random.default_rng(1)
    p = np.r_[rng.uniform(0, 0.02, 10), rng.uniform(0, 1, 500)]
    res = pd.DataFrame({"feature": [f"F{i}" for i in range(510)], "p": p, "hr_per_sd": 1.0})
    out = enrichment.compare(res, selected=[f"F{i}" for i in range(10)])
    assert out["mannwhitney_p"] < 0.001 and out["n_selected_tested"] == 10
    null = enrichment.compare(res, selected=[f"F{i}" for i in range(100, 110)])
    assert null["mannwhitney_p"] > 0.01


def test_direction_concordance_with_mr():
    res = pd.DataFrame(
        {"feature": ["A", "B", "C"], "hr_per_sd": [1.3, 0.8, 1.1], "p": [0.01, 0.02, 0.5]}
    )
    mr = pd.Series({"A": 0.2, "B": 0.1, "C": 0.05})
    out = enrichment.direction(res, mr)
    assert out["n"] == 3 and out["same_sign"] == 2
    assert out["binomial_p"] == pytest.approx(0.5, abs=1e-9)


def test_soma_column_to_seqid_and_metabolite_columns():
    from dementia_ml.omics import run

    assert run.seqid("X10000.28") == "10000-28"
    cols = ["RID", "EXAMDATE", "LOW_GLUCOSE", "TOTAL_C", "GLN", "S_HDL_TG_PCT", "update_stamp"]
    assert run.metabolite_columns(cols) == ["TOTAL_C", "GLN", "S_HDL_TG_PCT"]
