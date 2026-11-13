import json

import joblib
import numpy as np
import pytest

from dementia_ml.agent import tools

PROFILE = {"age": 74.0, "mmse": 27.0, "apoe4": 1.0, "adas13": 14.0, "horizon_months": 36}


@pytest.fixture(scope="module")
def reg(workspace):
    return tools.build_registry(kind="synthetic", root=workspace)


def test_registry_has_expected_tools(reg):
    assert set(reg) == {
        "describe_data",
        "cohort_summary",
        "predict_conversion_risk",
        "get_result",
        "explain_prediction",
    }
    for t in reg.values():
        assert t.parameters["type"] == "object" and t.description


def test_cohort_summary_refuses_small_cells(reg):
    r = reg["cohort_summary"].fn(filters={"AGE": [90, 200], "APOE4": 2}, variable="MMSE")
    assert r.refused and r.value is None


def test_cohort_summary_refuses_individual_ids(reg):
    r = reg["cohort_summary"].fn(filters={"RID": 10001}, variable="MMSE")
    assert r.refused and "individual" in r.refused


def test_cohort_summary_returns_aggregate(reg):
    r = reg["cohort_summary"].fn(filters={"APOE4": 0}, variable="MMSE")
    assert r.refused is None
    assert r.value["n"] >= tools.MIN_CELL
    assert 15 <= r.value["mean"] <= 30
    assert r.provenance["dataset"] == "synthetic"


def test_predict_risk_matches_model(reg, workspace):
    r = reg["predict_conversion_risk"].fn(**PROFILE)
    bundle = joblib.load(workspace / "models/synthetic/conversion_cox_full.joblib")
    x = tools.profile_frame(PROFILE, bundle["features"])
    fn = bundle["pipeline"].predict_survival_function(x)[0]
    assert r.value["probability"] == pytest.approx(1 - fn(36.0))
    lo, hi = r.uncertainty["ci_low"], r.uncertainty["ci_high"]
    assert 0 <= lo <= hi <= 1
    assert not r.value["warnings"]


def test_predict_risk_warns_out_of_range(reg):
    r = reg["predict_conversion_risk"].fn(**{**PROFILE, "mmse": 40.0})
    assert any("MMSE" in w for w in r.value["warnings"])


def test_get_result_reads_files(reg, workspace):
    r = reg["get_result"].fn(name="conversion_metrics")
    on_disk = json.loads((workspace / "results/synthetic/conversion_metrics.json").read_text())
    assert r.value["models"]["cox_full"]["c_index"] == on_disk["models"]["cox_full"]["c_index"]
    assert r.provenance["git_commit"]
    mr = reg["get_result"].fn(name="mr_candidates")
    assert mr.value["top"][0]["trait"] == "PROT_A"
    assert reg["get_result"].fn(name="nonexistent").refused


def test_explain_prediction_is_additive(reg, workspace):
    r = reg["explain_prediction"].fn(**PROFILE)
    contrib = r.value["contributions"]
    bundle = joblib.load(workspace / "models/synthetic/conversion_cox_full.joblib")
    x = tools.profile_frame(PROFILE, bundle["features"])
    lp = float(bundle["pipeline"].predict(x)[0])
    assert np.isclose(sum(c["log_hazard"] for c in contrib), lp, atol=1e-6)


def test_cohort_summary_refuses_continuous_range_filters(reg):
    r = reg["cohort_summary"].fn(filters={"AGE": [60, 93.8]}, variable="MMSE")
    assert r.refused and "categorical" in r.refused


def test_disclosure_check_blocks_differencing():
    import pandas as pd

    df = pd.DataFrame(
        {
            "age_band": ["85+"] * 15 + ["65-74"] * 40,
            "converted": [0.0] * 12 + [1.0] * 3 + [0.0] * 40,
        }
    )
    # {85+, converted=0} has 12 people, but its sibling {85+, converted=1} has 3:
    # subtracting the two answers would reveal those 3 people.
    assert tools.disclosure_problem(df, {"age_band": "85+", "converted": 0.0}) is not None
    assert tools.disclosure_problem(df, {"age_band": "65-74"}) is None


def test_cohort_summary_hides_quartiles_in_small_cells(reg):
    r = reg["cohort_summary"].fn(filters={"APOE4": 0}, variable="MMSE")
    assert r.value["n"] >= 20 and "q1" in r.value
    assert r.value["mean"] == round(r.value["mean"], 1)


def test_training_ranges_are_rounded_percentiles(workspace):
    bundle = joblib.load(workspace / "models/synthetic/conversion_cox_full.joblib")
    age = bundle["train_ranges"]["AGE"]
    assert set(age) == {"p01", "p99"}


def test_predict_risk_at_end_of_follow_up(reg):
    r = reg["predict_conversion_risk"].fn(**{**PROFILE, "horizon_months": 48})
    assert (
        0 <= r.uncertainty["ci_low"] <= r.value["probability"] <= r.uncertainty["ci_high"] <= 1
        or r.value["warnings"]
    )
    assert r.value["horizon_used_months"] <= 48


def test_cohort_summary_accepts_lowercase(reg):
    upper = reg["cohort_summary"].fn(filters={"APOE4": 0}, variable="MMSE")
    lower = reg["cohort_summary"].fn(filters={"apoe4": 0}, variable="mmse")
    assert lower.refused is None and lower.value["mean"] == upper.value["mean"]


def test_cohort_summary_refuses_lowercase_ids(reg):
    r = reg["cohort_summary"].fn(filters={"rid": 10001}, variable="mmse")
    assert r.refused and "individual" in r.refused
