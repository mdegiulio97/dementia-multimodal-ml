import json

import pytest

from dementia_ml.data import synthetic
from dementia_ml.survival import run

SMALL = {
    "n_subjects": 500,
    "seed": 0,
    "n_splits": 3,
    "n_boot": 30,
    "horizons": [24, 36, 48],
    "calibration_horizon": 36,
    "uno_tau": 36,
    "models": {"rsf": {"n_estimators": 50}, "gbsa": {"n_estimators": 50}},
    "n_boot_models": 5,
}


def test_pipeline_synthetic_end_to_end(tmp_path):
    summary = run.run("synthetic", SMALL, out_root=tmp_path)
    res = json.loads((tmp_path / "results/synthetic/conversion_metrics.json").read_text())
    assert res["provenance"]["git_commit"]
    assert set(res["models"]) == {"cox_baseline", "cox_full", "rsf", "gbsa"}
    for m in res["models"].values():
        c = m["c_index"]
        assert c["ci_low"] < c["estimate"] < c["ci_high"]
        assert m["n"] > 0 and m["n_events"] > 0
    for fig in [
        "conversion_km_by_risk_tertile.png",
        "conversion_calibration_36m.png",
        "conversion_cindex_forest.png",
    ]:
        assert (tmp_path / "figures/synthetic" / fig).stat().st_size > 1000
    assert (tmp_path / "models/synthetic/conversion_cox_full.joblib").exists()
    import joblib

    boot = joblib.load(tmp_path / "models/synthetic/conversion_cox_full_bootstrap.joblib")
    assert len(boot["pipelines"]) == 5
    assert summary["best_model"] in res["models"]


def test_pipeline_adni_writes_only_to_private_dirs(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    (data_root / "adni").mkdir(parents=True)
    synthetic.make_cohort(n_subjects=400, seed=5).to_csv(
        data_root / "adni/ADNIMERGE.csv", index=False
    )
    monkeypatch.setenv("DEMENTIA_ML_DATA", str(data_root))
    out = tmp_path / "repo"
    run.run("adni", SMALL, out_root=out)
    written = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    assert written
    assert all(w.split("/")[1] == "adni" for w in written), written


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        run.run("open", SMALL, out_root=tmp_path)
