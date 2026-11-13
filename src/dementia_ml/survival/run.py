"""End-to-end MCI-to-dementia conversion analysis (synthetic or ADNI)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.util import Surv

from dementia_ml import paths, provenance
from dementia_ml.data import adnimerge, conversion, synthetic
from dementia_ml.eval import bootstrap, metrics
from dementia_ml.survival import figures, models

KINDS = ("synthetic", "adni")


def _load(kind: str, config: dict) -> tuple[pd.DataFrame, str, str]:
    if kind == "synthetic":
        n, seed = config["n_subjects"], config["seed"]
        return synthetic.make_cohort(n, seed), "synthetic", f"make_cohort(n={n}, seed={seed})"
    path = paths.data_root() / config.get("adnimerge_file", "adni/ADNIMERGE.csv")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return adnimerge.load(path), "ADNI ADNIMERGE", f"sha256:{digest}"


def _with_ci(fn, n: int, n_boot: int, seed: int) -> dict:
    est, lo, hi = bootstrap.ci(fn, n=n, n_boot=n_boot, seed=seed)
    return {"estimate": est, "ci_low": lo, "ci_high": hi}


def _evaluate(data, oof, horizons, config) -> dict:
    t, e, risk = data.time, data.event, oof["risk"]
    n, nb, seed = len(t), config["n_boot"], config["seed"]
    out = {
        "n": int(n),
        "n_events": int(e.sum()),
        "c_index": _with_ci(lambda i: metrics.c_index(t[i], e[i], risk[i]), n, nb, seed),
    }
    tau = float(config["uno_tau"])
    if tau < t.max():
        out["uno_c"] = {
            "tau_months": tau,
            **_with_ci(lambda i: metrics.uno_c(t, e, t[i], e[i], risk[i], tau), n, nb, seed),
        }
    out["td_auc"], out["horizons_skipped"] = {}, []
    for h in horizons:
        if not (t.min() <= h < t.max()):
            out["horizons_skipped"].append(h)
            continue
        out["td_auc"][f"{h:g}"] = _with_ci(
            lambda i, h=h: float(metrics.td_auc(t, e, t[i], e[i], risk[i], [h])[0]), n, nb, seed
        )
    return out


def run(kind: str, config: dict, out_root: Path | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out_root = out_root or paths.REPO_ROOT
    res_dir = paths.results_dir(kind, out_root)
    fig_dir = paths.figures_dir(kind, out_root)
    mod_dir = paths.models_dir(kind, out_root)

    df, dataset, version = _load(kind, config)
    data = conversion.build(df, features=conversion.FULL_FEATURES)
    horizons = [float(h) for h in config["horizons"]]
    seed = config["seed"]
    y = Surv.from_arrays(data.event, data.time)

    results, oofs = {}, {}
    for name in models.MODELS:
        params = config.get("models", {}).get(name, {})
        oof = models.cross_validate(
            name, data, n_splits=config["n_splits"], seed=seed, horizons=horizons, **params
        )
        oofs[name] = oof
        results[name] = _evaluate(data, oof, horizons, config)
        final = models.make_pipeline(name, seed=seed, **params).fit(
            data.X[models.features_of(name)], y
        )
        joblib.dump(
            {
                "pipeline": final,
                "features": models.features_of(name),
                "horizons": horizons,
                # Rounded 1st/99th percentiles, never min/max: an extreme is one person's value.
                "train_ranges": {
                    col: {
                        "p01": round(float(data.X[col].quantile(0.01)), 1),
                        "p99": round(float(data.X[col].quantile(0.99)), 1),
                    }
                    for col in models.features_of(name)
                },
            },
            mod_dir / f"conversion_{name}.joblib",
        )

    # Bootstrap refits of the interpretable model: prediction intervals for the agent.
    rng = np.random.default_rng(seed)
    feats = models.features_of("cox_full")
    params = config.get("models", {}).get("cox_full", {})
    boot = []
    for _ in range(int(config.get("n_boot_models", 50))):
        idx = rng.integers(0, len(data.X), len(data.X))
        boot.append(
            models.make_pipeline("cox_full", seed=seed, **params).fit(
                data.X[feats].iloc[idx], y[idx]
            )
        )
    joblib.dump(
        {"pipelines": boot, "features": feats}, mod_dir / "conversion_cox_full_bootstrap.joblib"
    )

    best = max(results, key=lambda m: results[m]["c_index"]["estimate"])
    cal_h = float(config["calibration_horizon"])
    cal = metrics.calibration_at(
        data.time, data.event, oofs[best]["surv"][:, horizons.index(cal_h)], t=cal_h
    )
    figures.km_by_risk_tertile(
        data.time,
        data.event,
        oofs[best]["risk"],
        best,
        fig_dir / "conversion_km_by_risk_tertile.png",
    )
    figures.calibration(cal, cal_h, best, fig_dir / f"conversion_calibration_{cal_h:.0f}m.png")
    figures.cindex_forest(results, fig_dir / "conversion_cindex_forest.png")

    cohort = {
        **data.meta,
        "observed_per_feature": {c: int(data.X[c].notna().sum()) for c in data.X.columns},
    }
    payload = {
        "question": "Which participants with MCI convert to dementia, and when?",
        "cohort": cohort,
        "best_model": best,
        "calibration": {"model": best, "horizon_months": cal_h, "table": cal.to_dict("records")},
        "models": results,
        "provenance": provenance.block(dataset, version, config, seed),
    }
    (res_dir / "conversion_metrics.json").write_text(json.dumps(payload, indent=2))
    pred = pd.DataFrame(
        {
            "time": data.time,
            "event": data.event,
            **{f"risk_{m}": o["risk"] for m, o in oofs.items()},
        },
        index=pd.Index(data.X.index, name="subject_id"),
    )
    pred.to_parquet(res_dir / "conversion_predictions.parquet")
    return {"best_model": best, "c_index": results[best]["c_index"], "n": data.meta["n_included"]}
