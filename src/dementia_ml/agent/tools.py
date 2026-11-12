"""Tools the agent can call. Every number the agent reports comes from one of these.

Each tool returns a ToolResult with the value, its uncertainty when there is
one, and the provenance of the file or model it was computed from. Cohort
queries return aggregates only and refuse cells smaller than MIN_CELL or any
filter on participant identifiers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from dementia_ml import paths
from dementia_ml.data import adnimerge, conversion, synthetic

MIN_CELL = 10
ID_COLUMNS = {"RID", "PTID", "subject", "subject_id"}
PROFILE_COLUMNS = {
    "age": "AGE",
    "mmse": "MMSE",
    "apoe4": "APOE4",
    "education": "PTEDUCAT",
    "adas13": "ADAS13",
    "cdrsb": "CDRSB",
    "faq": "FAQ",
    "hippocampus": "Hippocampus",
    "icv": "ICV",
    "abeta": "ABETA",
    "tau": "TAU",
    "ptau": "PTAU",
}
RESULTS = {
    "conversion_metrics": ("kind", "conversion_metrics.json"),
    "trajectories": ("kind", "trajectories.json"),
    "mr_candidates": ("open", "mr_candidates.json"),
    "eeg_metrics": ("open", "eeg_metrics.json"),
}


@dataclass
class ToolResult:
    value: Any = None
    uncertainty: dict | None = None
    provenance: dict = field(default_factory=dict)
    refused: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., ToolResult]


def _clean(obj):
    """JSON-safe copy (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def profile_frame(profile: dict, features: list[str]) -> pd.DataFrame:
    row = {col: np.nan for col in features}
    for key, col in PROFILE_COLUMNS.items():
        if profile.get(key) is not None and col in row:
            row[col] = float(profile[key])
    return pd.DataFrame([row], columns=features)


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            k: {"type": "number", "description": f"baseline {v}"}
            for k, v in PROFILE_COLUMNS.items()
        },
        "horizon_months": {"type": "integer", "enum": [24, 36, 48]},
    },
    "required": ["age", "mmse", "apoe4"],
}


class _Context:
    def __init__(self, kind: str, root: Path):
        self.kind, self.root = kind, root
        self._cohort: pd.DataFrame | None = None

    def result_path(self, name: str) -> Path:
        where, file = RESULTS[name]
        return self.root / "results" / (self.kind if where == "kind" else "open") / file

    def read_json(self, name: str) -> dict:
        return json.loads(self.result_path(name).read_text())

    def model(self, suffix: str = "") -> dict:
        return joblib.load(self.root / "models" / self.kind / f"conversion_cox_full{suffix}.joblib")

    def cohort(self) -> pd.DataFrame:
        """Baseline table of the MCI cohort with outcome columns (never leaves the tools)."""
        if self._cohort is None:
            if self.kind == "synthetic":
                cfg = self.read_json("conversion_metrics")["provenance"]["config"]
                df = synthetic.make_cohort(cfg["n_subjects"], cfg["seed"])
            else:
                df = adnimerge.load(paths.data_root() / "adni" / "ADNIMERGE.csv")
            data = conversion.build(df, features=conversion.FULL_FEATURES)
            self._cohort = data.X.assign(
                converted=data.event.astype(float), follow_up_months=data.time
            ).reset_index(drop=True)
        return self._cohort


def build_registry(kind: str = "synthetic", root: Path | None = None) -> dict[str, Tool]:
    ctx = _Context(kind, root or paths.REPO_ROOT)

    def describe_data() -> ToolResult:
        conv = ctx.read_json("conversion_metrics")
        available = [n for n in RESULTS if ctx.result_path(n).exists()]
        return ToolResult(
            value={
                "data": kind,
                "cohort": conv["cohort"],
                "baseline_variables": conversion.FULL_FEATURES,
                "outcome": "first visit with a dementia diagnosis (MCI at baseline)",
                "results_available": available,
                "other_sources": {
                    "eeg": "OpenNeuro ds004504 (AD, FTD, controls)",
                    "mr": "UKB-PPP pQTL, NMR mQTL, AD GWAS GCST90027158",
                },
            },
            provenance=conv["provenance"],
        )

    def cohort_summary(filters: dict | None = None, variable: str = "MMSE") -> ToolResult:
        filters = filters or {}
        if ID_COLUMNS & (set(filters) | {variable}):
            return ToolResult(refused="individual-level queries are not allowed")
        df = ctx.cohort()
        unknown = (set(filters) | {variable}) - set(df.columns)
        if unknown:
            return ToolResult(
                refused=f"unknown variables: {sorted(unknown)}; available: {list(df.columns)}"
            )
        mask = pd.Series(True, index=df.index)
        for col, cond in filters.items():
            if isinstance(cond, list | tuple) and len(cond) == 2:
                mask &= df[col].between(cond[0], cond[1])
            else:
                mask &= df[col] == cond
        values = df.loc[mask, variable].dropna()
        if len(values) < MIN_CELL:
            return ToolResult(
                refused=f"fewer than {MIN_CELL} participants match; aggregate withheld"
            )
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        return ToolResult(
            value={
                "variable": variable,
                "filters": filters,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "median": float(med),
                "q1": float(q1),
                "q3": float(q3),
            },
            uncertainty={"se_mean": float(values.std() / np.sqrt(len(values)))},
            provenance={"dataset": kind, "table": "MCI baseline cohort", "min_cell": MIN_CELL},
        )

    def predict_conversion_risk(horizon_months: int = 36, **profile) -> ToolResult:
        bundle = ctx.model()
        x = profile_frame(profile, bundle["features"])
        fn = bundle["pipeline"].predict_survival_function(x)[0]
        h = float(horizon_months)
        prob = float(1 - fn(min(max(h, fn.domain[0]), fn.domain[1])))
        boot = ctx.model("_bootstrap")["pipelines"]
        draws = [1 - float(p.predict_survival_function(x)[0](h)) for p in boot]
        warnings = []
        for col, rng in bundle["train_ranges"].items():
            v = x[col].iloc[0]
            if not np.isnan(v) and not rng["min"] <= v <= rng["max"]:
                warnings.append(
                    f"{col}={v:g} outside training range [{rng['min']:g}, {rng['max']:g}]"
                )
        missing = [c for c in bundle["features"] if np.isnan(x[c].iloc[0])]
        return ToolResult(
            value={
                "probability": prob,
                "horizon_months": horizon_months,
                "model": "cox_full",
                "imputed_features": missing,
                "warnings": warnings,
            },
            uncertainty={
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "method": f"{len(boot)} bootstrap refits",
            },
            provenance=ctx.read_json("conversion_metrics")["provenance"],
        )

    def get_result(name: str) -> ToolResult:
        if name not in RESULTS or not ctx.result_path(name).exists():
            available = [n for n in RESULTS if ctx.result_path(n).exists()]
            return ToolResult(refused=f"unknown result {name!r}; available: {available}")
        payload = ctx.read_json(name)
        prov = payload.pop("provenance", {})
        if name == "mr_candidates":
            table = pd.read_csv(ctx.result_path(name).with_suffix(".csv"))
            payload["top"] = table.head(15).to_dict("records")
        return ToolResult(value=_clean(payload), provenance=prov)

    def explain_prediction(horizon_months: int = 36, **profile) -> ToolResult:
        bundle = ctx.model()
        pipe = bundle["pipeline"]
        x = profile_frame(profile, bundle["features"])
        xs = pipe.named_steps["pre"].transform(x)
        coef = pipe.named_steps["model"].coef_
        contrib = sorted(
            (
                {"feature": c, "log_hazard": float(b * v)}
                for c, b, v in zip(xs.columns, coef, xs.iloc[0], strict=True)
            ),
            key=lambda d: -abs(d["log_hazard"]),
        )
        return ToolResult(
            value={
                "contributions": contrib,
                "method": "exact linear SHAP of the Cox linear predictor (baseline = training mean)",
            },
            provenance=ctx.read_json("conversion_metrics")["provenance"],
        )

    return {
        "describe_data": Tool(
            "describe_data",
            "What data, cohort sizes, variables and results this project has.",
            {"type": "object", "properties": {}},
            describe_data,
        ),
        "cohort_summary": Tool(
            "cohort_summary",
            "Aggregate statistics (n, mean, sd, quartiles) of one baseline variable in the MCI "
            "cohort, optionally filtered. Filters: {column: value} or {column: [low, high]}. "
            "Columns include AGE, MMSE, APOE4, ADAS13, CDRSB, FAQ, converted, follow_up_months.",
            {
                "type": "object",
                "properties": {
                    "filters": {"type": "object"},
                    "variable": {"type": "string"},
                },
                "required": ["variable"],
            },
            cohort_summary,
        ),
        "predict_conversion_risk": Tool(
            "predict_conversion_risk",
            "Probability that a person with MCI and the given baseline values converts to "
            "dementia within the horizon, with a 95% bootstrap interval.",
            PROFILE_SCHEMA,
            predict_conversion_risk,
        ),
        "get_result": Tool(
            "get_result",
            "Read a stored analysis result: conversion_metrics (model C-index, AUC, "
            "calibration), trajectories (cognitive decline of converters vs stable), "
            "mr_candidates (Mendelian randomisation of proteins/metabolites on AD), "
            "eeg_metrics (EEG classification of AD, FTD, controls).",
            {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(RESULTS)}},
                "required": ["name"],
            },
            get_result,
        ),
        "explain_prediction": Tool(
            "explain_prediction",
            "Which baseline values push the predicted conversion risk up or down for a profile.",
            PROFILE_SCHEMA,
            explain_prediction,
        ),
    }
