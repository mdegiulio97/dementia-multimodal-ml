"""Cognitive trajectories of MCI converters vs stable participants."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dementia_ml import paths, provenance  # noqa: E402
from dementia_ml.data import conversion  # noqa: E402
from dementia_ml.longitudinal.lmm import fit_trajectory  # noqa: E402
from dementia_ml.survival.run import KINDS, _load  # noqa: E402


def _figure(res, outcome: str, path: Path) -> None:
    fe = res.fixed_effects["coef"]
    cov = res.model.cov_params().loc[fe.index, fe.index]
    years = np.linspace(0, 4, 50)
    fig, ax = plt.subplots(figsize=(5, 4))
    for g, label, color in [(0, "stable MCI", "#2a9d8f"), (1, "converters", "#e76f51")]:
        X = np.column_stack([np.ones_like(years), years, np.full_like(years, g), years * g])
        mean = X @ fe.to_numpy()
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov.to_numpy(), X))
        ax.plot(years, mean, color=color, label=label)
        ax.fill_between(years, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.2)
    ax.set_xlabel("Years from baseline")
    ax.set_ylabel(outcome)
    ax.set_title(f"{outcome}: mixed-model mean trajectory (95% CI)", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(kind: str, config: dict, out_root: Path | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out_root = out_root or paths.REPO_ROOT
    df, dataset, version = _load(kind, config)
    data = conversion.build(df, features=["AGE"])
    groups = dict(zip(data.X.index, data.event.astype(int), strict=True))
    long = df[df["RID"].isin(groups)].copy()
    long["converter"] = long["RID"].map(groups)

    out = {}
    for outcome in config.get("outcomes", ["MMSE", "ADAS13", "CDRSB"]):
        res = fit_trajectory(long, outcome)
        est, lo, hi = res.slope_difference
        out[outcome] = {
            "slope_difference": {"estimate": est, "ci_low": lo, "ci_high": hi, "units": "per year"},
            "fixed_effects": res.fixed_effects.reset_index(names="term").to_dict("records"),
            "n_subjects": res.n_subjects,
            "n_obs": res.n_obs,
        }
        _figure(res, outcome, paths.figures_dir(kind, out_root) / f"trajectories_{outcome}.png")

    payload = {
        "question": "How much faster do converters decline than stable MCI participants?",
        "model": "y ~ years * converter, random intercept and slope per participant (REML)",
        "outcomes": out,
        "provenance": provenance.block(dataset, version, config, config.get("seed", 0)),
    }
    (paths.results_dir(kind, out_root) / "trajectories.json").write_text(
        json.dumps(payload, indent=2)
    )
    return payload
