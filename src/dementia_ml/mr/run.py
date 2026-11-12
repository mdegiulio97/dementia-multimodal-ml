"""Proteome- and metabolome-wide MR screen for Alzheimer's disease (open data)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dementia_ml import paths, provenance  # noqa: E402
from dementia_ml.mr.screen import run_screen  # noqa: E402

CIS_ONLY = {"pqtl": True, "mqtl": False}


def select(res: pd.DataFrame, fdr: float) -> list[str]:
    """FDR-significant traits without evidence of directional pleiotropy."""
    ok = res["q_fdr"] < fdr
    no_pleio = res["egger_intercept_p"].isna() | (res["egger_intercept_p"] >= 0.05)
    return sorted(res.loc[ok & no_pleio, "trait"])


def _volcano(res: pd.DataFrame, fdr: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"pqtl": "#264653", "mqtl": "#e76f51"}
    for kind, g in res.groupby("kind"):
        sig = g["q_fdr"] < fdr
        ax.scatter(
            g.loc[~sig, "beta"], -np.log10(g.loc[~sig, "p"]), s=8, color=colors.get(kind), alpha=0.3
        )
        ax.scatter(
            g.loc[sig, "beta"],
            -np.log10(g.loc[sig, "p"]),
            s=18,
            color=colors.get(kind),
            label=f"{'proteins' if kind == 'pqtl' else 'metabolites'} (FDR<{fdr}: {sig.sum()})",
        )
    ax.set_xlabel("MR estimate (log-odds of AD per SD of exposure)")
    ax.set_ylabel("-log10 p")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _forest(res: pd.DataFrame, path: Path, top: int = 20) -> None:
    g = res.nsmallest(top, "p").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(g) + 1.2))
    y = np.arange(len(g))
    ax.errorbar(
        g["beta"],
        y,
        xerr=[g["beta"] - g["ci_low"], g["ci_high"] - g["beta"]],
        fmt="o",
        color="#264653",
    )
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(
        y,
        [f"{t} ({k}, {n} SNP)" for t, k, n in zip(g["trait"], g["kind"], g["n_snps"], strict=True)],
        fontsize=7,
    )
    ax.set_xlabel("MR estimate, 95% CI (log-odds of AD per SD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(
    exposures: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    config: dict,
    out_root: Path | None = None,
    dataset_names: dict | None = None,
    dataset_versions: dict | None = None,
) -> dict:
    out_root = out_root or paths.REPO_ROOT
    res_dir, fig_dir = paths.results_dir("open", out_root), paths.figures_dir("open", out_root)
    tables, instruments, logs = [], {}, []
    for kind, exp in exposures.items():
        res, inst, log = run_screen(
            exp,
            outcome,
            kind,
            p_threshold=config["p_threshold"],
            window_kb=config["window_kb"],
            cis_only=CIS_ONLY.get(kind, True),
            n_boot=config["n_boot"],
            seed=config["seed"],
            exclude_regions=config.get("exclude_regions"),
        )
        tables.append(res)
        instruments.update(inst)
        logs.append(log)
    res = pd.concat(tables, ignore_index=True).sort_values("p")
    selected = select(res, config["fdr"])
    res.to_csv(res_dir / "mr_candidates.csv", index=False)
    (res_dir / "mr_candidates_selected.txt").write_text("\n".join(selected) + "\n")
    log = pd.concat(logs, ignore_index=True)
    summary = {
        "question": "Which circulating proteins and metabolites have genetic evidence of a causal effect on Alzheimer's disease?",
        "n_traits_tested": {k: int((res["kind"] == k).sum()) for k in exposures},
        "n_fdr_significant": {
            k: int(((res["kind"] == k) & (res["q_fdr"] < config["fdr"])).sum()) for k in exposures
        },
        "selected_for_adni": selected,
        "instruments_dropped": log["dropped_reason"].value_counts().to_dict()
        if not log.empty
        else {},
        "provenance": provenance.block(
            dataset_names or {}, json.dumps(dataset_versions or {}), config, config["seed"]
        ),
    }
    (res_dir / "mr_candidates.json").write_text(json.dumps(summary, indent=2))
    _volcano(res, config["fdr"], fig_dir / "mr_volcano.png")
    _forest(res, fig_dir / "mr_top_forest.png")
    return {"table": res, "instruments": instruments, "selected": selected}
