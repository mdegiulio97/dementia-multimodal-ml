"""Test the MR-selected proteins (CSF SomaScan 7k) and metabolites (NMR) against conversion
in ADNI participants with MCI.

Outputs (aggregate statistics only) go to results/adni and figures/adni.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dementia_ml import paths, provenance  # noqa: E402
from dementia_ml.data import adnimerge, conversion  # noqa: E402
from dementia_ml.omics import enrichment, prep, screen  # noqa: E402

FIRST_NMR, LAST_NMR = "TOTAL_C", "S_HDL_TG_PCT"


def seqid(column: str) -> str:
    """SomaScan matrix column (X10000.28) -> SeqId (10000-28)."""
    return column.removeprefix("X").replace(".", "-")


def metabolite_columns(columns: list[str]) -> list[str]:
    cols = list(columns)
    return cols[cols.index(FIRST_NMR) : cols.index(LAST_NMR) + 1]


def _cohort(window_days: int):
    df = adnimerge.load(paths.data_root() / "adni" / "ADNIMERGE.csv")
    data = conversion.build(df, features=["AGE", "APOE4"])
    bl = df[df["VISCODE"] == "bl"].set_index("RID").loc[data.X.index]
    cov = pd.DataFrame(
        {
            "AGE": data.X["AGE"],
            "MALE": (bl["PTGENDER"] == "Male").astype(float),
            "APOE4": data.X["APOE4"],
        }
    )
    anchors = pd.to_datetime(bl["EXAMDATE"], errors="coerce").dropna()
    return cov, data.time, data.event, anchors


def _screen_block(values, features, cov, time, event, anchors, window_days):
    base = prep.baseline_samples(values, features, anchors, window_days)
    idx = cov.index.intersection(base.index)
    pos = cov.index.get_indexer(idx)
    res = screen.association_screen(base.loc[idx], cov.loc[idx], time[pos], event[pos])
    return res, len(idx), int(event[pos].sum())


def _qq(res: pd.DataFrame, selected_mask: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for mask, label, color, size in [
        (~selected_mask, "all other molecules", "#9aa5ab", 6),
        (selected_mask, "selected by MR", "#e76f51", 22),
    ]:
        p = np.sort(res.loc[mask, "p"].to_numpy())
        if not len(p):
            continue
        expected = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
        ax.scatter(expected, -np.log10(p), s=size, color=color, label=f"{label} (n={len(p)})")
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
    ax.set_xlabel("Expected -log10 p")
    ax.set_ylabel("Observed -log10 p (conversion, Cox)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _forest(res: pd.DataFrame, label_col: str, path: Path) -> None:
    g = res.sort_values("hr_per_sd")
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(g) + 1.3))
    y = np.arange(len(g))
    ax.errorbar(
        g["hr_per_sd"],
        y,
        xerr=[g["hr_per_sd"] - g["ci_low"], g["ci_high"] - g["hr_per_sd"]],
        fmt="o",
        color="#264653",
        ms=4,
    )
    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.set_yticks(y, g[label_col], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio of conversion per SD (95% CI), adjusted for age, sex, APOE4")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(raw_dir: Path, window_days: int = 180, out_root: Path | None = None) -> dict:
    out_root = out_root or paths.REPO_ROOT
    res_dir, fig_dir = paths.results_dir("adni", out_root), paths.figures_dir("adni", out_root)
    cov, time, event, anchors = _cohort(window_days)
    mr = pd.read_csv(paths.REPO_ROOT / "results" / "open" / "mr_candidates.csv")
    selected = (
        (paths.REPO_ROOT / "results" / "open" / "mr_candidates_selected.txt").read_text().split()
    )
    summary: dict = {"selected_mr": len(selected)}

    # Proteins: CSF SomaScan 7k
    soma = pd.read_csv(raw_dir / "CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv")
    info = pd.read_csv(
        raw_dir / "ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"
    )
    gene_of = dict(
        zip(info["SeqId"].astype(str), info["EntrezGeneSymbol"].astype(str), strict=False)
    )
    apt = [c for c in soma.columns if c.startswith("X") and c[1:2].isdigit()]
    prot, n_p, ev_p = _screen_block(soma, apt, cov, time, event, anchors, window_days)
    prot["gene"] = prot["feature"].map(lambda c: gene_of.get(seqid(c), ""))
    prot["selected"] = prot["gene"].isin(selected)
    mr_beta = mr.set_index("trait")["beta"]
    prot_mr = prot.assign(mr_beta=prot["gene"].map(mr_beta))
    summary["proteins"] = {
        "participants": n_p,
        "conversions": ev_p,
        "aptamers_tested": len(prot),
        "fdr_significant": int((prot["q_fdr"] < 0.05).sum()),
        "selected_genes_measured": int(prot.loc[prot["selected"], "gene"].nunique()),
        "enrichment": enrichment.compare(prot, prot.loc[prot["selected"], "feature"].tolist()),
        "direction": enrichment.direction(
            prot[prot["selected"]], prot_mr[prot_mr["selected"]].set_index("feature")["mr_beta"]
        ),
    }
    prot.to_csv(res_dir / "omics_proteins.csv", index=False)
    _qq(
        prot,
        prot["selected"],
        "CSF proteins (SomaScan 7k) vs conversion",
        fig_dir / "omics_proteins_qq.png",
    )
    sel_p = prot[prot["selected"]].assign(label=lambda d: d["gene"] + " (" + d["feature"] + ")")
    if len(sel_p):
        _forest(sel_p, "label", fig_dir / "omics_proteins_selected_forest.png")

    # Metabolites: Nightingale NMR
    nmr_file = sorted(raw_dir.glob("ADNINIGHTINGALE2_[0-9]*.csv"))[-1]
    nmr = pd.read_csv(nmr_file)
    met_cols = metabolite_columns(list(nmr.columns))
    for c in met_cols:
        nmr[c] = pd.to_numeric(nmr[c], errors="coerce")
    met, n_m, ev_m = _screen_block(nmr, met_cols, cov, time, event, anchors, window_days)
    sel_met = {prep.nmr_column(t): t for t in selected if prep.nmr_column(t) in met_cols}
    met["selected"] = met["feature"].isin(sel_met)
    met_beta = pd.Series({col: mr_beta.get(trait) for col, trait in sel_met.items()})
    summary["metabolites"] = {
        "participants": n_m,
        "conversions": ev_m,
        "tested": len(met),
        "fdr_significant": int((met["q_fdr"] < 0.05).sum()),
        "selected_measured": int(met["selected"].sum()),
        "enrichment": enrichment.compare(met, list(sel_met)),
        "direction": enrichment.direction(met[met["selected"]], met_beta),
    }
    met.to_csv(res_dir / "omics_metabolites.csv", index=False)
    _qq(
        met,
        met["selected"],
        "Serum NMR metabolites vs conversion",
        fig_dir / "omics_metabolites_qq.png",
    )

    summary["provenance"] = provenance.block(
        "ADNI CSF SomaScan 7k (CruchagaLab) + ADMC Nightingale NMR",
        raw_dir.name,
        {"window_days": window_days, "covariates": list(cov.columns)},
        0,
    )
    (res_dir / "omics_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
