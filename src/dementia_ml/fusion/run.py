"""ADNI multimodal fusion: which data add to the clinical prediction of conversion?"""

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
from dementia_ml.data.adnimerge2 import MRI_REGIONAL  # noqa: E402
from dementia_ml.fusion import model  # noqa: E402
from dementia_ml.omics import prep  # noqa: E402
from dementia_ml.omics.run import metabolite_columns  # noqa: E402

BASELINE = ["AGE", "MMSE", "APOE4"]
CLINICAL = ["AGE", "PTEDUCAT", "APOE4", "MMSE", "ADAS13", "CDRSB", "FAQ"]
MRI_HIPPOCAMPUS = ["Hippocampus", "ICV"]
CSF = ["ABETA", "TAU", "PTAU"]
N_COMPONENTS = {"proteomics": 20, "metabolomics": 10}
BIOMARKERS = ["mri_regional", "csf", "proteomics", "metabolomics"]
COMBOS = {
    "baseline": ["baseline"],
    "clinical": ["clinical"],
    "clinical+mri_hippocampus": ["clinical", "mri_hippocampus"],
    "clinical+mri_regional": ["clinical", "mri_regional"],
    "clinical+csf": ["clinical", "csf"],
    "clinical+proteomics": ["clinical", "proteomics"],
    "clinical+metabolomics": ["clinical", "metabolomics"],
    "baseline+biomarkers": ["baseline", *BIOMARKERS],
    "all": ["clinical", *BIOMARKERS],
}


def assemble(raw_dir: Path, window_days: int = 180):
    df = adnimerge.load(paths.data_root() / "adni" / "ADNIMERGE.csv")
    feats = list(dict.fromkeys(CLINICAL + MRI_HIPPOCAMPUS + MRI_REGIONAL + CSF))
    data = conversion.build(df, features=feats)
    bl = df[df["VISCODE"] == "bl"].set_index("RID").loc[data.X.index]
    anchors = pd.to_datetime(bl["EXAMDATE"], errors="coerce").dropna()

    soma = pd.read_csv(raw_dir / "CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv")
    apt = [c for c in soma.columns if c.startswith("X") and c[1:2].isdigit()]
    soma[apt] = np.log2(soma[apt].clip(lower=1e-3))
    prot = prep.baseline_samples(soma, apt, anchors, window_days)

    nmr = pd.read_csv(sorted(raw_dir.glob("ADNINIGHTINGALE2_[0-9]*.csv"))[-1])
    met = metabolite_columns(list(nmr.columns))
    nmr[met] = np.log1p(nmr[met].apply(pd.to_numeric, errors="coerce").clip(lower=0))
    metab = prep.baseline_samples(nmr, met, anchors, window_days)

    X = data.X.join(prot, how="left").join(metab, how="left")
    blocks = {
        "baseline": BASELINE,
        "clinical": CLINICAL,
        "mri_hippocampus": MRI_HIPPOCAMPUS,
        "mri_regional": MRI_HIPPOCAMPUS + MRI_REGIONAL,
        "csf": CSF,
        "proteomics": apt,
        "metabolomics": met,
    }
    coverage = {name: int(X[cols].notna().any(axis=1).sum()) for name, cols in blocks.items()}
    return X, data.time, data.event, blocks, coverage


def _figure(res: dict, path: Path) -> None:
    combos = list(res)
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(combos) + 1.4))
    for kind, color, dy in [("cox", "#264653", -0.12), ("gbsa", "#e76f51", 0.12)]:
        ys = np.arange(len(combos))[::-1] + dy
        est = [res[c][kind]["c_index"]["estimate"] for c in combos]
        lo = [res[c][kind]["c_index"]["ci_low"] for c in combos]
        hi = [res[c][kind]["c_index"]["ci_high"] for c in combos]
        ax.errorbar(
            est,
            ys,
            xerr=[np.subtract(est, lo), np.subtract(hi, est)],
            fmt="o",
            color=color,
            ms=4,
            label="ridge Cox" if kind == "cox" else "gradient-boosted survival",
        )
    ax.set_yticks(np.arange(len(combos))[::-1], combos, fontsize=8)
    ax.set_xlabel("C-index, out-of-fold (95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(raw_dir: Path, n_boot: int = 1000, out_root: Path | None = None) -> dict:
    out_root = out_root or paths.REPO_ROOT
    X, time, event, blocks, coverage = assemble(raw_dir)
    res = model.evaluate(
        X,
        time,
        event,
        blocks,
        COMBOS,
        N_COMPONENTS,
        ("cox", "gbsa"),
        n_boot=n_boot,
        reference="clinical",
    )
    # Sensitivity: participants with every block measured
    complete = np.all(
        [
            X[blocks[b]].notna().any(axis=1)
            for b in ["mri_regional", "csf", "proteomics", "metabolomics"]
        ],
        axis=0,
    )
    cc = None
    if complete.sum() >= 80:
        cc = model.evaluate(
            X[complete],
            time[complete],
            event[complete],
            blocks,
            {k: COMBOS[k] for k in ["baseline", "clinical", "baseline+biomarkers", "all"]},
            N_COMPONENTS,
            ("cox",),
            n_boot=n_boot,
            reference="clinical",
        )
    payload = {
        "question": "Which data add to the clinical prediction of MCI-to-dementia conversion?",
        "coverage": coverage,
        "n_complete_case": int(complete.sum()),
        "pca_components": N_COMPONENTS,
        "results": res,
        "complete_case": cc,
        "provenance": provenance.block(
            "ADNI (ADNIMERGE2, SomaScan 7k CSF, Nightingale NMR)",
            raw_dir.name,
            {"combos": COMBOS, "n_boot": n_boot},
            0,
        ),
    }
    (paths.results_dir("adni", out_root) / "fusion_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    _figure(res, paths.figures_dir("adni", out_root) / "fusion_cindex.png")
    return payload
