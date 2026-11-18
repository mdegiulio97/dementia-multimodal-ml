"""ACE in ADNI CSF: does the association with conversion survive adjustment for amyloid, tau
and blood-brain-barrier markers? (local; aggregate output)

    uv run python pipelines/ace_sensitivity.py --raw "$DEMENTIA_ML_DATA/adni/raw_2026-11-13"

Models (Cox, conversion MCI -> dementia; continuous variables inverse-normal transformed, so
hazard ratios are per SD):
  base      ACE + age + sex + APOE4
  +csf      base + CSF Abeta42 + p-tau181 (Elecsys)
  +barrier  base + CSF sPDGFRB + CSF albumin (SomaScan)
  +all      base + CSF + barrier
  +clinical base + MMSE + ADAS13 + CDR-SB + FAQ
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

from dementia_ml import paths, provenance
from dementia_ml.data import adnimerge, conversion
from dementia_ml.omics import prep

APTAMERS = {"ACE": "X10714.7", "PDGFRB": "X3459.49", "ALB": "X18380.78"}
CSF = ["ABETA", "PTAU", "TAU"]
CLINICAL = ["MMSE", "ADAS13", "CDRSB", "FAQ"]
MODELS = {
    "base": [],
    "+csf": ["ABETA", "PTAU"],
    "+barrier": ["PDGFRB", "ALB"],
    "+all": ["ABETA", "PTAU", "PDGFRB", "ALB"],
    "+clinical": CLINICAL,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    raw = ap.parse_args().raw

    df = adnimerge.load(paths.data_root() / "adni" / "ADNIMERGE.csv")
    data = conversion.build(df, features=["AGE", "APOE4", *CSF, *CLINICAL])
    bl = df[df["VISCODE"] == "bl"].set_index("RID").loc[data.X.index]
    anchors = pd.to_datetime(bl["EXAMDATE"], errors="coerce").dropna()
    soma = pd.read_csv(
        raw / "CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv",
        usecols=["RID", "EXAMDATE", *APTAMERS.values()],
    )
    prot = prep.baseline_samples(soma, list(APTAMERS.values()), anchors, 180)
    prot.columns = list(APTAMERS)

    frame = data.X.assign(
        MALE=(bl["PTGENDER"] == "Male").astype(float), time=data.time, event=data.event
    )
    frame = frame.join(prot, how="inner")
    for c in ["ACE", "PDGFRB", "ALB", *CSF, *CLINICAL, "AGE"]:
        frame[c] = prep.inverse_normal(frame[c])

    results = {}
    for name, extra in MODELS.items():
        cols = ["ACE", "AGE", "MALE", "APOE4", *extra]
        f = frame[cols + ["time", "event"]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = CoxPHFitter(penalizer=0.01).fit(f, "time", "event")
        s = fit.summary.loc["ACE"]
        results[name] = {
            "n": len(f),
            "events": int(f["event"].sum()),
            "hr_ace_per_sd": float(s["exp(coef)"]),
            "ci_low": float(s["exp(coef) lower 95%"]),
            "ci_high": float(s["exp(coef) upper 95%"]),
            "p": float(s["p"]),
            "covariates": cols[1:],
        }
        r = results[name]
        print(
            f"{name:10s} n={r['n']:4d} ev={r['events']:3d} HR {r['hr_ace_per_sd']:.3f} "
            f"({r['ci_low']:.3f}-{r['ci_high']:.3f}) p={r['p']:.4f}",
            flush=True,
        )

    corr = {}
    for c in ["TAU", "PTAU", "ABETA", "PDGFRB", "ALB"]:
        sub = frame[["ACE", c]].dropna()
        rho, pv = spearmanr(sub["ACE"], sub[c])
        corr[c] = {"rho": float(rho), "p": float(pv), "n": len(sub)}
    print("Spearman with ACE:", {k: round(v["rho"], 3) for k, v in corr.items()})

    out = {
        "question": "Does CSF ACE predict MCI-to-dementia conversion independently of amyloid, tau and barrier markers?",
        "models": results,
        "spearman_with_ace": corr,
        "provenance": provenance.block(
            "ADNI (ADNIMERGE2, SomaScan 7k CSF, Elecsys)", raw.name, {"models": MODELS}, 0
        ),
    }
    (paths.results_dir("adni") / "ace_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
