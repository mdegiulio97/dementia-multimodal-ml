"""Agreement with the R reference packages on identical inputs (susieR, coloc, MendelianRandomization).

    uv run python pipelines/benchmark_r_reference.py [--rscript PATH]

Exports the ACE and EPHA1 regions used by pipelines/coloc_susie.py (z-scores, betas, UK Biobank
LD) and every harmonised MR instrument of the screen (results/open/mr_harmonisation_log.csv),
runs benchmarks/r_reference.R on them, runs the package's own functions on the same inputs and
compares the two. Needs R with coloc, susieR, MendelianRandomization and jsonlite.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dementia_ml import paths, provenance
from dementia_ml.mr import coloc, estimators, susie

LOCI = {"ACE": "chr17_60000001_63000001", "EPHA1": "chr7_142000001_145000001"}
DEFAULT_RSCRIPT = r"C:\Program Files\R\R-4.6.2\bin\Rscript.exe"


def region_inputs(protein: str, stem: str):
    import scipy.sparse as sp

    m = pd.read_parquet(paths.data_root() / "mr" / "coloc_regions" / f"{protein}.parquet")
    base = paths.data_root() / "mr" / "ukbb_ld" / stem
    M = sp.load_npz(f"{base}.npz").tocsr()
    sign_p = np.where(m["ALLELE1"] == m["allele2"], 1.0, -1.0)
    sign_g = np.where(m["hm_effect_allele"] == m["allele2"], 1.0, -1.0)
    idx = m["ld_index"].to_numpy()
    sub = M[idx][:, idx].toarray()
    R = sub + sub.T
    np.fill_diagonal(R, 1.0)
    d = pd.DataFrame(
        {
            "z1": sign_p * m["BETA"] / m["SE"],
            "z2": sign_g * m["hm_beta"] / m["standard_error"],
            "beta1": m["BETA"],
            "se1": m["SE"],
            "beta2": m["hm_beta"],
            "se2": m["standard_error"],
        }
    )
    return d, R, int(m["N"].median()), int((m["n_cas"] + m["n_con"]).median())


def ours_locus(d, R, n1, n2):
    t0 = time.perf_counter()
    f1, f2 = susie.susie_rss(d["z1"].to_numpy(), R, n1), susie.susie_rss(d["z2"].to_numpy(), R, n2)
    t_susie = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairs = susie.coloc_susie(d["z1"].to_numpy(), d["z2"].to_numpy(), R, n1=n1, n2=n2)
    t_cs = time.perf_counter() - t0 - t_susie  # coloc_susie refits: subtract the fit time
    t0 = time.perf_counter()
    abf = coloc.coloc_abf(d)
    t_abf = time.perf_counter() - t0
    pip = lambda f: 1 - np.prod(1 - f.alpha[f.V > 0], axis=0)  # noqa: E731
    return {
        "pip1": pip(f1),
        "pip2": pip(f2),
        "cs1": [s["variants"] for s in susie.credible_sets(f1, R)],
        "cs2": [s["variants"] for s in susie.credible_sets(f2, R)],
        "coloc_susie": pairs,
        "abf": [abf[f"PP.H{i}"] for i in range(5)],
        "seconds": {"susie_two_traits": t_susie, "coloc_susie": max(t_cs, 0.0), "coloc_abf": t_abf},
    }


def published_overlap(protein: str, sets: list[list[int]]) -> int:
    """How many of ``sets`` share a variant with a published UKB-PPP SuSiE set (ST16)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench", Path(__file__).with_name("benchmark_susie_ukbppp.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    pub = bench.published_sets(protein)
    pos = pd.read_parquet(paths.data_root() / "mr" / "coloc_regions" / f"{protein}.parquet")[
        "pos37"
    ]
    return sum(any({int(pos[i]) for i in s} & p["members"] for p in pub) for s in sets)


def compare_sets(ours: list[list[int]], ref: list[list[int]]) -> dict:
    def best_jaccard(s, pool):
        s = set(s)
        return max((len(s & set(t)) / len(s | set(t)) for t in pool), default=0.0)

    return {
        "ours": len(ours),
        "reference": len(ref),
        "reference_sets_matched": sum(best_jaccard(s, ours) >= 0.5 for s in ref),
        "ours_matched": sum(best_jaccard(s, ref) >= 0.5 for s in ours),
        "median_jaccard": float(np.median([best_jaccard(s, ours) for s in ref])) if ref else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rscript", default=os.environ.get("RSCRIPT", DEFAULT_RSCRIPT))
    args = ap.parse_args()
    work = paths.data_root() / "benchmark_r"
    work.mkdir(parents=True, exist_ok=True)

    meta, ours = {"loci": {}}, {}
    for protein, stem in LOCI.items():
        d, R, n1, n2 = region_inputs(protein, stem)
        d.to_csv(work / f"{protein}_region.csv", index=False)
        np.asarray(R, "<f8").T.tofile(work / f"{protein}_ld.bin")  # column-major for R
        meta["loci"][protein] = {"n1": n1, "n2": n2, "variants": len(d)}
        ours[protein] = ours_locus(d, R, n1, n2)
        print(protein, "ours done", flush=True)
    (work / "meta.json").write_text(json.dumps(meta))

    log = pd.read_csv(paths.results_dir("open") / "mr_harmonisation_log.csv")
    h = log[log["dropped_reason"].isna()][["trait", "beta_exp", "se_exp", "beta_out", "se_out"]]
    h.to_csv(work / "mr_instruments.csv", index=False)
    t0 = time.perf_counter()
    ours_mr = {}
    for trait, x in h.groupby("trait", sort=False):
        a = [x[c].to_numpy() for c in ["beta_exp", "se_exp", "beta_out", "se_out"]]
        r = {"ivw": estimators.ivw(*a)}
        if len(x) >= 3:
            r["egger"] = estimators.egger(*a)
            r["weighted_median"] = estimators.weighted_median(*a, n_boot=1000, seed=0)
        ours_mr[trait] = r
    t_mr = time.perf_counter() - t0

    env = dict(os.environ)
    tmp = tempfile.gettempdir()
    if " " in tmp:  # R refuses a temporary directory whose path contains spaces
        tmp = os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "Rtmp")
        os.makedirs(tmp, exist_ok=True)
    env.update(TMPDIR=tmp, TMP=tmp, TEMP=tmp)
    subprocess.run([args.rscript, "benchmarks/r_reference.R", str(work)], check=True, env=env)
    ref = json.loads((work / "r_results.json").read_text())

    loci = {}
    for protein in LOCI:
        o, r = ours[protein], ref["loci"][protein]
        best_o = max(o["coloc_susie"], key=lambda p: p["PP.H4"]) if o["coloc_susie"] else None
        best_r = max(r["coloc_susie"], key=lambda p: p["PP.H4"]) if r["coloc_susie"] else None
        loci[protein] = {
            "variants": meta["loci"][protein]["variants"],
            "credible_sets_protein": compare_sets(o["cs1"], r["cs1"]),
            "credible_sets_ad": compare_sets(o["cs2"], r["cs2"]),
            "protein_sets_overlapping_published_ukbppp": {
                "ours": published_overlap(protein, o["cs1"]),
                "reference": published_overlap(protein, r["cs1"]),
            },
            "pip_max_abs_diff": float(
                max(
                    np.max(np.abs(o["pip1"] - np.array(r["pip1"]))),
                    np.max(np.abs(o["pip2"] - np.array(r["pip2"]))),
                )
            ),
            "pip_correlation_protein": float(np.corrcoef(o["pip1"], r["pip1"])[0, 1]),
            "coloc_susie_best_PP.H4": {
                "ours": best_o["PP.H4"] if best_o else None,
                "reference": best_r["PP.H4"] if best_r else None,
                "same_protein_lead": bool(best_o and best_r and best_o["lead1"] == best_r["hit1"]),
            },
            "coloc_abf_max_abs_diff": float(
                np.max(np.abs(np.array(o["abf"]) - np.array(list(r["abf"].values()))))
            ),
            "coloc_abf_PP.H4": {"ours": o["abf"][4], "reference": r["abf"]["PP.H4.abf"]},
            "seconds": {"ours": o["seconds"], "reference": r["seconds"]},
        }

    def diffs(method, idx_est, idx_se):
        de, ds, n = [], [], 0
        for trait, o in ours_mr.items():
            if method not in o:
                continue
            rr = ref["mr"][trait][method]
            de.append(abs(o[method].beta - rr[idx_est]))
            ds.append(abs(o[method].se - rr[idx_se]))
            n += 1
        return {"traits": n, "max_abs_diff_estimate": max(de), "max_abs_diff_se": max(ds)}

    out = {
        "question": "Do the package's Python implementations agree with the R reference packages on identical inputs?",
        "reference_versions": ref["versions"],
        "susie_coloc": loci,
        "mr": {
            "ivw": diffs("ivw", 0, 1),
            "egger": diffs("egger", 0, 1),
            "weighted_median": diffs("weighted_median", 0, 1),
            "seconds": {"ours": t_mr, "reference": ref["mr_seconds"]},
        },
        "provenance": provenance.block(
            "coloc_regions ACE/EPHA1 + mr_harmonisation_log.csv", "regional", {}, 0
        ),
    }
    path = paths.results_dir("open") / "r_reference_benchmark.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
