"""SuSiE-based colocalisation for the candidates that failed single-variant coloc (ACE, EPHA1).

    uv run python pipelines/coloc_susie.py

Needs: the UKB-PPP archive of each protein (downloaded with the Synapse token), the Bellenguez
2022 harmonised GWAS, and the UK Biobank LD matrices of the Alkes Price group
(s3://broad-alkesgroup-ukbb-ld, GRCh37, 3-Mb windows) in $DEMENTIA_ML_DATA/mr/ukbb_ld.
"""

import argparse
import gzip
import json
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from dementia_ml import paths, provenance
from dementia_ml.mr import coloc, susie

TARGETS = {
    # protein: (chromosome, LD window file stem); windows cover the lead +/- 500 kb in GRCh37
    "ACE": ("17", "chr17_60000001_63000001"),
    "EPHA1": ("7", "chr7_142000001_145000001"),
}
EUROPEAN_DISCOVERY = "syn51365303"
WINDOW = 500_000


def pqtl_region(syn, children, protein, chrom, lead37, dest):
    hit = next(c for c in children if c["name"].startswith(f"{protein}_"))
    entity = syn.get(hit["id"], downloadLocation=str(dest))
    with tarfile.open(entity.path) as tar:
        member = next(m for m in tar.getmembers() if f"_chr{chrom}_" in m.name)
        with gzip.open(tar.extractfile(member), "rt") as gz:
            df = pd.read_csv(
                gz, sep=" ", usecols=["GENPOS", "ID", "ALLELE0", "ALLELE1", "N", "BETA", "SE"]
            )
    df["pos37"] = df["ID"].str.split(":").str[1].astype(int)
    return df[(df["pos37"] - lead37).abs() <= WINDOW]


def gwas_region(chrom, lo38, hi38):
    path = paths.data_root() / "mr" / "35379992-GCST90027158-MONDO_0004975.h.tsv.gz"
    cols = [
        "hm_chrom",
        "hm_pos",
        "hm_other_allele",
        "hm_effect_allele",
        "hm_beta",
        "standard_error",
        "n_cas",
        "n_con",
    ]
    parts = []
    for chunk in pd.read_csv(
        path, sep="\t", usecols=lambda c: c.strip() in cols, dtype=str, chunksize=2_000_000
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        pos = pd.to_numeric(chunk["hm_pos"], errors="coerce")
        parts.append(chunk[(chunk["hm_chrom"] == chrom) & pos.between(lo38, hi38)])
    g = pd.concat(parts, ignore_index=True)
    for c in ["hm_pos", "hm_beta", "standard_error", "n_cas", "n_con"]:
        g[c] = pd.to_numeric(g[c].str.strip(), errors="coerce")
    return g.dropna(subset=["hm_pos", "hm_beta", "standard_error"])


def ld_matrix(stem):
    base = paths.data_root() / "mr" / "ukbb_ld" / stem
    variants = pd.read_csv(f"{base}.gz", sep="\t")
    M = sp.load_npz(f"{base}.npz").tocsr()
    return variants, M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=10)
    args = ap.parse_args()
    import synapseclient

    syn = synapseclient.Synapse(silent=True)
    syn.login(silent=True)
    children = list(syn.getChildren(EUROPEAN_DISCOVERY))
    st9 = pd.read_excel(
        paths.data_root() / "mr" / "41586_2023_6592_MOESM3_ESM.xlsx", sheet_name="ST9", header=4
    )
    st9 = st9[st9["cis/trans"].astype(str).str.lower().eq("cis")]
    results = {}
    for protein, (chrom, stem) in TARGETS.items():
        row = st9[st9["Assay Target"] == protein].sort_values("log10(p) (discovery)").iloc[-1]
        lead37 = int(str(row["Variant ID (CHROM:GENPOS (hg37):A0:A1:imp:v1)"]).split(":")[1])
        pq = pqtl_region(syn, children, protein, chrom, lead37, paths.data_root() / "mr" / "ukbppp")
        g = gwas_region(chrom, pq["GENPOS"].min(), pq["GENPOS"].max())
        m = pq.merge(g, left_on="GENPOS", right_on="hm_pos")
        m = m[
            [
                {a, b} == {c, d}
                for a, b, c, d in zip(
                    m["ALLELE0"],
                    m["ALLELE1"],
                    m["hm_other_allele"],
                    m["hm_effect_allele"],
                    strict=True,
                )
            ]
        ]
        variants, M = ld_matrix(stem)
        variants = variants.reset_index().rename(columns={"index": "ld_index"})
        m = m.merge(variants, left_on="pos37", right_on="position")
        m = m[
            [
                {a, b} == {c, d}
                for a, b, c, d in zip(
                    m["ALLELE0"], m["ALLELE1"], m["allele1"], m["allele2"], strict=True
                )
            ]
        ]
        m = m.drop_duplicates("pos37").sort_values("pos37").reset_index(drop=True)
        sign_p = np.where(m["ALLELE1"] == m["allele2"], 1.0, -1.0)
        sign_g = np.where(m["hm_effect_allele"] == m["allele2"], 1.0, -1.0)
        z1 = sign_p * m["BETA"] / m["SE"]
        z2 = sign_g * m["hm_beta"] / m["standard_error"]
        idx = m["ld_index"].to_numpy()
        sub = M[idx][:, idx].toarray()
        R = sub + sub.T
        np.fill_diagonal(R, 1.0)
        n1 = int(m["N"].median())
        n2 = int((m["n_cas"] + m["n_con"]).median())
        pairs = susie.coloc_susie(z1.to_numpy(), z2.to_numpy(), R, n1=n1, n2=n2, L=args.L)
        fit1, fit2 = (
            susie.susie_rss(z1.to_numpy(), R, n1, args.L),
            susie.susie_rss(z2.to_numpy(), R, n2, args.L),
        )
        abf = coloc.coloc_abf(
            m.assign(beta1=m["BETA"], se1=m["SE"], beta2=m["hm_beta"], se2=m["standard_error"])
        )
        best = max(pairs, key=lambda r: r["PP.H4"]) if pairs else None
        results[protein] = {
            "variants": len(m),
            "n_pqtl": n1,
            "n_gwas": n2,
            "credible_sets_protein": len(susie.credible_sets(fit1, R)),
            "credible_sets_ad": len(susie.credible_sets(fit2, R)),
            "pairs": [
                {k: v for k, v in p.items() if k.startswith("PP")}
                | {
                    "lead_protein_pos37": int(m.loc[p["lead1"], "pos37"]),
                    "lead_ad_pos37": int(m.loc[p["lead2"], "pos37"]),
                }
                for p in pairs
            ],
            "max_PP.H4": best["PP.H4"] if best else None,
            "abf_on_same_variants": {k: v for k, v in abf.items() if k.startswith("PP")},
        }
        print(
            protein,
            json.dumps({k: v for k, v in results[protein].items() if k != "pairs"}),
            flush=True,
        )
    out = {
        "method": "SuSiE-RSS (L=10) per trait + coloc on every pair of credible sets (p12=5e-6); UK Biobank LD (Broad, GRCh37)",
        "results": results,
        "provenance": provenance.block(
            "UKB-PPP + GCST90027158 + UKBB LD", "regional", {"L": args.L, "window": WINDOW}, 0
        ),
    }
    (paths.results_dir("open") / "coloc_susie.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
