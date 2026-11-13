"""Colocalisation of the strongest MR candidates with Alzheimer's disease (coloc ABF).

    uv run python pipelines/coloc_candidates.py

Needs a Synapse account with access to UKB-PPP (syn51364943) and ~/.synapseConfig with a
personal access token. Downloads one UKB-PPP archive per protein (European discovery), keeps
the chromosome of the cis lead variant, and tests a +/- 500 kb window against Bellenguez 2022.
"""

import argparse
import gzip
import json
import tarfile
from pathlib import Path

import pandas as pd

from dementia_ml import paths, provenance
from dementia_ml.mr import coloc

PROTEINS = ["ACE", "CR1", "EPHA1", "PILRA", "PILRB", "C1S", "C1R", "CD33", "IL34"]
EUROPEAN_DISCOVERY = "syn51365303"
WINDOW = 500_000


def cis_leads(st9: Path) -> pd.DataFrame:
    raw = pd.read_excel(st9, sheet_name="ST9", header=4)
    raw = raw[raw["cis/trans"].astype(str).str.lower().eq("cis")]
    raw = raw.sort_values("log10(p) (discovery)", ascending=False).drop_duplicates("Assay Target")
    return raw.set_index("Assay Target")[["CHROM", "GENPOS (hg38)"]].rename(
        columns={"CHROM": "chrom", "GENPOS (hg38)": "pos"}
    )


def fetch_region(syn, folder_children, protein, chrom, lo, hi, dest: Path) -> pd.DataFrame | None:
    hits = [c for c in folder_children if c["name"].startswith(f"{protein}_")]
    if not hits:
        return None
    entity = syn.get(hits[0]["id"], downloadLocation=str(dest))
    with tarfile.open(entity.path) as tar:
        member = next(m for m in tar.getmembers() if f"_chr{chrom}_" in m.name)
        fh = tar.extractfile(member)
        with gzip.open(fh, "rt") as gz:
            df = pd.read_csv(
                gz, sep=" ", usecols=["CHROM", "GENPOS", "ALLELE0", "ALLELE1", "BETA", "SE"]
            )
    Path(entity.path).unlink()  # keep only the region we need
    return df[(df["GENPOS"] >= lo) & (df["GENPOS"] <= hi)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proteins", nargs="*", default=PROTEINS)
    args = ap.parse_args()
    import synapseclient

    syn = synapseclient.Synapse(silent=True)
    syn.login(silent=True)
    children = list(syn.getChildren(EUROPEAN_DISCOVERY))
    data = paths.data_root() / "mr"
    leads = cis_leads(data / "41586_2023_6592_MOESM3_ESM.xlsx")

    regions = {}
    for prot in args.proteins:
        if prot not in leads.index:
            print(f"{prot}: no cis lead in ST9")
            continue
        chrom, pos = str(leads.loc[prot, "chrom"]), int(leads.loc[prot, "pos"])
        region = fetch_region(
            syn, children, prot, chrom, pos - WINDOW, pos + WINDOW, data / "ukbppp"
        )
        if region is not None:
            regions[prot] = (chrom, pos, region)
            print(f"{prot}: chr{chrom}:{pos}, {len(region)} variants", flush=True)

    # One pass over the AD GWAS for all regions
    gwas_path = data / "35379992-GCST90027158-MONDO_0004975.h.tsv.gz"
    cols = [
        "hm_chrom",
        "hm_pos",
        "hm_other_allele",
        "hm_effect_allele",
        "hm_beta",
        "standard_error",
    ]
    parts = []
    for chunk in pd.read_csv(
        gwas_path, sep="\t", usecols=lambda c: c.strip() in cols, dtype=str, chunksize=2_000_000
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        pos = pd.to_numeric(chunk["hm_pos"], errors="coerce")
        keep = pd.Series(False, index=chunk.index)
        for chrom, lead, _ in regions.values():
            keep |= (chunk["hm_chrom"] == chrom) & pos.between(lead - WINDOW, lead + WINDOW)
        parts.append(chunk[keep])
    gwas = pd.concat(parts, ignore_index=True)

    results = {}
    for prot, (chrom, lead, region) in regions.items():
        merged = coloc.merge_region(region, gwas[gwas["hm_chrom"] == chrom])
        pp = coloc.coloc_abf(merged)
        top = merged.iloc[pp.pop("top_shared_snp_index")]
        results[prot] = {
            **pp,
            "chrom": chrom,
            "lead_pos_hg38": lead,
            "top_shared_pos_hg38": int(top["pos"]),
        }
        print(
            f"{prot:6s} n={pp['n_snps']:5d} H3={pp['PP.H3']:.3f} H4={pp['PP.H4']:.3f}", flush=True
        )

    out = {
        "method": "coloc ABF (Giambartolomei 2014), p1=p2=1e-4, p12=1e-5, prior sd 0.15/0.2, +/-500 kb",
        "exposure": "UKB-PPP European discovery (Sun et al. 2023)",
        "outcome": "Bellenguez et al. 2022 (GCST90027158)",
        "results": results,
        "provenance": provenance.block(
            "UKB-PPP syn51364943 + GCST90027158", "regional", {"window": WINDOW}, 0
        ),
    }
    (paths.results_dir("open") / "coloc_candidates.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
