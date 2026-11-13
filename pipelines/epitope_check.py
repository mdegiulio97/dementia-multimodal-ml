"""Epitope-effect check of the colocalised MR candidates (Ensembl VEP + 1000G EUR LD).

uv run python pipelines/epitope_check.py
"""

import json

import pandas as pd

from dementia_ml import paths, provenance
from dementia_ml.mr.epitope import EnsemblClient, protein_altering_in_gene, verdict

COLOCALISED = ["CR1", "PILRA", "PILRB", "IL34", "C1S", "C1R", "CD33"]
ACE_COLOC_POS37 = 61548948  # colocalising ACE instrument from coloc_susie (GRCh37)


def instruments() -> dict[str, str]:
    data = paths.data_root() / "mr"
    st9 = pd.read_excel(data / "41586_2023_6592_MOESM3_ESM.xlsx", sheet_name="ST9", header=4)
    st9 = st9[st9["cis/trans"].astype(str).str.lower().eq("cis")]
    out = {}
    for prot in COLOCALISED:
        row = st9[st9["Assay Target"] == prot].sort_values("log10(p) (discovery)").iloc[-1]
        out[prot] = str(row["rsID"])
    ld = pd.read_csv(data / "ukbb_ld" / "chr17_60000001_63000001.gz", sep="\t")
    out["ACE"] = str(ld.loc[ld["position"] == ACE_COLOC_POS37, "rsid"].iloc[0])
    return out


def main() -> None:
    client = EnsemblClient(paths.data_root() / "mr" / "ensembl_cache")
    rows = []
    for prot, rsid in instruments().items():
        inst_alt = protein_altering_in_gene(client.vep(rsid), prot)
        proxies = [p for p in client.ld_proxies(rsid) if p != rsid]
        altering = [p for p in proxies if protein_altering_in_gene(client.vep(p), prot)]
        rows.append(
            {
                "protein": prot,
                "instrument": rsid,
                "instrument_protein_altering": inst_alt,
                "n_proxies_r2_08": len(proxies),
                "protein_altering_proxies": ";".join(altering),
                "verdict": verdict(inst_alt, altering),
            }
        )
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    res = paths.results_dir("open")
    table.to_csv(res / "epitope_check.csv", index=False)
    (res / "epitope_check.json").write_text(
        json.dumps(
            {
                "method": "Ensembl VEP (GRCh38) on the instrument and its 1000G EUR proxies (r2 >= 0.8); "
                "protein-altering = missense, stop, start-lost, frameshift, in-frame indel, splice donor/acceptor in the target gene",
                "results": rows,
                "provenance": provenance.block("Ensembl REST", "current release", {"r2": 0.8}, 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
