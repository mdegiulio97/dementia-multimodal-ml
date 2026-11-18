"""Benchmark: our SuSiE-RSS credible sets vs the published UKB-PPP SuSiE credible sets.

    uv run python pipelines/benchmark_susie_ukbppp.py

UKB-PPP (Sun et al. 2023, Supplementary Table 16) ran SuSiE on individual-level data; we ran
our Python SuSiE-RSS on summary statistics with UK Biobank LD (pipelines/coloc_susie.py). A
published credible set is "recovered" when the lead variant of one of our credible sets falls
inside it.
"""

import json

import openpyxl

from dementia_ml import paths, provenance

PROTEINS = ["ACE", "EPHA1"]


def published_sets(protein: str) -> list[dict]:
    wb = openpyxl.load_workbook(
        paths.data_root() / "mr" / "41586_2023_6592_MOESM3_ESM.xlsx", read_only=True
    )
    out = []
    for row in wb["ST16"].iter_rows(min_row=6, values_only=True):
        if row[0] and str(row[0]).startswith(f"{protein}:") and row[9] == "cis":
            members = {int(v.split(":")[1]) for v in str(row[13]).split(", ") if v.count(":") >= 3}
            out.append(
                {"lead_pos37": int(str(row[1]).split(":")[1]), "rsid": row[2], "members": members}
            )
    return out


def main() -> None:
    ours = json.loads((paths.results_dir("open") / "coloc_susie.json").read_text())["results"]
    summary = {}
    for protein in PROTEINS:
        leads = sorted({p["lead_protein_pos37"] for p in ours[protein]["pairs"]})
        pub = published_sets(protein)
        recovered = [s for s in pub if s["members"] & set(leads)]
        identical = [s for s in pub if s["lead_pos37"] in leads]
        extra = [p for p in leads if not any(p in s["members"] for s in pub)]
        summary[protein] = {
            "published_sets": len(pub),
            "our_sets": len(leads),
            "published_recovered": len(recovered),
            "identical_lead": len(identical),
            "our_sets_not_in_published": len(extra),
        }
        print(protein, summary[protein])
    out = {
        "question": "Do our SuSiE-RSS credible sets match the published UKB-PPP SuSiE credible sets?",
        "reference": "Sun et al. 2023 Nature, doi:10.1038/s41586-023-06592-6, Supplementary Table 16",
        "results": summary,
        "provenance": provenance.block("UKB-PPP ST16 + coloc_susie.json", "regional", {}, 0),
    }
    (paths.results_dir("open") / "susie_benchmark_ukbppp.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
