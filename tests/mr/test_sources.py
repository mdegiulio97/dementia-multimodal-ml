import gzip

import numpy as np
import pandas as pd
import pytest

from dementia_ml.mr import sources


def _excel_with_offset(path, sheet, df, blank_rows):
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, sheet_name=sheet, startrow=blank_rows, index=False)


def test_load_pqtl_parses_variant_id_and_neglog10(tmp_path):
    df = pd.DataFrame(
        {
            "Variant ID (CHROM:GENPOS (hg37):A0:A1:imp:v1)": [
                "2:27730940:T:C:imp:v1",
                "6:31000000:A:G:imp:v1",
            ],
            "CHROM": [2, 6],
            "MHC": [0, 1],
            "Assay Target": ["A1BG", "HLA_X"],
            "rsID": ["rs1260326", "rs99"],
            "A1FREQ (discovery)": [0.6, 0.3],
            "BETA (discovery, wrt. A1)": [-0.13, 0.2],
            "SE (discovery)": [0.007, 0.01],
            "log10(p) (discovery)": [79.2, 30.0],
            "cis/trans": ["cis", "cis"],
        }
    )
    path = tmp_path / "st.xlsx"
    _excel_with_offset(path, "ST9", df, blank_rows=4)
    out = sources.load_pqtl(path)
    row = out.iloc[0]
    assert len(out) == 1  # MHC variant excluded
    assert (row["effect_allele"], row["other_allele"]) == ("C", "T")
    assert row["pval"] == pytest.approx(10**-79.2)
    assert row["pos"] == 27730940 and row["chrom"] == 2 and bool(row["cis"])
    assert row["trait"] == "A1BG"


def test_load_mqtl_uppercases_alleles(tmp_path):
    df = pd.DataFrame(
        {
            "Chromosome": [1],
            "Lead variant position": [1000],
            "Metabolite": ["Total-C"],
            "Rsid": ["rs5"],
            "Allele1": ["a"],
            "Allele2": ["g"],
            "Freq1": [0.2],
            "Effect": [0.05],
            "StdErr": [0.005],
            "P-value": [1e-20],
        }
    )
    path = tmp_path / "s5.xlsx"
    _excel_with_offset(path, "TableS5_all_lead_SNP_assoc", df, blank_rows=1)
    out = sources.load_mqtl(path)
    assert out.iloc[0][["effect_allele", "other_allele"]].tolist() == ["A", "G"]
    assert not out["cis"].any()


def test_load_outcome_filters_snps_and_strips_cr(tmp_path):
    path = tmp_path / "gwas.h.tsv.gz"
    header = "hm_rsid\thm_effect_allele\thm_other_allele\thm_effect_allele_frequency\thm_beta\tstandard_error\tp_value"
    lines = [header, "rs1\tA\tG\t0.2\t0.1\t0.01\t1e-5", "rs2\tC\tT\t0.3\t-0.2\t0.02\t0.5"]
    with gzip.open(path, "wt", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    out = sources.load_outcome(path, snps={"rs2"}, chunksize=1)
    assert out["snp"].tolist() == ["rs2"]
    assert out["pval"].iloc[0] == 0.5 and np.isclose(out["beta"].iloc[0], -0.2)
