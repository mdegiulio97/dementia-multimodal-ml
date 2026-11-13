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


def test_cached_outcome_reuses_cache_for_searched_snps(tmp_path):
    calls = []

    def loader(snps):
        calls.append(set(snps))
        return pd.DataFrame({"snp": sorted(s for s in snps if s != "rs_absent"), "beta": 0.1})

    cache = tmp_path / "outcome.parquet"
    first = sources.cached_outcome(cache, {"rs1", "rs_absent"}, loader)
    again = sources.cached_outcome(cache, {"rs1", "rs_absent"}, loader)
    assert len(calls) == 1 and first.equals(again)
    sources.cached_outcome(cache, {"rs1", "rs2"}, loader)
    assert len(calls) == 2 and calls[-1] == {"rs1", "rs2", "rs_absent"}


KUNKLE_COLUMNS = {
    "snp": "MarkerName",
    "effect_allele": "Effect_allele",
    "other_allele": "Non_Effect_allele",
    "eaf": None,
    "beta": "Beta",
    "se": "SE",
    "pval": "Pvalue",
}


def _kunkle_file(tmp_path):
    path = tmp_path / "Kunkle_etal_Stage1_results.txt"
    path.write_text(
        "Chromosome Position MarkerName Effect_allele Non_Effect_allele Beta SE Pvalue\n"
        "1 100 rs1 a g 0.1 0.02 1e-5\n"
        "1 200 rs5 a t 0.2 0.03 1e-3\n"
        "2 300 rs9 c t -0.05 0.01 0.4\n"
    )
    return path


def test_load_outcome_kunkle_columns(tmp_path):
    out = sources.load_outcome(
        _kunkle_file(tmp_path), {"rs1", "rs5"}, columns=KUNKLE_COLUMNS, sep=" "
    )
    assert out["snp"].tolist() == ["rs1", "rs5"]
    assert out["eaf"].isna().all()
    assert out.loc[0, "effect_allele"] in {"a", "A"} and out.loc[0, "beta"] == 0.1


def test_kunkle_palindromes_dropped_without_eaf(tmp_path):
    from dementia_ml.mr.harmonise import harmonise

    out = sources.load_outcome(
        _kunkle_file(tmp_path), {"rs1", "rs5"}, columns=KUNKLE_COLUMNS, sep=" "
    )
    exp = pd.DataFrame(
        [["rs1", "A", "G", 0.2, 0.3, 0.01], ["rs5", "A", "T", 0.2, 0.3, 0.01]],
        columns=["snp", "effect_allele", "other_allele", "eaf", "beta", "se"],
    )
    h = harmonise(exp, out).set_index("snp")
    assert pd.isna(h.loc["rs1", "dropped_reason"])
    assert h.loc["rs5", "dropped_reason"] == "palindrome_no_eaf"
