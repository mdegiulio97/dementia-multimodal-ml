import numpy as np
import pandas as pd

from dementia_ml.mr.harmonise import harmonise


def _df(rows):
    return pd.DataFrame(rows, columns=["snp", "effect_allele", "other_allele", "eaf", "beta", "se"])


def test_harmonise_flips_swapped_alleles():
    exp = _df([["rs1", "A", "G", 0.3, 0.2, 0.01]])
    out = _df([["rs1", "G", "A", 0.7, -0.05, 0.01]])
    h = harmonise(exp, out)
    row = h.iloc[0]
    assert pd.isna(row["dropped_reason"])
    assert row["beta_out"] == 0.05 and np.isclose(row["eaf_out"], 0.3)


def test_harmonise_handles_strand_flip_and_case():
    exp = _df([["rs2", "a", "g", 0.3, 0.2, 0.01]])
    out = _df([["rs2", "T", "C", 0.3, 0.04, 0.01]])  # complement strand
    h = harmonise(exp, out)
    assert pd.isna(h.iloc[0]["dropped_reason"]) and h.iloc[0]["beta_out"] == 0.04


def test_harmonise_drops_ambiguous_palindromes():
    exp = _df([["rs3", "A", "T", 0.48, 0.2, 0.01], ["rs4", "C", "G", 0.10, 0.2, 0.01]])
    out = _df([["rs3", "A", "T", 0.50, 0.04, 0.01], ["rs4", "C", "G", 0.90, 0.04, 0.01]])
    h = harmonise(exp, out).set_index("snp")
    assert h.loc["rs3", "dropped_reason"] == "ambiguous_palindrome"
    # rs4: frequencies on opposite sides of 0.5 -> outcome reported on the other strand
    assert pd.isna(h.loc["rs4", "dropped_reason"]) and h.loc["rs4", "beta_out"] == -0.04


def test_harmonise_drops_palindrome_without_eaf_and_mismatch():
    exp = _df([["rs5", "A", "T", 0.2, 0.2, 0.01], ["rs6", "A", "G", 0.2, 0.2, 0.01]])
    out = _df([["rs5", "A", "T", np.nan, 0.04, 0.01], ["rs6", "A", "C", 0.2, 0.04, 0.01]])
    h = harmonise(exp, out).set_index("snp")
    assert h.loc["rs5", "dropped_reason"] == "palindrome_no_eaf"
    assert h.loc["rs6", "dropped_reason"] == "allele_mismatch"


def test_harmonise_reports_missing_in_outcome():
    exp = _df([["rs7", "A", "G", 0.2, 0.2, 0.01]])
    out = _df([["rs8", "A", "G", 0.2, 0.04, 0.01]])
    h = harmonise(exp, out)
    assert h.iloc[0]["dropped_reason"] == "missing_in_outcome"


def test_palindrome_swapped_letters_same_strand():
    exp = _df([["rs9", "A", "T", 0.2, 0.2, 0.01]])
    out = _df([["rs9", "T", "A", 0.8, -0.04, 0.01]])  # same strand, alleles listed the other way
    row = harmonise(exp, out).iloc[0]
    assert pd.isna(row["dropped_reason"])
    assert row["beta_out"] == 0.04 and np.isclose(row["eaf_out"], 0.2)


def test_palindrome_swapped_letters_strand_flip():
    exp = _df([["rs10", "A", "T", 0.2, 0.2, 0.01]])
    out = _df([["rs10", "T", "A", 0.2, 0.04, 0.01]])  # reverse strand: outcome T is exposure A
    row = harmonise(exp, out).iloc[0]
    assert pd.isna(row["dropped_reason"]) and row["beta_out"] == 0.04


def test_palindrome_ambiguous_outcome_eaf_dropped():
    exp = _df([["rs11", "A", "T", 0.30, 0.2, 0.01]])
    out = _df([["rs11", "A", "T", 0.49, 0.04, 0.01]])
    assert harmonise(exp, out).iloc[0]["dropped_reason"] == "ambiguous_palindrome"
