"""Align exposure and outcome summary statistics to the same effect allele."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPLEMENT = str.maketrans("ACGT", "TGCA")
PALINDROMES = {frozenset("AT"), frozenset("CG")}
AMBIGUOUS_EAF = (0.42, 0.58)
COLUMNS = ["snp", "effect_allele", "other_allele", "eaf", "beta", "se"]


def _upper(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    for c in ("effect_allele", "other_allele"):
        df[c] = df[c].astype(str).str.upper()
    return df


def _align(ea_x, oa_x, eaf_x, ea_y, oa_y, eaf_y) -> tuple[int, str | None]:
    """Return (sign to apply to the outcome beta, reason if dropped)."""
    if frozenset((ea_x, oa_x)) in PALINDROMES:
        if pd.isna(eaf_x) or pd.isna(eaf_y):
            return 0, "palindrome_no_eaf"
        if AMBIGUOUS_EAF[0] < eaf_x < AMBIGUOUS_EAF[1]:
            return 0, "ambiguous_palindrome"
        if {ea_y, oa_y} != {ea_x, oa_x}:
            return 0, "allele_mismatch"
        # Same-side frequencies mean the same allele is the minor one in both studies.
        same_side = (eaf_x < 0.5) == (eaf_y < 0.5)
        if ea_y == ea_x:
            return (1 if same_side else -1), None
        return (-1 if same_side else 1), None
    if (ea_y, oa_y) == (ea_x, oa_x):
        return 1, None
    if (ea_y, oa_y) == (oa_x, ea_x):
        return -1, None
    ea_c, oa_c = ea_y.translate(COMPLEMENT), oa_y.translate(COMPLEMENT)
    if (ea_c, oa_c) == (ea_x, oa_x):
        return 1, None
    if (ea_c, oa_c) == (oa_x, ea_x):
        return -1, None
    return 0, "allele_mismatch"


def harmonise(exposure: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    """One row per exposure SNP; ``dropped_reason`` is NaN for usable instruments."""
    x, y = _upper(exposure), _upper(outcome)
    m = x.merge(y, on="snp", how="left", suffixes=("_exp", "_out"))
    signs, reasons = [], []
    for r in m.itertuples(index=False):
        if pd.isna(r.beta_out):
            signs.append(0)
            reasons.append("missing_in_outcome")
            continue
        s, why = _align(
            r.effect_allele_exp,
            r.other_allele_exp,
            r.eaf_exp,
            r.effect_allele_out,
            r.other_allele_out,
            r.eaf_out,
        )
        signs.append(s)
        reasons.append(why)
    signs = np.asarray(signs, float)
    m["beta_out"] = m["beta_out"] * np.where(signs == 0, 1, signs)
    m["eaf_out"] = np.where(signs == -1, 1 - m["eaf_out"], m["eaf_out"])
    m["dropped_reason"] = pd.Series(reasons, index=m.index, dtype=object)
    return m[
        [
            "snp",
            "effect_allele_exp",
            "other_allele_exp",
            "eaf_exp",
            "beta_exp",
            "se_exp",
            "eaf_out",
            "beta_out",
            "se_out",
            "dropped_reason",
        ]
    ].rename(columns={"effect_allele_exp": "effect_allele", "other_allele_exp": "other_allele"})
