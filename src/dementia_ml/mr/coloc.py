"""Approximate Bayes factor colocalisation (Giambartolomei et al. 2014, PLoS Genet,
doi:10.1371/journal.pgen.1004383), as in the R package ``coloc`` (``coloc.abf``).

Assumes at most one causal variant per trait in the region. Hypotheses:
H0 no association; H1 trait 1 only; H2 trait 2 only; H3 both, different variants;
H4 both, one shared variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def log_abf(beta: np.ndarray, se: np.ndarray, prior_sd: float) -> np.ndarray:
    """Wakefield (2009) log approximate Bayes factor per SNP."""
    v = np.asarray(se, float) ** 2
    w = prior_sd**2
    r = w / (w + v)
    z = np.asarray(beta, float) / np.asarray(se, float)
    return 0.5 * (np.log(1 - r) + r * z**2)


def _logdiff(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) for a > b."""
    return a + np.log1p(-np.exp(b - a))


def posterior_from_lbf(
    l1: np.ndarray, l2: np.ndarray, p1: float = 1e-4, p2: float = 1e-4, p12: float = 1e-5
) -> np.ndarray:
    """PP.H0..H4 from per-SNP log Bayes factors of two traits (coloc.bf_bf)."""
    s1, s2, s12 = logsumexp(l1), logsumexp(l2), logsumexp(l1 + l2)
    lh = np.array(
        [
            0.0,
            np.log(p1) + s1,
            np.log(p2) + s2,
            np.log(p1) + np.log(p2) + _logdiff(s1 + s2, s12),
            np.log(p12) + s12,
        ]
    )
    return np.exp(lh - logsumexp(lh))


def coloc_abf(
    df: pd.DataFrame,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
    sd1: float = 0.15,
    sd2: float = 0.2,
) -> dict:
    """``df`` has beta1, se1 (trait 1, e.g. a protein) and beta2, se2 (trait 2, e.g. AD log-odds).

    ``sd1`` = 0.15 is coloc's default for a quantitative trait, ``sd2`` = 0.2 for a case-control
    log-odds ratio.
    """
    l1 = np.asarray(log_abf(df["beta1"], df["se1"], sd1))
    l2 = np.asarray(log_abf(df["beta2"], df["se2"], sd2))
    pp = posterior_from_lbf(l1, l2, p1, p2, p12)
    out = {f"PP.H{i}": float(p) for i, p in enumerate(pp)}
    out["n_snps"] = len(df)
    out["top_shared_snp_index"] = int(np.argmax(l1 + l2))
    return out


def merge_region(pqtl: pd.DataFrame, gwas: pd.DataFrame) -> pd.DataFrame:
    """Join a UKB-PPP REGENIE region (CHROM, GENPOS, ALLELE0, ALLELE1, BETA, SE) with the
    harmonised AD GWAS (hm_chrom, hm_pos, hm_other_allele, hm_effect_allele, hm_beta,
    standard_error) on position and allele pair, both in GRCh38.

    Allele orientation does not matter for ABF colocalisation (it uses z squared), so only the
    unordered allele pair has to match.
    """
    p = pqtl.assign(
        chrom=pqtl["CHROM"].astype(str),
        pos=pqtl["GENPOS"].astype(int),
        pair=[frozenset((a, b)) for a, b in zip(pqtl["ALLELE0"], pqtl["ALLELE1"], strict=True)],
        beta1=pqtl["BETA"].astype(float),
        se1=pqtl["SE"].astype(float),
    )[["chrom", "pos", "pair", "beta1", "se1"]]
    g = gwas.assign(
        chrom=gwas["hm_chrom"].astype(str),
        pos=pd.to_numeric(gwas["hm_pos"], errors="coerce"),
        pair=[
            frozenset((a, b))
            for a, b in zip(gwas["hm_other_allele"], gwas["hm_effect_allele"], strict=True)
        ],
        beta2=pd.to_numeric(gwas["hm_beta"], errors="coerce"),
        se2=pd.to_numeric(gwas["standard_error"], errors="coerce"),
    ).dropna(subset=["pos", "beta2", "se2"])[["chrom", "pos", "pair", "beta2", "se2"]]
    g["pos"] = g["pos"].astype(int)
    m = p.merge(g, on=["chrom", "pos"])
    m = m[m["pair_x"] == m["pair_y"]].drop(columns=["pair_x", "pair_y"])
    return m.drop_duplicates(["chrom", "pos"]).sort_values("pos").reset_index(drop=True)


def wald_for_variant(row: pd.Series):
    """Wald ratio at one variant: UKB-PPP effect (per ALLELE1) and AD effect aligned to ALLELE1."""
    from dementia_ml.mr.estimators import wald_ratio

    sign = 1.0 if row["hm_effect_allele"] == row["ALLELE1"] else -1.0
    return wald_ratio(row["BETA"], row["SE"], sign * row["hm_beta"], row["standard_error"])
