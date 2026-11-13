import numpy as np
import pandas as pd
import pytest

from dementia_ml.mr import coloc


def _region(n=300, causal1=150, causal2=150, b1=0.15, b2=0.08, se1=0.01, se2=0.012, seed=0):
    """Two traits over n SNPs with an AR(1) LD-like correlation between neighbours."""
    rng = np.random.default_rng(seed)
    pos = np.arange(n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    beta1 = se1 * z1
    beta2 = se2 * z2
    decay = 0.7 ** np.abs(pos - causal1)
    beta1 = beta1 + b1 * decay
    if causal2 is not None:
        beta2 = beta2 + b2 * 0.7 ** np.abs(pos - causal2)
    return pd.DataFrame({"beta1": beta1, "se1": se1, "beta2": beta2, "se2": se2})


def test_shared_causal_variant_supports_h4():
    pp = coloc.coloc_abf(_region(causal2=150))
    assert pp["PP.H4"] > 0.8
    assert sum(pp[f"PP.H{i}"] for i in range(5)) == pytest.approx(1.0)


def test_distinct_causal_variants_support_h3():
    pp = coloc.coloc_abf(_region(causal1=80, causal2=220))
    assert pp["PP.H3"] > 0.8


def test_signal_in_one_trait_only_supports_h1():
    pp = coloc.coloc_abf(_region(causal2=None))
    assert pp["PP.H1"] > 0.8


def test_log_abf_is_wakefield():
    # Wakefield (2009): log ABF = 0.5 * (log(1 - r) + r z^2), r = W / (W + V)
    labf = coloc.log_abf(np.array([0.1]), np.array([0.02]), prior_sd=0.15)
    v, w = 0.02**2, 0.15**2
    r = w / (w + v)
    z = 0.1 / 0.02
    assert labf[0] == pytest.approx(0.5 * (np.log(1 - r) + r * z**2))


def test_merge_region_matches_position_and_allele_set():
    pq = pd.DataFrame(
        {
            "CHROM": [17, 17, 17],
            "GENPOS": [100, 200, 300],
            "ALLELE0": ["A", "C", "G"],
            "ALLELE1": ["G", "T", "A"],
            "BETA": [0.1, 0.2, 0.3],
            "SE": [0.01, 0.02, 0.03],
        }
    )
    gw = pd.DataFrame(
        {
            "hm_chrom": ["17", "17", "17"],
            "hm_pos": [100, 200, 300],
            "hm_other_allele": ["G", "C", "T"],
            "hm_effect_allele": ["A", "T", "C"],
            "hm_beta": [0.05, 0.06, 0.07],
            "standard_error": [0.01, 0.01, 0.01],
        }
    )
    m = coloc.merge_region(pq, gw)
    assert m["pos"].tolist() == [100, 200]  # 300: allele set G/A vs T/C does not match
    assert {"beta1", "se1", "beta2", "se2"} <= set(m.columns)
