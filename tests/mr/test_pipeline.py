import json

import numpy as np
import pandas as pd

from dementia_ml.mr import run, screen

CAUSAL = {"PROT_A": 0.3, "PROT_B": -0.25, "MET_C": 0.2}


def _fixture(seed=0):
    """3 causal traits and 20 null traits; 1-4 independent instruments each."""
    rng = np.random.default_rng(seed)
    traits = list(CAUSAL) + [f"NULL_{i}" for i in range(20)]
    exp_rows, out_rows, snp_id = [], [], 0
    for t in traits:
        chrom = int(rng.integers(1, 23))
        for j in range(int(rng.integers(1, 5))):
            snp_id += 1
            bx = rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
            eaf = rng.uniform(0.1, 0.4)
            snp = f"rs{snp_id}"
            exp_rows.append(
                [t, snp, "A", "G", eaf, bx, 0.01, 1e-30, True, chrom, 1_000_000 * (j + 1)]
            )
            by = CAUSAL.get(t, 0.0) * bx + rng.normal(0, 0.01)
            out_rows.append([snp, "A", "G", eaf, by, 0.01, 0.5])
    # one weak (non-significant) instrument and one cis-window duplicate that must be pruned
    chrom_a = next(r[9] for r in exp_rows if r[0] == "PROT_A")
    exp_rows.append(["PROT_A", "rs_weak", "A", "G", 0.3, 0.01, 0.01, 0.2, True, chrom_a, 9_000_000])
    exp_rows.append(
        ["PROT_A", "rs_near", "A", "G", 0.3, 0.2, 0.01, 1e-10, True, chrom_a, 1_000_100]
    )
    exp_cols = [
        "trait",
        "snp",
        "effect_allele",
        "other_allele",
        "eaf",
        "beta",
        "se",
        "pval",
        "cis",
        "chrom",
        "pos",
    ]
    out_cols = ["snp", "effect_allele", "other_allele", "eaf", "beta", "se", "pval"]
    exposures = pd.DataFrame(exp_rows, columns=exp_cols)
    exposures.loc[exposures["trait"].str.startswith("MET"), "cis"] = False
    out_rows += [
        ["rs_weak", "A", "G", 0.3, 0.5, 0.01, 0.5],
        ["rs_near", "A", "G", 0.3, 0.5, 0.01, 0.5],
    ]
    return exposures, pd.DataFrame(out_rows, columns=out_cols)


def test_prune_by_distance_keeps_most_significant():
    exp, _ = _fixture()
    exp = exp[exp["trait"] == "PROT_A"]
    kept = screen.prune_by_distance(exp[exp["pval"] < 5e-8], window_kb=500)
    assert "rs_near" not in set(kept["snp"])
    assert (
        kept.groupby("chrom")["pos"]
        .apply(lambda p: (p.sort_values().diff().dropna() > 500_000).all())
        .all()
    )


def test_pipeline_on_fixtures(tmp_path):
    exp, out = _fixture()
    config = {"p_threshold": 5e-8, "window_kb": 500, "fdr": 0.05, "n_boot": 100, "seed": 0}
    res = run.run(
        {"pqtl": exp[exp["cis"]], "mqtl": exp[~exp["cis"]]},
        out,
        config,
        out_root=tmp_path,
        dataset_names={"pqtl": "fixture", "mqtl": "fixture", "outcome": "fixture"},
    )
    csv = pd.read_csv(tmp_path / "results/open/mr_candidates.csv")
    expected_cols = {
        "trait",
        "kind",
        "method",
        "beta",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "q_fdr",
        "n_snps",
        "egger_intercept_p",
    }
    assert expected_cols <= set(csv.columns)
    hits = set(csv.loc[csv["q_fdr"] < 0.05, "trait"])
    assert set(CAUSAL) <= hits
    assert len(hits - set(CAUSAL)) <= 1
    a = csv.set_index("trait").loc["PROT_A"]
    assert "rs_weak" not in res["instruments"]["PROT_A"]
    assert abs(a["beta"] - 0.3) < 0.05
    prov = json.loads((tmp_path / "results/open/mr_candidates.json").read_text())["provenance"]
    assert prov["dataset"] == {"pqtl": "fixture", "mqtl": "fixture", "outcome": "fixture"}
    selected = (tmp_path / "results/open/mr_candidates_selected.txt").read_text().split()
    assert set(CAUSAL) <= set(selected)
    assert (tmp_path / "figures/open/mr_volcano.png").exists()
    assert (tmp_path / "figures/open/mr_top_forest.png").exists()


def test_excluded_region_instruments_are_dropped():
    exp, out = _fixture()
    chrom_a = exp.loc[exp["trait"] == "PROT_A", "chrom"].iloc[0]
    res, inst, log = screen.run_screen(
        exp[exp["cis"]],
        out,
        "pqtl",
        exclude_regions=[{"chrom": int(chrom_a), "start": 0, "end": 10_000_000}],
    )
    assert "PROT_A" not in inst
    assert (log.loc[log["trait"] == "PROT_A", "dropped_reason"] == "excluded_region").all()
