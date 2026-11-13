import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dementia_ml.fusion import model


def _blocks_data(n=500, n_omics=40, seed=0, omics_missing=0.3):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    clin = pd.DataFrame({"AGE": rng.normal(73, 7, n), "MMSE": rng.normal(27, 2, n)})
    omics = pd.DataFrame({f"O{i}": 0.8 * latent + rng.normal(size=n) for i in range(n_omics)})
    missing = rng.random(n) < omics_missing
    omics.loc[missing, :] = np.nan
    X = pd.concat([clin, omics], axis=1)
    X.index = pd.Index(np.arange(n) + 1, name="RID")
    hazard = np.exp(1.0 * latent + 0.02 * (clin["AGE"] - 73))
    t = rng.exponential(60 / hazard)
    c = rng.uniform(12, 96, n)
    blocks = {"clinical": ["AGE", "MMSE"], "omics": [f"O{i}" for i in range(n_omics)]}
    return X, np.minimum(t, c), t <= c, blocks


def test_pca_is_fitted_on_training_rows_only():
    X, _, _, blocks = _blocks_data()
    train, test = X.iloc[:300], X.iloc[300:]
    pre = model.BlockPreprocessor(blocks, n_components={"omics": 3}).fit(train)
    present = train[blocks["omics"]].dropna()
    filled = present.fillna(present.median())
    ref = PCA(3, random_state=0).fit(StandardScaler().fit_transform(filled))
    assert np.allclose(np.abs(pre.pca_["omics"].components_), np.abs(ref.components_), atol=1e-6)
    a = pre.transform(test)
    b = pre.transform(test.assign(O0=test["O0"] + 100))
    assert not np.allclose(a["omics_pc1"], b["omics_pc1"])  # transform applies, does not refit


def test_missing_block_gets_indicator_and_zero_components():
    X, _, _, blocks = _blocks_data()
    pre = model.BlockPreprocessor(blocks, n_components={"omics": 3}).fit(X)
    out = pre.transform(X)
    no_omics = X[blocks["omics"]].isna().all(axis=1)
    assert (out.loc[no_omics, "has_omics"] == 0).all() and (
        out.loc[~no_omics, "has_omics"] == 1
    ).all()
    assert (out.loc[no_omics, ["omics_pc1", "omics_pc2", "omics_pc3"]] == 0).all().all()
    assert not out.isna().any().any()


def test_paired_bootstrap_delta_detects_better_risk():
    rng = np.random.default_rng(0)
    n = 400
    risk_true = rng.normal(size=n)
    time = rng.exponential(np.exp(-risk_true)) * 30
    event = np.ones(n, dtype=bool)
    d = model.paired_delta(time, event, risk_true, rng.normal(size=n), n_boot=300)
    assert d["ci_low"] > 0 and d["estimate"] > 0.1


def test_fusion_with_informative_block_beats_clinical():
    X, time, event, blocks = _blocks_data(n=700)
    combos = {"clinical": ["clinical"], "clinical+omics": ["clinical", "omics"]}
    res = model.evaluate(
        X,
        time,
        event,
        blocks,
        combos,
        n_components={"omics": 3},
        kinds=("cox",),
        n_splits=4,
        n_boot=100,
        reference="clinical",
    )
    c_clin = res["clinical"]["cox"]["c_index"]["estimate"]
    c_fused = res["clinical+omics"]["cox"]["c_index"]["estimate"]
    assert c_fused > c_clin + 0.05
    assert res["clinical+omics"]["cox"]["delta_vs_reference"]["ci_low"] > 0
