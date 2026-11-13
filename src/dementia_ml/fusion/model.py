"""Multimodal fusion for MCI-to-dementia conversion with missing modalities.

Each block (clinical, MRI, CSF, proteomics, metabolomics) is median-imputed and scaled; the
high-dimensional blocks are reduced with PCA. Everything is fitted on the training fold only.
Participants without a block get zeros for it and a ``has_<block> = 0`` indicator, so one
model serves the whole cohort.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from dementia_ml.eval import bootstrap, metrics
from dementia_ml.splits import subject_folds


class BlockPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, blocks: dict[str, list[str]], n_components: dict[str, int] | None = None):
        self.blocks = blocks
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y=None):
        n_comp = self.n_components or {}
        self.median_, self.mean_, self.std_, self.pca_, self.indicator_ = {}, {}, {}, {}, {}
        for name, cols in self.blocks.items():
            block = X[cols]
            present = block.notna().any(axis=1)
            self.indicator_[name] = bool((~present).any())
            rows = block[present]
            med = rows.median()
            filled = rows.fillna(med)
            mean, std = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
            self.median_[name], self.mean_[name], self.std_[name] = med, mean, std
            if name in n_comp:
                k = min(n_comp[name], filled.shape[1], max(len(filled) - 1, 1))
                self.pca_[name] = PCA(k, random_state=0).fit(((filled - mean) / std).to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for name, cols in self.blocks.items():
            block = X[cols]
            present = block.notna().any(axis=1).to_numpy()
            z = ((block.fillna(self.median_[name]) - self.mean_[name]) / self.std_[name]).to_numpy()
            if name in self.pca_:
                values = self.pca_[name].transform(z)
                names = [f"{name}_pc{i + 1}" for i in range(values.shape[1])]
            else:
                values, names = z, list(cols)
            values = np.where(present[:, None], values, 0.0)
            part = pd.DataFrame(values, index=X.index, columns=names)
            if self.indicator_[name]:
                part[f"has_{name}"] = present.astype(float)
            parts.append(part)
        return pd.concat(parts, axis=1)


def make_model(kind: str, blocks, n_components, seed: int = 0) -> Pipeline:
    if kind == "cox":
        est = CoxPHSurvivalAnalysis(alpha=1.0)
    elif kind == "gbsa":
        est = GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed
        )
    else:
        raise ValueError(kind)
    return Pipeline([("blocks", BlockPreprocessor(blocks, n_components)), ("model", est)])


def oof_risk(X, time, event, blocks, n_components, kind, n_splits=5, seed=0) -> np.ndarray:
    y = Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float))
    risk = np.full(len(X), np.nan)
    for train, test in subject_folds(X.index.to_numpy(), n_splits, stratify=event, seed=seed):
        pipe = make_model(kind, blocks, n_components, seed).fit(X.iloc[train], y[train])
        risk[test] = pipe.predict(X.iloc[test])
    return risk


def paired_delta(time, event, risk_a, risk_b, n_boot: int = 1000, seed: int = 0) -> dict:
    """C-index of ``risk_a`` minus C-index of ``risk_b`` on the same bootstrap resamples."""
    t, e = np.asarray(time, float), np.asarray(event, bool)

    def diff(i):
        return metrics.c_index(t[i], e[i], risk_a[i]) - metrics.c_index(t[i], e[i], risk_b[i])

    est, lo, hi = bootstrap.ci(diff, n=len(t), n_boot=n_boot, seed=seed)
    return {"estimate": est, "ci_low": lo, "ci_high": hi}


def evaluate(
    X: pd.DataFrame,
    time,
    event,
    blocks: dict[str, list[str]],
    combos: dict[str, Sequence[str]],
    n_components: dict[str, int] | None = None,
    kinds: Sequence[str] = ("cox", "gbsa"),
    n_splits: int = 5,
    n_boot: int = 1000,
    reference: str = "clinical",
    seed: int = 0,
) -> dict:
    t, e = np.asarray(time, float), np.asarray(event, bool)
    n = len(t)
    risks: dict[tuple[str, str], np.ndarray] = {}
    out: dict = {}
    for combo, names in combos.items():
        sub_blocks = {b: blocks[b] for b in names}
        comp = {b: k for b, k in (n_components or {}).items() if b in sub_blocks}
        out[combo] = {}
        for kind in kinds:
            r = oof_risk(X, t, e, sub_blocks, comp, kind, n_splits, seed)
            risks[(combo, kind)] = r
            c = bootstrap.ci(
                lambda i, r=r: metrics.c_index(t[i], e[i], r[i]), n=n, n_boot=n_boot, seed=seed
            )
            entry = {
                "blocks": list(names),
                "c_index": {"estimate": c[0], "ci_low": c[1], "ci_high": c[2]},
                "n": n,
                "n_events": int(e.sum()),
            }
            if t.max() > 36.0:
                entry["td_auc_36"] = float(metrics.td_auc(t, e, t, e, r, [36.0])[0])
            out[combo][kind] = entry
    for (combo, kind), r in risks.items():
        if combo != reference and (reference, kind) in risks:
            out[combo][kind]["delta_vs_reference"] = paired_delta(
                t, e, r, risks[(reference, kind)], n_boot=n_boot, seed=seed
            )
    return out
