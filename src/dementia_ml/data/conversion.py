"""Build a leakage-safe survival dataset for MCI-to-dementia conversion.

Population: participants with baseline diagnosis EMCI or LMCI.
Event: first visit whose diagnosis is Dementia (later reversions are ignored).
Censoring: last visit with a recorded diagnosis.
Features: values from the baseline visit only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dementia_ml.data import schema
from dementia_ml.data.adnimerge import parse_bounds

BASELINE_FEATURES = ["AGE", "MMSE", "APOE4"]
# Raw columns; the training-fold preprocessor derives Hippocampus_z from
# Hippocampus and ICV (see dementia_ml.survival.preprocess).
FULL_FEATURES = BASELINE_FEATURES + [
    "PTEDUCAT",
    "ADAS13",
    "CDRSB",
    "FAQ",
    "Hippocampus",
    "ICV",
    "ABETA",
    "TAU",
    "PTAU",
]


@dataclass
class ConversionData:
    X: pd.DataFrame
    time: np.ndarray
    event: np.ndarray
    meta: dict = field(default_factory=dict)


def build(df: pd.DataFrame, features: list[str]) -> ConversionData:
    schema.validate(df)
    df = parse_bounds(df)
    excluded: Counter[str] = Counter()
    keep_rows, times, events = [], [], []

    for _rid, visits in df.sort_values(["RID", "M"]).groupby("RID", sort=True):
        if visits["DX_bl"].iloc[0] not in schema.DX_BL_MCI:
            excluded["not_mci_at_baseline"] += 1
            continue
        baseline = visits[visits["VISCODE"] == schema.BASELINE_VISCODE]
        if baseline.empty:
            excluded["no_baseline_visit"] += 1
            continue
        follow = visits[(visits["M"] > 0) & visits["DX"].notna()]
        if follow.empty:
            excluded["no_follow_up"] += 1
            continue
        dementia = follow[follow["DX"] == schema.DX_DEMENTIA]
        if dementia.empty:
            times.append(float(follow["M"].iloc[-1]))
            events.append(False)
        else:
            times.append(float(dementia["M"].iloc[0]))
            events.append(True)
        keep_rows.append(baseline.iloc[0])

    base = pd.DataFrame(keep_rows)
    cols = list(dict.fromkeys(features))
    X = base.set_index("RID")[cols] if keep_rows else pd.DataFrame(columns=cols)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X.index = X.index.astype(int)
    return ConversionData(
        X=X,
        time=np.asarray(times, dtype=float),
        event=np.asarray(events, dtype=bool),
        meta={
            "n_subjects": int(df["RID"].nunique()),
            "n_included": len(X),
            "n_events": int(np.sum(events)),
            "excluded": dict(excluded),
        },
    )
